//===- HotColdSplitting.cpp -- Outline Cold Regions -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// The goal of hot/cold splitting is to improve the memory locality of code.
/// The splitting pass does this by identifying cold blocks and moving them into
/// separate functions.
///
/// When the splitting pass finds a cold block (referred to as "the sink"), it
/// grows a maximal cold region around that block. The maximal region contains
/// all blocks (post-)dominated by the sink [*]. In theory, these blocks are as
/// cold as the sink. Once a region is found, it's split out of the original
/// function provided it's profitable to do so.
///
/// [*] In practice, there is some added complexity because some blocks are not
/// safe to extract.
///
/// TODO: Use the PM to get domtrees, and preserve BFI/BPI.
/// TODO: Reorder outlined functions.
///
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/HotColdSplitting.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/BlockFrequency.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <algorithm>
#include <cassert>
#include <string>
#include <set>
#include <cmath>

#define DEBUG_TYPE "hotcoldsplit"

STATISTIC(NumColdRegionsFound, "Number of cold regions found.");
STATISTIC(NumColdRegionsOutlined, "Number of cold regions outlined.");

using namespace llvm;

static cl::opt<bool> EnableStaticAnalyis("hot-cold-static-analysis",
                              cl::init(true), cl::Hidden);

static cl::opt<int>
    SplittingThreshold("hotcoldsplit-threshold", cl::init(2), cl::Hidden,
                       cl::desc("Base penalty for splitting cold code (as a "
                                "multiple of TCC_Basic)"));

static cl::opt<bool> EnableColdSection(
    "enable-cold-section", cl::init(false), cl::Hidden,
    cl::desc("Enable placement of extracted cold functions"
             " into a separate section after hot-cold splitting."));

static cl::opt<std::string>
    ColdSectionName("hotcoldsplit-cold-section-name", cl::init("__llvm_cold"),
                    cl::Hidden,
                    cl::desc("Name for the section containing cold functions "
                             "extracted by hot-cold splitting."));

namespace {
// Same as blockEndsInUnreachable in CodeGen/BranchFolding.cpp. Do not modify
// this function unless you modify the MBB version as well.
//
/// A no successor, non-return block probably ends in unreachable and is cold.
/// Also consider a block that ends in an indirect branch to be a return block,
/// since many targets use plain indirect branches to return.
bool blockEndsInUnreachable(const BasicBlock &BB) {
  if (!succ_empty(&BB))
    return false;
  if (BB.empty())
    return true;
  const Instruction *I = BB.getTerminator();
  return !(isa<ReturnInst>(I) || isa<IndirectBrInst>(I));
}

bool unlikelyExecuted(BasicBlock &BB, ProfileSummaryInfo *PSI,
                      BlockFrequencyInfo *BFI) {
  // Exception handling blocks are unlikely executed.
  if (BB.isEHPad() || isa<ResumeInst>(BB.getTerminator()))
    return true;

  // The block is cold if it calls/invokes a cold function. However, do not
  // mark sanitizer traps as cold.
  for (Instruction &I : BB)
    if (auto *CB = dyn_cast<CallBase>(&I))
      if (CB->hasFnAttr(Attribute::Cold) && !CB->getMetadata("nosanitize"))
        return true;

  // The block is cold if it has an unreachable terminator, unless it's
  // preceded by a call to a (possibly warm) noreturn call (e.g. longjmp);
  // in the case of a longjmp, if the block is cold according to
  // profile information, we mark it as unlikely to be executed as well.
  if (blockEndsInUnreachable(BB)) {
    if (auto *CI =
            dyn_cast_or_null<CallInst>(BB.getTerminator()->getPrevNode()))
      if (CI->hasFnAttr(Attribute::NoReturn)) {
        if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(CI))
          return (II->getIntrinsicID() != Intrinsic::eh_sjlj_longjmp) ||
                 (BFI && PSI->isColdBlock(&BB, BFI));
        return !CI->getCalledFunction()->getName().contains("longjmp") ||
               (BFI && PSI->isColdBlock(&BB, BFI));
      }
    return true;
  }

  return false;
}

/// Check whether it's safe to outline \p BB.
static bool mayExtractBlock(const BasicBlock &BB) {
  // EH pads are unsafe to outline because doing so breaks EH type tables. It
  // follows that invoke instructions cannot be extracted, because CodeExtractor
  // requires unwind destinations to be within the extraction region.
  //
  // Resumes that are not reachable from a cleanup landing pad are considered to
  // be unreachable. It’s not safe to split them out either.
  auto Term = BB.getTerminator();
  return !BB.hasAddressTaken() && !BB.isEHPad() && !isa<InvokeInst>(Term) &&
         !isa<ResumeInst>(Term);
}

/// Mark \p F cold. Based on this assumption, also optimize it for minimum size.
/// If \p UpdateEntryCount is true (set when this is a new split function and
/// module has profile data), set entry count to 0 to ensure treated as cold.
/// Return true if the function is changed.
static bool markFunctionCold(Function &F, bool UpdateEntryCount = false) {
  assert(!F.hasOptNone() && "Can't mark this cold");
  bool Changed = false;
  if (!F.hasFnAttribute(Attribute::Cold)) {
    F.addFnAttr(Attribute::Cold);
    Changed = true;
  }
  if (!F.hasFnAttribute(Attribute::MinSize)) {
    F.addFnAttr(Attribute::MinSize);
    Changed = true;
  }
  if (UpdateEntryCount) {
    // Set the entry count to 0 to ensure it is placed in the unlikely text
    // section when function sections are enabled.
    F.setEntryCount(0);
    Changed = true;
  }

  return Changed;
}

class HotColdSplittingLegacyPass : public ModulePass {
public:
  static char ID;
  HotColdSplittingLegacyPass() : ModulePass(ID) {
    initializeHotColdSplittingLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<BlockFrequencyInfoWrapperPass>();
    AU.addRequired<ProfileSummaryInfoWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addUsedIfAvailable<AssumptionCacheTracker>();
  }

  bool runOnModule(Module &M) override;
};

} // end anonymous namespace

/// Check whether \p F is inherently cold.
bool HotColdSplitting::isFunctionCold(const Function &F) const {
  if (F.hasFnAttribute(Attribute::Cold))
    return true;

  if (F.getCallingConv() == CallingConv::Cold)
    return true;

  if (PSI->isFunctionEntryCold(&F))
    return true;

  return false;
}

// Returns false if the function should not be considered for hot-cold split
// optimization.
bool HotColdSplitting::shouldOutlineFrom(const Function &F) const {
  if (F.hasFnAttribute(Attribute::AlwaysInline))
    return false;

  if (F.hasFnAttribute(Attribute::NoInline))
    return false;

  // A function marked `noreturn` may contain unreachable terminators: these
  // should not be considered cold, as the function may be a trampoline.
  if (F.hasFnAttribute(Attribute::NoReturn))
    return false;

  if (F.hasFnAttribute(Attribute::SanitizeAddress) ||
      F.hasFnAttribute(Attribute::SanitizeHWAddress) ||
      F.hasFnAttribute(Attribute::SanitizeThread) ||
      F.hasFnAttribute(Attribute::SanitizeMemory))
    return false;

  return true;
}

/// Get the benefit score of outlining \p Region.
static int getOutliningBenefit(ArrayRef<BasicBlock *> Region,
                               TargetTransformInfo &TTI) {
  // Sum up the code size costs of non-terminator instructions. Tight coupling
  // with \ref getOutliningPenalty is needed to model the costs of terminators.
  int Benefit = 0;
  for (BasicBlock *BB : Region)
    for (Instruction &I : BB->instructionsWithoutDebug())
      if (&I != BB->getTerminator())
        Benefit +=
            TTI.getInstructionCost(&I, TargetTransformInfo::TCK_CodeSize);

  return Benefit;
}

/// Get the penalty score for outlining \p Region.
static int getOutliningPenalty(ArrayRef<BasicBlock *> Region,
                               unsigned NumInputs, unsigned NumOutputs) {
  int Penalty = SplittingThreshold;
  LLVM_DEBUG(dbgs() << "Applying penalty for splitting: " << Penalty << "\n");

  // If the splitting threshold is set at or below zero, skip the usual
  // profitability check.
  if (SplittingThreshold <= 0)
    return Penalty;

  // The typical code size cost for materializing an argument for the outlined
  // call.
  LLVM_DEBUG(dbgs() << "Applying penalty for: " << NumInputs << " inputs\n");
  const int CostForArgMaterialization = TargetTransformInfo::TCC_Basic;
  Penalty += CostForArgMaterialization * NumInputs;

  // The typical code size cost for an output alloca, its associated store, and
  // its associated reload.
  LLVM_DEBUG(dbgs() << "Applying penalty for: " << NumOutputs << " outputs\n");
  const int CostForRegionOutput = 3 * TargetTransformInfo::TCC_Basic;
  Penalty += CostForRegionOutput * NumOutputs;

  // Find the number of distinct exit blocks for the region. Use a conservative
  // check to determine whether control returns from the region.
  bool NoBlocksReturn = true;
  SmallPtrSet<BasicBlock *, 2> SuccsOutsideRegion;
  for (BasicBlock *BB : Region) {
    // If a block has no successors, only assume it does not return if it's
    // unreachable.
    if (succ_empty(BB)) {
      NoBlocksReturn &= isa<UnreachableInst>(BB->getTerminator());
      continue;
    }

    for (BasicBlock *SuccBB : successors(BB)) {
      if (find(Region, SuccBB) == Region.end()) {
        NoBlocksReturn = false;
        SuccsOutsideRegion.insert(SuccBB);
      }
    }
  }

  // Apply a `noreturn` bonus.
  if (NoBlocksReturn) {
    LLVM_DEBUG(dbgs() << "Applying bonus for: " << Region.size()
                      << " non-returning terminators\n");
    Penalty -= Region.size();
  }

  // Apply a penalty for having more than one successor outside of the region.
  // This penalty accounts for the switch needed in the caller.
  if (!SuccsOutsideRegion.empty()) {
    LLVM_DEBUG(dbgs() << "Applying penalty for: " << SuccsOutsideRegion.size()
                      << " non-region successors\n");
    Penalty += (SuccsOutsideRegion.size() - 1) * TargetTransformInfo::TCC_Basic;
  }

  return Penalty;
}

Function *HotColdSplitting::extractColdRegion(
    const BlockSequence &Region, const CodeExtractorAnalysisCache &CEAC,
    DominatorTree &DT, BlockFrequencyInfo *BFI, TargetTransformInfo &TTI,
    OptimizationRemarkEmitter &ORE, AssumptionCache *AC, unsigned Count) {
  assert(!Region.empty());

  // TODO: Pass BFI and BPI to update profile information.
  CodeExtractor CE(Region, &DT, /* AggregateArgs */ false, /* BFI */ nullptr,
                   /* BPI */ nullptr, AC, /* AllowVarArgs */ false,
                   /* AllowAlloca */ false,
                   /* Suffix */ "cold." + std::to_string(Count));

  // Perform a simple cost/benefit analysis to decide whether or not to permit
  // splitting.
  SetVector<Value *> Inputs, Outputs, Sinks;
  CE.findInputsOutputs(Inputs, Outputs, Sinks);
  int OutliningBenefit = getOutliningBenefit(Region, TTI);
  int OutliningPenalty =
      getOutliningPenalty(Region, Inputs.size(), Outputs.size());
  LLVM_DEBUG(dbgs() << "Split profitability: benefit = " << OutliningBenefit
                    << ", penalty = " << OutliningPenalty << "\n");
  if (OutliningBenefit <= OutliningPenalty)
    return nullptr;

  Function *OrigF = Region[0]->getParent();
  if (Function *OutF = CE.extractCodeRegion(CEAC)) {
    User *U = *OutF->user_begin();
    CallInst *CI = cast<CallInst>(U);
    NumColdRegionsOutlined++;
    if (TTI.useColdCCForColdCall(*OutF)) {
      OutF->setCallingConv(CallingConv::Cold);
      CI->setCallingConv(CallingConv::Cold);
    }
    CI->setIsNoInline();

    if (EnableColdSection)
      OutF->setSection(ColdSectionName);
    else {
      if (OrigF->hasSection())
        OutF->setSection(OrigF->getSection());
    }

    markFunctionCold(*OutF, BFI != nullptr);

    LLVM_DEBUG(llvm::dbgs() << "Outlined Region: " << *OutF);
    ORE.emit([&]() {
      return OptimizationRemark(DEBUG_TYPE, "HotColdSplit",
                                &*Region[0]->begin())
             << ore::NV("Original", OrigF) << " split cold code into "
             << ore::NV("Split", OutF);
    });
    return OutF;
  }

  ORE.emit([&]() {
    return OptimizationRemarkMissed(DEBUG_TYPE, "ExtractFailed",
                                    &*Region[0]->begin())
           << "Failed to extract region at block "
           << ore::NV("Block", Region.front());
  });
  return nullptr;
}

/// A pair of (basic block, score).
using BlockTy = std::pair<BasicBlock *, unsigned>;

namespace {
/// A maximal outlining region. This contains all blocks post-dominated by a
/// sink block, the sink block itself, and all blocks dominated by the sink.
/// If sink-predecessors and sink-successors cannot be extracted in one region,
/// the static constructor returns a list of suitable extraction regions.
class OutliningRegion {
  /// A list of (block, score) pairs. A block's score is non-zero iff it's a
  /// viable sub-region entry point. Blocks with higher scores are better entry
  /// points (i.e. they are more distant ancestors of the sink block).
  SmallVector<BlockTy, 0> Blocks = {};

  /// The suggested entry point into the region. If the region has multiple
  /// entry points, all blocks within the region may not be reachable from this
  /// entry point.
  BasicBlock *SuggestedEntryPoint = nullptr;

  /// Whether the entire function is cold.
  bool EntireFunctionCold = false;

  /// If \p BB is a viable entry point, return \p Score. Return 0 otherwise.
  static unsigned getEntryPointScore(BasicBlock &BB, unsigned Score) {
    return mayExtractBlock(BB) ? Score : 0;
  }

  /// These scores should be lower than the score for predecessor blocks,
  /// because regions starting at predecessor blocks are typically larger.
  static constexpr unsigned ScoreForSuccBlock = 1;
  static constexpr unsigned ScoreForSinkBlock = 1;

  OutliningRegion(const OutliningRegion &) = delete;
  OutliningRegion &operator=(const OutliningRegion &) = delete;

public:
  OutliningRegion() = default;
  OutliningRegion(OutliningRegion &&) = default;
  OutliningRegion &operator=(OutliningRegion &&) = default;

  static std::vector<OutliningRegion> create(BasicBlock &SinkBB,
                                             const DominatorTree &DT,
                                             const PostDominatorTree &PDT) {
    std::vector<OutliningRegion> Regions;
    SmallPtrSet<BasicBlock *, 4> RegionBlocks;

    Regions.emplace_back();
    OutliningRegion *ColdRegion = &Regions.back();

    auto addBlockToRegion = [&](BasicBlock *BB, unsigned Score) {
      RegionBlocks.insert(BB);
      ColdRegion->Blocks.emplace_back(BB, Score);
    };

    // The ancestor farthest-away from SinkBB, and also post-dominated by it.
    unsigned SinkScore = getEntryPointScore(SinkBB, ScoreForSinkBlock);
    ColdRegion->SuggestedEntryPoint = (SinkScore > 0) ? &SinkBB : nullptr;
    unsigned BestScore = SinkScore;

    // Visit SinkBB's ancestors using inverse DFS.
    auto PredIt = ++idf_begin(&SinkBB);
    auto PredEnd = idf_end(&SinkBB);
    while (PredIt != PredEnd) {
      BasicBlock &PredBB = **PredIt;
      bool SinkPostDom = PDT.dominates(&SinkBB, &PredBB);

      // If the predecessor is cold and has no predecessors, the entire
      // function must be cold.
      if (SinkPostDom && pred_empty(&PredBB)) {
        ColdRegion->EntireFunctionCold = true;
        return Regions;
      }

      // If SinkBB does not post-dominate a predecessor, do not mark the
      // predecessor (or any of its predecessors) cold.
      if (!SinkPostDom || !mayExtractBlock(PredBB)) {
        PredIt.skipChildren();
        continue;
      }

      // Keep track of the post-dominated ancestor farthest away from the sink.
      // The path length is always >= 2, ensuring that predecessor blocks are
      // considered as entry points before the sink block.
      unsigned PredScore = getEntryPointScore(PredBB, PredIt.getPathLength());
      if (PredScore > BestScore) {
        ColdRegion->SuggestedEntryPoint = &PredBB;
        BestScore = PredScore;
      }

      addBlockToRegion(&PredBB, PredScore);
      ++PredIt;
    }

    // If the sink can be added to the cold region, do so. It's considered as
    // an entry point before any sink-successor blocks.
    //
    // Otherwise, split cold sink-successor blocks using a separate region.
    // This satisfies the requirement that all extraction blocks other than the
    // first have predecessors within the extraction region.
    if (mayExtractBlock(SinkBB)) {
      addBlockToRegion(&SinkBB, SinkScore);
      if (pred_empty(&SinkBB)) {
        ColdRegion->EntireFunctionCold = true;
        return Regions;
      }
    } else {
      Regions.emplace_back();
      ColdRegion = &Regions.back();
      BestScore = 0;
    }

    // Find all successors of SinkBB dominated by SinkBB using DFS.
    auto SuccIt = ++df_begin(&SinkBB);
    auto SuccEnd = df_end(&SinkBB);
    while (SuccIt != SuccEnd) {
      BasicBlock &SuccBB = **SuccIt;
      bool SinkDom = DT.dominates(&SinkBB, &SuccBB);

      // Don't allow the backwards & forwards DFSes to mark the same block.
      bool DuplicateBlock = RegionBlocks.count(&SuccBB);

      // If SinkBB does not dominate a successor, do not mark the successor (or
      // any of its successors) cold.
      if (DuplicateBlock || !SinkDom || !mayExtractBlock(SuccBB)) {
        SuccIt.skipChildren();
        continue;
      }

      unsigned SuccScore = getEntryPointScore(SuccBB, ScoreForSuccBlock);
      if (SuccScore > BestScore) {
        ColdRegion->SuggestedEntryPoint = &SuccBB;
        BestScore = SuccScore;
      }

      addBlockToRegion(&SuccBB, SuccScore);
      ++SuccIt;
    }

    return Regions;
  }

  /// Whether this region has nothing to extract.
  bool empty() const { return !SuggestedEntryPoint; }

  /// The blocks in this region.
  ArrayRef<std::pair<BasicBlock *, unsigned>> blocks() const { return Blocks; }

  /// Whether the entire function containing this region is cold.
  bool isEntireFunctionCold() const { return EntireFunctionCold; }

  /// Remove a sub-region from this region and return it as a block sequence.
  BlockSequence takeSingleEntrySubRegion(DominatorTree &DT) {
    assert(!empty() && !isEntireFunctionCold() && "Nothing to extract");

    // Remove blocks dominated by the suggested entry point from this region.
    // During the removal, identify the next best entry point into the region.
    // Ensure that the first extracted block is the suggested entry point.
    BlockSequence SubRegion = {SuggestedEntryPoint};
    BasicBlock *NextEntryPoint = nullptr;
    unsigned NextScore = 0;
    auto RegionEndIt = Blocks.end();
    auto RegionStartIt = remove_if(Blocks, [&](const BlockTy &Block) {
      BasicBlock *BB = Block.first;
      unsigned Score = Block.second;
      bool InSubRegion =
          BB == SuggestedEntryPoint || DT.dominates(SuggestedEntryPoint, BB);
      if (!InSubRegion && Score > NextScore) {
        NextEntryPoint = BB;
        NextScore = Score;
      }
      if (InSubRegion && BB != SuggestedEntryPoint)
        SubRegion.push_back(BB);
      return InSubRegion;
    });
    Blocks.erase(RegionStartIt, RegionEndIt);

    // Update the suggested entry point.
    SuggestedEntryPoint = NextEntryPoint;

    return SubRegion;
  }
};
} // namespace

// Edge structure for the undirected
// multigraph version of control-flow graph
// required by the cycle-equivalence algorithm.
struct MultiEdge {
  // Source vertex
  unsigned U;

  // Sink vertex
  unsigned V;

  // Unique edge ID
  unsigned ID;

  // Equivalence class of edge
  unsigned Class;

  // most recent size of stack which
  // this edge is on top.
  unsigned RecentSize;

  // most recent equivalence class.
  unsigned RecentClass;

  // Is this edge a reversed edge 
  bool Rev;

  // Is this edge a capping edge
  bool IsCappingEdge;

  // A reverse pointer into the
  // container carrying this edge inside
  // the linked list of back-edges.
  // There is a one-to-one correspondence
  // between a MultiEdge and a Bracket.
  Bracket *B;
  
  void Init() {
    this->Class = 0;
    this->RecentSize = 0;
    this->RecentClass = 0;
    this->IsCappingEdge = false;
    this->B = nullptr;
  }
  
  MultiEdge() {
    U = 0; V = 0; Rev = false; ID = 0;
    Init();
  }

  MultiEdge(unsigned U, unsigned V) {
    this->U = U;
    this->V = V;
    this->Rev = false;
    this->ID = 0;
    Init();
  }

  MultiEdge(unsigned U, unsigned V, unsigned ID) {
    this->U = U;
    this->V = V;
    this->Rev = false;
    this->ID = ID;
    Init();
  }

  bool operator==(const MultiEdge& RHS) {
    return this->ID == RHS.ID;
  }

  Bracket* getBracket() const {
    assert(this->B != nullptr && "Getting a null bracket from a MultiEdge.");
    return this->B;
  }

  void setBracket(Bracket* B) {
    this->B = B;
  }
};

// A container for backedges inside the
// linked list (BracketList) of back-edges maintained
// by the cycle equivalence algorithm.
struct Bracket {

  // A pointer to the edge we're carrying.
  MultiEdge *E;

  // Next in list.
  Bracket *Next;

  // Previous in list.
  Bracket *Prev;

  Bracket() {
    Next = Prev = nullptr;
  }
  
  Bracket (MultiEdge *E) {
    this->E = E;
    Next = Prev = nullptr;
  }

  Bracket (MultiEdge *E, Bracket *Next, Bracket *Prev) {
    this->E = E;
    this->Next = Next;
    this->Prev = Prev;
  }
};

// A linked list structure of back-edges, required
// by the cycle equivalence algorithm supporting O(1) deletion.
// All back-edges from a vertex u's descendants that reach over u
// are called brackets.
struct BracketList {

  // The topmost element in list.
  Bracket * Top;

  // The bottom-most element in list.
  Bracket * Bottom;

  // The size of the current list.
  size_t Size;

  BracketList() {
    this->Top = nullptr;
    this->Bottom = nullptr;
    this->Size = 0;
  }

  // Polymorphic Push() method. Pushes
  // a heap-allocated bracket onto the list.  
  void Push(Bracket *B) {
    Bracket *T = this->Top;
    this->Top = B;
    B->Prev = nullptr;
    B->Next = T;
    T->Prev = B;
    if (Bottom == nullptr)
      Bottom = B;
  }

  // Polymorphically receive a MultiEdge
  // reference; build its container Bracket
  // on the heap and insert it into the list.
  // A deleted bracket is always de-allocated
  // automatically in the Delete() method.
  void Push(MultiEdge& E) {
    Bracket *B = new Bracket(&E);
    this->Push(B);
    E.setBracket(B);
  }
  
  // Deletes a bracket pointer B in O(1)-time
  // by unlinking it from the list. 
  // Also deallocates its memory since all 
  // bracket structures are heap-allocated.
  void Delete(Bracket *B) {
    if (B->Prev == nullptr) {
      // B is Top node.
      this->Top = B->Next;
      B->Next->Prev = nullptr;
      B->Prev = B->Next = nullptr;      
    } else if (B->Next == nullptr) {
      // B is Bottom node.
      this->Bottom = B->Prev;
      B->Prev->Next = nullptr;
      B->Prev = B->Next = nullptr;
    } else {
      // B is somewhere in the middle.
      B->Next->Prev = B->Prev;
      B->Prev->Next = B->Next;
      B->Prev = B->Next = nullptr;      
    }
    delete B;
  }

  // Concatenates a sublist into this current list.
  // After concatenation, the sublist structures are
  // permanently damaged and cannot be reused.
  void Concat(const BracketList& NextList) {
    // Trivial case.
    if (NextList.Size == 0) return;     
    this->Size += NextList.Size;
    this->Bottom->Next = NextList.Top;
    NextList.Top->Prev = this->Bottom;
    this->Bottom = NextList.Bottom; 
  }
};

// Use DFS to calculate DFS numbers, parent pointers, and back-edges for
// the cycle-equivalence algorithm.
void UndirectedDepthFirstTraversal(const std::vector< std::vector<MultiEdge> >& UGraph,
                                   std::vector< std::vector<MultiEdge> >& BackEdgesInto, 
                                   std::vector<unsigned>& Parent,
                                   std::vector<unsigned>& DFSNum,
                                   std::vector<unsigned>& DFSOrder, 
                                   unsigned& DFSCounter, int U) {
  // Visited?
  if (DFSNum[U] != 0) 
    return;
  
  DFSNum[U] = DFSCounter++;
  DFSOrder.push_back(U);

  // Visit our neighbors.
  for (const MultiEdge &E : UGraph[U]) {
    if (DFSNum[E.V] != 0) {
      // Back edge found. Mark it.
      BackEdgesInto[E.V].push_back(E);
    }
    // Otherwise, visit the unvisited edge.
    Parent[E.V] = E.U;
    UndirectedDepthFirstTraversal(UGraph, BackEdgesInto, Parent, DFSNum, DFSOrder, DFSCounter, E.V);
  }
}

// O(|E|)-time canonical SESE region calculation using cycle-equivalence algorithm.
std::vector< std::set<BasicBlock*> > HotColdSplitting::calculateCanonicalSESERegions(Function &F) {
  // Establish a bijective mapping between BBs in the CFG to natural numbers.
  std::map<BasicBlock *, unsigned> VertexNumbering;
  std::vector<BasicBlock*> VertexNumberToBB;
  // Use the numbering to calculate directed/undirected adjacency lists
  // for the cycle equivalence algorithm.
  std::vector< std::vector<MultiEdge> > DirectedAdjList;
  std::vector< std::vector<MultiEdge> > UndirectedAdjList;
  std::vector<MultiEdge> DirectedEdgeList;
  unsigned VertexNumber = 1;
  unsigned EdgeNumber = 1;
  ReversePostOrderTraversal<Function *> RPOT(&F);
  
  // First, map each BB to numbers.
  for (BasicBlock *BB : RPOT) 
    VertexNumbering[BB] = VertexNumber++;

  // Initialize adjacency list.
  DirectedAdjList.assign(VertexNumber + 1, std::vector<MultiEdge>());
  UndirectedAdjList.assign(VertexNumber + 1, std::vector<MultiEdge>());

  // Next, populate the adjacency list using CFG information.
  for(BasicBlock *BB : RPOT) {
    unsigned SrcVertex = VertexNumbering[BB];
    VertexNumberToBB.push_back(BB);
    for (unsigned I = 0,
         NSucc = BB->getTerminator()->getNumSuccessors(); I < NSucc; ++I) {
      BasicBlock *Succ = BB->getTerminator()->getSuccessor(I);
      unsigned DstVertex = VertexNumbering[Succ];
      DirectedAdjList[SrcVertex].push_back(MultiEdge(SrcVertex, DstVertex, EdgeNumber));
      DirectedEdgeList.push_back(MultiEdge(SrcVertex, DstVertex, EdgeNumber++));
    }
  }

  // Establish source, sink nodes of the CFG.
  // If the CFG itself is SEME, use 0 as a "virtual" terminating block.
  unsigned Source = VertexNumbering[&F.getEntryBlock()], Sink;
  unsigned NumExits = 0;
  for (unsigned I = 1; I < DirectedAdjList.size(); ++I)
    if (DirectedAdjList[I].size() == 0){
      ++NumExits;
      Sink = I;
    } 
  
  assert(NumExits > 0 && "Number of control-flow graph exits must > 0.");
  
  if (NumExits > 1) {
    Sink = 0;
    // Link all exit nodes to 0.
    for (unsigned I = 1; I < DirectedAdjList.size(); ++I)
      if (DirectedAdjList[I].size() == 0){
        DirectedAdjList[I].push_back(MultiEdge(I, Sink, EdgeNumber));
        DirectedEdgeList.push_back(MultiEdge(I, Sink, EdgeNumber++));
      }
  }

  // Now convert the directed graph into an undirected (multi)-graph for finding
  // SESE regions, and add edge from Sink to Source.
  for (MultiEdge& E : DirectedEdgeList) {
    MultiEdge ReverseE = MultiEdge(E.V, E.U, E.ID);
    ReverseE.Rev = true;
    UndirectedAdjList[E.U].push_back(E);
    UndirectedAdjList[E.V].push_back(ReverseE);
  }
  MultiEdge STEdge(Source, Sink, 0);
  MultiEdge TSEdge(Sink, Source, 0);
  TSEdge.Rev = true;
  UndirectedAdjList[Source].push_back(STEdge);
  UndirectedAdjList[Sink].push_back(TSEdge);

  // Set up vertex attributes maintained by our algorithm.
  std::vector<unsigned> UndirectedDFSNum(UndirectedAdjList.size());
  std::vector<BracketList> BList(UndirectedAdjList.size());
  std::vector<unsigned> Hi(UndirectedAdjList.size());
  std::vector<unsigned> Parent(UndirectedAdjList.size());

  // Set up information maintained by DFS traversal of graph.
  std::vector<unsigned> DFSNum(UndirectedAdjList.size());
  std::vector<unsigned> DFSOrder;
  std::vector< std::vector<MultiEdge> > BackEdgesInto(UndirectedAdjList.size());
  // Counter for assigning DFS number to each vertex.
  unsigned DFSCounter = 1;
  // Counter for registering equivalence classes of SESE regions.
  unsigned EquivalenceClassCounter = 1;

  // Do a DFS traversal of the graph to set up
  Parent[Source] = 0;
  UndirectedDepthFirstTraversal(UndirectedAdjList, BackEdgesInto, Parent, DFSNum, DFSOrder, DFSCounter, Source);

  // Calculate SESE regions in reverse depth-first order.
  std::reverse(DFSOrder.begin(), DFSOrder.end());
  for (const unsigned& N : DFSOrder) {
    // Calculate Hi0-2 and HiChild for capping back-edges.
    unsigned Hi0 = 0xFFFFFFFEU, Hi1 = 0xFFFFFFFEU, HiChild, Hi2 = 0xFFFFFFFEU;
    for (const MultiEdge& E : UndirectedAdjList[N]) {
      if (DFSNum[E.V] > DFSNum[E.U] && Hi0 >= DFSNum[E.V]) {
        // Update Hi0 := min{dfsnum(t); (n, t) backedge}
        Hi0 = DFSNum[E.V];
      } else if (DFSNum[E.V] < DFSNum[E.U] && Hi1 >= Hi[E.V]) {
        // V is child of U.
        // This case is always triggered inductively. The base case
        // will never rest in this if-condition, since the base
        // case is a leaf in the DFS spanning tree.
        // Here, update Hi1 = min{c.Hi; c child of N}
        Hi1 = Hi[E.V];
      }
    }

    // Hi of N = min{Hi0, Hi1} 
    if (Hi0 < Hi1) 
      Hi[N] = Hi0;
    else 
      Hi[N] = Hi1;

    // Compute HiChild = any child c of N having Hi[c] = Hi[N]
    for(const MultiEdge& E : UndirectedAdjList[N])
      if (DFSNum[E.V] > DFSNum[E.U] && Hi[E.V] == Hi[E.U]) {
        HiChild = E.V;
        break;        
      }
    
    // Compute Hi2 = min{C.Hi; C is child of N other than HiChild}
    for(const MultiEdge& E : UndirectedAdjList[N]) 
      if (DFSNum[E.V] > DFSNum[E.U] && E.V != HiChild && Hi2 > Hi[E.V]) 
        Hi2 = Hi[E.V];

    // Next, compute the BracketList of node N.
    // First, concatenate together all bracket lists of children of N
    // into the bracket list of the current node.
    // Note that, the bracket lists of children will be destroyed
    // and no longer valid after we consolidated everything into the
    // current bracket list.
    for(const MultiEdge& E : UndirectedAdjList[N])
      if (DFSNum[E.V] > DFSNum[E.U])
        BList[N].Concat(BList[E.V]);
    // Next, delete back-edges from descendants of N to N.
    // If a back-edge B is a capping back-edge, we simply delete
    // it. Otherwise, we assign a new equivalence class to it
    // after deleting it.   
    for(MultiEdge& E : BackEdgesInto[N]) {
      BList[N].Delete(E.getBracket());
      E.setBracket(nullptr);
      if (!E.IsCappingEdge && E.Class == 0)
        E.Class = EquivalenceClassCounter++;
    }
    // For each backedge B out of N,
    // Push B into the BracketList of N.
    for(MultiEdge& E : UndirectedAdjList[N]) 
      if (DFSNum[E.V] > DFSNum[E.U]) {
        BList[N].Push(E);
      }
    // If Hi2 < Hi0 for the current node N,
    // then we must create a capping back-edge from 
    // N to node Hi2.
    if (Hi2 < Hi0) {
      MultiEdge CappingEdge(N, Hi2);
      CappingEdge.IsCappingEdge = true;
      BList[N].Push(CappingEdge);
    }

    // Finally, determine the equivalence class for the in-edge
    // from Parent[N] to node N (if N is not Source).
    if (Parent[N] != 0) {
      for (MultiEdge& E : UndirectedAdjList[N]) 
        if (E.V == Parent[N]) {
          Bracket * B = BList[N].Top;
          assert(B != nullptr && "Empty Bracket List");
          if (B->E->RecentSize != BList[N].Size) {
            B->E->RecentSize = BList[N].Size;
            B->E->RecentClass = EquivalenceClassCounter++;
          }
          E.Class = B->E->RecentClass;
          // Check for E, B equivalence:
          if (B->E->RecentSize == 1) 
            B->E->Class = E.Class;
          break;
        }
    }
  }

  // After finished calculating cycle equivalence,
  // map equivalence classes of SESE regions back to pairs
  // of LLVM BasicBlock objects.

  std::vector< std::set<BasicBlock*> > EquivalentCycles(EquivalenceClassCounter);
  for (unsigned I = 1; I < UndirectedAdjList.size(); ++I) {
    for (const MultiEdge& E : UndirectedAdjList[I]) {
      unsigned EquivClass = E.Class;
      BasicBlock *SrcBlock = VertexNumberToBB[E.U], *SinkBlock = VertexNumberToBB[E.V];
      EquivalentCycles[EquivClass].insert(SrcBlock);
      EquivalentCycles[EquivClass].insert(SinkBlock);
    }
  }
  return EquivalentCycles;
}

bool HotColdSplitting::outlineColdRegions(Function &F, bool HasProfileSummary) {
  bool Changed = false;

  // The set of cold blocks.
  SmallPtrSet<BasicBlock *, 4> ColdBlocks;

  // The worklist of non-intersecting regions left to outline.
  SmallVector<OutliningRegion, 2> OutliningWorklist;

  // Set up an RPO traversal. Experimentally, this performs better (outlines
  // more) than a PO traversal, because we prevent region overlap by keeping
  // the first region to contain a block.
  ReversePostOrderTraversal<Function *> RPOT(&F);

  // Calculate domtrees lazily. This reduces compile-time significantly.
  std::unique_ptr<DominatorTree> DT;
  std::unique_ptr<PostDominatorTree> PDT;

  // Calculate BFI lazily (it's only used to query ProfileSummaryInfo). This
  // reduces compile-time significantly. TODO: When we *do* use BFI, we should
  // be able to salvage its domtrees instead of recomputing them.
  BlockFrequencyInfo *BFI = nullptr;
  if (HasProfileSummary)
    BFI = GetBFI(F);

  TargetTransformInfo &TTI = GetTTI(F);
  OptimizationRemarkEmitter &ORE = (*GetORE)(F);
  AssumptionCache *AC = LookupAC(F);

  // Find all cold regions.
  for (BasicBlock *BB : RPOT) {
    // This block is already part of some outlining region.
    if (ColdBlocks.count(BB))
      continue;

    bool Cold = (BFI && PSI->isColdBlock(BB, BFI)) ||
                (EnableStaticAnalyis && unlikelyExecuted(*BB, PSI, BFI));
    if (!Cold)
      continue;

    LLVM_DEBUG({
      dbgs() << "Found a cold block:\n";
      BB->dump();
    });

    if (!DT)
      DT = std::make_unique<DominatorTree>(F);
    if (!PDT)
      PDT = std::make_unique<PostDominatorTree>(F);

    auto Regions = OutliningRegion::create(*BB, *DT, *PDT);
    for (OutliningRegion &Region : Regions) {
      if (Region.empty())
        continue;

      if (Region.isEntireFunctionCold()) {
        LLVM_DEBUG(dbgs() << "Entire function is cold\n");
        return markFunctionCold(F);
      }

      // If this outlining region intersects with another, drop the new region.
      //
      // TODO: It's theoretically possible to outline more by only keeping the
      // largest region which contains a block, but the extra bookkeeping to do
      // this is tricky/expensive.
      bool RegionsOverlap = any_of(Region.blocks(), [&](const BlockTy &Block) {
        return !ColdBlocks.insert(Block.first).second;
      });
      if (RegionsOverlap)
        continue;

      OutliningWorklist.emplace_back(std::move(Region));
      ++NumColdRegionsFound;
    }
  }

  if (OutliningWorklist.empty())
    return Changed;

  // Outline single-entry cold regions, splitting up larger regions as needed.
  unsigned OutlinedFunctionID = 1;
  // Cache and recycle the CodeExtractor analysis to avoid O(n^2) compile-time.
  CodeExtractorAnalysisCache CEAC(F);
  do {
    OutliningRegion Region = OutliningWorklist.pop_back_val();
    assert(!Region.empty() && "Empty outlining region in worklist");
    do {
      BlockSequence SubRegion = Region.takeSingleEntrySubRegion(*DT);
      LLVM_DEBUG({
        dbgs() << "Hot/cold splitting attempting to outline these blocks:\n";
        for (BasicBlock *BB : SubRegion)
          BB->dump();
      });

      Function *Outlined = extractColdRegion(SubRegion, CEAC, *DT, BFI, TTI,
                                             ORE, AC, OutlinedFunctionID);
      if (Outlined) {
        ++OutlinedFunctionID;
        Changed = true;
      }
    } while (!Region.empty());
  } while (!OutliningWorklist.empty());

  return Changed;
}

bool HotColdSplitting::run(Module &M) {
  bool Changed = false;
  bool HasProfileSummary = (M.getProfileSummary(/* IsCS */ false) != nullptr);
  for (auto It = M.begin(), End = M.end(); It != End; ++It) {
    Function &F = *It;

    // Do not touch declarations.
    if (F.isDeclaration())
      continue;

    // Do not modify `optnone` functions.
    if (F.hasOptNone())
      continue;

    // Detect inherently cold functions and mark them as such.
    if (isFunctionCold(F)) {
      Changed |= markFunctionCold(F);
      continue;
    }

    if (!shouldOutlineFrom(F)) {
      LLVM_DEBUG(llvm::dbgs() << "Skipping " << F.getName() << "\n");
      continue;
    }

    LLVM_DEBUG(llvm::dbgs() << "Outlining in " << F.getName() << "\n");
    Changed |= outlineColdRegions(F, HasProfileSummary);
  }
  return Changed;
}

bool HotColdSplittingLegacyPass::runOnModule(Module &M) {
  if (skipModule(M))
    return false;
  ProfileSummaryInfo *PSI =
      &getAnalysis<ProfileSummaryInfoWrapperPass>().getPSI();
  auto GTTI = [this](Function &F) -> TargetTransformInfo & {
    return this->getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
  };
  auto GBFI = [this](Function &F) {
    return &this->getAnalysis<BlockFrequencyInfoWrapperPass>(F).getBFI();
  };
  std::unique_ptr<OptimizationRemarkEmitter> ORE;
  std::function<OptimizationRemarkEmitter &(Function &)> GetORE =
      [&ORE](Function &F) -> OptimizationRemarkEmitter & {
    ORE.reset(new OptimizationRemarkEmitter(&F));
    return *ORE.get();
  };
  auto LookupAC = [this](Function &F) -> AssumptionCache * {
    if (auto *ACT = getAnalysisIfAvailable<AssumptionCacheTracker>())
      return ACT->lookupAssumptionCache(F);
    return nullptr;
  };

  return HotColdSplitting(PSI, GBFI, GTTI, &GetORE, LookupAC).run(M);
}

PreservedAnalyses
HotColdSplittingPass::run(Module &M, ModuleAnalysisManager &AM) {
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  auto LookupAC = [&FAM](Function &F) -> AssumptionCache * {
    return FAM.getCachedResult<AssumptionAnalysis>(F);
  };

  auto GBFI = [&FAM](Function &F) {
    return &FAM.getResult<BlockFrequencyAnalysis>(F);
  };

  std::function<TargetTransformInfo &(Function &)> GTTI =
      [&FAM](Function &F) -> TargetTransformInfo & {
    return FAM.getResult<TargetIRAnalysis>(F);
  };

  std::unique_ptr<OptimizationRemarkEmitter> ORE;
  std::function<OptimizationRemarkEmitter &(Function &)> GetORE =
      [&ORE](Function &F) -> OptimizationRemarkEmitter & {
    ORE.reset(new OptimizationRemarkEmitter(&F));
    return *ORE.get();
  };

  ProfileSummaryInfo *PSI = &AM.getResult<ProfileSummaryAnalysis>(M);

  if (HotColdSplitting(PSI, GBFI, GTTI, &GetORE, LookupAC).run(M))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

char HotColdSplittingLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(HotColdSplittingLegacyPass, "hotcoldsplit",
                      "Hot Cold Splitting", false, false)
INITIALIZE_PASS_DEPENDENCY(ProfileSummaryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(BlockFrequencyInfoWrapperPass)
INITIALIZE_PASS_END(HotColdSplittingLegacyPass, "hotcoldsplit",
                    "Hot Cold Splitting", false, false)

ModulePass *llvm::createHotColdSplittingPass() {
  return new HotColdSplittingLegacyPass();
}
