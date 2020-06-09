#!/bin/bash
set -e # exit on failure
set -x # enable command printing
LLVM_ROOT="/home/rjf/Projects/gsoc/llvm-project"
LLVM_BUILD_BASE="build"
LLVM_BUILD_PGO="build-instrumented"
LLVM_BUILD_OPT="build-optimized"
LLVM_BUILD_HCS="build-hcs"
LLVM_BUILD_HCS_PGO="build-hcs-instrumented"
LLVM_BUILD_LTO="build-lto"
LLVM_MAKE_FLAGS="-j4"
BENCHMARK_ROOT="/home/rjf/Projects/gsoc/benchmarks"
QEMU_MAKE_FLAGS="-j4"

function build_llvm_base {
  echo "Building LLVM base..."
  mkdir $LLVM_ROOT/$LLVM_BUILD_BASE
  cd $LLVM_ROOT/$LLVM_BUILD_BASE
  cmake -DLLVM_ENABLE_PROJECTS="clang;lld;lldb;clang-tools-extra;compiler-rt" -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles" ../llvm
  make $LLVM_MAKE_FLAGS  
}

function build_llvm_pgo {
  echo "Building LLVM with PGO..."
  mkdir $LLVM_ROOT/$LLVM_BUILD_PGO
  cd $LLVM_ROOT/$LLVM_BUILD_PGO
  cmake -DLLVM_ENABLE_PROJECTS="clang;lld;lldb;clang-tools-extra;compiler-rt" -G "Unix Makefiles" -DLLVM_BUILD_INSTRUMENTED=IR -DLLVM_BUILD_RUNTIME=No -DCMAKE_C_COMPILER=$LLVM_ROOT/$LLVM_BUILD_BASE/bin/clang -DCMAKE_CXX_COMPILER=$LLVM_ROOT/$LLVM_BUILD_BASE/bin/clang++ ../llvm
  make $LLVM_MAKE_FLAGS
}

function clone_and_build_qemu {
  echo "Cloning and building QEMU..."
  cd $BENCHMARK_ROOT
  rm -rf qemu/
  git clone --depth=1 https://github.com/qemu/qemu
  cd qemu/
  ./configure --cc=$LLVM_ROOT/$LLVM_BUILD_PGO/bin/clang --cxx=$LLVM_ROOT/$LLVM_BUILD_PGO/bin/clang++
  # disable build warnings; otherwise qemu build fails using newest edition of clang.
  sed -i "s/\-Werror//g" config-host.mak
  make $QEMU_MAKE_FLAGS
}

function clone_and_build_qemu_timed {
  cc=$1
  cxx=$2
  mflags=$3
  echo "Building QEMU with cc="$cc" and cxx="$cxx" and make flags="$mflags
  cd $BENCHMARK_ROOT
  rm -rf qemu/
  git clone --depth=1 https://github.com/qemu/qemu
  cd qemu/
  ./configure --cc=$cc --cxx=$cxx
  # disable build warnings; otherwise qemu build fails using newest edition of clang.
  sed -i "s/\-Werror//g" config-host.mak
  echo " ***** Building QEMU *****"
  time make $mflags
}

function cleanup_profiles {
  echo "Cleaning old profiles..."
  cd $LLVM_ROOT/$LLVM_BUILD_PGO/profiles
  rm -rf *.profraw
}

function convert_profiles {
  echo "Converting profiles..."
  cd $LLVM_ROOT/$LLVM_BUILD_PGO/profiles
  $LLVM_ROOT/$LLVM_BUILD_BASE/bin/llvm-profdata merge -output=merged.profdata *.profraw
}

function build_llvm_optimized {
  echo "Using PGO information to build optimized LLVM..."
  mkdir $LLVM_ROOT/$LLVM_BUILD_OPT
  cd $LLVM_ROOT/$LLVM_BUILD_OPT
  cmake -DLLVM_ENABLE_PROJECTS="clang;lld;lldb;clang-tools-extra;compiler-rt" -DCMAKE_BUILD_TYPE=Release -DLLVM_PROFDATA_FILE=$LLVM_ROOT/$LLVM_BUILD_PGO/profiles/merged.profdata -DCMAKE_C_COMPILER=$LLVM_ROOT/$LLVM_BUILD_BASE/bin/clang -DCMAKE_CXX_COMPILER=$LLVM_ROOT/$LLVM_BUILD_BASE/bin/clang++ -G "Unix Makefiles" ../llvm
  make $LLVM_MAKE_FLAGS
}

function build_llvm_optimized_with_hcs {
  echo "Using PGO information and HCS to build optimized LLVM..."
  echo "Using PGO information to build optimized LLVM..."
  mkdir $LLVM_ROOT/$LLVM_BUILD_HCS
  cd $LLVM_ROOT/$LLVM_BUILD_HCS
  cmake -DLLVM_ENABLE_PROJECTS="clang;lld;lldb;clang-tools-extra;compiler-rt" -DCMAKE_BUILD_TYPE=Release -DLLVM_PROFDATA_FILE=$LLVM_ROOT/$LLVM_BUILD_PGO/profiles/merged.profdata -DCMAKE_C_COMPILER=$LLVM_ROOT/$LLVM_BUILD_BASE/bin/clang -DCMAKE_CXX_COMPILER=$LLVM_ROOT/$LLVM_BUILD_BASE/bin/clang++ -DCMAKE_C_FLAGS="-O2 -mllvm -hot-cold-split=true" -DCMAKE_CXX_FLAGS="-O2 -mllvm -hot-cold-split=true" -G "Unix Makefiles" ../llvm
  make $LLVM_MAKE_FLAGS
}

function build_llvm_hcs_instrumented {
  echo "Using PGO information and HCS to build optimized LLVM..."
  echo "Using PGO information to build optimized LLVM..."
  mkdir $LLVM_ROOT/$LLVM_BUILD_HCS_PGO
  cd $LLVM_ROOT/$LLVM_BUILD_HCS_PGO
  cmake -DLLVM_ENABLE_PROJECTS="clang;lld;lldb;clang-tools-extra;compiler-rt" -DLLVM_PROFDATA_FILE=$LLVM_ROOT/$LLVM_BUILD_PGO/profiles/merged.profdata -DCMAKE_C_COMPILER=$LLVM_ROOT/$LLVM_BUILD_BASE/bin/clang -DCMAKE_CXX_COMPILER=$LLVM_ROOT/$LLVM_BUILD_BASE/bin/clang++ -DCMAKE_C_FLAGS="-fprofile-arcs -ftest-coverage -mllvm -hot-cold-split=true" -DLLVM_BUILD_INSTRUMENTED=IR -DLLVM_BUILD_RUNTIME=No -DCMAKE_CXX_FLAGS="-fprofile-arcs -ftest-coverage -g -mllvm -hot-cold-split=true" -G "Unix Makefiles" ../llvm
  make $LLVM_MAKE_FLAGS
}


echo "starting script..."
build_llvm_base
build_llvm_pgo
cleanup_profiles
clone_and_build_qemu
convert_profiles
build_llvm_optimized
build_llvm_optimized_with_hcs
echo "*** timed trial qemu, baseline (PGO only)"
clone_and_build_qemu_timed $LLVM_ROOT/$LLVM_BUILD_OPT/bin/clang $LLVM_ROOT/$LLVM_BUILD_OPT/bin/clang++ "-j4"
echo "*** timed trial qemu, HCS and PGO enabled"
clone_and_build_qemu_timed $LLVM_ROOT/$LLVM_BUILD_HCS/bin/clang $LLVM_ROOT/$LLVM_BUILD_OPT/bin/clang++ "-j4"
#build_llvm_hcs_instrumented
