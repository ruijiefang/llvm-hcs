# place to put QEMU and other benchmark-related folders
QEMU_BASE_DIR=./
# directory to LLVM11 binaries
LLVM_DIR=/home/rjf/Projects/gsoc/llvm-project/build/bin/
LLVM_HCS=/home/rjf/Projects/gsoc/llvm-project/build-hcs/bin/

# benchmark rounds
os_rounds=8
psql_rounds=3

set -x

function clone_and_build_hcs_llvm {
  echo "cloning HCS-enabled LLVM"
  git clone https://github.com/ruijiefang/llvm-hcs 
  cd ./llvm-hcs
  mkdir build
  cd build/
  cmake -DLLVM_ENABLE_PROJECTS="clang;lld;lldb;clang-tools-extra;compiler-rt" -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles" ../llvm
  make -j4
  echo "build complete; exiting..."
  cd ../../
}

function clone_and_build_qemu_instrumented {
  echo "compiling qemu"
  git clone https://github.com/ruijiefang/qemu-x86_64-linux-benchmark ./qemu --depth=1
  cd ./qemu
  ./configure --wno-error --cc=$LLVM_DIR/clang --cxx=$LLVM_DIR/clang++ --extra-cflags="-gline-tables-only -fprofile-generate -O2" --extra-cxxflags="-gline-tables-only -fprofile-generate -O2" --target-list="x86_64-linux-user,x86_64-softmmu"
  make -j4
  echo "finished compiling qemu"
  cd ../
}

function instrument_system_startup {
  echo "setting up system for the first time. Log in via root/root to set up crontasks to be run via qemu."
  sudo ./debootstrap-ubuntu-qemu.sh
  echo "finished setting up. Now running benchmark run..."
  sudo time ./run-prof.sh
  echo "finished running benchmark. moving profiles to profiles/"
  mv *.profraw profiles/
}

function instrument_postgres_compilation {
  echo "instrumenting postgresql compilation"
  wget https://ftp.postgresql.org/pub/source/v12.3/postgresql-12.3.tar.gz 
  tar xvf ./postgresql-12.3.tar.gz
  cd postgresql-12.3/
  CC="../qemu/x86_64-linux-user/qemu-x86_64 $LLVM_DIR/clang" CXX="../qemu/x86_64-linux-user/qemu-x86_64 $LLVM_DIR/clang++" ./configure 
  make -j4
  mv ./*.profraw ../profiles/
  cd ../
}

function build_qemu_pgo {
  echo "compiling baseline qemu"
  git clone https://github.com/ruijiefang/qemu-x86_64-linux-benchmark ./qemu-pgo --depth=1
  cd ./qemu-pgo
  ./configure --wno-error --cc=$LLVM_DIR/clang --cxx=$LLVM_DIR/clang++ --extra-cflags="-gline-tables-only -fprofile-generate -O2" --extra-cxxflags="-gline-tables-only -fprofile-use=../profiles/default.profdata -O2" --target-list="x86_64-linux-user,x86_64-softmmu"
  make -j4
  echo "build complete"
  cd ../
}


function benchmark_qemu {
  q=$1
  echo "benchmark_qemu: benchmarking using "$q
  echo "benchmark_qemu: benchmarking OS startup for "$os_rounds" rounds"
  for (( i=0; i<=$os_rounds; i++ )) ; do
    echo " * run "$i
    sudo time ./run.sh $q
  done
  echo "benchmark_qemu: icache data for "$q" + OS startup"
  sudo perf stats -v ./run.sh
  echo "benchmark_qemu: benchmarking postgres compilation for "$psql_rounds" rounds"
  for (( i=0; i<=$psql_rounds; i++ )) ; do
    echo " * run"$i
    cd postgresql-12.3/
    time CC="$q $LLVM_DIR/clang" CXX="$q $LLVM_DIR/clang++" ./configure
    time make -j4
    make clean
    cd ../
  done
}

function build_qemu_hcs {
  echo "compiling qemu hcs"
  git clone https://github.com/ruijiefang/qemu-x86_64-linux-benchmark ./qemu-hcs --depth=1
  cd ./qemu-hcs
  ./configure --wno-error --cc=$LLVM_HCS/clang --cxx=$LLVM_HCS/clang++ --extra-cflags="-gline-tables-only -fprofile-use=../profiles/default.profdata -O2 -mllvm --debug-only=hotcoldsplit" --extra-cxxflags="-gline-tables-only -fprofile-use=../profiles/default.profdata -O2 -mllvm --debug-only=hotcoldsplit" --target-list="x86_64-linux-user,x86_64-softmmu"
  rm -rf *.dot # ignore these dot files; they're generated by autotools and meaningless
  make &>> OUT.log # append all debug messages to a log.
  echo "build complete!"
  echo "converting dotfiles into png!"
  mkdir ../dotfiles
  mv ./*.dot ../dotfiles
  cd ../dotfiles
  for x in `ls ./`; do
    dot -Tpng -o$x.png $x
  end
  echo "png conversion complete! exiting..."
  cd ../
}

clone_and_build_qemu_instrumented 
mkdir ./profiles/
instrument_system_startup
echo "merging profiles..."
cd profiles/
$LLVM_DIR/llvm-profdata merge ./*.profraw --output default.profdata 
echo "*** QEMU PGO Build"
build_qemu_pgo
benchmark_qemu "./qemu-pgo"
build_qemu_hcs
benchmark_qemu "./qemu-hcs"
