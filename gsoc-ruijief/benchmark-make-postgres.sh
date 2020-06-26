#!/bin/bash

POSTGRES_DIR="/home/rjf/Projects/gsoc/benchmarks/postgresql-12.3"


# properly configures machine for benchmarking purposes.
# references:
# https://easyperf.net/blog/2019/08/02/Perf-measurement-environment-on-Linux
# https://llvm.org/docs/Benchmarking.html#linux

# note: the following configurations are all for my local machine, running
# an Intel Xeon E5 with 4 cores, SMT+Hyperthreading, and 3.1GHz maxfreq.
# You'll have to adjust certain settings, e.g. CPU frequency, according
# to your own platform.

# required packages/commands:
# cpufreq-info, cpufreq-set, cset/cpuset (all installable via Ubuntu apt)

set -x

# 1: set all cores to 3.1GHz, the maximum frequency.
sudo cpufreq-set -r -f 3.1GHz
sudo cpufreq-info

# 2: disable address space randomization
echo 0 > /proc/sys/kernel/randomize_va_space

# 3: use cpuset to reserve CPUs #1, #2 for benchmarking purposes
sudo cset shield -c 1,2 -k on

# 4: disable SMT pair for #1, #2 (we're a 4-core cpu so no need to do this.)


# 5: Run benchmark with:
# sudo cset shield --exec -- perf stat -o perf.log -v -r 10 <cmd>
cd $POSTGRES_DIR
make clean
sudo cset shield --exec -- \
	perf stat -o perf.log -r 6 -e task-clock,context-switches,cpu-migrations,page-faults,cycles,instructions,branches,branch-misses,L1-icache-load-misses,icache.misses,icache.hit,icache.ifdata_stall,icache.ifetch_stall \
	nice -n -20 ionice -c 2 -n 0 \
	./execute.sh
# task-clock,context-switches,cpu-migrations,page-faults,cycles,instructions,branches,branch-misses,L1-icache-load-misses,icache.misses,icache.hit,icache.ifdata_stall,icache.ifetch_stall
