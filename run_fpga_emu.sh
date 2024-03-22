#!/bin/bash

#PBS -l nodes=1:fpga_compile:ppn=2
#PBS -d .

source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
make run_emu -f Makefile.fpga
