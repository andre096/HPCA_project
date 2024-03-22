#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
make run_omp
make run_sycl
