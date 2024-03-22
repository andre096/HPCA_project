#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
make build_omp
make build_sycl
