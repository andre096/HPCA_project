
CXX = icpx

OMP_CXXFLAGS = -fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__ -g -o
OMP_LDFLAGS = 
OMP_EXE_NAME = block_matrix_mul_omp
OMP_SOURCES = src/block_matrix_mul_omp.cpp


build_omp:
	$(CXX) $(OMP_CXXFLAGS) $(OMP_EXE_NAME) $(OMP_SOURCES) $(OMP_LDFLAGS)


run_omp:
	./$(OMP_EXE_NAME)


clean: 
	rm -rf $(OMP_EXE_NAME)