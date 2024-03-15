#include <float.h>
#include <math.h>
#include <omp.h>
#include <iostream>
#include <limits>

#define M 100
#define N 100
#define P 100
#define BLOCK_SIZE 10 // Define the block size

using namespace std;

float a[M][N];
float b[N][P];
float c[M][P];

void MatrixMulBlock(float (*a)[N], float (*b)[P], float (*c)[P]);

int VerifyResult(float (*c_back)[P]);

int main(void) {
	int Result1;
	
    MatrixMulBlock(a, b, c);
	
	
	cout << "Result of matrix multiplication using OpenMP: ";
    Result1 = VerifyResult(c);	
	

    return 0;
}

void MatrixMulBlock(float (*a)[N], float (*b)[P], float (*c)[P]) {
    int i, j, k, ii, jj, kk;

    // Each element of matrix a is 1.
    for (i = 0; i < M; i++)
        for (j = 0; j < N; j++) a[i][j] = 1.0f;

    // Each column of b is the sequence 1,2,...,N
    for (i = 0; i < N; i++)
        for (j = 0; j < P; j++) b[i][j] = i + 1.0f;

    for (i = 0; i < M; i++)
        for (j = 0; j < P; j++) c[i][j] = 0.0f;
	
	
	double itime, ftime, exec_time;
	itime = omp_get_wtime();
	
    // Perform block matrix multiplication
	#pragma omp parallel for private(i, j, k, ii, jj, kk) shared(a, b, c)
    for (ii = 0; ii < M; ii += BLOCK_SIZE) {
        for (jj = 0; jj < P; jj += BLOCK_SIZE) {
            for (kk = 0; kk < N; kk += BLOCK_SIZE) {
                // Multiply block of A and B
                for (i = ii; i < ii + BLOCK_SIZE && i < M; i++) {
                    for (j = jj; j < jj + BLOCK_SIZE && j < P; j++) {
                        for (k = kk; k < kk + BLOCK_SIZE && k < N; k++) {
                            c[i][j] += a[i][k] * b[k][j];
                        }
                    }
                }
            }
        }
    }
	ftime = omp_get_wtime();
	exec_time = ftime-itime;
	printf("Time taken for parallelized block multiplication is %f \n", exec_time);
}


bool ValueSame(float a, float b) {
	return fabs(a - b) < numeric_limits<float>::epsilon();
}


int verifyResult(float (*c_back)[P]){
	// Check that the results are correct by comparing with host computing.
	int i, j, k, ii, jj, kk;

	float(*a_host)[N] = new float[M][N];
	float(*b_host)[P] = new float[N][P];
	float(*c_host)[P] = new float[M][P];
	
	
	// Each element of matrix a is 1.
    for (i = 0; i < M; i++)
		for (j = 0; j < N; j++) a_host[i][j] = 1.0f;

	// Each column of b_host is the sequence 1,2,...,N
	for (i = 0; i < N; i++)
		for (j = 0; j < P; j++) b_host[i][j] = i + 1.0f;

	// c_host is initialized to zero.
	for (i = 0; i < M; i++)
		for (j = 0; j < P; j++) c_host[i][j] = 0.0f;
	
	for (ii = 0; ii < M; ii += BLOCK_SIZE) {
        for (jj = 0; jj < P; jj += BLOCK_SIZE) {
            for (kk = 0; kk < N; kk += BLOCK_SIZE) {
                // Multiply block of A and B
                for (i = ii; i < ii + BLOCK_SIZE && i < M; i++) {
                    for (j = jj; j < jj + BLOCK_SIZE && j < P; j++) {
                        for (k = kk; k < kk + BLOCK_SIZE && k < N; k++) {
                            c_host[i][j] += a_host[i][k] * b_host[k][j];
                        }
                    }
                }
            }
        }
    }
	
	bool mismatch_found = false;
	
	int print_count = 0;

	  for (i = 0; i < M; i++) {
		for (j = 0; j < P; j++) {
		  if (!ValueSame(c_back[i][j], c_host[i][j])) {
			cout << "Fail - The result is incorrect for element: [" << i << ", "
				 << j << "], expected: " << c_host[i][j]
				 << ", but found: " << c_back[i][j] << "\n";
			mismatch_found = true;
			print_count++;
			if (print_count == 5) break;
		  }
		}

		if (print_count == 5) break;
	  }
	  delete[] a_host;
	  delete[] b_host;
	  delete[] c_host;

	  if (!mismatch_found) {
		cout << "Success - The results are correct!\n";
		return 0;
	  } else {
		cout << "Fail - The results mismatch!\n";
		return -1;
	  }
		
}
















