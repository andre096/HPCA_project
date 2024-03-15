#include <iostream>
#include <vector>
#include <omp.h>

#define M 100
#define N 100
#define P 100
#define BLOCK_SIZE 10 // Define the block size

using namespace std;



void MatrixMulBlock(float (*a)[N], float (*b)[P], float (*c)[P]);

int main(void) {
    float a[M][N];
    float b[N][P];
    float c[M][P];

    // Initialize matrices a and b (as before)
    // ...

    MatrixMulBlock(a, b, c);

    // Print the result matrix c
    // printf("Result Matrix c:\n");
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < P; j++) {
    //         printf("%f ", c[i][j]);
    //     }
    //     printf("\n");
    // }

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

    for (i = 0; i < M; i++) {
        for (j = 0; j < P; j++) {
            c[i][j] = 0.0f; // Initialize each element of c to zero
        }
    }
	
    // Perform block matrix multiplication
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
	printf("\n Time taken is %f", exec_time);
}