#include <iostream>
#include <vector>
#include <omp.h>

#define M 3
#define N 3
#define P 3
#define BLOCK_SIZE 2 // Define the block size

using namespace std;



void MatrixMulBlock(float (*a)[N], float (*b)[P], float (*c)[P]);

int main(void) {
    float a[M][N] = {{1.0,2.0,3.0},{4.0,5.0,6.0},{7.0,8.0,9.0}};
    float b[N][P] = {{9.0,8.0,7.0},{6.0,5.0,4.0},{3.0,2.0,1.0}};
    float c[M][P];

    // Initialize matrices a and b (as before)
    // ...

    MatrixMulBlock(a, b, c);

    // Print the result matrix c
    printf("Result Matrix c:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            printf("%f ", c[i][j]);
        }
        printf("\n");
    }

    return 0;
}

void MatrixMulBlock(float (*a)[N], float (*b)[P], float (*c)[P]) {
    int i, j, k, ii, jj, kk;
	
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