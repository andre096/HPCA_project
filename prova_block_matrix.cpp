#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

// Function to perform block matrix multiplication
vector<vector<int>> blockMatrixMultiply(const vector<vector<int>>& A, const vector<vector<int>>& B, int blockSize) {
    int n = A.size();
    vector<vector<int>> C(n, vector<int>(n, 0));

    #pragma omp parallel for collapse(3) // parallelize the outer loop
    for (int i = 0; i < n; i += blockSize) {
        for (int j = 0; j < n; j += blockSize) {
            for (int k = 0; k < n; k += blockSize) {
                // Multiply block A(i,k) with block B(k,j)
                for (int ii = i; ii < min(i + blockSize, n); ++ii) {
                    for (int jj = j; jj < min(j + blockSize, n); ++jj) {
                        for (int kk = k; kk < min(k + blockSize, n); ++kk) {
                            C[ii][jj] += A[ii][kk] * B[kk][jj];
                        }
                    }
                }
            }
        }
    }

    return C;
}

int main() {
    vector<vector<int>> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    vector<vector<int>> B = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
    int blockSize = 2;

    vector<vector<int>> result = blockMatrixMultiply(A, B, blockSize);

    // Output the result
    for (const auto& row : result) {
        for (int elem : row) {
            cout << elem << " ";
        }
        cout << endl;
    }

    return 0;
}