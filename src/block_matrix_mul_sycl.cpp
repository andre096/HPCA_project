#include <CL/sycl.hpp>
#include <iostream>
#include <chrono>

#define M 1000
#define N 1000
#define P 1000
#define BLOCK_SIZE 10 // Define the block size

using namespace std;
using namespace cl::sycl;
using namespace std::chrono;

class MatrixMul;

void MatrixMulBlock(float *a, float *b, float *c) {
    queue q;
    buffer<float, 2> bufferA(a, range<2>(M, N));
    buffer<float, 2> bufferB(b, range<2>(N, P));
    buffer<float, 2> bufferC(c, range<2>(M, P));

    auto R = range<2>(M, P);
    q.submit([&](handler &h) {
        auto accessA = bufferA.get_access<access::mode::read>(h);
        auto accessB = bufferB.get_access<access::mode::read>(h);
        auto accessC = bufferC.get_access<access::mode::write>(h);

        h.parallel_for<MatrixMul>(R, [=](id<2> idx) {
            int i = idx[0];
            int j = idx[1];
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += accessA[i][k] * accessB[k][j];
            }
            accessC[idx] = sum;
        });
    });

    q.wait();
}

int main() {
    float a[M][N], b[N][P], c[M][P];

    // Fill matrices a and b
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            a[i][j] = 1.0f;
        }
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            b[i][j] = i + 1.0f;
        }
    }

    	// Measure the execution time
        auto start_time = high_resolution_clock::now();

    // Perform matrix multiplication
    MatrixMulBlock(reinterpret_cast<float *>(a),
                   reinterpret_cast<float *>(b),
                   reinterpret_cast<float *>(c));

    // Measure the execution time
        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end_time - start_time);

        cout << "Execution time: " << duration.count() << " milliseconds" << "\n";

    return 0;
}