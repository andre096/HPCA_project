#include <sycl/sycl.hpp>
#include <iostream>
#include <limits>
#include <chrono>


using namespace std;
using namespace sycl;
using namespace std::chrono;

// Matrix size constants.
constexpr int M = 1000;
constexpr int N = 1000;
constexpr int P = 1000;
constexpr int BLOCK_SIZE = 10;


void VerifyResult(float (*c_back)[P]);

int main() {
	float(*c_back)[P] = new float[M][P];

	  // Intialize c_back
	  for (int i = 0; i < M; i++)
		for (int j = 0; j < P; j++) c_back[i][j] = 0.0f;
  try {
	    
    queue q(default_selector_v);

    cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";


    buffer<float, 2> a_buf(range(M, N));
    buffer<float, 2> b_buf(range(N, P));
    buffer c_buf(reinterpret_cast<float *>(c_back), range(M, P));

    cout << "Problem size: c(" << M << "," << P << ") = a(" << M << "," << N
         << ") * b(" << N << "," << P << ")\n";


    q.submit([&](auto &h) {
      accessor a(a_buf, h, write_only);

      h.parallel_for(range(M, N), [=](auto index) {
        a[index] = 1.0f;
      });
    });

    q.submit([&](auto &h) {
      accessor b(b_buf, h, write_only);

      h.parallel_for(range(N, P), [=](auto index) {
        b[index] = index[0] + 1.0f;
      });
    });
	
    //q.submit([&](auto &h) {
    //  accessor c(c_buf, h, write_only);

      //h.parallel_for(range(M, P), [=](auto index) {
        //c[index] = index[0] + 0.0f;
      //});
    //});
        auto start_time = high_resolution_clock::now();

			q.submit([&](auto &h) {
				accessor a(a_buf, h, read_only);
				accessor b(b_buf, h, read_only);
				accessor c(c_buf, h, write_only);

				h.parallel_for(range(M, P), [=](auto index) {
					size_t row = index[0];
					size_t col = index[1];

					float sum = 0.0f;

					// Calculate the starting indices of the current block
					size_t start_row = row - row % BLOCK_SIZE;
					size_t start_col = col - col % BLOCK_SIZE;

					// Perform block matrix multiplication
					for (size_t k = 0; k < N; k += BLOCK_SIZE) {
						for (size_t i = start_row, ii = 0; i < start_row + BLOCK_SIZE; ++i, ++ii) {
							for (size_t j = start_col, jj = 0; j < start_col + BLOCK_SIZE; ++j, ++jj) {
								sum += a[{i, k + jj}] * b[{k + ii, j}];
							}
						}
					}

					c[index] = sum;
				});
			});
		q.wait();

        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end_time - start_time);

        cout << "Execution time parallelized: " << duration.count() << " milliseconds" << "\n";

		


		} catch (sycl::exception const &e) {
			cout << "An exception is caught while multiplying matrices.\n";
			terminate();
		}
	  cout << "Result of matrix multiplication using SYCL: ";
	  VerifyResult(c_back);
	  delete[] c_back;

return 0;
}


bool ValueSame(float a, float b) {
  return fabs(a - b) < numeric_limits<float>::epsilon();
}

void VerifyResult(float (*c_back)[P]){
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
	
	auto start_time = high_resolution_clock::now();
	
	for (size_t row = 0; row < P; ++row) {
		for (size_t col = 0; col < P; ++col) {
			float sum = 0.0f;

			// Calculate the starting indices of the current block
			size_t start_row = row - row % BLOCK_SIZE;
			size_t start_col = col - col % BLOCK_SIZE;

			// Perform block matrix multiplication
			for (size_t k = 0; k < N; k += BLOCK_SIZE) {
				for (size_t i = start_row, ii = 0; i < start_row + BLOCK_SIZE; ++i, ++ii) {
					for (size_t j = start_col, jj = 0; j < start_col + BLOCK_SIZE; ++j, ++jj) {
						sum += a_host[{i, k + jj}] * b_host[{k + ii, j}];
					}
				}
			}

			c_host[{row, col}] = sum;
		}
	}
	auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time);

    cout << "Execution time unparallelized: " << duration.count() << " milliseconds" << "\n";
	
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
	  } else {
		cout << "Fail - The results mismatch!\n";
	  }
		
}
