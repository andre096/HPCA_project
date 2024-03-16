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

int main() {
  try {
    queue q(default_selector_v);

    cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";


    buffer<float, 2> a_buf(range(M, N));
    buffer<float, 2> b_buf(range(N, P));
    buffer<float, 2> c_buf(range(M, P));

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
	
    q.submit([&](auto &h) {
      accessor c(c_buf, h, write_only);

      h.parallel_for(range(M, P), [=](auto index) {
        c[index] = index[0] + 0.0f;
      });
    });
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

        cout << "Execution time: " << duration.count() << " milliseconds" << "\n";

		} catch (sycl::exception const &e) {
			cout << "An exception is caught while multiplying matrices.\n";
			terminate();
		}

return 0;
}
