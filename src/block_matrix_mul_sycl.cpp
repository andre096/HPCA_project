//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

/**
 * Matrix_mul multiplies two large matrices both the CPU and the offload device,
 * then compares results. If the code executes on both CPU and the offload
 * device, the name of the offload device and a success message are displayed.
 *
 * For comprehensive instructions regarding SYCL Programming, go to
 * https://software.intel.com/en-us/oneapi-programming-guide and search based on
 * relevant terms noted in the comments.
 */

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
  // Initialize the device queue with the default selector. The device queue is
  // used to enqueue kernels. It encapsulates all states needed for execution.
  try {
    queue q(default_selector_v);

    cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";

    // Create 2D buffers for matrices, buffer c is bound with host memory c_back

    buffer<float, 2> a_buf(range(M, N));
    buffer<float, 2> b_buf(range(N, P));
    buffer<float, 2> c_buf(range(M, P));

    cout << "Problem size: c(" << M << "," << P << ") = a(" << M << "," << N
         << ") * b(" << N << "," << P << ")\n";

    // Using three command groups to illustrate execution order. The use of
    // first two command groups for initializing matrices is not the most
    // efficient way. It just demonstrates the implicit multiple command group
    // execution ordering.

    // Submit command group to queue to initialize matrix a
    q.submit([&](auto &h) {
      // Get write only access to the buffer on a device.
      accessor a(a_buf, h, write_only);

      // Execute kernel.
      h.parallel_for(range(M, N), [=](auto index) {
        // Each element of matrix a is 1.
        a[index] = 1.0f;
      });
    });

    // Submit command group to queue to initialize matrix b
    q.submit([&](auto &h) {
      // Get write only access to the buffer on a device
      accessor b(b_buf, h, write_only);

      // Execute kernel.
      h.parallel_for(range(N, P), [=](auto index) {
        // Each column of b is the sequence 1,2,...,N
        b[index] = index[0] + 1.0f;
      });
    });
	
	    // Submit command group to queue to initialize matrix c
    q.submit([&](auto &h) {
      // Get write only access to the buffer on a device
      accessor c(c_buf, h, write_only);

      // Execute kernel.
      h.parallel_for(range(M, P), [=](auto index) {
        // Each column of c is the sequence 1,2,...,N
        c[index] = index[0] + 0.0f;
      });
    });
	// Measure the execution time
        auto start_time = high_resolution_clock::now();

	// Submit command group to queue to perform block matrix multiplication: c = a * b
			q.submit([&](auto &h) {
				accessor a(a_buf, h, read_only);
				accessor b(b_buf, h, read_only);
				accessor c(c_buf, h, write_only);

				h.parallel_for(range(M, P), [=](auto index) {
				  size_t row = index[0];
				  size_t col = index[1];

				  // Vectorize these variables based on the chosen vector type (e.g., float4)
				  float4 sum = 0.0f;
				  size_t start_row = row - row % BLOCK_SIZE;
				  size_t start_col = col - col % BLOCK_SIZE;

				  // Loop over tiles with vectorized strides
				  for (size_t k = 0; k < N; k += BLOCK_SIZE/sizeof(float4)) {
					// Load vector elements from matrices A and B
					float4 a_vec = load(a_buf, {start_row, k});
					float4 b_vec = load(b_buf, {k, start_col});

					// Perform vectorized dot product (replace with intrinsic function or custom implementation)
					sum += dot(a_vec, b_vec);

					// Update starting indices for the next tile
					start_row += sizeof(float4);
				  }

				  // Store the final result (considering vector type)
				  store(c_buf, index, sum);
				});
		// Wait for the queue to finish
		q.wait();

        // Measure the execution time
        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end_time - start_time);

        cout << "Execution time: " << duration.count() << " milliseconds" << "\n";

		} catch (sycl::exception const &e) {
			cout << "An exception is caught while multiplying matrices.\n";
			terminate();
		}

return 0;
}
