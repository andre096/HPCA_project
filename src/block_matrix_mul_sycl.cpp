#include <CL/sycl.hpp>

// Define block size for efficient use of device resources
const int BLOCK_SIZE = 32; // Adjust based on your hardware

template <typename T>
void blocked_matrix_multiply_sycl(sycl::queue& queue, const sycl::buffer<T, 1>& a_buffer,
                                 const sycl::buffer<T, 1>& b_buffer, sycl::buffer<T, 1>& c_buffer,
                                 int M, int N, int P) {
  // Calculate grid dimensions for efficient work distribution
  int grid_size_M = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int grid_size_N = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int grid_size_P = (P + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // Define work range for parallel execution
  sycl::range work_range(grid_size_M, grid_size_N, grid_size_P);

  // Submit the kernel for execution on the SYCL device
  queue.submit([&](sycl::handler& h) {
    // Access buffers on the device for efficient data movement
    auto a_data = a_buffer.get_access<sycl::access::mode::read>(h);
    auto b_data = b_buffer.get_access<sycl::access::mode::read>(h);
    auto c_data = c_buffer.get_access<sycl::access::mode::write>(h);

    // Initialize a_data and b_data with 1.0f on the device
    h.parallel_for<sycl::nd_range>(a_buffer.get_range(), [=](sycl::id idx) {
      a_data[idx] = T(1.0f);
    });
    h.parallel_for<sycl::nd_range>(b_buffer.get_range(), [=](sycl::id idx) {
      b_data[idx] = T(1.0f);
    });

    // Initialize c_data with 0.0f on the device
    h.parallel_for<sycl::nd_range>(c_buffer.get_range(), [=](sycl::id idx) {
      c_data[idx] = T(0.0f);
    });



    // Declare local space for potential optimizations
    // (e.g., tile storage or partial sums)
    sycl::accessor<T, 1, sycl::access::mode::read_write, sycl::target::local> shared_data(BLOCK_SIZE * BLOCK_SIZE, h);

    // Execute the kernel in parallel using a 3D work-group structure
    h.parallel_for<sycl::nd_range>(work_range, [=](sycl::id idx) {
      int i = idx[0] * BLOCK_SIZE;
      int j = idx[2] * BLOCK_SIZE;

      // Calculate block bounds within the global matrix dimensions
      int i_limit = std::min(i + BLOCK_SIZE, M);
      int j_limit = std::min(j + BLOCK_SIZE, P);

      for (int k = 0; k < N; k += BLOCK_SIZE) {
        int k_limit = std::min(k + BLOCK_SIZE, N);

        // Potentially use local space for optimizations within blocks
        T sum = 0; // Initialize partial sum for potential reduction
        for (int kk = k; kk < k_limit; ++kk) {
          sum += a_data[i * N + kk] * b_data[kk * P + j];
        }

        // Local reduction (optional for certain SYCL implementations)
        // ... (code for reduction within work-group)

        // Write partial or final result to global memory
        for (int ii = i; ii < i_limit; ++ii) {
          for (int jj = j; jj < j_limit; ++jj) {
            c_data[ii * P + jj] += sum;
          }
        }
      }
    });

    // Copy c_data back to host memory for verification
    std::vector<T> c_host(M * P);  // Allocate host memory to hold results
    h.copy(c_data, c_host.data());  // Copy device data to host vector

    // Print c_data on the host (outside the SYCL kernel)
    h.sync();  // Wait for the copy to finish before printing
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < P; ++j) {
        std::cout << c_host[i * P + j] << " ";  // Print each element
      }
      std::cout << std::endl;
    }
  });
}
