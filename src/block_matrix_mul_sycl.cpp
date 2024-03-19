#include <CL/sycl.hpp>
#include <sycl/sycl.hpp>
#include <iostream>
#include <limits>
#include <chrono>


using namespace std;
using namespace sycl;
using namespace std::chrono;

// Matrix size constants.
constexpr int M = 4;
constexpr int N = 4;
constexpr int P = 4;

const int BLOCK_SIZE = 32; // Adjust based on your hardware
const int TILE_SIZE = 4;  // Adjust based on cache size and blocking strategy

template <typename T>
void blocked_matrix_multiply_sycl(sycl::queue& queue, const sycl::buffer<T, 1>& a_buffer,
                                 const sycl::buffer<T, 1>& b_buffer, sycl::buffer<T, 1>& c_buffer,
                                 int M, int N, int P) {
  // ... (rest of the code for grid dimensions and work range)

  // Submit the kernel for execution on the SYCL device
  queue.submit([&](sycl::handler& h) {
    // Access buffers on the device for efficient data movement
    auto a_data = a_buffer.get_access<sycl::access::mode::read_only>(h);
    auto b_data = b_buffer.get_access<sycl::access::mode::read_only>(h);
    auto c_data = c_buffer.get_access<sycl::access::mode::write_only>(h);

    // Execute the blocked matrix multiplication kernel (improved)
    h.parallel_for<sycl::nd_range>(range(M / TILE_SIZE, N / TILE_SIZE, P / TILE_SIZE), [=](sycl::id idx) {
      int i_block = idx[0] * TILE_SIZE;
      int j_block = idx[2] * TILE_SIZE;

      for (int k = 0; k < N; k += BLOCK_SIZE) {
        for (int i_tile = 0; i_tile < TILE_SIZE; ++i_tile) {
          int i = i_block + i_tile;
          if (i >= M) continue;  // Skip out-of-bounds elements

          for (int j_tile = 0; j_tile < TILE_SIZE; ++j_tile) {
            int j = j_block + j_tile;
            if (j >= P) continue;  // Skip out-of-bounds elements

            float sum = 0.0f;
            for (int kk = k; kk < k + BLOCK_SIZE; ++kk) {
              sum += a_data[i * N + kk] * b_data[kk * P + j];
            }
            c_data[i * P + j] = sum;
          }
        }
      }
    });
  });
}

int main() {
  // ... (SYCL queue creation, etc.)
  sycl::queue q(sycl::default_selector_v);

  // Allocate host memory for input matrices (optional)
  std::vector<float> a_data(M * N);
  std::vector<float> b_data(N * P);

  // Initialize input matrices (optional)
  //fill a_data
  std::fill(a_data.begin(), a_data.end(), 1.0f);
  //fill b_data
  for (int i = 0; i < N; ++i) {
  for (int j = 0; j < P; ++j) {
    int index = i * P + j;
    b_data[index] = i + 1.0f;  // Values start from 1.0f for each row
  }
}

  // Allocate device memory for result (optional)
  sycl::buffer<float, 1> c_buffer(range(M, P));

  // Create buffers on the device
  sycl::buffer<float, 2> a_buf(range(M, N), a_data.data());
  sycl::buffer<float, 2> b_buf(range(N, P), b_data.data());

  // Call blocked matrix multiplication
  blocked_matrix_multiply_sycl(q, a_buf, b_buf, c_buffer, M, N, P);

  // Wait for SYCL execution (optional)
  q.wait();

   Access results (optional)
   If using host memory for c_data:
   std::cout << "Result matrix: " << std::endl;
   for (int i = 0; i < M; ++i) {
     for (int j = 0; j < P; ++j) {
       std::cout << c_data[i * P + j] << " ";
    }
     std::cout << std::endl;
   }

  // If using device memory for c_data:
  // ... (copy data back to host and then access it)

  return 0;
}
