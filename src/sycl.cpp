#include <CL/sycl.hpp>
#include <iomanip>
#include <iostream>
#include <ctime>
#include <limits>
#include <chrono>

using namespace sycl;
using namespace std;
using namespace sycl;
using namespace std::chrono;

//Matrix size constants
constexpr int m = 1000;
constexpr int n = 1000;
constexpr int p = 1000;
const int block_size = 10;

// Kernel for blocked matrix multiplication
class mxm_kernel {
public:
  mxm_kernel(int m, int n, int p, const buffer<float>& A_data, const buffer<float>& B_data, buffer<float>& C_data)
      : m_(m), n_(n), p_(p), A_(A_data), B_(B_data), C_(C_data) {}

  void operator()(sycl::nd_item it) const {
    int i = it.get_global_id(0) * block_size;
    int j = it.get_global_id(1) * block_size;

    for (int k = 0; k < p_; k += block_size_) {
      int k_limit = std::min(k + block_size, p_);

      float sum = 0.0f;
      for (int kk = k; kk < k_limit; ++kk) {
        // Access elements from A and B within their blocks
        float A_val = A_[i * p_ + kk];
        float B_val = B_[kk * n_ + j];
        sum += A_val * B_val;
      }
      C_[i * n_ + j] = sum;
    }
  }

private:
  int m_, n_, p_;
  buffer<float> A_, B_, C_;
};

int main() {
  // Define matrix dimensions
  //int m = 1024;
  //int n = 1024;
  //int p = 1024;

  // Allocate host memory for matrices
  std::vector<float> A(m * p), B(p * n), C(m * n);

  // Initialize matrices (replace with your initialization logic)
  for (int i = 0; i < m * p; ++i) {
    A[i] = 1.0f;
  }
  for (int i = 0; i < p * n; ++i) {
    B[i] = 2.0f;
  }

  // Create SYCL device and queue
  device dev = device::gpu_selector{};
  context ctx(dev);
  queue q(ctx);

  // Allocate SYCL buffers for matrices
  buffer<float, 1> A_data(A.data(), A.size());
  buffer<float, 1> B_data(B.data(), B.size());
  buffer<float, 1> C_data(C.data(), C.size());

  // Measure execution time
  auto start_time = std::chrono::high_resolution_clock::now();

  // Create and run the kernel
  q.submit([&](handler& h) {
    h.parallel_for(nd_range(range(m / block_size, n / block_size), block(block_size, block_size)), mxm_kernel(m, n, p, A_data, B_data, C_data));
  });

  q.wait();

  auto end_time = std::chrono::high_resolution_clock::now();
  double execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;

  // Print execution time
  std::cout << "Execution time: " << std::fixed << std::setprecision(3) << execution_time << " seconds" << std::endl;

  // (Optional) Copy result back to host memory (if needed)
  // q.read(C_data, C.data());

  return 0;
}
