#include <sycl/sycl.hpp>
#include <iostream>
#include <limits>
#include <chrono>
#include <sycl/ext/intel/fpga_extensions.hpp>

#define M 812
#define N 812
#define P 812
#define BLOCK_SIZE 7

#define SUB_GROUP_SIZE 8

using namespace std;
using namespace sycl;
using namespace std::chrono;

void VerifyResult(float (*c_back)[P]);

int main() {
    #if FPGA_EMULATOR
  // DPC++ extension: FPGA emulator selector on systems without FPGA card.
  ext::intel::fpga_emulator_selector d_selector;
    #elif FPGA
  // DPC++ extension: FPGA selector on systems with FPGA card.
  ext::intel::fpga_selector d_selector;
    #else
  // The default device selector will select the most performant device.
  default_selector d_selector;
    #endif
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

	cout << "Block size: "<<BLOCK_SIZE<<"\n";
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
	
        auto start_time = high_resolution_clock::now();
			q.submit([&](auto &h) {
                                //sycl::stream out(1024,256,h);
				accessor a(a_buf, h, read_only);
				accessor b(b_buf, h, read_only);
				accessor c(c_buf, h, write_only);
				range global {M, P};
				range local {1, BLOCK_SIZE};
				
				h.parallel_for(nd_range{global,local}, [=](nd_item<2> index)[[intel::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
					auto sg = index.get_sub_group();

					//int size = sg.get_local_range()[0];
					//int maxSize = sg.get_max_local_range()[0];
					//out<<"size: "<<size<<" "<<"; maxSize :"<<maxSize<<sycl::endl;
					

					size_t row = index.get_global_id()[0];
					size_t col = index.get_global_id()[1];
					
					size_t i = index.get_local_id()[1];
					float sum = 0.0f;
					
					for(size_t l = 0; l < P; l+=BLOCK_SIZE){
						size_t tileA = a[row][l+i];
						for(size_t k = 0; k < BLOCK_SIZE; k++){
							sum+= group_broadcast(sg, tileA, k) * b[l + k][col];
						}
					}
					c[row][col] = sum;
				});
			});
		q.wait();

        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end_time - start_time);
		
		cout << "Result of matrix multiplication using SYCL: "<<"\n";
        cout << "Execution time parallelized: " << duration.count() << " milliseconds" << "\n";

		


		} catch (sycl::exception const &e) {
			cout << "An exception is caught while multiplying matrices.\n";
			terminate();
		}
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
	
    for (ii = 0; ii < M; ii += BLOCK_SIZE) {
        for (jj = 0; jj < P; jj += BLOCK_SIZE) {
            for (kk = 0; kk < N; kk += BLOCK_SIZE) {
                // Multiply block of A and B
                for (i = ii; i < ii + BLOCK_SIZE && i < M; i++) {
                    for (j = jj; j < jj + BLOCK_SIZE && j < P; j++) {
                        for (k = kk; k < kk + BLOCK_SIZE && k < N; k++) {
                            c_host[i][j] += a_host[i][k] * b_host[k][j];
                        }
                    }
                }
            }
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
