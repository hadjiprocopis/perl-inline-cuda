use 5.006;
use strict;
use warnings;

our $VERSION = 0.16;

use Test::More;

use File::Spec;
use FindBin;

use Inline CUDA => Config =>
	host_code_language => 'C++',
	clean_after_build => 0,
	BUILD_NOISY => 1,
	warnings => 10,
	# C includes
	INC =>
		# include/ is in current test dir (e.g. t/include) and has files needed for testing
		'-I"'.File::Spec->catdir($FindBin::RealBin, 'include').'"'
		.' '
		# C/ contains kernels, headers and is installed along Inline::CUDA in its shared-dir
		.'-I"'.File::Spec->catdir($FindBin::Bin, '..', 'C', 'C').'"'
	,
;

use Inline CUDA => <<'EOC';
// from https://www.quantstart.com/articles/Matrix-Matrix-Multiplication-on-the-GPU-with-Nvidia-CUDA/
#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include <stdlib.h>

using namespace std;
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "dev_array.h"
#include <math.h>

using namespace std;

class XYZ {
	public:
		void matrixMultiplication(float *A, float *B, float *C, int N);
};

__global__ void matrixMultiplicationKernel(float* A, float* B, float* C, int N) {
	int ROW = blockIdx.y*blockDim.y+threadIdx.y;
	int COL = blockIdx.x*blockDim.x+threadIdx.x;

	float tmpSum = 0;

	if (ROW < N && COL < N) {
		// each thread computes one element of the block sub-matrix
		for (int i = 0; i < N; i++) {
			tmpSum += A[ROW * N + i] * B[i * N + COL];
		}
	}
	C[ROW * N + COL] = tmpSum;
}

void XYZ::matrixMultiplication(float *A, float *B, float *C, int N){

	// declare the number of blocks per grid and the number of threads per block
	// use 1 to 512 threads per block
	dim3 threadsPerBlock(N, N);
	dim3 blocksPerGrid(1, 1);
		if (N*N > 512){
			threadsPerBlock.x = 512;
			threadsPerBlock.y = 512;
			blocksPerGrid.x = ceil(double(N)/double(threadsPerBlock.x));
			blocksPerGrid.y = ceil(double(N)/double(threadsPerBlock.y));
		}

	matrixMultiplicationKernel<<<blocksPerGrid,threadsPerBlock>>>(A, B, C, N);
}

int cuda_harness()
{
	XYZ *myXYZ = new XYZ();

	// Perform matrix multiplication C = A*B
	// where A, B and C are NxN matrices
	int N = 2;
	int SIZE = N*N;

	// Allocate memory on the host
	vector<float> h_A(SIZE);
	vector<float> h_B(SIZE);
	vector<float> h_C(SIZE);

	// Initialize matrices on the host
	for (int i=0; i<N; i++){
		for (int j=0; j<N; j++){
			h_A[i*N+j] = (float )drand48();
			h_B[i*N+j] = (float )drand48();
		}
	}

	// Allocate memory on the device
	dev_array<float> d_A(SIZE);
	dev_array<float> d_B(SIZE);
	dev_array<float> d_C(SIZE);

	d_A.set(&h_A[0], SIZE);
	d_B.set(&h_B[0], SIZE);

	myXYZ->matrixMultiplication(d_A.getData(), d_B.getData(), d_C.getData(), N);
	cudaDeviceSynchronize();

	d_C.get(&h_C[0], SIZE);
	cudaDeviceSynchronize();

	float *cpu_C;
	cpu_C=new float[SIZE];

	printf("Matrix A:\n");
	for (int row=0; row<N; row++){
		for (int col=0; col<N; col++){
			printf("%lf ", h_A[row*N+col]);
		}
		printf("\n");
	}
	printf("Matrix B:\n");
	for (int row=0; row<N; row++){
		for (int col=0; col<N; col++){
			printf("%lf ", h_B[row*N+col]);
		}
		printf("\n");
	}
	printf("Matrix C:\n");
	for (int row=0; row<N; row++){
		for (int col=0; col<N; col++){
			printf("%lf ", h_C[row*N+col]);
		}
		printf("\n");
	}
/*
	// Now do the matrix multiplication on the CPU
	float sum;
	for (int row=0; row<N; row++){
		for (int col=0; col<N; col++){
			sum = 0.f;
			for (int n=0; n<N; n++){
				sum += h_A[row*N+n]*h_B[n*N+col];
			}
			cpu_C[row*N+col] = sum;
		}
	}

	double err = 0;
	// Check the result and make sure it is correct
	for (int ROW=0; ROW < N; ROW++){
		for (int COL=0; COL < N; COL++){
			err += cpu_C[ROW * N + COL] - h_C[ROW * N + COL];
		printf("
		}
	}

	cout << "Error: " << err << endl;
*/
	return 0;
}

EOC

my $ret = cuda_harness();
is($ret, 0, "cuda_harness() : called.");
done_testing();
