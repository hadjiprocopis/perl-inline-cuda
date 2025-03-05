use 5.006;
use strict;
use warnings;

our $VERSION = 0.16;

use Test::More;

# this test will instruct Inline::CUDA to
# use the system compiler, and nvcc to
# not check compiler versions
# it can well fail

use Inline CUDA => Config =>
	host_code_language => 'c++',
	cc => '/usr/bin/cc',
	ld => '/usr/bin/cc',
	# this tells nvcc to not check compiler version
	nvccflags => '--allow-unsupported-compiler',
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

#include <stdio.h>

//
// Nearly minimal CUDA example.
// Compile with:
//
// nvcc -o example example.cu
//

#define N 1000

//
// A function marked __global__
// runs on the GPU but can be called from
// the CPU.
//
// This function adds the elements of two vectors (element-wise)
//
// The entire computation can be thought of as running
// with one thread per array element with blockIdx.x
// identifying the thread.
//
// The comparison i<N is because often it isn't convenient
// to have an exact 1-1 correspondence between threads
// and array elements. Not strictly necessary here.
//
// Note how we're mixing GPU and CPU code in the same source
// file. An alternative way to use CUDA is to keep
// C/C++ code separate from CUDA code and dynamically
// compile and load the CUDA code at runtime, a little
// like how you compile and load OpenGL shaders from
// C/C++ code.
//
__global__
void add(int *a, int *b) {
    int i = blockIdx.x;
    if (i<N) {
        b[i] = a[i]+b[i];
	}
}

int cuda_harness() {
	cudaError_t err;
	//
	// Create int arrays on the CPU.
	// ('h' stands for "host".)
	//
	int ha[N], hb[N];

	//
	// Create corresponding int arrays on the GPU.
	// ('d' stands for "device".)
	//
	int *da, *db;
	if( (err=cudaMalloc((void **)&da, N*sizeof(int))) != cudaSuccess ){ fprintf(stderr, "cuda_harness(): error, call to cudaMalloc() has failed for %zu bytes for da: %s\n", N*sizeof(int), cudaGetErrorString(err)); return 1; }
	if( (err=cudaMalloc((void **)&db, N*sizeof(int))) != cudaSuccess ){ fprintf(stderr, "cuda_harness(): error, call to cudaMalloc() has failed for %zu bytes for db: %s\n", N*sizeof(int), cudaGetErrorString(err)); return 1; }

	//
	// Initialise the input data on the CPU.
	//
	for (int i = 0; i<N; ++i) {
		ha[i] = i;
	}

	//
	// Copy input data to array on GPU.
	//
	if( (err=cudaMemcpy(da, ha, N*sizeof(int), cudaMemcpyHostToDevice)) != cudaSuccess ){ fprintf(stderr, "cuda_harness(): error, call to cudaMemcpy(cudaMemcpyHostToDevice) has failed for %zu bytes for ha->da: %s\n", N*sizeof(int), cudaGetErrorString(err)); return 1; }

	//
	// Launch GPU code with N threads, one per
	// array element.
	//
	add<<<N, 1>>>(da, db);
	if( (err=cudaGetLastError()) != cudaSuccess ){ fprintf(stderr, "cuda_harness(): error, failed to launch the kernel into the device: %s\n", cudaGetErrorString(err)); return 1; }

	//
	// Copy output array from GPU back to CPU.
	//
	if( (err=cudaMemcpy(hb, db, N*sizeof(int), cudaMemcpyDeviceToHost)) != cudaSuccess ){ fprintf(stderr, "cuda_harness(): error, call to cudaMemcpy(cudaMemcpyDeviceToHost) has failed for %zu bytes for db->ha: %s\n", N*sizeof(int), cudaGetErrorString(err)); return 1; }

/*
	for (int i = 0; i<N; ++i) {
		printf("%d\n", hb[i]);
	}
*/
	//
	// Free up the arrays on the GPU.
	//
	if( (err=cudaFree(da)) != cudaSuccess ){ fprintf(stderr, "cuda_harness(): error, call to cudaFree() has failed for da: %s\n", cudaGetErrorString(err)); return 1; }
	if( (err=cudaFree(db)) != cudaSuccess ){ fprintf(stderr, "cuda_harness(): error, call to cudaFree() has failed for db: %s\n", cudaGetErrorString(err)); return 1; }

	return 0;
}

EOC

my $retcode = cuda_harness();

is($retcode, 0, "cuda_harness() : called.");
done_testing();
