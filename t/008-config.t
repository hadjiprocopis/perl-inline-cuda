use 5.006;
use strict;
use warnings;

our $VERSION = 0.14;

use Test::More;

use File::Spec;
use FindBin;

# test with not specifying cc,
# it should just work if it was installed correctly
# if it fails it most likely mean that the version
# of the NVIDIA CUDA toolkit installed is not compatible
# with the C compiler found/user-specified during
# installation.
# This is a big headache 

use Inline CUDA => Config =>
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
#define N 10
__global__
void lametest(int *a) {
    int i = blockIdx.x;
    if (i<N) a[i] = 2.0 * a[i];
}
int main(){
	cudaError_t err;
	int ha[N], i, *da;
	for(i=0;i<N;i++) ha[i] = i;
	if( (err=cudaMalloc((void **)&da, N*sizeof(int))) != cudaSuccess ){ fprintf(stderr, "main(): error, call to cudaMalloc() has failed for %zu bytes for db: %s\n", N*sizeof(int), cudaGetErrorString(err)); return 1; }
	if( (err=cudaMemcpy(da, ha, N*sizeof(int), cudaMemcpyHostToDevice)) != cudaSuccess ){ fprintf(stderr, "main(): error, call to cudaMemcpy(cudaMemcpyHostToDevice) has failed for %zu bytes for ha->da: %s\n", N*sizeof(int), cudaGetErrorString(err)); return 1; }
	// triple angle brackets !!! <<< >>>
	lametest<<<N,1>>>(da);
	if( (err=cudaGetLastError()) != cudaSuccess ){ fprintf(stderr, "main(): error, failed to launch the kernel into the device: %s\n", cudaGetErrorString(err)); return 1; }
	return 0;
}
EOC
my $retcode = main();
is($retcode, 0, "main() : called.");
done_testing();

