use strict;
use warnings;

our $VERSION = 0.16;

use Test::More;
use Benchmark qw/timethese cmpthese :hireswallclock/;

use constant VERBOSE => 1; # prints out several junk

use Inline CUDA => Config =>
	host_code_language => 'C++',
	BUILD_NOISY => 1,
	warnings => 10,
	# C includes
	INC =>
		# include/ is in current test dir (e.g. t/include) and has files needed for testing
		'-I"'.File::Spec->catdir($FindBin::RealBin, 'include').'"'
		.' '
		# C/ contains kernels, headers and is installed along Inline::CUDA in its shared-dir
		.'-I"'.File::Spec->catdir($FindBin::Bin, '..', '..', 'C', 'C').'"'
	,
;
use Inline CUDA => <<'EOC';
// from https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/
// NOTE: don't use cuda_harness(void), use cuda_harness()!!!
#include <stdio.h>

AV *saxpy_cuda(int N, double a, SV *_x, SV *_y);
int array_numelts(SV *array);

__global__
void saxpy(int n, double a, double *x, double *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if( i < n ) y[i] = a * x[i] + y[i];
}

int array_numelts(SV *array){
	int numelts;
	if( (!SvROK(array))
	 || (SvTYPE(SvRV(array)) != SVt_PVAV)
	 || ((numelts = av_len((AV *)SvRV(array))) < 0)
	) return -1;
	return numelts;
}

/* returns an arrayref of results */
AV* saxpy_cuda(
	int N,
	double a,
	SV *_x,
	SV *_y
){
	cudaError_t err;
	double *x, *y, *gpu_x, *gpu_y;
	int nX, nY, i;

	AV *ret = newAV();
	sv_2mortal((SV*)ret);

	if( N <= 0 ){ fprintf(stderr, "saxpy_cuda(): error, N (the vector size) must be positive.\n"); return NULL; }

	if( ((nX=array_numelts(_x))<0)
	  ||((nY=array_numelts(_y))<0)
	){ fprintf(stderr, "saxpy_cuda(): error allocating perl array for returning results.\n"); return NULL; }

	if( (x=(double*)malloc(N*sizeof(double))) == NULL ){ fprintf(stderr, "saxpy_cuda(): error, failed to allocate %zu bytes for x.\n", N*sizeof(double)); return NULL; }
	if( (y=(double*)malloc(N*sizeof(double))) == NULL ){ fprintf(stderr, "saxpy_cuda(): error, failed to allocate %zu bytes for y.\n", N*sizeof(double)); return NULL; }

	AV *deref_x = (AV *)SvRV(_x),
	   *deref_y = (AV *)SvRV(_y);
	SV **dummy;
	for(i=0;i<N;i++){
	  dummy = av_fetch(deref_x, i, 0);
	  x[i] = SvNV(*dummy);
	  dummy = av_fetch(deref_y, i, 0);
	  y[i] = SvNV(*dummy);
	  //fprintf(stdout, "saxpy_cuda() : perl sent these data: x[%d]=%lf and y[%d]=%lf\n", i, x[i], i, y[i]);
	}

	if( (err=cudaMalloc(&gpu_x, N*sizeof(double))) != cudaSuccess ){ fprintf(stderr, "saxpy_cuda(): error, call to cudaMalloc() has failed for %zu bytes for gpu_x: %s\n", N*sizeof(double), cudaGetErrorString(err)); return NULL; }
	if( (err=cudaMalloc(&gpu_y, N*sizeof(double))) != cudaSuccess ){ fprintf(stderr, "saxpy_cuda(): error, call to cudaMalloc() has failed for %zu bytes for gpu_y: %s\n", N*sizeof(double), cudaGetErrorString(err)); return NULL; }

	if( (err=cudaMemcpy(gpu_x, x, N*sizeof(double), cudaMemcpyHostToDevice)) != cudaSuccess ){ fprintf(stderr, "saxpy_cuda(): error, call to cudaMemcpy(cudaMemcpyHostToDevice) (x->gpu_x) has failed for %zu bytes for gpu_x: %s\n", N*sizeof(double), cudaGetErrorString(err)); return NULL; }
	if( (err=cudaMemcpy(gpu_y, y, N*sizeof(double), cudaMemcpyHostToDevice)) != cudaSuccess ){ fprintf(stderr, "saxpy_cuda(): error, call to cudaMemcpy(cudaMemcpyHostToDevice) (y->gpu_y) has failed for %zu bytes for gpu_y: %s\n", N*sizeof(double), cudaGetErrorString(err)); return NULL; }

	int blockSize, gridSize;
 	// Number of threads in each thread block
	blockSize = 1024;
 	// Number of thread blocks in grid
	gridSize = (int)ceil((float)N/blockSize);

	//fprintf(stdout, "saxpy_cuda(): calculated gridSize=%d for blockSize=%d.\n", gridSize, blockSize);

	// Perform SAXPY on 1M elements
	saxpy<<<gridSize, blockSize>>>(N, a, gpu_x, gpu_y);
	if( (err=cudaGetLastError()) != cudaSuccess ){ fprintf(stderr, "saxpy_cuda(): error, failed to launch the kernel into the device: %s\n", cudaGetErrorString(err)); return NULL; }

	// this copies data from GPU (dy) onto CPU memory, we use y because
	// it's just sitting there and no longer needed
	if( (err=cudaMemcpy(y, gpu_y, N*sizeof(double), cudaMemcpyDeviceToHost)) != cudaSuccess ){ fprintf(stderr, "saxpy_cuda(): error, call to cudaMemcpy(cudaMemcpyDeviceToHost) (gpu_y->y) has failed for %zu bytes for gpu_y (getting back results after successfully launching the kernel): %s\n", N*sizeof(double), cudaGetErrorString(err)); return NULL; }

	/* put the result into ret for perl */
	for(i=0;i<N;i++){
		//fprintf(stdout, "saxpy_cuda(): got back result from device: y[%d]=%lf\n", i, y[i]);
		av_push(ret, newSVnv(y[i]));
	}

	if( (err=cudaFree(gpu_x)) != cudaSuccess ){ fprintf(stderr, "saxpy_cuda(): error, call to cudaFree() has failed for gpu_x: %s\n", cudaGetErrorString(err)); return NULL; }
	if( (err=cudaFree(gpu_y)) != cudaSuccess ){ fprintf(stderr, "saxpy_cuda(): error, call to cudaFree() has failed for gpu_y: %s\n", cudaGetErrorString(err)); return NULL; }
	free(x);
	free(y);

	//fprintf(stdout, "done, success.\n");
	return ret;
}
EOC

my $num_repeats = 50;
my $NUM_DATAPOINTS = 1000000;

# shamelessly ripped off App::Benchmark
cmpthese(timethese($num_repeats, {
	'perl : vector size '.$NUM_DATAPOINTS.', repeats '.$num_repeats.':' => \&do_saxpy_perl,
	'cuda : vector size '.$NUM_DATAPOINTS.', repeats '.$num_repeats.':' => \&do_saxpy_cuda
}));

sub saxpy_perl {
	my ($n, $A, $x, $y) = @_;
	for(my $i=0;$i<$n;$i++){ $y->[$i] = $A * $x->[$i] + $y->[$i] }
}
sub do_saxpy_perl {
	my @x = map { rand() } 1..$NUM_DATAPOINTS;
	my @y = map { rand() } 1..$NUM_DATAPOINTS;
	my $A = 3.0;
	saxpy_perl($NUM_DATAPOINTS, $A, \@x, \@y);
}
sub do_saxpy_cuda {
	my @x = map { rand() } 1..$NUM_DATAPOINTS;
	my @y = map { rand() } 1..$NUM_DATAPOINTS;
	my $A = 3.0;
	my $result = saxpy_cuda($NUM_DATAPOINTS, $A, \@x, \@y);
}

ok(1,"OK");
done_testing();
