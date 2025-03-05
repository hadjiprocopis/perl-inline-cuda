use 5.006;
use strict;
use warnings;

our $VERSION = 0.16;

# this tries to run Inline in try/catch just for the sake of it, do not use in production

# WARNING: these tests are run under try/catch which is not recommended
# for production code, it works but ...

use Test::More;

use File::Spec;
use FindBin;
use Try::Tiny;

# DO NOT CHANGE N=10 (we change N via regex in the $code below)

my ($retcode, $failed);
my $expected_sum_N10 = 90;  # this is the expected result of the cuda_harness() for specific N=10
my $expected_sum_N20 = 380; # this is the expected result of the cuda_harness() for specific N=20
my $expected_sum_N30 = 870; # this is the expected result of the cuda_harness() for specific N=30
my $code =<<'EOC';
#include <stdio.h>
#define N 10
__global__
// gets an array of numbers and doubles each item 2->4
void lametest(int *a) {
    int i = blockIdx.x;
    if (i<N) a[i] = 2.0 * a[i];
}
// returns -1 on failure or >0 (a sum) on success, for N=10 it returns 90
int cuda_harness(){
	fprintf(stdout, "cuda_harness() : called ...\n");
	cudaError_t err;
	int ha[N], i, *da;
	for(i=0;i<N;i++) ha[i] = i;
	if( (err=cudaMalloc((void **)&da, N*sizeof(int))) != cudaSuccess ){ fprintf(stderr, "cuda_harness(): error, call to cudaMalloc() has failed for %zu bytes for db: %s\n", N*sizeof(int), cudaGetErrorString(err)); return -1; }
	if( (err=cudaMemcpy(da, ha, N*sizeof(int), cudaMemcpyHostToDevice)) != cudaSuccess ){ fprintf(stderr, "cuda_harness(): error, call to cudaMemcpy(cudaMemcpyHostToDevice) has failed for %zu bytes for ha->da: %s\n", N*sizeof(int), cudaGetErrorString(err)); return -1; }
	// triple angle brackets !!! <<< >>>
	lametest<<<N,1>>>(da);
	if( (err=cudaGetLastError()) != cudaSuccess ){ fprintf(stderr, "cuda_harness(): error, failed to launch the kernel into the device: %s\n", cudaGetErrorString(err)); return -1; }
	if( (err=cudaMemcpy(ha, da, N*sizeof(int), cudaMemcpyDeviceToHost)) != cudaSuccess ){ fprintf(stderr, "cuda_harness(): error, call to cudaMemcpy(cudaMemcpyHostToDevice) has failed for %zu bytes for ha->da: %s\n", N*sizeof(int), cudaGetErrorString(err)); return -1; }

	int sum = 0;
	for(i=0;i<N;i++){
	 sum += ha[i];
	}
	return sum;
}
EOC

$failed = undef;
try {
  use Inline;
  Inline->bind( CUDA => $code,
	BUILD_NOISY => 1,
	clean_after_build => 0,
	warnings => 10,
  );
} catch { $failed = $_ };
ok(! defined $failed, "returned good result.") or BAIL_OUT("failed: $failed");
$retcode = cuda_harness();
is($retcode, $expected_sum_N10, "returned expected ${expected_sum_N10}") or BAIL_OUT("no it returned $retcode");

#### Now let's modify the source code and see if the change is coming back for calling it another time...
$failed = undef;
$code =~ s/called/called again!!!!/g;
$code =~ s/define N \d+/define N 20/;
try {
  use Inline;
  Inline->bind( CUDA => $code,
	BUILD_NOISY => 1,
	clean_after_build => 0,
	warnings => 10,
  );
} catch { $failed = $_ };
ok(! defined $failed, "returned good result.") or BAIL_OUT("failed: $failed");
$retcode = cuda_harness();
is($retcode, $expected_sum_N20, "returned expected ${expected_sum_N20}") or BAIL_OUT("no it returned $retcode");

### this must fail because of bogus configuration
$failed = undef;
try {
  use Inline;
  Inline->bind( CUDA => $code,
	BUILD_NOISY => 1,
	clean_after_build => 0,
	warnings => 10,
	# this contains bogus CC, compilation will fail
	CONFIGURATION_FILE => File::Spec->catdir($FindBin::RealBin, 't-data', 'Inline-CUDA.fail.conf'),
  );
} catch { $failed = $_ };
ok(defined $failed, "failed as expected because of bogus configuration.") or BAIL_OUT("failed to fail");

#### and let's see with this expected success following the failure:
#### Now let's modify the source code and see if the change is coming back for calling it another time...
$failed = undef;
$code =~ s/called/called again!!!!/g;
$code =~ s/define N \d+/define N 30/;
try {
  use Inline;
  Inline->bind( CUDA => $code,
	BUILD_NOISY => 1,
	clean_after_build => 0,
	warnings => 10,
  );
} catch { $failed = $_ };
ok(! defined $failed, "returned good result.") or BAIL_OUT("failed: $failed");
$retcode = cuda_harness();
is($retcode, $expected_sum_N30, "returned expected ${expected_sum_N30}") or BAIL_OUT("no it returned $retcode");

done_testing();

