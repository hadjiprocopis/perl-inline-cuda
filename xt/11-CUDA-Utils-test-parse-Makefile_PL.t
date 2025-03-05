use 5.006;
use strict;
use warnings;

use File::Temp;
use File::Path qw/rmtree/;

our $VERSION = 0.16;

use Test::More;

use 5.006;
use strict;
use warnings;

our $VERSION = 0.16;

use Test::More;
use File::Temp qw/tempdir/;

use Inline::CUDA::Utils;

my ($tmpdir, $o);
BEGIN {
	$tmpdir = tempdir();
}

use Inline CUDA => Config =>
directory => '/tmp/shit1',
	clean_after_build => 0,
	DIRECTORY => $tmpdir, # dir must exist
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
	pass_me_o => \$o, # get the effing Inline obj
;

use Inline CUDA => <<'EOCUDA';
#include <stdio.h>

#define N 1000

__global__
void add(int *a, int *b) {
    int i = blockIdx.x;
    if (i<N) {
        b[i] = a[i]+b[i];
	}
}

int cuda_harness() {
	cudaError_t err;
	int ha[N], hb[N];
	int *da, *db;
	if( (err=cudaMalloc((void **)&da, N*sizeof(int))) != cudaSuccess ){ fprintf(stderr, "cuda_harness(): error, call to cudaMalloc() has failed for %zu bytes for da: %s\n", N*sizeof(int), cudaGetErrorString(err)); return 1; }
	if( (err=cudaMalloc((void **)&db, N*sizeof(int))) != cudaSuccess ){ fprintf(stderr, "cuda_harness(): error, call to cudaMalloc() has failed for %zu bytes for db: %s\n", N*sizeof(int), cudaGetErrorString(err)); return 1; }
	for (int i = 0; i<N; ++i) {
		ha[i] = i;
	}
	if( (err=cudaMemcpy(da, ha, N*sizeof(int), cudaMemcpyHostToDevice)) != cudaSuccess ){ fprintf(stderr, "cuda_harness(): error, call to cudaMemcpy(cudaMemcpyHostToDevice) has failed for %zu bytes for ha->da: %s\n", N*sizeof(int), cudaGetErrorString(err)); return 1; }
	add<<<N, 1>>>(da, db);
	if( (err=cudaGetLastError()) != cudaSuccess ){ fprintf(stderr, "cuda_harness(): error, failed to launch the kernel into the device: %s\n", cudaGetErrorString(err)); return 1; }

	if( (err=cudaMemcpy(hb, db, N*sizeof(int), cudaMemcpyDeviceToHost)) != cudaSuccess ){ fprintf(stderr, "cuda_harness(): error, call to cudaMemcpy(cudaMemcpyDeviceToHost) has failed for %zu bytes for db->ha: %s\n", N*sizeof(int), cudaGetErrorString(err)); return 1; }

/*
	for (int i = 0; i<N; ++i) {
		printf("%d\n", hb[i]);
	}
*/
	if( (err=cudaFree(da)) != cudaSuccess ){ fprintf(stderr, "cuda_harness(): error, call to cudaFree() has failed for da: %s\n", cudaGetErrorString(err)); return 1; }
	if( (err=cudaFree(db)) != cudaSuccess ){ fprintf(stderr, "cuda_harness(): error, call to cudaFree() has failed for db: %s\n", cudaGetErrorString(err)); return 1; }

	return 0;
}
EOCUDA

ok(defined($tmpdir), "tempdir() called.");
ok(-d $tmpdir, "tempdir '$tmpdir' exists.");
my $retcode = cuda_harness();
ok(defined($retcode), "cuda_harness() : called.");
is($retcode, 0, "cuda_harness() : returned success.");

my $build_dir = $o->{API}->{build_dir};
ok(defined($build_dir), "\$o->{API}->{build_dir} is defined.");
ok(-d $build_dir, "\$o->{API}->{build_dir} is '$build_dir' and is a valid dir.");
my $mfile = File::Spec->catdir($build_dir, 'Makefile.PL');
ok(-f $mfile, "Makefile.PL ($mfile) is a valid file.");

# now we have a build dir ($tmpdir/_Inline/...) and a Makefile.PL somewhere in there
my $flags = Inline::CUDA::Utils::enquire_Makefile_PL(
	$build_dir,
	'Makefile.PL',
	1 # noisy
);
ok(defined($flags), "enquire_Makefile_PL() : called.");

for my $k (@Inline::CUDA::Utils::MAKEFILE_PL_FLAGS_KEYS){
	# for each *FLAGS (e.g. CCFLAGS) we have $flags->{CCFLAGS}
	# as a string of flags (space separated)
	ok(	exists($flags->{$k})
		&& defined($flags->{$k})
		&& ($k!~/^\s*$/)
	  , "key '$k' exists and is not empty."
	) and diag("$k => ".$flags->{$k});

	# but we also have split the string of flags to separate flags
	# each key in e.g. 'hash-CCFLAGS' is a flag on its own
	my $K = 'hash-'.$k;
	ok(	exists($flags->{$K})
		&& defined($flags->{$K})
		&& (ref($flags->{$K}) eq 'HASH')
	  , "key '$K' exists and is not empty."
	);
	diag("$K => <<".join(">>, <<", sort keys %{$flags->{$K}}).">>");
}
# rmtree $dir,$verbose,$safe
#ok(rmtree($tmpdir,0,0), "removed tmpdir '$tmpdir'.");

done_testing();
