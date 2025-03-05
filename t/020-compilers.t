use strict;
use warnings;

# This tests that compiler executables contained within
# the configuration file exist on their specified path
# are executables

our $VERSION = 0.16;

use Test::More;
use FindBin;
use File::Spec;

use Inline::CUDA::Utils;

my ($current_user, @dummy) = getpwuid($<);

my $conf = Inline::CUDA::Utils::read_configuration();
ok(defined($conf), "Inline::CUDA::Utils::read_configuration() : called.")
  or BAIL_OUT("can not continue without configuration")
;

for my $k (qw/cc cxx ld nvcc nvlink/){
	ok(exists($conf->{$k}) && defined($conf->{$k}), "key '$k' exists in configuration file.");
	my $aexe = File::Spec->catdir($FindBin::Bin, 'bin', 'dummy cc with spaces .pl');
	ok(-f $aexe, "File '$aexe' exists.");
	ok(-x $aexe, "File '$aexe' is executable by current user ($current_user) - warning, it may not be for other users!.");
#	$conf->{$k} = Inline::CUDA::Utils::quoteme($aexe);
}
my $inlinecuda_dir = File::Spec->catdir($FindBin::Bin, '..', 'C', 'C');
my $ret = Inline::CUDA::Utils::test_compilers($inlinecuda_dir, $conf);
is($ret, 0, "Inline::CUDA::Utils::test_compilers(): called.");
if( $ret != 0 ){
	# I need to print this multi-line message to user
	print STDERR <<EOEM;
failed to compile a test CUDA program.
This means one or all of the following:

1) Cuda SDK was not installed
   (see https://developer.nvidia.com/cuda-toolkit)
2) Cuda compiler (nvcc) could not be found on your
   system although it exists because the executable (nvcc)
   or the headers (*.h) or the libraries that it needs
   does not know where they were installed.

   Make sure that in the Inline::CUDA configuration file
   (at the stage of testing this should be located at
      blib/lib/auto/share/dist/Inline-CUDA/Inline-CUDA.conf
    after installation it is copied to
      <INSALLDIR>/lib/perl5/auto/share/dist/Inline-CUDA/Inline-CUDA.conf
   )
   there is an entry specifying the path to the EXECUTABLE nvcc file, like:
       nvcc=/usr/local/cuda/bin/nvcc
   the executable expects headers and libraries to be found relative to it
   so, nothing else to do if your installation is correct.
   It is worth checking if these files/dirs exist
   (note substitute /usr/local/cuda with the CUDA INSTALL DIR)
    /usr/local/cuda/include/cuda.h
    /usr/local/cuda/lib64
       or
    /usr/local/cuda/lib
   ).
   It could be easier to just re-install your Cuda SDK and be done with it.
   Additionally, check if you can compile a test program:

   // begin test program (save it as 'test.cuda'):
#include <stdio.h>
#define N 10
__global__
void lametest(int *a) {
    int i = blockIdx.x;
    if (i<N) a[i] = 2.0 * a[i];
}
int cuda_harness(){
	cudaError_t err;
	int ha[N], i, *da;
	for(i=0;i<N;i++) ha[i] = i;
	if( (err=cudaMalloc((void **)&da, N*sizeof(int))) != cudaSuccess ){ fprintf(stderr, "cuda_harness(): error, call to cudaMalloc() has failed for %zu bytes for db: %s\n", N*sizeof(int), cudaGetErrorString(err)); return 1; }
	if( (err=cudaMemcpy(da, ha, N*sizeof(int), cudaMemcpyHostToDevice)) != cudaSuccess ){ fprintf(stderr, "cuda_harness(): error, call to cudaMemcpy(cudaMemcpyHostToDevice) has failed for %zu bytes for ha->da: %s\n", N*sizeof(int), cudaGetErrorString(err)); return 1; }
	// triple angle brackets !!! <<< >>>
	lametest<<<N,1>>>(da);
	if( (err=cudaGetLastError()) != cudaSuccess ){ fprintf(stderr, "cuda_harness(): error, failed to launch the kernel into the device: %s\n", cudaGetErrorString(err)); return 1; }
	return 0; # success
}
   // end test program

   and then:
   nvcc test.cuda 
3) This is usually the most common cause of failures:
   There is no C/C++ compiler on your system (or in the PATH)
   and/or the C/C++ compiler *found* is not compatible with your
   version of Cuda SDK. For example, in Linux Cuda SDK 10.1 or 10.2
   require a gcc version of 8.* (I do not know for other compilers).
   In Linux, your system compiler is upgraded by your package manager
   regularly (and currently mine is at 11.2 and growing).
   However, because of GPU hardware versions you are most likely
   restricted to specific Cuda SDK versions. See this for more:
    https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version
   So, you must now install a new C/C++ compiler at the version
   Cuda SDK requires and place it in your system so that it does not
   interfere (or overwrites!no no!) your current system compiler.
   It is preferred to install it without being in the PATH, i.e.
   in an obscure, public dir (e.g. /usr/local/gcc82) AND THEN LET
   nvcc know where this compiler is located. Which can be achieved
   by editing Inline::CUDA config file (mentioned above) and
   amending all entries pertaining to the C/C++ compiler, linker, etc.
   Most likely these (change the path to your /usr/local/gcc82/bin/...):
     cc=/usr/bin/cc
     cxx=/usr/bin/c++
     ld=/usr/bin/c++

   Again, you should change /usr/bin/ to /usr/local/gcc82/bin
   (for example, '/usr/local/gcc82' was an example)
   AND ALSO MAKE SURE that /usr/local/gcc82/bin/cc does exist
   or change its name to /usr/local/gcc82/bin/gcc
   or make a link /usr/local/gcc82/bin/cc -> /usr/local/gcc82/bin/gcc

   If you can re-install the package, then the easiest way would be
   to specify the paths for 'cc', 'cxx', 'ld' using ENVIRONMENT VARIABLES
   when creating the Makefile.

   This should do it using temporary ENV vars in a bash-based terminal:
   CC=/usr/local/gcc82/bin/gcc \
   CXX=/usr/local/gcc82/bin/g++ \
   LD=/usr/local/gcc82/bin/g++ \
   perl Makefile.PL

   That's the end. If none of these work then drop me a line or file a bug.
EOEM
	BAIL_OUT("can not continue");
}

done_testing;
