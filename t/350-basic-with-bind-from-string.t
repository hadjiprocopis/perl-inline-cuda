use 5.006;
use strict;
use warnings;

our $VERSION = 0.16;

use Test::More;

use File::Spec;
use FindBin;

use Inline CUDA => Config =>
	clean_after_build => 1,
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

my $codestr = <<'EOC';
/* this is C code with CUDA extensions */
#include <stdio.h>

#define N 1000

#define CUDA_GLOBAL __global__
/* This is the CUDA Kernel which nvcc compiles: */
CUDA_GLOBAL
void add(int *a, int *b) {
        int i = blockIdx.x;
        if (i<N) b[i] = a[i]+b[i];
}
/* this function can be called from Perl.
   It returns 0 on success or 1 on failure.
   This simple code does not support passing parameters in,
   which is covered elsewhere.
*/
int do_add() {
        cudaError_t err;

        // Create int arrays on the CPU.
        // ('h' stands for "host".)
        int ha[N], hb[N];

        // Create corresponding int arrays on the GPU.
        // ('d' stands for "device".)
        int *da, *db;
        if( (err=cudaMalloc((void **)&da, N*sizeof(int))) != cudaSuccess ){
                fprintf(stderr, "do_add(): error, call to cudaMalloc() has failed for %zu bytes for da: %s\n",
                        N*sizeof(int), cudaGetErrorString(err)
                );
                return 1;
        }
        if( (err=cudaMalloc((void **)&db, N*sizeof(int))) != cudaSuccess ){
                fprintf(stderr, "do_add(): error, call to cudaMalloc() has failed for %zu bytes for db: %s\n",
                        N*sizeof(int), cudaGetErrorString(err)
                );
                return 1;
        }

        // Initialise the input data on the CPU.
        for (int i = 0; i<N; ++i) ha[i] = i;

        // Copy input data to array on GPU.
        if( (err=cudaMemcpy(da, ha, N*sizeof(int), cudaMemcpyHostToDevice)) != cudaSuccess ){
                fprintf(stderr, "do_add(): error, call to cudaMemcpy(cudaMemcpyHostToDevice) has failed for %zu bytes for ha->da: %s\n",
                        N*sizeof(int), cudaGetErrorString(err)
                );
                return 1;
        }

        // Launch GPU code with N threads, one per array element.
        add<<<N, 1>>>(da, db);
        if( (err=cudaGetLastError()) != cudaSuccess ){
                fprintf(stderr, "do_add(): error, failed to launch the kernel into the device: %s\n",
                        cudaGetErrorString(err)
                );
                return 1;
        }

        // Copy output array from GPU back to CPU.
        if( (err=cudaMemcpy(hb, db, N*sizeof(int), cudaMemcpyDeviceToHost)) != cudaSuccess ){
                fprintf(stderr, "do_add(): error, call to cudaMemcpy(cudaMemcpyDeviceToHost) has failed for %zu bytes for db->ha: %s\n",
                        N*sizeof(int), cudaGetErrorString(err)
                );
                return 1;
        }

        //for (int i = 0; i<N; ++i) printf("%d\n", hb[i]); // print results

        // Free up the arrays on the GPU.
        if( (err=cudaFree(da)) != cudaSuccess ){
                fprintf(stderr, "do_add(): error, call to cudaFree() has failed for da: %s\n",
                        cudaGetErrorString(err)
                );
                return 1;
        }
        if( (err=cudaFree(db)) != cudaSuccess ){
                fprintf(stderr, "do_add(): error, call to cudaFree() has failed for db: %s\n",
                        cudaGetErrorString(err)
                );
                return 1;
        }

        return 0;
}
EOC

Inline->bind(CUDA => $codestr);

my $retcode = do_add();

is($retcode, 0, "do_add() : called.");
done_testing();


