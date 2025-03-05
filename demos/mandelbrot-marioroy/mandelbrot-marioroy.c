#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

// helper functions to facilitate the bridge with perl
#include "inlinecuda/perlbridge.c"

#include "mandelbrot-marioroy.h"

#include "mandelbrot-marioroy.cuda"

/* what Perl should be able to see and call */
int mandelbrot_main(
	double min_x, double min_y,
	double step_x, double step_y,
	SV *perl_result, SV *perl_colors,
	int iters,
	int width, int height,
	int bDimX, int bDimY,
	int gDimX, int gDimY
);

/* Returns 1 on failure,
	   0 on success
*/
int mandelbrot_main(
	double min_x, double min_y,
	double step_x, double step_y,
	SV *perl_result, SV *perl_colors,
	int iters,
	int width, int height,
	int bDimX, int bDimY,
	int gDimX, int gDimY
){
	cudaError_t err;
	size_t i;

	fprintf(stdout, "mandelbrot_main() : called for %d x %d ...\n", width, height);

	size_t colors_sz;
	if( array_numelts_1D(perl_colors, &colors_sz) ){ fprintf(stderr, "mandelbrot_main() : error, input parameter 'perl_colors' is not an array-ref.\n"); return 1; }
	fprintf(stdout, "mandelbrot_main() : a palette of %zu colors (triplets) provided.\n", colors_sz/3);

	// check the perl_result which will take results back to caller
	// we expect an arrayref of a totally empty array, e.g.: my @R; xx(\@R);
	// or a scalar where to place the ref to the new array we allocate here, e.g. my $x; xx($x);
	fprintf(stdout, "mandelbrot_main() : checking what kind of perl data structure we have for returning back the results in %p ...\n", perl_result);
	AV *av;
	size_t asz;
	if( is_array_ref(perl_result, &asz) ){
		// we have an array-ref to store and send back results
		av = (AV *)SvRV(perl_result);
		if( asz > 0 ){
			// non-empty content, clear it
			av_clear(av);
		}
		fprintf(stdout, "mandelbrot_main() : we have been given an array-ref.\n");
	} else if( SvROK(perl_result) ){
		// we have a scalar ref for passing back the results
		// make it a ref to a new array
		av = newAV();
		sv_setrv(SvRV(perl_result), (SV *)av);
		fprintf(stdout, "mandelbrot_main() : we have been given a scalar-ref.\n");
	} else {
		// we have a scalar for passing back the results
		// make it a ref to a new array
		av = newAV();
		sv_setrv(SvRV(perl_result), (SV *)av);
		fprintf(stdout, "mandelbrot_main() : we have been given a scalar.\n");
	}
	fprintf(stdout, "mandelbrot_main() : done, checked input perl data structure.\n");

	// we are now sure that av = (AV *)SvRV(perl_R) will be an empty
	// array ready to be filled with subarrays, when we get results

	/* allocate temp */
	size_t temp_sz = width * height;
	uchar4 *h_temp;
	if( (err=cudaMallocHost((void **) &h_temp, temp_sz*sizeof(uchar4))) != cudaSuccess ){ fprintf(stderr, "mandelbrot_main() : error, failed to allocate %zu bytes (%zu items) for h_temp via cudaMallocHost(), status was %d.\n", temp_sz*sizeof(uchar4), temp_sz, err); return 1; }

	/* allocate and transfer into colors the items from perl_colors, we need to allocate a new array */
	short *h_colors;
	if( (err=cudaMallocHost((void **) &h_colors, colors_sz*sizeof(short))) != cudaSuccess ){ fprintf(stderr, "mandelbrot_main() : error, failed to allocate %zu bytes (%zu items) for h_colors via cudaMallocHost(), status was %d.\n", colors_sz*sizeof(short), colors_sz, err); return 1; }

	/* and allocate some for the device */
	uchar4 *d_temp;
	if( (err=cudaMalloc((void **) &d_temp, temp_sz*sizeof(uchar4))) != cudaSuccess ){ fprintf(stderr, "mandelbrot_main() : error, failed to allocate %zu bytes (%zu items) for h_colors via cudaMalloc(), status was %d.\n", temp_sz*sizeof(uchar4), temp_sz, err); return 1; }
	short *d_colors;
	if( (err=cudaMalloc((void **) &d_colors, colors_sz*sizeof(short))) != cudaSuccess ){ fprintf(stderr, "mandelbrot_main() : error, failed to allocate %zu bytes (%zu items) for h_colors via cudaMalloc(), status was %d.\n", colors_sz*sizeof(short), colors_sz, err); return 1; }
	/* zero d_temp */
	cudaMemset(d_temp, 0, temp_sz*sizeof(uchar4));

	/* read data from Perl into device */
	SV **subav;
	short *a_color_p = &(h_colors[0]);
	av = (AV *)SvRV(perl_colors);
	for(i=colors_sz;i-->0;a_color_p++){
		subav = av_fetch(av, i, FALSE);
		*a_color_p = (short )SvNV(*subav);
	}

	/* copy data from host to device */
	if( (err=cudaMemcpy(d_colors, h_colors, colors_sz*sizeof(short), cudaMemcpyHostToDevice)) != cudaSuccess ){ fprintf(stderr, "mandelbrot_main() : error, failed to transfer h_colors from host to device, status was %d.\n", err); return 1; }
	
	dim3 dimGrid(gDimX, gDimY);
	dim3 dimBlock(bDimX, bDimY);

	fprintf(stdout, "mandelbrot_main() : launching the kernel 'mandelbrot1' ...\n");
	// Launch kernel
	mandelbrot1<<<dimGrid, dimBlock>>>(
		min_x, min_y,
		step_x, step_y,
		d_temp, d_colors,
		iters,
		width, height
	);
	if( (err=cudaGetLastError()) != cudaSuccess ){ fprintf(stderr, "mandelbrot_main() : error, failed to launch cuda kernel 'mandelbrot1' (block:%dx%d, grid:%dz%d).\n", bDimX, bDimY, gDimX, gDimY); return 1; }
	fprintf(stdout, "mandelbrot_main() : done, launched and executed the kernel 'mandelbrot1'.\n");

	/* copy result d_temp from device to host */
	if( (err=cudaMemcpy(h_temp, d_temp, temp_sz*sizeof(uchar4), cudaMemcpyDeviceToHost)) != cudaSuccess ){ fprintf(stderr, "mandelbrot_main() : error, failed to transfer d_temp from device to host, status was %d.\n", err); return 1; }
	/* and now transfer host into Perl (ouphh!) */
	uint32_t *a_temp_p = (uint32_t *)(&(h_temp[0]));
	av = (AV *)SvRV(perl_result);
	for(i=temp_sz;i-->0;a_temp_p++){
		av_push(av, newSVnv(*a_temp_p));
	}
	/* done!, Perl arrayref should now be filled with the result */
	fprintf(stdout, "mandelbrot_main() : done, and returning.\n");
	return 0;
}
