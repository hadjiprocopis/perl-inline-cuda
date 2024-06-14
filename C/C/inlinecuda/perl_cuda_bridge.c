#ifndef _PERL_CUDA_BRIDGE_C
#define _PERL_CUDA_BRIDGE_C

/* our $VERSION = 0.05; */

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <inlinecuda/utils.h>

#include <inlinecuda/cuda_kernels/matrix_multiplication_lzhengchun_int.cu>
#include <inlinecuda/cuda_kernels/matrix_multiplication_lzhengchun_float.cu>
#include <inlinecuda/cuda_kernels/matrix_multiplication_lzhengchun_double.cu>

#include <inlinecuda/perl_cuda_bridge.h>

// MATRIX NOTATION: size(m x n) means m rows, n cols
// perl matrices in should be [row][col]

// Note A's width must be equal to B's height for Result = A x B
// meaning A's cols = B's rows
// and Result will have height=A's height and width = B's width
// meaning Result's rows = A's rows, Result's cols = B's cols
// i.e. (m x n) x (n x k) = (m x k)
int inline_cuda_matrix_multiply(
	SV *perl_A,
	SV *perl_B,
	SV *perl_R,
	int noisy
){
	// AH = m = rows
	// AW = n = cols
	size_t AH, AW, *AWs = NULL, // input matrix A, AWs is the size of each row (e.g. width)
	       BH, BW, *BWs = NULL, // input matrix B
	       RH, RW, // result
	       i, j, asz
	;
	float *host_A, *host_B, *host_R,
	      *device_A, *device_B, *device_R
	;
	SV *subav, *subsubav, **ssubav;

	// events to count the execution time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// start to count execution time
	cudaEventRecord(start, 0);

	if( array_numelts_2D(perl_A, &AH, &AWs) ){ fprintf(stderr, "inline_cuda_matrix_multiply() : error, call to array_numelts_2D() has failed for input matrix A.\n"); return 1; }
	if( array_numelts_2D(perl_B, &BH, &BWs) ){ fprintf(stderr, "inline_cuda_matrix_multiply() : error, call to array_numelts_2D() has failed for input matrix B.\n"); return 1; }

	AW = AWs[0]; for(i=AH;i-->0;){ if( AWs[i] != AW ){ fprintf(stderr, "inline_cuda_matrix_multiply() : error, input matrix A is not uniform in its dimensions, row %zu has size %zu instead of %zu (the size of the 1st row).\n", i, AWs[i], AW); return 1; } }
	BW = BWs[0]; for(i=BH;i-->0;){ if( BWs[i] != BW ){ fprintf(stderr, "inline_cuda_matrix_multiply() : error, input matrix B is not uniform in its dimensions, row %zu has size %zu instead of %zu (the size of the 1st row).\n", i, BWs[i], BW); return 1; } }

	// (m x n) (n x k) = (m x k)
	if( AW != BH ){ fprintf(stderr, "inline_cuda_matrix_multiply() : error, matrix sizes are inconsistent for multiplication, A's width (%zu) != B's height (%zu).\n", AW, BH); return 1; }

	RH = AH; RW = BW; // the Result's dimensions

	if( noisy>0 ){
		fprintf(stdout, "inline_cuda_matrix_multiply() : input matrix dimensions  (h/rows,w/cols): A(%zu,%zu), B(%zu,%zu).\n", AH, AW, BH, BW);
		fprintf(stdout, "inline_cuda_matrix_multiply() : output matrix dimensions (h/rows,w/cols): R(%zu,%zu).\n", RH, RW);
	}

	// check the perl_R matrix which will take results back to caller
	// we expect an array which is totally empty. e.g. my @R; func(\@R);
	// or a scalar where to place array ref we allocate here
	// e.g. my $x; func($x);
	if( is_array_ref(perl_R, &asz) ){
		// array ref: func(\@R);
 		if( noisy>0 ){ fprintf(stdout, "inline_cuda_matrix_multiply() : we have an array ref for passing back the results ...\n"); }
		if( asz > 0 ){
			if( noisy>0 ){ fprintf(stdout, "inline_cuda_matrix_multiply() : clearing contents of array or passing back results, it has %zu items there already:", asz); }
			AV *avd = (AV *)SvRV(perl_R);
			for(i=0;i<asz;i++){
				ssubav = av_fetch(avd, i, FALSE);
				if( ssubav != NULL ){
					if( noisy>0 ){ fprintf(stdout, " %d,", SvIV(*ssubav)); }
				}
			}
			if( noisy>0 ){ fprintf(stdout, "\n"); }
			av_clear(avd); // clear the array
		}
	} else if( SvROK(perl_R) ){
		// scalar to place our array ref in there: my $x; func($x);
		if( noisy>0 ){ fprintf(stdout, "inline_cuda_matrix_multiply() : we have a scalar ref for passing back the results ...\n"); }
		// LeoNerd's suggestion:
		sv_setrv(SvRV(perl_R), (SV *)newAV());
	} else {
		if( noisy>0 ){ fprintf(stdout, "inline_cuda_matrix_multiply() : we have a scalar for passing back the results ...\n"); }
		// LeoNerd's suggestion:
		sv_setrv(perl_R, (SV *)newAV());
	}
	// ABOVE, the array will be extended later to fit the results

	// we are now sure that av = (AV *)SvRV(perl_R) will be an empty
	// array ready to be filled with subarrays, when we get results

	if( noisy > 0 ){ fprintf(stdout, "inline_cuda_matrix_multiply() : allocating host memory for a total of %zu bytes ...\n", sizeof(float)*AW*AH+sizeof(float)*BW*BH+sizeof(float)*RW*RH); }
	/* unfortunately we need to allocate memory to copy the input SV ... */
	cudaError_t cudret;
	if( (cudret=cudaMallocHost((void **) &host_A, sizeof(float)*AW*AH)) != cudaSuccess ){ fprintf(stderr, "inline_cuda_matrix_multiply() : error, call to cudaMallocHost() has failed for host_A for %zu bytes (matrix size: (h/rows,w/cols)=(%zu,%zu): %s\n", sizeof(float)*AW*AH, AH, AW, cudaGetErrorString(cudret)); return 1; }
	if( (cudret=cudaMallocHost((void **) &host_B, sizeof(float)*BW*BH)) != cudaSuccess ){ fprintf(stderr, "inline_cuda_matrix_multiply() : error, call to cudaMallocHost() has failed for host_B for %zu bytes (matrix size: (h/rows,w/cols)=(%zu,%zu): %s\n", sizeof(float)*BW*BH, BH, BW, cudaGetErrorString(cudret)); return 1; }
	if( (cudret=cudaMallocHost((void **) &host_R, sizeof(float)*RW*RH)) != cudaSuccess ){ fprintf(stderr, "inline_cuda_matrix_multiply() : error, call to cudaMallocHost() has failed for host_R for %zu bytes (matrix size: (h/rows,w/cols)=(%zu,%zu): %s\n", sizeof(float)*RW*RH, RH, RW, cudaGetErrorString(cudret)); return 1; }
	if( noisy > 0 ){ fprintf(stdout, "inline_cuda_matrix_multiply() : success allocating total host memory of %zu bytes ...\n", sizeof(float)*AW*AH+sizeof(float)*BW*BH+sizeof(float)*RW*RH); }

	// allocate again for the device
	if( noisy > 0 ){ fprintf(stdout, "inline_cuda_matrix_multiply() : allocating device memory for a total of %zu bytes ...\n", sizeof(float)*AW*AH+sizeof(float)*BW*BH+sizeof(float)*RW*RH); }
	if( (cudret=cudaMalloc((void **) &device_A, sizeof(float)*AW*AH)) != cudaSuccess ){ fprintf(stderr, "inline_cuda_matrix_multiply() : error, call to cudaMalloc() has failed for device_A for %zu bytes (matrix size: (h/rows,w/cols)=(%zu,%zu): %s\n", sizeof(float)*AW*AH, AH, AW, cudaGetErrorString(cudret)); return 1; }
	if( (cudret=cudaMalloc((void **) &device_B, sizeof(float)*BW*BH)) != cudaSuccess ){ fprintf(stderr, "inline_cuda_matrix_multiply() : error, call to cudaMalloc() has failed for device_B for %zu bytes (matrix size: (h/rows,w/cols)=(%zu,%zu): %s\n", sizeof(float)*BW*BH, BH, BW, cudaGetErrorString(cudret)); return 1; }
	if( (cudret=cudaMalloc((void **) &device_R, sizeof(float)*RW*RH)) != cudaSuccess ){ fprintf(stderr, "inline_cuda_matrix_multiply() : error, call to cudaMalloc() has failed for device_R for %zu bytes (matrix size: (h/rows,w/cols)=(%zu,%zu): %s\n", sizeof(float)*RW*RH, RH, RW, cudaGetErrorString(cudret)); return 1; }
	if( noisy > 0 ){ fprintf(stdout, "inline_cuda_matrix_multiply() : success allocating total device memory of %zu bytes ...\n", sizeof(float)*AW*AH+sizeof(float)*BW*BH+sizeof(float)*RW*RH); }

	// copy all input (extract from SVs) to device_*, if they were on an array we could
	// used cudaMemcpy(cudaMemcpyHostToDevice)!
	// input matrix A -> device A
	AV *av, *av2;
	float *pd = &(host_A[0]);
	av = (AV *)SvRV(perl_A);
	if( noisy>1 ){ fprintf(stdout, "inline_cuda_matrix_multiply() : matrix A:\n"); }

	for(i=0;i<AH;i++){ // for each row
		ssubav = av_fetch(av, i, FALSE);
		if( ssubav == NULL ){ fprintf(stderr, "inline_cuda_matrix_multiply() : error, input matrix A does not contain valid row at i=%d\n", i); return 1; }
		subav = *ssubav;
		for(j=0;j<AW;j++){ // for the cols of that row
			ssubav = av_fetch((AV *)SvRV(subav), j, FALSE);
			if( ssubav == NULL ){ fprintf(stderr, "inline_cuda_matrix_multiply() : error, input matrix A does not contain valid column at i=%d, j=%d\n", i, j); return 1; }
			subsubav = *ssubav;
			*pd = SvNV(subsubav);
			if( noisy>1 ){ fprintf(stdout, "%f ", *pd); }
			pd++;
		}
		if( noisy>1 ){ fprintf(stdout, "\n"); }
	}
	// transfer results from host to device for A
	if( (cudret=cudaMemcpy(device_A, host_A, AW*AH*sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess ){ fprintf(stderr, "inline_cuda_matrix_multiply() : error, call to cudaMemcpy(cudaMemcpyHostToDevice) has failed for transfering results from host_A to device: %s\n", cudaGetErrorString(cudret)); return 1; }

	// input matrix B -> device B
	pd = &(host_B[0]);
	av = (AV *)SvRV(perl_B);
	if( noisy>1 ){ fprintf(stdout, "inline_cuda_matrix_multiply() : matrix B:\n"); }
	for(i=0;i<BH;i++){ // for each row
		ssubav = av_fetch(av, i, FALSE);
		if( ssubav == NULL ){ fprintf(stderr, "inline_cuda_matrix_multiply() : error, input matrix B does not contain valid row at i=%d\n", i); return 1; }
		subav = *ssubav;
		for(j=0;j<BW;j++){ // for the cols of that row
			ssubav = av_fetch((AV *)SvRV(subav), j, FALSE);
			if( ssubav == NULL ){ fprintf(stderr, "inline_cuda_matrix_multiply() : error, input matrix B does not contain valid column at i=%d, j=%d\n", i, j); return 1; }
			subsubav = *ssubav;
			*pd = SvNV(subsubav);
			if( noisy>1 ){ fprintf(stdout, "%f ", *pd); }
			pd++;
		}
		if( noisy>1 ){ fprintf(stdout, "\n"); }
	}
	// transfer results from host to device for B
	if( (cudret=cudaMemcpy(device_B, host_B, BW*BH*sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess ){ fprintf(stderr, "inline_cuda_matrix_multiply() : error, call to cudaMemcpy(cudaMemcpyHostToDevice) has failed for transfering results from host_B to device: %s\n", cudaGetErrorString(cudret)); return 1; }

	unsigned int grid_rows = (RH + BLOCK_SIZE - 1) / BLOCK_SIZE;
	unsigned int grid_cols = (RW + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 dimGrid(grid_cols, grid_rows);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

	// Launch kernel 
	if(AH == AW && BH == BW){
		if( noisy>0 ){ fprintf(stdout, "inline_cuda_matrix_multiply() : launching kernel 'gpu_square_matrix_mult_lzhengchun_float()' for (%zu,%zu) x (%zu,%zu) ...\n", AH, AW, BH, BW); }
		gpu_square_matrix_mult_lzhengchun_float<<<dimGrid, dimBlock>>>(device_A, device_B, device_R, AW);
	} else {
		if( noisy>0 ){ fprintf(stdout, "inline_cuda_matrix_multiply() : launching kernel 'gpu_matrix_mult_float_lzhengchun()' for (%zu,%zu) x (%zu,%zu) ...\n", AH, AW, BH, BW); }
		gpu_matrix_mult_lzhengchun_float<<<dimGrid, dimBlock>>>(device_A, device_B, device_R, AH, AW, BW);
	}

	if( (cudret=cudaGetLastError()) != cudaSuccess ){ fprintf(stderr, "inline_cuda_matrix_multiply() : error, launching kernel has failed: %s\n", cudaGetErrorString(cudret)); return 1; }
	if( noisy>0 ){ fprintf(stdout, "inline_cuda_matrix_multiply() : done kernel with this code: %s\n", cudaGetErrorString(cudret)); }

	// free A and B from Host and device
	if( (cudret=cudaFreeHost(host_A)) != cudaSuccess ){ fprintf(stderr, "inline_cuda_matrix_multiply() : error, call to cudaFreeHost(host_A) has failed: %s\n", cudaGetErrorString(cudret)); return 1; }
	if( (cudret=cudaFreeHost(host_B)) != cudaSuccess ){ fprintf(stderr, "inline_cuda_matrix_multiply() : error, call to cudaFreeHost(host_B) has failed: %s\n", cudaGetErrorString(cudret)); return 1; }
	if( (cudret=cudaFree((void *)device_A)) != cudaSuccess ){ fprintf(stderr, "inline_cuda_matrix_multiply() : error, call to cudaFree(device_A) has failed: %s\n", cudaGetErrorString(cudret)); return 1; }
	if( (cudret=cudaFree((void *)device_B)) != cudaSuccess ){ fprintf(stderr, "inline_cuda_matrix_multiply() : error, call to cudaFree(device_B) has failed: %s\n", cudaGetErrorString(cudret)); return 1; }

	// transfer results from device to host
	if( (cudret=cudaMemcpy(host_R, device_R, RW*RH*sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess ){ fprintf(stderr, "inline_cuda_matrix_multiply() : error, call to cudaMemcpy(cudaMemcpyDeviceToHost) has failed for transfering results to host device: %s\n", cudaGetErrorString(cudret)); return 1; }
	if( (cudret=cudaDeviceSynchronize()) != cudaSuccess ){ fprintf(stderr, "inline_cuda_matrix_multiply() : error, call to cudaDeviceSynchronize() has failed: %s\n", cudaGetErrorString(cudret)); return 1; }

	// Transfer results from host to perl(!)
	// device R => input matrix R
	pd = &(host_R[0]);
	av = (AV *)SvRV(perl_R);
	// extend it to hold RH items
	// and then each item will be an arrayref (of RW items capacity)
	// the subarrays are created below in the loop
	av_extend(av, RH);
	if( noisy>1 ){ fprintf(stdout, "inline_cuda_matrix_multiply() : matrix R (result):\n"); }
	for(i=0;i<RH;i++){ // for each row
		av2 = newAV(); // make a new array for each row
		av_extend(av2, RW); // extend it to hold #cols items (RW)
		// LeoNerd's suggestion
		av_push(av, newRV_noinc((SV *)av2)); // insert it into the top Array
		for(j=0;j<RW;j++){ // for the cols of that row
			av_store(av2, j, newSVnv(*pd));
			if( noisy>1 ){ fprintf(stdout, "%f ", *pd); }
			pd++;
		}
		if( noisy>1 ){ fprintf(stdout, "\n"); }
	}

	// free device memory for result
	if( (cudret=cudaFree((void *)device_R)) != cudaSuccess ){ fprintf(stderr, "inline_cuda_matrix_multiply() : error, call to cudaFree(device_R) has failed: %s\n", cudaGetErrorString(cudret)); return 1; }
	// free result on host
	if( (cudret=cudaFreeHost((void *)host_R)) != cudaSuccess ){ fprintf(stderr, "inline_cuda_matrix_multiply() : error, call to cudaFreeHost(host_R) has failed: %s\n", cudaGetErrorString(cudret)); return 1; }

	// compute time elapse on GPU computing
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float gpu_elapsed_time_ms;
	cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
	if( noisy > 0 ){ fprintf(stdout, "inline_cuda_matrix_multiply() : time elapsed on matrix multiplication of (%zu,%zu) x (%zu,%zu) on GPU: %f ms.\n\n", AH,AW,BH,BW, gpu_elapsed_time_ms); }

	return 0; // success
}
#endif
