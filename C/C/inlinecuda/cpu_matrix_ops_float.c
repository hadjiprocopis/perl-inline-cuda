#ifndef CPU_MATRIX_OPS_FLOAT_C
#define CPU_MATRIX_OPS_FLOAT_C

/* our $VERSION = 0.05; */

#include <inlinecuda/utils.h>
#include <inlinecuda/cpu_matrix_ops_float.h>

/* NOTE: this code was copied verbatim from
   https://github.com/lzhengchun/matrix-cuda/blob/master/matrix_cuda.cu
   and is originally written by
   *****************
   *** Zhengchun Liu
   *** github user: https://github.com/lzhengchun
   *****************
   Zhengchun Liu holds all credit and copyrights for the contents of this file
   Thank you Zhengchun Liu
*/

/*
*********************************************************************
function name: cpu_matrix_mult
description: dot product of two matrix (not only square) in CPU, 
             for validating GPU results
parameters: 
            &a CPU host pointer to a m X n matrix (A)
            &b CPU host pointer to a n X k matrix (B)
            &c CPU host output purpose pointer to a m X k matrix (C) 
            to store the result
return: 1 on failure, 0 on success
*********************************************************************
*/
int cpu_matrix_mult_float(
	float *h_a,
	float *h_b,
	float *h_result,
	size_t m,
	size_t n,
	size_t k,
	int noisy
) {
	struct timespec TSTARTED;
	clock_gettime(CLOCK_MONOTONIC_RAW, &TSTARTED);

	for (size_t i = 0; i < m; ++i) 
	{
		for (size_t j = 0; j < k; ++j) 
		{
			float tmp = 0.0;
			for (size_t h = 0; h < n; ++h) 
			{
				tmp += h_a[i * n + h] * h_b[h * k + j];
			}
			h_result[i * k + j] = tmp;
		}
	}
	if( noisy > 0 ){
		struct timespec TENDED;
		clock_gettime(CLOCK_MONOTONIC_RAW, &TENDED);
		fprintf(stdout, "cpu_matrix_mult_float() : done matrix multiplication of (%zu,%zu) x (%zu,%zu) in %lf seconds.\n", m, n, n, k, time_difference_seconds(TSTARTED, TENDED));
	}
	return 0; // success
}

#endif
