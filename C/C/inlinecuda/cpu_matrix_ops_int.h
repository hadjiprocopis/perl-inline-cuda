#ifndef CPU_MATRIX_OPS_INT_H
#define CPU_MATRIX_OPS_INT_H

/* our $VERSION = 0.05; */

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
function name: cpu_matrix_mult_int
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
int cpu_matrix_mult_int(int *h_a, int *h_b, int *h_result, size_t m, size_t n, size_t k, int noisy);
#endif
