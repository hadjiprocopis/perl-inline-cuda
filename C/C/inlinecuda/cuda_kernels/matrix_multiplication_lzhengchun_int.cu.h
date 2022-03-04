#ifndef _MATRIX_MULTIPLICATION_LZHENGCHUN_INT_CU_H
#define _MATRIX_MULTIPLICATION_LZHENGCHUN_INT_CU_H

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

/* You must specify a #define BLOCK_SIZE before loading this */

/*
*********************************************************************
function name: gpu_matrix_mult_lzhengchun_int
description: dot product of two matrix (not only square)
parameters: 
            &a GPU device pointer to a m X n matrix (A)
            &b GPU device pointer to a n X k matrix (B)
            &c GPU device output purpose pointer to a m X k matrix (C) 
            to store the result
Note:
    grid and block should be configured as:
        dim3 dimGrid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    further sppedup can be obtained by using shared memory to decrease global memory access times
return: none
*********************************************************************
*/
__global__
void gpu_matrix_mult_lzhengchun_int(int *a,int *b, int *c, int m, int n, int k);

/*
*********************************************************************
function name: gpu_square_matrix_mult_lzhengchun_int
description: dot product of two matrix (not only square) in GPU
parameters: 
            &a GPU device pointer to a n X n matrix (A)
            &b GPU device pointer to a n X n matrix (B)
            &c GPU device output purpose pointer to a n X n matrix (C) 
            to store the result
Note:
    grid and block should be configured as:
        dim3 dim_grid((n - 1) / BLOCK_SIZE + 1, (n - 1) / BLOCK_SIZE + 1, 1);
        dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
return: none
*********************************************************************
*/
__global__
void gpu_square_matrix_mult_lzhengchun_int(int *d_a, int *d_b, int *d_result, int n);

/*
*********************************************************************
function name: gpu_matrix_transpose_lzhengchun_int
description: matrix transpose
parameters: 
            &mat_in GPU device pointer to a rows X cols matrix
            &mat_out GPU device output purpose pointer to a cols X rows matrix 
            to store the result
Note:
    grid and block should be configured as:
        dim3 dim_grid((n - 1) / BLOCK_SIZE + 1, (n - 1) / BLOCK_SIZE + 1, 1);
        dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
return: none
*********************************************************************
*/
__global__
void gpu_matrix_transpose_lzhengchun_int(int* mat_in, int* mat_out, unsigned int rows, unsigned int cols);
#endif
