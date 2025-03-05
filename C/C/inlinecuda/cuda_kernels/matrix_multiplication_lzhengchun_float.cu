#ifndef _MATRIX_MULTIPLICATION_LZHENGCHUN_FLOAT_CU
#define _MATRIX_MULTIPLICATION_LZHENGCHUN_FLOAT_CU

/* our $VERSION = 0.05; */

#include <inlinecuda/cuda_kernels/matrix_multiplication_lzhengchun_float.cu.h>

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
function name: gpu_matrix_mult_lzhengchun_float
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
__global__ void gpu_matrix_mult_lzhengchun_float(float *a,float *b, float *c, size_t m, size_t n, size_t k)
{ 
    size_t row = blockIdx.y * blockDim.y + threadIdx.y; 
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if( col < k && row < m) 
    {
        for(size_t i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
} 

/*
*********************************************************************
function name: gpu_square_matrix_mult_lzhengchun_float
description: dot product of two square matrices in GPU
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
void gpu_square_matrix_mult_lzhengchun_float(
	float *d_a,
	float *d_b,
	float *d_result,
	size_t n
) {
    __shared__ float tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tile_b[BLOCK_SIZE][BLOCK_SIZE];

    size_t row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    size_t col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float tmp = 0;
    int idx;

    for (size_t sub = 0; sub < gridDim.x; ++sub) 
    {
        idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
        if(idx >= n*n)
        {
            // n may not divisible by BLOCK_SIZE
            tile_a[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
        }

        idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
        if(idx >= n*n)
        {
            tile_b[threadIdx.y][threadIdx.x] = 0;
        }  
        else
        {
            tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
        }
        __syncthreads();

        for (size_t k = 0; k < BLOCK_SIZE; ++k) 
        {
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    if(row < n && col < n)
    {
        d_result[row * n + col] = tmp;
    }
}

/*
*********************************************************************
function name: gpu_matrix_transpose_lzhengchun_float
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
__global__ void gpu_matrix_transpose_lzhengchun_float(float* mat_in, float* mat_out, size_t rows, size_t cols)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) 
    {
        size_t pos = idy * cols + idx;
        size_t trans_pos = idx * rows + idy;
        mat_out[trans_pos] = mat_in[pos];
    }
}
#endif
