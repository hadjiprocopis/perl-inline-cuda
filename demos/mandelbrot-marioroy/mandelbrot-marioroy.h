#ifndef _MANDELBROT_MARIOROY_H
#define _MANDELBROT_MARIOROY_H

#include <math.h>

// FROM: https://github.com/marioroy/mandelbrot-python/blob/main/app/mandel_cuda.c
// NOTE: don't use main(void), use main()!!!
// 
// CUDA C code for computing the Mandelbrot Set on the GPU.
// This requires double-precision capabilities on the device.
// 
// Optimization flags defined in ../mandel_cuda.py:
//   MIXED_PREC1  integer comparison; this may run faster depending on GPU
//     #if defined(MIXED_PREC1) || defined(MIXED_PREC2)
//       if ( (zreal.x == sreal.x) && (zreal.y == sreal.y) &&
//            (zimag.x == simag.x) && (zimag.y == simag.y) ) {
//     #else
//       if ( (zreal.d == sreal.d) && (zimag.d == simag.d) ) {
//     #endif
//   MIXED_PREC2  includes MIXED_PREC1; single-precision addition (hi-bits)
//     #if defined(MIXED_PREC2)
//       a.i = (zreal_sqr.y & 0xc0000000) | ((zreal_sqr.y & 0x7ffffff) << 3);
//       b.i = (zimag_sqr.y & 0xc0000000) | ((zimag_sqr.y & 0x7ffffff) << 3);
//       if (a.f + b.f > ESCAPE_RADIUS_2) {
//     #else
//       if (zreal_sqr.d + zimag_sqr.d > ESCAPE_RADIUS_2) {
//     #endif
// 
// Depending on the GPU, mixed_prec=1 may run faster than 2.
// But definitely try 2 for possibly better results.
// GeForce 2070 RTX 1280x720 auto-zoom (press x).
//   mixed_prec=0 fma=0    9.0 seconds
//   mixed_prec=1 fma=0    8.4 seconds
//   mixed_prec=2 fma=0    7.1 seconds
//   mixed_prec=2 fma=1    6.3 seconds
//

/* some constants */
#define RADIUS 16.0
#define GRADIENT_LENGTH 256
#define MATRIX_LENGTH (((int )(ceil(0.25)) * 2)*2+1)

#if defined(FMA_ON)
#define _mad(a,b,c) fma(a,b,c)
#else
#define _mad(a,b,c) a * b + c
#endif

#define ESCAPE_RADIUS_2 (float) (RADIUS * RADIUS)
#define INSIDE_COLOR1 make_uchar4(0x01,0x01,0x01,0xff)
#define INSIDE_COLOR2 make_uchar4(0x8d,0x02,0x1f,0xff)

#if !defined(M_LN2)
#define M_LN2 0.69314718055994530942  // log_e 2
#endif

#if defined(MIXED_PREC2)
typedef union {
    float f;  // value
    unsigned int i;  // bits
} ufloat_t;
#endif

typedef union {
    double d;  // value
    struct { unsigned int x, y; };  // bits
} udouble_t;

#endif
