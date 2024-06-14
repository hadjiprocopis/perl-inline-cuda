#ifndef _INLINE_CUDA_CPULIB_UTILS_H
#define _INLINE_CUDA_CPULIB_UTILS_H

/* our $VERSION = 0.05; */

// obtain time like:
//   struct timespec _TSTARTED; clock_gettime(CLOCK_MONOTONIC_RAW, &_TSTARTED);
// given 2 timespec structs, returns time difference in seconds with microsecond accuracy
#define time_difference_seconds(_TSTARTED,_TENDED) (((_TENDED.tv_sec - _TSTARTED.tv_sec) * 1000000 + (_TENDED.tv_nsec - _TSTARTED.tv_nsec) / 1000.0)/1000000.0)


#endif
