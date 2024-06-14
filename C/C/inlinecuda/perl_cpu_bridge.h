#ifndef _PERL_CPU_BRIDGE_H
#define _PERL_CPU_BRIDGE_H

/* our $VERSION = 0.05; */

// MATRIX NOTATION: size(m x n) means m rows, n cols
// perl matrices in should be [row][col]

// Note A's width must be equal to B's height for Result = A x B
// meaning A's cols = B's rows
// and Result will have height=A's height and width = B's width
// meaning Result's rows = A's rows, Result's cols = B's cols
// i.e. (m x n) x (n x k) = (m x k)
int cpu_matrix_multiply(
	SV *perl_A,
	SV *perl_B,
	SV *perl_R,
	int noisy
);
#endif
