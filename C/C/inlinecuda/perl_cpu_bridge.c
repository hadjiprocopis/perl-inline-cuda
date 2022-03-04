#ifndef _PERL_CPU_BRIDGE_C
#define _PERL_CPU_BRIDGE_C

/* our $VERSION = 0.05; */

#include <sys/time.h> // TODO win/unix compatible timespec!

#include <inlinecuda/utils.h>

#include <inlinecuda/cpu_matrix_ops_int.c>
#include <inlinecuda/cpu_matrix_ops_float.c>
#include <inlinecuda/cpu_matrix_ops_double.c>

#include <inlinecuda/perl_cpu_bridge.h>

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
){
	struct timespec TSTARTED, TENDED;
	clock_gettime(CLOCK_MONOTONIC_RAW, &TSTARTED);

	// AH = m = rows
	// AW = n = cols
	size_t AH, AW, *AWs = NULL, // input matrix A, AWs is the size of each row (e.g. width)
	       BH, BW, *BWs = NULL, // input matrix B
	       RH, RW, // result
	       i, j, asz
	;
	float *host_A, *host_B, *host_R;
	SV *subav, *subsubav;

	if( array_numelts_2D(perl_A, &AH, &AWs) ){ fprintf(stderr, "cpu_matrix_multiply() : error, call to array_numelts_2D() has failed for input matrix A.\n"); return 1; }
	if( array_numelts_2D(perl_B, &BH, &BWs) ){ fprintf(stderr, "cpu_matrix_multiply() : error, call to array_numelts_2D() has failed for input matrix B.\n"); return 1; }

	AW = AWs[0]; for(i=AH;i-->0;){ if( AWs[i] != AW ){ fprintf(stderr, "cpu_matrix_multiply() : error, input matrix A is not uniform in its dimensions, row %zu has size %zu instead of %zu (the size of the 1st row).\n", i, AWs[i], AW); return 1; } }
	BW = BWs[0]; for(i=BH;i-->0;){ if( BWs[i] != BW ){ fprintf(stderr, "cpu_matrix_multiply() : error, input matrix B is not uniform in its dimensions, row %zu has size %zu instead of %zu (the size of the 1st row).\n", i, BWs[i], BW); return 1; } }

	// (m x n) (n x k) = (m x k)
	if( AW != BH ){ fprintf(stderr, "cpu_matrix_multiply() : error, matrix sizes are inconsistent for multiplication, A's width (%zu) != B's height (%zu).\n", AW, BH); return 1; }

	RH = AH; RW = BW; // the Result's dimensions

	if( noisy>0 ){
		fprintf(stdout, "cpu_matrix_multiply() : input matrix dimensions  (h/rows,w/cols): A(%zu,%zu), B(%zu,%zu).\n", AH, AW, BH, BW);
		fprintf(stdout, "cpu_matrix_multiply() : output matrix dimensions (h/rows,w/cols): R(%zu,%zu).\n", RH, RW);
	}

	// check the perl_R matrix which will take results back to caller
	// we expect an array which is totally empty. e.g. my @R; xx(\@R);
	// or a scalar where to place array ref we allocate here, e.g. my $x; xx($x);
	AV *av, *av2, *avres;
	if( is_array_ref(perl_R, &asz) ){
		if( noisy>0 ){ fprintf(stdout, "cpu_matrix_multiply() : we have an array ref for passing back the results ...\n"); }
		avres = (AV *)SvRV(perl_R);
		if( asz > 0 ){
			if( noisy>0 ){ fprintf(stdout, "cpu_matrix_multiply() : clearing contents of array or passing back results, it has %zu items there already ...\n", asz); }
			av_clear(avres);
		}
	} else if( SvROK(perl_R) ){
		if( noisy>0 ){ fprintf(stdout, "cpu_matrix_multiply() : we have a scalar ref for passing back the results ...\n"); }
		avres = newAV();
		// LeoNerd's suggestion:
		sv_setrv(SvRV(perl_R), (SV *)avres);
	} else {
		if( noisy>0 ){ fprintf(stdout, "cpu_matrix_multiply() : we have a scalar for passing back the results ...\n"); }
		avres = newAV();
		// LeoNerd's suggestion:
		sv_setrv(perl_R, (SV *)avres);
	}
	// we are now sure that av = (AV *)SvRV(perl_R) will be an empty
	// array ready to be filled with subarrays, when we get results

	if( noisy > 0 ){ fprintf(stdout, "cpu_matrix_multiply() : allocating memory for a total of %zu bytes ...\n", sizeof(float)*AW*AH+sizeof(float)*BW*BH+sizeof(float)*RW*RH); }
	if( (host_A=(float *)malloc(AH*AW*sizeof(float))) == NULL ){ fprintf(stderr, "cpu_matrix_multiply() : error, failed to allocate %zu bytes for matrix A (%zu,%zu).\n", AH*AW*sizeof(float), AH,AW); return 1; }
	if( (host_B=(float *)malloc(BH*BW*sizeof(float))) == NULL ){ fprintf(stderr, "cpu_matrix_multiply() : error, failed to allocate %zu bytes for matrix B (%zu,%zu).\n", BH*BW*sizeof(float), BH,BW); return 1; }
	if( (host_R=(float *)malloc(RH*RW*sizeof(float))) == NULL ){ fprintf(stderr, "cpu_matrix_multiply() : error, failed to allocate %zu bytes for matrix R (%zu,%zu).\n", RH*RW*sizeof(float), RH,RW); return 1; }

	// input matrix A -> host A
	float *pd = &(host_A[0]);
	av = (AV *)SvRV(perl_A);
	if( noisy>1 ){ fprintf(stdout, "cpu_matrix_multiply() : matrix A:\n"); }
	for(i=0;i<AH;i++){ // for each row
		subav = *av_fetch(av, i, FALSE);
		for(j=0;j<AW;j++){ // for the cols of that row
			subsubav = *av_fetch((AV *)SvRV(subav), j, FALSE);
			*pd = SvNV(subsubav);
			if( noisy>1 ){ fprintf(stdout, "%f ", *pd); }
			pd++;
		}
		if( noisy>1 ){ fprintf(stdout, "\n"); }
	}

	// input matrix B -> host B
	pd = &(host_B[0]);
	av = (AV *)SvRV(perl_B);
	if( noisy>1 ){ fprintf(stdout, "cpu_matrix_multiply() : matrix B:\n"); }
	for(i=0;i<BH;i++){ // for each row
		subav = *av_fetch(av, i, FALSE);
		for(j=0;j<BW;j++){ // for the cols of that row
			subsubav = *av_fetch((AV *)SvRV(subav), j, FALSE);
			*pd = SvNV(subsubav);
			if( noisy>1 ){ fprintf(stdout, "%f ", *pd); }
			pd++;
		}
		if( noisy>1 ){ fprintf(stdout, "\n"); }
	}

	// call it
	if( cpu_matrix_mult_float(
		host_A, host_B, host_R,
		AH, AW, BW,
		noisy
	) ){ fprintf(stderr, "cpu_matrix_multiply() : error, call to cpu_matrix_mult_float() has failed.\n"); return 1; }

	// Transfer results from host to perl(!)
	// host R => perl R
	pd = &(host_R[0]);
	if( noisy>1 ){ fprintf(stdout, "cpu_matrix_multiply() : matrix R (result):\n"); }
	for(i=0;i<RH;i++){ // for each row
		av2 = newAV(); // make a new array for each row
		av_extend(av2, RW); // extend it to hold #cols items (RW)
		// LeoNerd's suggestion
		av_push(avres, newRV_noinc((SV *)av2)); // insert it into the top Array
		for(j=0;j<RW;j++){ // for the cols of that row
			av_store(av2, j, newSVnv(*pd));
			if( noisy>1 ){ fprintf(stdout, "%f ", *pd); }
			pd++;
		}
		if( noisy>1 ){ fprintf(stdout, "\n"); }
	}

	free(host_A); free(host_B); free(host_R);

	clock_gettime(CLOCK_MONOTONIC_RAW, &TENDED);
	if( noisy > 0 ){ fprintf(stdout, "cpu_matrix_multiply() : time elapsed on matrix multiplication of (%zu,%zu) x (%zu,%zu) on GPU: %lf ms.\n\n", AH,AW,BH,BW, time_difference_seconds(TSTARTED,TENDED)); }

	return 0; // success
}
#endif
