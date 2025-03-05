use 5.006;
use strict;
use warnings;

our $VERSION = 0.16;

use Test::More;

use File::Spec;
use FindBin;

use Data::Roundtrip qw/perl2dump/;

use Inline CUDA => Config =>
	host_code_language => 'c++',
	clean_after_build => 0,
	BUILD_NOISY => 1,
	warnings => 10,
	# C includes
	INC =>
		# include/ is in current test dir (e.g. t/include) and has files needed for testing
		'-I"'.File::Spec->catdir($FindBin::RealBin, 'include').'"'
		.' '
		# C/ contains kernels, headers and is installed along Inline::CUDA in its shared-dir
		.'-I"'.File::Spec->catdir($FindBin::Bin, '..', 'C', 'C').'"'
	,
;

use Inline CUDA => <<'EOC';
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>

/* kernel specific param required prior to using any of the cuda kernels */
#define BLOCK_SIZE 16

// helper functions to facilitate the bridge with perl
#include <inlinecuda/perlbridge.h>
#include <inlinecuda/perlbridge.c>

#include <inlinecuda/perl_cpu_bridge.h>
#include <inlinecuda/perl_cpu_bridge.c>

/* It shows how to deal with passed parameters as an ARRAYref
   the 1st as input data and the 2nd as data to send back to
   caller. I prefer not to return results back from such
   functions but just a return code. If 0 then success, else
   some error.
*/
int testfunc(
	SV *perl_INP_2D,
	SV *perl_OUT_2D
){
	struct timespec TSTARTED, TENDED;
	clock_gettime(CLOCK_MONOTONIC_RAW, &TSTARTED);

	// width (rows) and an array of widths (row-size) for INP and OUT
	size_t 	INP_H, *INP_Ws = NULL,
		OUT_H, OUT_W;
	;

	/* this function from perlbridge.c will take in an SV* and check
	   that it is an ARRAYref to a 2D Perl array. It will then fill
	   the number of rows (INP_H) and the size of each row (INP_Ws[arowidx])
	*/
	if( array_numelts_2D(perl_INP_2D, &INP_H, &INP_Ws) ){ fprintf(stderr, "testfunc() : error, call to array_numelts_2D() has failed for input matrix perl_INP_2D.\n"); return 1; }

	// to keep things simple, we expect a matrix with fix number of cols:
	size_t INP_W = INP_Ws[0];
	OUT_H = INP_H;
	OUT_W = INP_W;

	/* this function from perlbridge.c will take in an SV* and check
	   that it is either 1) an ARRAYref or 2) a SCALAR or a SCALARref
	   so that we create a 2D array here
	   in order to hold the results of this hypothetical calculation
	   The 1st case is:
		my @Data = ([1,2,3],[3,4,5]);
		my @R;
		testfunc(\@Data, \@R) or die;
	   The 2nd case is:
		my @Data = ([1,2,3],[3,4,5]);
		my $R; # << this will come back as a 2D array ref
		testfunc(\@Data, $R) or die;
	*/
	size_t asz;
	AV *avres;
	if( is_array_ref(perl_OUT_2D, &asz) ){
		/* it is an arrayref, asz will be its size. We expect asz
		   to be zero so that we allocate this user-sent array with its
		   rows as sub-arrays.
		   If size (asz) it is not zero, then we will clear it
		*/
		avres = (AV *)SvRV(perl_OUT_2D);
		if( asz > 0 ){ av_clear(avres); }
		fprintf(stdout, "testfunc() : passing back results as an ARRAYref.\n");
	} else if( SvROK(perl_OUT_2D) ){ 
		/* it is a scalarref, we will initialise it to a Perl 2D array */
		avres = newAV(); // create a new Array
		sv_setrv(SvRV(perl_OUT_2D), (SV *)avres); // set that to the scalarref's contents
		fprintf(stdout, "testfunc() : passing back results as a SCALARref (set to an ARRAYref).\n");
	} else {
		/* it is a scalar, we will initialise it to a Perl 2D array */
		avres = newAV();
		sv_setrv(perl_OUT_2D, (SV *)avres);
		fprintf(stdout, "testfunc() : passing back results as a SCALAR (set to an ARRAYref).\n");
	}

	// we are now sure that av = (AV *)SvRV(perl_OUT_2D) will be an empty
	// array ready to be filled with subarrays, when we get results

	float *host_INP_2D = (float *)malloc(INP_H*INP_W*sizeof(float));
	if( host_INP_2D == NULL ){ fprintf(stderr, "testfunc() : error, call to malloc has failed for host_INP_2D for %zu bytes (matrix size: (h/rows,w/cols)=(%zu,%zu).\n", sizeof(float)*INP_W*INP_H, INP_H, INP_W); return 1; }
	float *host_OUT_2D = (float *)malloc(OUT_H*OUT_W*sizeof(float));
	if( host_OUT_2D == NULL ){ fprintf(stderr, "testfunc() : error, call to malloc has failed for host_OUT_2D for %zu bytes (matrix size: (h/rows,w/cols)=(%zu,%zu).\n", sizeof(float)*OUT_W*OUT_H, OUT_H, OUT_W); return 1; }

	// Is there a way to copy memory from Perl array in bulk as a block? a-la memcpy?
	float *pd = &(host_INP_2D[0]);
	AV *av = (AV *)SvRV(perl_INP_2D);
	SV *subsv, *subsubsv;
	size_t i, j;
	for(i=0;i<INP_H;i++){ // for each row
		subsv = *av_fetch(av, i, FALSE);
		for(j=0;j<INP_W;j++){ // for the cols of that row
			subsubsv = *av_fetch((AV *)SvRV(subsv), j, FALSE);
			*pd = SvNV(subsubsv);
			pd++;
		}
	}

	// call it
	if( cpu_matrix_mult_float(
		host_INP_2D, host_INP_2D, host_OUT_2D,
		INP_H, INP_W, INP_W,
		1
	) ){ fprintf(stderr, "testfunc() : error, call to cpu_matrix_mult_float() has failed.\n"); return 1; }

	// Transfer results from host to perl(!)
	// device OUT_2D => 
	pd = &(host_OUT_2D[0]);
	AV *subav;
	// avres is the array to push results into, see above
	for(i=0;i<OUT_H;i++){ // for each row
		subav = newAV(); // make a new array for each row
		av_extend(subav, OUT_W); // extend it to hold #cols items (OUT_W)
		// LeoNerd's suggestion
		av_push(avres, newRV_noinc((SV *)subav)); // insert it into the top Array
		for(j=0;j<OUT_W;j++){ // for the cols of that row
			av_store(subav, j, newSVnv(*pd));
			pd++;
		}
	}

	free(host_INP_2D);
	free(host_OUT_2D);

	clock_gettime(CLOCK_MONOTONIC_RAW, &TENDED);
	fprintf(stdout, "testfunc() : time elapsed %f ms.\n", time_difference_seconds(TSTARTED,TENDED));
	return 0; // success
}

EOC

my @inp = ([1,2,3], [4,5,6], [7,8,9]);
my @expected_result = (
	[30,36,42],
	[66,81,96],
	[102,126,150]
);
my @out1 = ();
my $ret = testfunc(\@inp, \@out1);
is($ret, 0, "testfunc() : called, result is an empty array.");
is_deeply(\@out1, \@expected_result, "result is correct.");

my @out2 = ([1,2,3]);
$ret = testfunc(\@inp, \@out2);
is($ret, 0, "testfunc() : called, result is a non-empty array.");
is_deeply(\@out1, \@expected_result, "result is correct.");

my $out3 = undef;
$ret = testfunc(\@inp, $out3);
is($ret, 0, "testfunc() : called, result is a scalar.");

$ret = testfunc(\@inp, \$out3);
is($ret, 0, "cuda_harness() : called, result is a scalar-ref.");
is_deeply(\@out1, \@expected_result, "result is correct.");

done_testing();


#result:
#30	36	42
#66	81	96
#102	126	150
