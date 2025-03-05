use 5.006;
use strict;
use warnings;

our $VERSION = 0.16;

use Test::More;

use File::Spec;
use FindBin;

use Inline C => Config =>
	clean_after_build => 0,
	BUILD_NOISY => 10,
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

use Inline C => <<'EOC';

#include <stdio.h>

#include <inlinecuda/perlbridge.c>

int test_passing_array_1D_is_array_ref(
	SV *array,
	SV *array_sz
);
int test_passing_array_2D_is_array_ref(
	SV *array,
	SV *array_sz
);
int test_passing_array_1D(
	SV *array,
	SV *array_sz
);
int test_passing_array_2D(
	SV *array,
	SV *dim1,
	SV *dim2
);

int test_passing_array_1D_is_array_ref(
	SV *array,
	SV *array_sz
){
	size_t asz;
	if( is_array_ref(array, &asz) == 0 ){
		fprintf(stderr, "test_passing_array_1D_is_array_ref() : error, call to is_array_ref() has failed.\n");
		sv_setiv(array_sz, 0);
		return 1;
	}
	sv_setiv(array_sz, asz);
	return 0;
}

int test_passing_array_2D_is_array_ref(
	SV *array,
	SV *array_sz
){
	size_t dim2sz;
	if( is_array_ref(array, &dim2sz) == 0 ){
		fprintf(stderr, "test_passing_array_2D_is_array_ref() : error, call to is_array_ref() has failed.\n");
		sv_setiv(array_sz, 0);
		return 1;
	}
	sv_setiv(array_sz, dim2sz);
	return 0;
}

int test_passing_array_1D(
	SV *array,
	SV *array_sz
){
	size_t asz;
	if( array_numelts_1D(array, &asz) ){ fprintf(stderr, "test_passing_array_1D() : error, call to array_numelts_1D() has failed.\n"); return 1; }
	sv_setiv(array_sz, asz);
	return 0;
}

int test_passing_array_2D(
	SV *array,
	SV *dim1,
	SV *dim2
){
	size_t dim2sz;
	if( ! is_array_ref(dim2, &dim2sz) ){ fprintf(stderr, "test_passing_array_2D() : error, call to is_array_ref() has failed.\n"); return 1; }

	size_t d1, *d2 = NULL;
	if( array_numelts_2D(array, &d1, &d2) ){ fprintf(stderr, "test_passing_array_2D() : error, call to array_numelts_2D() has failed.\n"); return 1; }

	sv_setiv(dim1, d1);

	if( array_of_unsigned_int_into_AV(d2, d1, dim2) ){ fprintf(stderr, "test_passing_array_2D() : error, call to array_of_unsigned_int_into_AV() has failed.\n"); return 1; }

	return 0;
}

EOC

my ($W, $H, $gotW, @gotH, @inp1D, @inp2D, $retcode, $notarray);


### check is_array_ref:
$W = 5;
@inp1D = (1..$W);
$retcode = test_passing_array_1D_is_array_ref(\@inp1D, $gotW);
ok(defined($retcode),"test_passing_array_1D_is_array_ref() : called.") or BAIL_OUT;
is($retcode, 0, "test_passing_array_1D_is_array_ref() : returned success.") or BAIL_OUT;
is($gotW, $W, "test_passing_array_1D_is_array_ref() : returned correct array size (original:".$W.", got:$gotW).") or BAIL_OUT;
# and a failure
$notarray = {'a'=>1};
$retcode = test_passing_array_1D_is_array_ref($notarray, $gotW);
ok(defined($retcode),"test_passing_array_1D_is_array_ref() : called.") or BAIL_OUT;
is($retcode, 1, "test_passing_array_1D_is_array_ref() : returned failure AS EXPECTED.") or BAIL_OUT;
is($gotW, 0, "test_passing_array_1D_is_array_ref() : returned zero as the array size because input is not an array.") or BAIL_OUT;

$W = 5; $H = 8;
@inp2D = ();
for(my $i=0;$i<$W;$i++){
	push @inp2D, [ map { $i*10+$_ } 1..$H ];
}
@gotH = ();
$retcode = test_passing_array_2D_is_array_ref(\@inp2D, $gotW);
ok(defined($retcode),"test_passing_array_2D_is_array_ref() : called.") or BAIL_OUT;
is($retcode, 0, "test_passing_array_2D_is_array_ref() : returned success.") or BAIL_OUT;
is($gotW, $W, "test_passing_array_2D_is_array_ref() : returned correct array size (original:".$W.", got:$gotW).") or BAIL_OUT;
# and a failure
$notarray = {'a'=>1};
@gotH = ();
$retcode = test_passing_array_2D_is_array_ref($notarray, $gotW);
ok(defined($retcode),"test_passing_array_2D_is_array_ref() : called.") or BAIL_OUT;
is($retcode, 1, "test_passing_array_2D_is_array_ref() : returned failure AS EXPECTED.") or BAIL_OUT;
is($gotW, 0, "test_passing_array_2D_is_array_ref() : returned array size as zero because it is not an array") or BAIL_OUT;

### check xxx
$W = 5;
@inp1D = (1..$W);
$retcode = test_passing_array_1D(\@inp1D, $gotW);
ok(defined($retcode),"test_passing_array_1D() : called.") or BAIL_OUT;
is($retcode, 0, "test_passing_array_1D() : returned success.") or BAIL_OUT;
is($gotW, $W, "test_passing_array_1D() : returned correct array size (original:".$W.", got:$gotW).") or BAIL_OUT;

$W = 5; $H = 8;
@inp2D = ();
for(my $i=0;$i<$W;$i++){
	push @inp2D, [ map { $i*10+$_ } 1..$H ];
}
$retcode = test_passing_array_2D(\@inp2D, $gotW, \@gotH);
ok(defined($retcode),"test_passing_array_2D() : called.") or BAIL_OUT;
is($retcode, 0, "test_passing_array_2D() : returned success.") or BAIL_OUT;
is($gotW, $W, "test_passing_array_2D() : returned correct array size (original:".$W.", got:$gotW).") or BAIL_OUT;
for(my $i=0;$i<$W;$i++){
	is($gotH[$i], $H, "test_passing_array_2D() : returned correct array size for item $i (original:".$H.", got:".$gotH[$i].").") or BAIL_OUT;
}


done_testing();
