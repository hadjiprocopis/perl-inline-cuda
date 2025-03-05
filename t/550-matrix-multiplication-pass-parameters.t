use 5.006;
use strict;
use warnings;

our $VERSION = 0.16;

use Test::More;
use File::Temp qw/tempfile/;
use Time::HiRes qw/time/;
use Math::Matrix;
use FindBin;
use File::Spec;

use Inline CUDA => Config =>
	host_code_language => 'C',
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
;

my $ERROR_TOLERANCE = 1E-04;

# we are going to read some functions from the library, but Inline::C does
# not parse #include's
# so we are basically including some C file so as not to repeat code
my $code;
BEGIN {
	$code = <<'EOCODE';
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>

// helper functions to facilitate the bridge with perl
#include "inlinecuda/perlbridge.h"
#include "inlinecuda/perlbridge.c"

#define BLOCK_SIZE 16
#include "inlinecuda/cuda_kernels/matrix_multiplication_lzhengchun_float.cu.h"
#include "inlinecuda/cuda_kernels/matrix_multiplication_lzhengchun_float.cu"

EOCODE
	for my $afname (
		File::Spec->catdir($FindBin::Bin, '..', 'C', 'C', 'inlinecuda', 'perl_cuda_bridge.c')
	){
		my $FH;
		if( ! open($FH, '<', $afname) ){ die "error, failed to read '$afname', bailing out..." }
		$code .= "/* IMPORT FROM: $afname */\n";
		{local $/ = undef; $code .= <$FH> } close($FH);
		$code .= "/* END IMPORT FROM: $afname */\n";
	}
}

use Inline CUDA => $code;

my $ret1 = small_test();

# the small test is allowed to report all results, failed or success
# but the large test will have its succeeded tests redirected to a file
# because they are too many for large matrices, only failed tests will appear
# neat hack
my $test_more_filehandle = Test::More->builder->output; # save the old fh

my ($tmpfh, $tmpfilename) = tempfile();
#Test::More->builder->output($tmpfh);
#Test::More->builder->failure_output($test_more_filehandle);
my $ret2 = large_test();
Test::More->builder->output($test_more_filehandle);
close($tmpfh);
unlink($tmpfilename);

ok(1, "DONE here is a summary:\n".join("\n", @$ret1, @$ret2));

sub small_test {
	my (@retstr, $astr, $ret);
	my $testname = 'small_test';
	my $startTime = time;
	my $noisy = 1; # 2 is too verbose, it prints all the matrices
	my $AW = 8; my $AH = 5;
	my $BW = 3; my $BH = 8;
	my $RW = $BW; my $RH = $AH;
	is($AW, $BH,"$testname : A's width is the same as B's height.");
	my (@A, @B, @R, $i);
	for($i=0;$i<$AH;$i++){ push @A, [ map { rand() } 1..$AW ] }
	for($i=0;$i<$BH;$i++){ push @B, [ map { rand() } 1..$BW ] }
	
	# do the multiplication in Perl
	my $mA = Math::Matrix->new(@A);
	ok(defined($mA),"$testname : Math::Matrix->new() : called.");
	my $mB = Math::Matrix->new(@B);
	ok(defined($mB),"$testname : Math::Matrix->new() : called.");
	my $startTime1 = time;
	my $mR = $mA->multiply($mB);
	$astr = "$testname : done Math::Matrix() multiplication of ($AH,$AW) x ($BH,$BW) in ".(time-$startTime)." seconds.";
	push @retstr, $astr;
	ok(defined($mR),$astr);
	push @R, 1..3;
	my $T = 'Case1(sending \@R)';
	
	$startTime1 = time;
	$ret = inline_cuda_matrix_multiply(\@A, \@B, \@R, $noisy);
	$astr = "$testname : $T: inline_cuda_matrix_multiply() called for ($AH,$AW) x ($BH,$BW), success in ".(time-$startTime1)." seconds.";
	is($ret, 0, $astr) or BAIL_OUT("can not continue, cuda failed.");
	push @retstr, $astr;

	is(scalar(@R), $RH,"$testname : $T: rows are $RH");
	for(my $i=0;$i<$RH;$i++){
		ok(ref($R[$i])eq'ARRAY',"$testname : $T : item $i is ARRAYref.");
		is(scalar(@{$R[$i]}), $RW,"$testname : $T : it has $RW elements: ".scalar(@{$R[$i]}));
	}
	my $average_error = 0.0;
	# the most stupid way to access Math::Matrix elements!
	$mR->map(sub {
		my $I = $_[0];
		my $J = $_[1];
		my $anerror = abs($_ - $R[$I][$J]);
		ok($anerror <= $ERROR_TOLERANCE,"$testname : $T : discrepancy [$I][$J] $anerror <= $ERROR_TOLERANCE");
		$average_error += $anerror;
	});
	$average_error /= ($RH*$RW);
	$astr = "$testname : $T : total discrepancy $average_error <= $ERROR_TOLERANCE";
	push @retstr, $astr;
	ok($average_error <= $ERROR_TOLERANCE,"$astr");
	$astr = "$testname : done matrix multiplication of ($AH,$AW) x ($BH,$BW) in ".(time-$startTime)." seconds.";
	ok(1, $astr);
	push @retstr, $astr;
	return \@retstr;
}
sub large_test {
	my (@retstr, $astr, $ret);
	my $testname = 'large_test';
	my $startTime = time;
	my $noisy = 1; # 2 is too verbose, it prints all the matrices
	my $AW = 80; my $AH = 50;
	my $BW = 30; my $BH = 80;
	my $RW = $BW; my $RH = $AH;
	is($AW, $BH,"$testname : A's width is the same as B's height.");
	my (@A, @B, @R, $i);
	for($i=0;$i<$AH;$i++){ push @A, [ map { rand() } 1..$AW ] }
	for($i=0;$i<$BH;$i++){ push @B, [ map { rand() } 1..$BW ] }
	
	# do the multiplication in Perl
	my $mA = Math::Matrix->new(@A);
	ok(defined($mA),"$testname : Math::Matrix->new() : called.");
	my $mB = Math::Matrix->new(@B);
	ok(defined($mB),"$testname : Math::Matrix->new() : called.");
	my $startTime1 = time;
	my $mR = $mA->multiply($mB);
	$astr = "$testname : done Math::Matrix() multiplication of ($AH,$AW) x ($BH,$BW) in ".(time-$startTime)." seconds.";
	push @retstr, $astr;
	ok(defined($mR),$astr);
	push @R, 1..3;
	my $T = 'Case1(sending \@R)';
	
	$startTime1 = time;
	$ret = inline_cuda_matrix_multiply(\@A, \@B, \@R, $noisy);
	$astr = "$testname : $T: inline_cuda_matrix_multiply() called for ($AH,$AW) x ($BH,$BW), success in ".(time-$startTime1)." seconds.";
	is($ret, 0, $astr) or BAIL_OUT("can not continue, cuda failed.");
	push @retstr, $astr;

	is(scalar(@R), $RH,"$testname : $T: rows are $RH");
	for(my $i=0;$i<$RH;$i++){
		ok(ref($R[$i])eq'ARRAY',"$testname : $T : item $i is ARRAYref.");
		is(scalar(@{$R[$i]}), $RW,"$testname : $T : it has $RW elements: ".scalar(@{$R[$i]}));
	}
	my $average_error = 0.0;
	# the most stupid way to access Math::Matrix elements!
	$mR->map(sub {
		my $I = $_[0];
		my $J = $_[1];
		my $anerror = abs($_ - $R[$I][$J]);
		ok($anerror <= $ERROR_TOLERANCE,"$testname : $T : discrepancy [$I][$J] $anerror <= $ERROR_TOLERANCE");
		$average_error += $anerror;
	});
	$average_error /= ($RH*$RW);
	$astr = "$testname : $T : total discrepancy $average_error <= $ERROR_TOLERANCE";
	push @retstr, $astr;
	ok($average_error <= $ERROR_TOLERANCE,"$astr");
	$astr = "$testname : done matrix multiplication of ($AH,$AW) x ($BH,$BW) in ".(time-$startTime)." seconds.";
	ok(1, $astr);
	push @retstr, $astr;
	return \@retstr;
}

done_testing();
