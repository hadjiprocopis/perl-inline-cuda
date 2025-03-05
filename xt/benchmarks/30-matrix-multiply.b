use strict;
use warnings;

our $VERSION = 0.16;

use Test::More;
use Benchmark qw/timethese cmpthese :hireswallclock/;
use constant VERBOSE => 1; # prints out several junk

use File::Temp qw/tempfile/;
use Time::HiRes qw/time/;
use Data::Dumper;
use Devel::Peek 'Dump';
use Math::Matrix;
use FindBin;
use File::Spec;

my $ERROR_TOLERANCE = 1E-04;

# specify what tests to run to multiply A x B,
# AH: number of rows, AW: number of cols
# if a test is too slow, then skip it, see below
my %tests = (
	'small'  => {'AH'=>10,'AW'=>12,'BW'=>11, 'iterations' => 10},
	'medium' => {'AH'=>100,'AW'=>120,'BW'=>110, 'iterations' => 2},
	'large'  => {'AH'=>1000,'AW'=>1200,'BW'=>1100, 'iterations' => 1},
);
my @TEST_NAMES_TO_DO = qw/small medium large/;

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

/* kernel specific param required prior to using any of the cuda kernels */
#define BLOCK_SIZE 16

// helper functions to facilitate the bridge with perl
#include <inlinecuda/perlbridge.h>
#include <inlinecuda/perlbridge.c>

// these contain numerical recipes for CPU matrix operations
#include <inlinecuda/perl_cpu_bridge.h>
#include <inlinecuda/perl_cpu_bridge.c>

// these contain numerical recipes for GPU matrix operations
#include <inlinecuda/perl_cuda_bridge.h>
#include <inlinecuda/perl_cuda_bridge.c>
// perl does not see cpu_matrix_multiply() because it was added via a #include
// so resort to this hack:
int call_cpu_matrix_multiply(
	SV *perl_A,
	SV *perl_B,
	SV *perl_R,
	int noisy
){ return cpu_matrix_multiply(perl_A, perl_B, perl_R, noisy); }
int call_inline_cuda_matrix_multiply(
	SV *perl_A,
	SV *perl_B,
	SV *perl_R,
	int noisy
){ return inline_cuda_matrix_multiply(perl_A, perl_B, perl_R, noisy); }

EOCODE
=pod
	# we call high level functions to do CPU/GPU matrix multiplication
	# which are located in C/C/inlinecuda/*.c etc.
	# we can compile them as libraries and link them
	# or we can #include the C code (as above) which has the disadvantage
	# of Perl not "finding" the C functions via Inline::CUDA
	# or just append the C code to the code to run as below:
	for my $afname (
		File::Spec->catdir($FindBin::Bin, '..', '..', 'C', 'C', 'inlinecuda', 'cpu_matrix_ops_float.c'),
		File::Spec->catdir($FindBin::Bin, '..', '..', 'C', 'C', 'inlinecuda', 'perl_cuda_bridge.c'),
		File::Spec->catdir($FindBin::Bin, '..', '..', 'C', 'C', 'inlinecuda', 'perl_cpu_bridge.c'),
	){
		my $FH;
		if( ! open($FH, '<', $afname) ){ die "error, failed to read '$afname', bailing out..." }
		$code .= "/* IMPORT FROM: $afname */\n";
		{local $/ = undef; $code .= <$FH> } close($FH);
		$code .= "/* END IMPORT FROM: $afname */\n";
	}
=cut
}

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
		.'-I"'.File::Spec->catdir($FindBin::Bin, '..', '..', 'C', 'C').'"'
	,
;

use Inline CUDA => $code;

for my $tname (@TEST_NAMES_TO_DO){
	my $tv = $tests{$tname};
	my $iterations = $tv->{'iterations'};
	my $AH = $tv->{AH};
	my $AW = $tv->{AW};
	my $BH = $AW;
	my $BW = $tv->{BW};
	ok(1, "Doing test '$tname' for $iterations iterations ...\n");
	my $mytests = {
		"CPU  $tname dataset, iters ${iterations}:" => sub { do_cpu_matrix_multi($AH, $AW, $BH, $BW) },
		"CUDA $tname dataset, iters ${iterations}:" => sub { do_inline_cuda_matrix_multi($AH, $AW, $BH, $BW) },
	};
	# PERL is too slow for any test other than 'small' so do only for that
	if( $tname eq 'small' ){
		$mytests->{"PERL $tname dataset, iters ${iterations}:"} = sub { do_perl_matrix_multi($AH, $AW, $BH, $BW) }
	}
	# shamelessly ripped off App::Benchmark
	cmpthese(timethese($iterations, $mytests));
}

sub do_inline_cuda_matrix_multi {
	my ($AH, $AW, $BH, $BW) = @_;
	my $RW = $BW; my $RH = $AH;
	my $A = random_matrix($AH, $AW);
	my $B = random_matrix($BH, $BW);
	my $R = [];
	my $ret = call_inline_cuda_matrix_multiply($A, $B, $R, 0);
	if( ! defined $R ){ die "error, inline_cuda_matrix_multiply() failed for ($AH,$AW)x($BH,$BW)." }
	undef $R, $A, $B;
}
sub do_perl_matrix_multi {
	my ($AH, $AW, $BH, $BW) = @_;
	my $RW = $BW; my $RH = $AH;
	my $mA = matrix2mathmatrix(random_matrix($AH, $AW));
	my $mB = matrix2mathmatrix(random_matrix($BH, $BW));
	my $mR = $mA->multiply($mB);
	if( ! defined $mR ){ die "error, multiply() failed (Math::Matrix) for ($AH,$AW)x($BH,$BW)." }
	undef $mA, $mB, $mR;
}
sub do_cpu_matrix_multi {
	my ($AH, $AW, $BH, $BW) = @_;
	my $RW = $BW; my $RH = $AH;
	my $A = random_matrix($AH, $AW);
	my $B = random_matrix($BH, $BW);
	my $R = [];
	my $ret = call_cpu_matrix_multiply($A, $B, $R, 0);
	if( ! defined $R ){ die "error, cpu_matrix_mult() failed for ($AH,$AW)x($BH,$BW)." }
	undef $R, $A, $B;
}
sub random_matrix {
	my ($H, $W, $seed) = @_;
	srand($seed) if $seed;
	return [ map { [ map { rand } 1..$W ] } 1..$H ]
}
sub make_and_initialise_matrix {
	my ($H, $W, $v) = @_;
	return [ map { [ ($v)x$W ] } 1..$H ]
}
sub matrix2mathmatrix { return Math::Matrix->new(@{$_[0]}) }

ok(1,"OK");
done_testing();
