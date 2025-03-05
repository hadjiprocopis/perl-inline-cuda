use strict;
use warnings;

use Image::PNG::Libpng ':all';
use Image::PNG::Const ':all';

use Data::Roundtrip qw/perl2dump/;

use File::ShareDir qw/dist_dir/;

our $VERSION = 0.11;

use constant VERBOSE => 1; # prints out several junk

# see https://github.com/marioroy/mandelbrot-python

use Inline CUDA => Config =>
	host_code_language => 'C',
	BUILD_NOISY => 1,
	warnings => 10,
	clean_after_build => 0,
#	inc => '-I' . File::Spec->catdir(
#		File::ShareDir::dist_dir('Inline-CUDA'),
#		'C'
#	),
;

use Inline CUDA => <<'EOC';
#include "mandelbrot-marioroy.c"
int mandelbrot_caller(
        double min_x, double min_y,
        double step_x, double step_y,
        SV *perl_result, SV *perl_colors,
        int iters,
        int width, int height,
        int bDimX, int bDimY,
        int gDimX, int gDimY
);
int mandelbrot_caller(
        double min_x, double min_y,
        double step_x, double step_y,
        SV *perl_result, SV *perl_colors,
        int iters,
        int width, int height,
        int bDimX, int bDimY,
        int gDimX, int gDimY
){ return mandelbrot_main(min_x, min_y, step_x, step_y, perl_result, perl_colors, iters, width, height, bDimX, bDimY, gDimX, gDimY); }

EOC

my $PARAMS = {
	'GRADIENT_LENGTH' => 256,
	'min_x' => 0.0,
	'min_y' => 0.0,
	'step_x' => 0.005,
	'step_y' => 0.005,
	'iters' => 1000,
	'width'  => 500,
	'height' => 500,
	'bDimX' => 8,
	'bDimY' => 4,
	'shared' => 0,
};
$PARAMS->{'gDimX'} = divide_up($PARAMS->{'width'}, $PARAMS->{'bDimX'});
$PARAMS->{'gDimY'} = divide_up($PARAMS->{'height'}, $PARAMS->{'bDimY'});
#$PARAMS->{'block'} = [ $PARAMS->{'bDimX'}, $PARAMS->{'bDimY'}, 1 ];
#$PARAMS->{'grid'} = [ $PARAMS->{'gDimX'}, $PARAMS->{'gDimY'} ];

$PARAMS->{'colors'} = init_colors($PARAMS, 1);
$PARAMS->{'colors1d'} = colors_rgb21d($PARAMS->{'colors'});
$PARAMS->{'compression_level'} = 9;
mandelbrot_cuda($PARAMS);

sub mandelbrot_cuda {
	my $params = $_[0];
	# this will become an arrayref to hold the result
	# which is 4 bytes packed as an uint32_t (uchar4)
	my $result = [];
	#print "Colors:\n".perl2dump($params->{'colors1d'})."\n";
	my $ret = mandelbrot_caller(
		$params->{'min_x'}, $params->{'min_y'},
		$params->{'step_x'}, $params->{'step_y'},
		$result, $params->{'colors1d'},
		$params->{'iters'},
		$params->{'width'}, $params->{'height'},
		$params->{'bDimX'}, $params->{'bDimY'},
		$params->{'gDimX'}, $params->{'gDimY'}
	);
	print "Creating the image ...\n";
	my @rows;
	for my $x (0..$params->{'width'}-1){
		my $col = "";
		for my $y (0..$params->{'height'}-1){
			my $argb = $result->[$x + $y * $params->{'width'}];
			my $R = ($argb & 0x00FF0000) >> 16;
			my $G = ($argb & 0x0000FF00) >> 8;
			my $B = ($argb & 0x000000FF);
			$col .= pack 'CCC', $R, $G, $B;
		}
		push @rows, $col;
	}

	my %ihdr = (
	    width => $params->{'width'},
	    height => $params->{'height'},
	    bit_depth => 8,
	    # one of: PNG_COLOR_TYPE_GRAY, PNG_COLOR_TYPE_GRAY_ALPHA, PNG_COLOR_TYPE_PALETTE, PNG_COLOR_TYPE_RGB, PNG_COLOR_TYPE_RGB_ALPHA.
	    color_type => PNG_COLOR_TYPE_RGB,
	);
	my $outimage = 'outmandelbrot.png';
	my $png = create_writer($outimage);
	$png->set_compression_level($params->{'compression_level'});
	$png->set_IHDR(\%ihdr);
	$png->set_rows(\@rows);
	$png->write_png();
	print "Output image written to '$outimage' ...\n";
}	

# convert an array of [R,G,B]'s into a 1d array
sub colors_rgb21d {
	my @colors_in = @{$_[0]}; # an arrayref of colors as [R,G,B] each.

	my @colors1d;
	push @colors1d, @$_ foreach @colors_in;
	return \@colors1d;
}

sub fill_linear {
	my $params = $_[0];
	# input: an arrayref, of siize $num_colors equal to the number of colors
	# it contains, each as an arrayref of [R,G,B]
	my @palette = @{$_[1]};

	my $num_colors = scalar @palette - 1;
	my @colors; # returning this array (as an arrayref)
	for my $i (0..($params->{'GRADIENT_LENGTH'}-1)){
		my $mu = ($i*1.0) / $params->{'GRADIENT_LENGTH'};
		$mu *= $num_colors;
		my $i_mu = int($mu);
		my $dx = $mu - $i_mu;
		my $c1 = $palette[$i_mu];  # this is an [R,G,B] triplet
		my $c2 = $palette[$i_mu+1];# same as this
		$colors[$i] = [
			int($dx * ($c2->[0] - $c1->[0]) + $c1->[0]),
			int($dx * ($c2->[1] - $c1->[1]) + $c1->[1]),
			int($dx * ($c2->[2] - $c1->[2]) + $c1->[2])
		];
	}
	return \@colors;
}

sub init_colors {
	my $params = $_[0];
	my $color_scheme = $_[1]; # 1, 2, 3,
	my $palette; # returning this as an arrayref of [R,G,B]
	if( $color_scheme == 1 ){
		$palette = [
			[0x0,0x0,0x0],
			[0x44,0x77,0xaa],
			[0x66,0xcc,0xee],
	    		[0x22,0x88,0x33],
	    		[0xcc,0xbb,0x44],
	    		[0xee,0x66,0x77],
	    		[0xaa,0x33,0x77],
			[0x8,0x8,0x8],
		];
	} elsif( $color_scheme == 2 ){
		$palette = [
			[0x0,0x0,0x0],
	    		[0x65,0xbf,0xa1],
	    		[0x2d,0x95,0xeb],
	    		[0xee,0x4b,0x2f],
			[0x8,0x8,0x8],
		];
	} elsif( $color_scheme == 3 ){
		$palette = [
			[0x0,0x0,0x0],
	    		[0xfb,0xc9,0x65],
	    		[0x65,0xcb,0xca],
	    		[0xf8,0x64,0x64],
			[0x8,0x8,0x8],
		];
	} elsif( $color_scheme == 4 ){
		$palette = [
			[0x0,0x0,0x0],
	    		[0x77,0xaa,0xdd],
	    		[0xff,0xaa,0xbb],
	    		[0xaa,0xaa,0x00],
			[0x8,0x8,0x8],
		];
	}

	return fill_linear($params, $palette);
}
sub divide_up {
	my ($dividend, $divisor) = @_;
	# Helper funtion to get the next up value for integer division.
	# from https://github.com/marioroy/mandelbrot-python/app/base.py
	# translated from this python code:
	#   return dividend // divisor + 1 if dividend % divisor else dividend // divisor

	return
		$dividend % $divisor
	?
		int($dividend / $divisor) + 1
	:
		int($dividend / $divisor)
}
