use 5.006;
use strict;
use warnings;

use File::Temp;

our $VERSION = 0.16;

use Test::More;

use Inline::CUDA::Utils;

local $ENV{__CRAP} = '123';
ok($ENV{__CRAP} eq Inline::CUDA::Utils::envfind('__CRAP'), "read value from ENV with envfind().");
ok($^X eq Inline::CUDA::Utils::exefind('perl'), "found executable ($^X) using exefind().");

my $deps = Inline::CUDA::Utils::find_dependencies();
ok(defined($deps), "find_dependencies() : called.");
for my $ak (qw/cc nvcc nvlink ld cxx/){
	ok(exists($deps->{$ak}) && defined($deps->{$ak}) && ($deps->{$ak}!~/^\s*$/), "find_dependencies() : contains entry '$ak' => '".$deps->{$ak}."'.");
}
done_testing();
