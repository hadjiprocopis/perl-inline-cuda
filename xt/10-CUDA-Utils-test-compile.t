use 5.006;
use strict;
use warnings;

use FindBin;
use File::Spec;

our $VERSION = 0.16;

use Test::More;

use Inline::CUDA::Utils;

my $inlinecuda_dir = File::Spec->catdir($FindBin::Bin, '..', 'C', 'C');
# load default site configuration from the shared-dir file
# if we have not yet run make install, it resides in ./blib/...
my $conf = Inline::CUDA::Utils::read_configuration();
ok(defined($conf), "Inline::CUDA::Utils::read_configuration() : called.");
ok(1, join("\n", map { $_ => $conf->{$_} } keys %$conf));
ok(0==Inline::CUDA::Utils::test_compilers($inlinecuda_dir, $conf), "Inline::CUDA::Utils::test_compilers() : called.");

done_testing();
