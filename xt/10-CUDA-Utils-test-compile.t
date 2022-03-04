use 5.006;
use strict;
use warnings;

use File::Temp;

our $VERSION = 0.14;

use Test::More;

use Inline::CUDA::Utils;

# load default site configuration from the shared-dir file
# if we have not yet run make install, it resides in ./blib/...
my $conf = Inline::CUDA::Utils::load_configuration();
ok(defined($conf), "Inline::CUDA::Utils::load_configuration() : called.");
ok(1, join("\n", map { $_ => $conf->{$_} } keys %$conf));
ok(0==Inline::CUDA::Utils::test_compilers($conf), "Inline::CUDA::Utils::test_compilers() : called.");

done_testing();
