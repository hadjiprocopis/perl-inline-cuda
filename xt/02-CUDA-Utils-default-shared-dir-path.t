use 5.006;
use strict;
use warnings;

our $VERSION = 0.16;

use Test::More;

use Inline::CUDA::Utils;

my $default_conf_file = Inline::CUDA::Utils::path_of_configuration_file_in_shared_dir();
ok(defined($default_conf_file), "Inline::CUDA::Utils::path_of_configuration_file_in_shared_dir() : called.");
ok(-e $default_conf_file, "Inline::CUDA::Utils::path_of_configuration_file_in_shared_dir() : conf location exists.");
ok(-f $default_conf_file, "Inline::CUDA::Utils::path_of_configuration_file_in_shared_dir() : conf-file is '$default_conf_file'.");

done_testing();
