use 5.006;
use strict;
use warnings;

use File::Temp;

our $VERSION = 0.16;

use Test::More;

use Inline::CUDA::Utils;

my $conf_file = Inline::CUDA::Utils::path_of_configuration_file_in_shared_dir();
ok(defined($conf_file), "path_of_configuration_file_in_shared_dir() : called.");
ok(-f "$conf_file", "path_of_configuration_file_in_shared_dir() : '$conf_file', file exists.");

my $conf = Inline::CUDA::Utils::read_configuration_file_in_shared_dir();
ok(defined($conf), "read_configuration_file_in_shared_dir() : called.");

for my $ak (qw/cc nvcc nvlink ld cxx/){
	ok(exists($conf->{$ak}), "read_configuration_file_in_shared_dir() : contains key '$ak'.");
	my $anexe = $conf->{$ak};
	ok(defined($anexe) && ($anexe!~/^\s*$/), "read_configuration_file_in_shared_dir() : contains entry '$ak' => '".$anexe."'.");
	ok(-x "$anexe", "read_configuration_file_in_shared_dir() : file exists and is executable '$anexe'.");
}
done_testing();
