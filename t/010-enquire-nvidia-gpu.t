use strict;
use warnings;

# This test does a basic `use` check on all the code.
our $VERSION = 0.16;

use Test::More;

use nvidia2::ml::Utils;

my $ret = nvidia2::ml::Utils::enquire_gpu_hardware();
ok(defined($ret), "nvidia2::ml::Utils::enquire_gpu_hardware() : called.");

ok(exists($ret->{'nvmlDeviceGetCount'}) && defined($ret->{'nvmlDeviceGetCount'}) && ref($ret->{'nvmlDeviceGetCount'})eq'ARRAY', "nvmlDeviceGetCount() returned some value.");
my $numGPU = $ret->{'nvmlDeviceGetCount'}->[0];
ok(defined($numGPU)&&($numGPU>0), "Found $numGPU GPUs.") or BAIL_OUT("no cuda-capable GPUs found, can not continue.");

ok(exists($ret->{'nvmlSystemGetDriverVersion'}) && defined($ret->{'nvmlSystemGetDriverVersion'}) && ref($ret->{'nvmlSystemGetDriverVersion'})eq'ARRAY', "nvmlSystemGetDriverVersion() returned some value.");
my $driverversion = $ret->{'nvmlSystemGetDriverVersion'}->[0];
ok(defined($driverversion)&&($driverversion!~/^\s*$/), "nvmlSystemGetDriverVersion is not empty.") or BAIL_OUT("Failed to get the GPU Driver Version, can not continue.");

diag("Found $numGPU GPUs and the GPU Driver Version is $driverversion.");

ok(exists($ret->{GPU}) && defined($ret->{GPU}) && ref($ret->{GPU})eq'ARRAY', "Found key 'GPU' and is an ARRAYref.");
ok(scalar(@{$ret->{GPU}})==$numGPU, "Key 'GPU' contains an array with $numGPU elements, as many as the GPUs found.");
my ($minor, $major);
for my $aGPUindex (0..($numGPU-1)){
	my $RG = $ret->{GPU}->[$aGPUindex];
	ok(defined($RG) && ref($RG)eq'HASH', "Found array item for GPU $aGPUindex, for key 'GPU' and is a HASHref.");
	ok(exists($RG->{nvmlDeviceGetName}) && defined($RG->{nvmlDeviceGetName}) && ref($RG->{nvmlDeviceGetName})eq'ARRAY', "Found key 'nvmlDeviceGetName' for GPU index $aGPUindex.");
	my $aGPUname = $RG->{nvmlDeviceGetName}->[0];
	ok(defined($aGPUname)&&($aGPUname!~/^\s*$/), "Found the name for GPU with index $aGPUindex.");
	ok(exists($RG->{nvmlDeviceGetCudaComputeCapability}) && defined($RG->{nvmlDeviceGetCudaComputeCapability}) && ref($RG->{nvmlDeviceGetCudaComputeCapability})eq'ARRAY', "Found key 'nvmlDeviceGetCudaComputeCapability' for GPU index $aGPUindex.");
	my ($major, $minor) = @{$RG->{nvmlDeviceGetCudaComputeCapability}};
	ok(defined($major)&&($major!~/^\s*$/), "Found valid compute capability.");
	ok(defined($minor)&&($minor!~/^\s*$/), "Found valid compute capability.");
	diag("GPU index ${aGPUindex}, '$aGPUname': compute capability: '$major.$minor'.");
}

done_testing;
