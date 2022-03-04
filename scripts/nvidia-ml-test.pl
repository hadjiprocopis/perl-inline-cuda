#!/usr/bin/perl

use strict;
use warnings;

use nvidia::ml qw(:all);


our $VERSION = '0.14';

nvmlInit();

my ($ret, $version, $count, $i, $handle, $speed, $info, $total, $name);
 
($ret, $version) = nvmlSystemGetDriverVersion();
die nvmlErrorString($ret) unless $ret == $nvidia::ml::bindings::NVML_SUCCESS;
print "Driver version: " . $version . "\n";
 
($ret, $count) = nvmlDeviceGetCount();
die nvmlErrorString($ret) unless $ret == $nvidia::ml::bindings::NVML_SUCCESS;
 
for ($i=0; $i<$count; $i++)
{
    ($ret, $handle) = nvmlDeviceGetHandleByIndex($i);
    next if $ret != $nvidia::ml::bindings::NVML_SUCCESS;
  
    ($ret, $name) = nvmlDeviceGetName($handle);
    next if $ret != $nvidia::ml::bindings::NVML_SUCCESS;
	print "Name: $name\n";

    ($ret, $speed) = nvmlDeviceGetFanSpeed($handle);
    next if $ret != $nvidia::ml::bindings::NVML_SUCCESS;
    print "Device " . $i . " fan speed: " . $speed . "%\n";
  
    ($ret, $info) = nvmlDeviceGetMemoryInfo($handle);
    next if $ret != $nvidia::ml::bindings::NVML_SUCCESS;
    $total = ($info->{"total"} / 1024 / 1024);
    print "Device " . $i . " total memory: " . $total . " MB\n";
}
 
nvmlShutdown();

