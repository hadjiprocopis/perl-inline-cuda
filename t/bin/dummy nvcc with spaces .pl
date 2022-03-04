#!/usr/bin/env perl

use Getopt::Long;
use File::Temp;

my $compiler_bindir;
my @Xcompiler;
my @Xlinker;

Getopt::Long::GetOptions(
	'compiler-bindir=s' => \$compiler_bindir,
	'Xcompiler=s' => \@Xcompiler,
	'Xlinker=s' => \@Xlinker,
) or die "Error in command line arguments";

if( defined $compiler_bindir ){
	my ($fh, $filename) = File::Temp::tempfile(SUFFIX => '.c');
	print $fh <<'EOC';
#include <stdio.h>
int main(void){ return 0; }
EOC
	close($fh);
	my $cmd = <<EOC;
${compiler_bindir} "$filename"
EOC
	print STDOUT "$cmd\n\n$0 : testing '--compiler-bindir' executable '${compiler_bindir} with above command ...\n";
	my $ret = qx/${cmd}/;
	unlink($filename);
	if( $? != 0 ){
		print STDERR "$ret\n\nError executing compiler-bindir command with above output: '${compiler_bindir}'.\n";
		exit(1);
	}
}

1;
