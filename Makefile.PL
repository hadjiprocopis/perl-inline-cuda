use 5.006;
use strict;
use warnings;
use ExtUtils::MakeMaker;
use File::ShareDir::Install;
use File::Spec;
use FindBin;

use Data::Roundtrip qw/perl2dump/;

my %WriteMakefileArgs = (
# remove before dist!!!!!!!
#INSTALL_BASE => 'xxx',
	NAME		   => 'Inline::CUDA',
	AUTHOR		   => q{Andreas Hadjiprocopis <bliako@cpan.org>},
	VERSION_FROM	   => 'lib/Inline/CUDA.pm',
	ABSTRACT_FROM	   => 'lib/Inline/CUDA.pm',
	EXE_FILES 	   => [qw!script/nvidia-ml-test.pl!],
	LICENSE		   => 'artistic_2',
	MIN_PERL_VERSION   => '5.006',
	CONFIGURE_REQUIRES => {
		'ExtUtils::MakeMaker' => '0',
		'File::Which' => '0',
		'File::Find' => '0',
		'FindBin' => '0',
		'File::Spec' => '0',
		'File::ShareDir::Install' => '0',
		'Inline' => '0.56',
		'Inline::C' => '0.80',
		'Try::Tiny' => '0',
		'Data::Roundtrip' => '0',
		'nvidia2::ml' => '0',
	},
	TEST_REQUIRES => {
		'nvidia2::ml' => '0',
		'nvidia2::ml::Utils' => '0',
		'Test::More' => '0',
		'Try::Tiny' => '0',
		'Config' => '0',
		'Data::Roundtrip' => '0',
		'ExtUtils::Manifest' => '0',
		'File::Copy' => '0',
		'File::Find' => '0',
		'File::Path' => '0',
		'File::Spec' => '0',
		'File::Temp' => '0',
		'FindBin' => '0',
		'Inline' => '0',
		'Math::Matrix' => '0',
		'Time::HiRes' => '0',
	},
	PREREQ_PM => {
		'Inline' => '0.56',
		'Inline::C' => '0.80',
		'Config' => '0',
		'Carp' => '0',
		'Cwd' => '0',
		'FindBin' => '0',
		'Data::Roundtrip' => '0',
		'Getopt::Long' => '0',
		'Test::Pod::Coverage' => '0',
		'File::ShareDir' => '0',
		'File::Basename' => '0',
		'File::Which' => '0',
		'Image::PNG::Const' => '0',
		'Image::PNG::Libpng' => '0',
		'Text::CSV_XS' => '0',
		'Try::Tiny' => '0',
		'File::Spec' => '0',
		'File::Copy' => '0',
		'File::Temp' => '0',
		'ShellQuote::Any' => '0',
		'Data::Roundtrip' => '0',
		'nvidia2::ml' => '0',
	},
	META_MERGE => {
		'meta-spec' => { version => 2 },
		requires	=> { perl	=> '5.10.0' },
		resources   => {
			license	=> 'artistic_2',
			homepage   => 'https://metacpan.org/module/Inline::CUDA',
			repository => {
				url  => 'https://github.com/hadjiprocopis/perl-inline-cuda',
				web  => 'https://github.com/hadjiprocopis/perl-inline-cuda',
				type => 'git',
			},
			# no i prefer cpan's
			#bugtracker => {
			#	web => 'https://github.com/hadjiprocopis/perl-inline-cuda/issues',
			#}
		},
	},
	dist  => { COMPRESS => 'gzip -9f', SUFFIX => 'gz', },
	clean => { FILES => 'Inline-CUDA-* _Inline demos/_Inline config/Inline-CUDA.conf Makefile.old Makefile.bak' },
	postamble => { BENCHMARK_FILES => 'xt/benchmarks/*.b' }
);

# there are some static subs in Inline::CUDA::Utils.pm
# we need, so I came up with this ...
require File::Spec->catdir($FindBin::Bin, 'lib', 'Inline', 'CUDA', 'Utils.pm');

# install all files in config dir (they will be created below)
# this must be called before WriteMakefile()
my $config_dir = File::Spec->catdir($FindBin::Bin, 'config');
if( ! -d $config_dir ){
	mkdir $config_dir;
	die "failed to find or create a config dir at '$config_dir'." unless -d $config_dir;
	print STDOUT "$0 : created config dir '$config_dir'.\n";
}
File::ShareDir::Install::install_share dist => $config_dir;

my $C_dir = File::Spec->catdir($FindBin::Bin, 'C');
File::ShareDir::Install::install_share dist => $C_dir;

my $dependencies = Inline::CUDA::Utils::find_dependencies();
if( ! defined $dependencies ){ print STDERR "$0 : error, call to Inline::CUDA::Utils::find_dependencies() has failed.\n"; exit(1) }
print "find_dependencies() : report:\n";
print "  $_ : ".$dependencies->{$_}."\n" for (sort keys %$dependencies);

my $dist_name = $WriteMakefileArgs{NAME}; $dist_name =~ s/::/-/g;
my $conf_file_base = $dist_name . '.conf';
my $conf_file = File::Spec->catdir($config_dir, $conf_file_base);
my $conf_file_mods_base = $conf_file_base . '.mods';
my $conf_file_mods = File::Spec->catdir($FindBin::Bin, 'config', $conf_file_mods_base);

# check if there is a config/Inline-CUDA.conf.mods which will contain
# user-modifications
# to the dependencies we found and will overwrite any config params with the amended ones.
my $mods_conf;
if( -f $conf_file_mods ){
	print STDOUT "$0 : found and loading user-modified config file: '$conf_file_mods' ...\n";
	$mods_conf = Inline::CUDA::Utils::read_configuration($conf_file_mods);
	if( ! defined $mods_conf ){ print STDERR "$0 : error, call to Inline::CUDA::Utils::read_configuration() has failed for user-modified config file '$conf_file_mods', check that it is valid and looks like '$conf_file'.\n"; exit(1); }
	# ok, so we now have a hash with mods to overwrite the standard config
	for my $k (keys %$dependencies){
		if( exists($mods_conf->{$k}) && defined($mods_conf->{$k}) ){
			# overwrite
			print STDOUT "$0 : overwriting configuration key '$k' (='".$dependencies->{$k}."') with user-modified value '".$mods_conf->{$k}."'\n";
			$dependencies->{$k} = $mods_conf->{$k};
		}
	}
} else { print STDOUT "$0 : no user-modified config file found (searched for '$conf_file_mods'), that's OK. (Note: use this file to include modifications to configuration, another way to achieve this is via Environment variables set during this installation procedure).\n"; }

if( Inline::CUDA::Utils::save_configuration($dependencies, $conf_file) ){ print STDERR "$0 : error, call to Inline::CUDA::Utils::save_configuration() has failed to save into '$conf_file'.\n"; exit(1) }
print STDOUT perl2dump($dependencies)."\n\n$0 : configuration (see above) written to '$conf_file'.\n";

my $newconf = Inline::CUDA::Utils::read_configuration($conf_file);
if( ! defined $newconf ){ print STDERR "$0 : error, call to Inline::CUDA::Utils::read_configuration() has failed for configuration file '$conf_file'. Can not verify that the configuration file was written correctly.\n"; exit(1) }

# Now, check configuration by compiling a test cuda program
# I am not sure if this should be here as
# there exists a test file in t/002-compilers.t
# which does that, but hey not everyone runs tests during installation
# so, I will stick that test here too:
print STDOUT "testing C and CUDA compilers by trying them on a small script ...\n";
if(  Inline::CUDA::Utils::test_compilers(File::Spec->catdir($FindBin::Bin, 'C', 'C'), $newconf) ){
	print STDERR "begin configuration:\n".perl2dump($newconf)."\nend configuration.\n$0 : error, call to Inline::CUDA::Utils::test_compilers() has failed using above configuration. Assuming all executables dependencies (e.g. nvcc via Cuda SDK and host compiler, e.g. gcc) were found correctly, then the most likely reason for this failure is that host compiler version is incompatible with current Cuda SDK and installed GPU version (assuming a GPU does exist!). It is likely that Inline::CUDA installation procedure found *a* set of host compilers in PATH and did not complain, but their version is not compatible. So, if you have ANOTHER set of host-compiler tools, e.g. a specific version of gcc which you installed in a non-PATH location, then specify those paths using environment variables (this is the easiest way, but there are others) during installation, e.g.: 'CC=/a/b/c CXX=/x/y/z/ LD=/1/2/3 perl Makefile.PL' .  Also, make sure that LD is pointing to a C++ compiler, e.g. that specified in CXX, unless you know what you are doing. For example this works for CUDA SDK v10.2 which requires a GCC maximum version of 8.2 (and assuming you installed GCC 8.2 at /usr/local/gcc82):\n". <<'EOP';
      CC=/usr/local/gcc82/bin/gcc82 \
      CXX=/usr/local/gcc82/bin/g++82 \
      LD=/usr/local/gcc82/bin/g++82 \
      perl Makefile.PL && make all && make test
EOP
	exit(1);
}
# Compatibility with old versions of ExtUtils::MakeMaker
unless (eval { ExtUtils::MakeMaker->VERSION('6.64'); 1 }) {
    my $test_requires = delete $WriteMakefileArgs{TEST_REQUIRES} || {};
    @{$WriteMakefileArgs{PREREQ_PM}}{keys %$test_requires} = values %$test_requires;
}

unless (eval { ExtUtils::MakeMaker->VERSION('6.55_03'); 1 }) {
    my $build_requires = delete $WriteMakefileArgs{BUILD_REQUIRES} || {};
    @{$WriteMakefileArgs{PREREQ_PM}}{keys %$build_requires} = values %$build_requires;
}

delete $WriteMakefileArgs{CONFIGURE_REQUIRES}
    unless eval { ExtUtils::MakeMaker->VERSION('6.52'); 1 };
delete $WriteMakefileArgs{MIN_PERL_VERSION}
    unless eval { ExtUtils::MakeMaker->VERSION('6.48'); 1 };
delete $WriteMakefileArgs{LICENSE}
    unless eval { ExtUtils::MakeMaker->VERSION('6.31'); 1 };


WriteMakefile(%WriteMakefileArgs);

print STDOUT "\n=============================\nFinished!\nDetected configuration:\n".perl2dump($dependencies, {terse=>1})."\nAbove configuration of Inline::CUDA *seems* to be working OK and will be written to '$conf_file' and then in an installation-wide dir for wide access.\nWhat you need next is to\n  make all\nAnd then, in order to be absolutely sure, about successful initialisation please run the tests, by e.g.\n  make test\nFinally, you can do a\n  make install\nwhich will install this module including the configuration file just created.\nThere are also some demo scripts you can run in demos/\n";

{
package MY;
use File::ShareDir::Install;

sub libscan {
    my( $mm, $file ) = @_;
    return if $file =~ /(push_to_GIT)|(\.lock)/;
    #warn "@_\n";
    return $file;
}

sub postamble {
	my ($self, %h) = @_;
	my @ret = File::ShareDir::Install::postamble( $self );
	# ... add more things to @ret;
	my $retstr = join "\n", @ret;
	# WARNING: USE REAL TABS in the Here-doc below, it goes into Makefile!!!
	$retstr .=
	  "BENCHMARK_FILES=$h{BENCHMARK_FILES}\n"
	. <<'POSTAMBLE';
TEST_D = $(ABSPERLRUN) -MExtUtils::Command -e test_d --

demo :: demos/run-demo.sh
	demos/run-demo.sh

benchmark :: $(BENCHMARK_FILES)
	prove --blib $(INST_LIB) --blib $(INST_ARCHLIB) --verbose $^

bench2 :: $(BENCHMARK_FILES)
	$(TEST_D) xt && $(MAKE) test TEST_FILES='$(BENCHMARK_FILES)'

author-test :: all xt/*.t
	prove -Iblib/lib -vl xt/*.t

POSTAMBLE


}
}
1;
