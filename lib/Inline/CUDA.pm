package Inline::CUDA;

####
# NOTE: ILSM = Inline Support Language Module
####

use strict;
use warnings;

our $VERSION = '0.16';

use Inline 0.56;
use Config;
use Data::Roundtrip qw/perl2dump/;
use Carp;
use Cwd;
use File::Spec;
use File::Copy;
use File::ShareDir qw/dist_dir/;
use ShellQuote::Any;
use version;

use Inline::CUDA::Utils;

use constant IS_WIN32 => $^O eq 'MSWin32';
use if !IS_WIN32, Fcntl => ':flock';
use if IS_WIN32, 'Win32::Mutex';
 
our @ISA = qw(Inline);

# This module is heavily based on Inline::C it uses a lot of its
# subs verbatim. The names of these subs are stored below in:
#   @IMPORT_SUBS_FROM_INLINE_C
# These subs will be imported into this module.
# Whatever subs are not present in this array MUST BE implemented
# in this module.
# REALLY we would love to inherit from Inline::C but it is
# not made for this. The philosophy is that you inherit from Inline
# but Inline::C is so much more convenient for lazy me.
# So I am creating a new fresh module (this one) and copy subs
# (see the for-loop below) from Inline::C, then I implement those
# few that need to be overriden because they are CUDA-specific etc.

# the array of subs to import from Inline::C verbatim:
my @IMPORT_SUBS_FROM_INLINE_C = qw/add_list add_string add_text info config build call preprocess parse get_parser get_maps get_types ValidProtoString TrimWhitespace TidyType C_string write_XS xs_generate xs_includes xs_struct_macros xs_code xs_struct_code xs_boot xs_bindings write_Inline_headers write_Makefile_PL compile make_install cleanup system_call build_error_message fix_make quote_space _parser_test usage_validate/;

# import Inline::C subs (see above @IMPORT_SUBS_FROM_INLINE_C):
require Inline::C; 
for my $asub (@IMPORT_SUBS_FROM_INLINE_C){
	my $src = "Inline::C::$asub";
	my $dst = "Inline::CUDA::$asub";
	my $subref = \&{ $src };
	no strict 'refs';
	*{ $dst } = $subref;
}

# these are compile options (found in Makefile or user-specified in CCFLAGS etc.)
# which must be replaced as per the matrix below,
# '' means replace with nothing, i.e. remove them
my @compile_options_to_be_modified = (
	#[from, to]
#	[qr/\-Werror=format\-security(?=\s|$)/,''],
#	[qr/\-m[0-9]+(?=\s|$)/,''],
#	[qr/\-mtune=.+?(?=\s|$)/,''],
#	[qr/\-grecord\-gcc\-switches(?=\s|$)/,''],
#	[qr/\-pipe(?=\s|$)/,''],
	[qr/\-Wp,\-U_FORTIFY_SOURCE,-D_FORTIFY_SOURCE=[0-9]+(?=\s|$)/,''],
	[qr/\-D_FORTIFY_SOURCE=[0-9]+(?=\s|$)/,''],
#	[qr/\-Wp,\-D_GLIBCXX_ASSERTIONS(?=\s|$)/,''],
#	[qr/\-DVERSION=[^ ]+(?=\s|$)/,''],
#	[qr/\-DXS_VERSION=[^ ]+(?=\s|$)/,''],
	# let's hope there are no spaces in the spec:
	[qr/\-specs=.+?(?=\n|$)/,''],
	[qr/\-iquote/,'-I'],
	[qr/\-Wl,\-\-as-needed/,''],
	[qr/\-Wl,\-z,relro/,''],
	[qr/\-Wl,\-z,now/,''],

	# this exists here for testing purposes, no harm, no remove
	[qr/\s*\\"\-test\-bliako\-A/,''],
);
# these need to be prepended with '-Xcompiler' which nvcc understands
# to mean to pass to the host compiler (e.g. gcc)
my @compile_options_to_be_passed_to_the_host_compiler = (
#	qr/(\-flto=auto)(?=\s|$)/,
#	qr/(\-ffat\-lto\-objects)(?=\s|$)/,
#	qr/(\-fexceptions)(?=\s|$)/,
#	qr/(\-fstack\-protector\-strong)(?=\s|$)/,
#	qr/(\-fasynchronous\-unwind\-tables)(?=\s|$)/,
#	qr/(\-fstack\-clash\-protection)(?=\s|$)/,
#	qr/(\-fcf\-protection)(?=\s|$)/,
#	qr/(\-fwrapv)(?=\s|$)/,
#	qr/(\-fPIC)(?=\s|$)/,
#	qr/(\-fno\-strict\-aliasing)(?=\s|$)/,
#	qr/(\-Wall)(?=\s|$)/,
);
# these need to be prepended with '-Xlinker' which nvcc understands
# to mean to pass to the host linker (e.g. gcc)
my @compile_options_to_be_passed_to_the_host_linker = (
	# this exists here for testing purposes, no harm, no remove
	qr/(\-test\-bliako\-C)(?=\s|$)/,
);
# and these should not be touched, just pass them as they are to nvcc
my %compile_options_not_to_be_modified = (
	qr/\-I[\s"']*/ => 1,
	qr/\-iquote[\s"']*/ => 1,
);
# and these should be duplicated and sent to both compiler and linker
my %compile_options_to_be_duplicated = (
	'-shared' => 1,
);

sub _do_compile_options_to_be_modified {
	my $o = $_[0];
	my $to_be_modified_ref = $_[1];

        my $parent = (caller(1))[3]; if( ! defined($parent) ){ $parent = 'N/A' }
        my $whoami = ( caller(0) )[3];

	my ($m, $k, $K);
	# perform replacements from the @compile_options_to_be_modified array
	for $k (@compile_options_to_be_modified){
		my ($w1,$w2) = @$k;
		if( $$to_be_modified_ref =~ s/$w1/$w2/g ){ print STDERR "_do_compile_options_to_be_modified(): modifying compile-option '$w1' with '$w2'.\n" if $o->{CONFIG}{BUILD_NOISY} }
		else { print STDERR "_do_compile_options_to_be_modified(): no match for '$w1', no problem.\n" }
	}

	my $mpl = $o->{ILSM}->{MAKEFILE_PL};
	for $k (@Inline::CUDA::Utils::MAKEFILE_PL_FLAGS_KEYS_COMPILER){
		$K = 'hash-'.$k; # we also have a hash whose keys are each directive: -Wall etc.
		if( ! exists($mpl->{$K}) || ! defined($m=$mpl->{$K}) ){ print STDERR "_do_compile_options_to_be_modified() : error, there is no key '$K' under ".'$o->{ILSM}->{MAKEFILE_PL}'.".\n"; return 1 }
		AFLAGL:
		for my $aflag (keys %{$mpl->{$K}}){
			for my $nbm (keys %compile_options_not_to_be_modified){
				next unless 1 == $compile_options_not_to_be_modified{$nbm};
				next AFLAGL if $aflag =~ /$nbm/;
			}
			_send_flag_to_compiler(
				$to_be_modified_ref,
				$aflag,
				exists($compile_options_to_be_duplicated{$aflag}) ? $compile_options_to_be_duplicated{$aflag} : 0
			);
		}
	}
	for my $k (@Inline::CUDA::Utils::MAKEFILE_PL_FLAGS_KEYS_LINKER){
		$K = 'hash-'.$k; # we also have a hash whose keys are each directive: -Wall etc.
		if( ! exists($mpl->{$K}) || ! defined($m=$mpl->{$K}) ){ print STDERR "_do_compile_options_to_be_modified() : error, there is no key '$K' under ".'$o->{ILSM}->{MAKEFILE_PL}'.".\n"; return 1 }
		for my $aflag (keys %{$mpl->{$K}}){
			next if exists $compile_options_not_to_be_modified{$aflag};
			_send_flag_to_linker(
				$to_be_modified_ref,
				$aflag,
				exists($compile_options_to_be_duplicated{$aflag}) ? $compile_options_to_be_duplicated{$aflag} : 0
			);
		}
	}

	return 0; # success
}

#sub _send_flag_to_compiler { ${$_[0]} =~ s/(?:^|\s+)($_[1])(?=\s|$)/$_[2]==1 ? " $1 -Xcompiler \\\"$1\\\" ": " -Xcompiler \\\"$1\\\" "/ge }
#sub _send_flag_to_linker { ${$_[0]} =~ s/(?:^|\s+)($_[1])(?=\s|$)/$_[2]==1 ? " $1 -Xlinker \\\"$1\\\" ": " -Xlinker \\\"$1\\\" "/ge }

sub _send_flag_to_compiler { ${$_[0]} =~ s/(?:^|\s+)($_[1])(?=\s|$)/$_[2]==1 ? " $1 -Xcompiler ".ShellQuote::Any::shell_quote(['"'.$1.'"'])." ": " -Xcompiler ".ShellQuote::Any::shell_quote(['"'.$1.'"'])." "/ge }
sub _send_flag_to_linker { ${$_[0]} =~ s/(?:^|\s+)($_[1])(?=\s|$)/$_[2]==1 ? " $1 -Xlinker ".ShellQuote::Any::shell_quote(['"'.$1.'"'])." ": " -Xlinker ".ShellQuote::Any::shell_quote(['"'.$1.'"'])." "/ge }

# Our sub to modify the Makefile in the _Inline/build/* dir
# remove options which gcc does not like and prepend -Xcompiler to others
# that nvcc does not understand (and so it passes them on to gcc).
# Input is a hashref of params. 'o' is our Inline::CUDA object
# 'Makefile_base' is the name of the Makefile (optional, default is 'Makefile')
# Ideally it should be called within make(), before the 'make -f Makefile' is shelled-out
# It returns 1 on failure and 0 on success.
sub _fix_make_cuda {
	my $params = $_[0];
	my $o = $params->{'o'};
	my $build_dir = $o->{'API'}->{'build_dir'};
	my $module = $o->{'API'}->{'module'}; # full path to the module name, append a .cu to get source-code filename
	my $makefile_base = (exists($params->{'Makefile_base'}) && defined($params->{'Makefile_base'}))
		? $params->{'Makefile_base'} : 'Makefile'
	;
	my $makefile = File::Spec->catdir($build_dir, $makefile_base);

	print STDERR "_fix_make_cuda() : called for '$makefile'...\n" if $o->{CONFIG}{BUILD_NOISY};

	my $fh;
	if( ! open($fh, '<', $makefile) ){ print STDERR "_fix_make_cuda() : error, failed to open '$makefile' for reading, $!"; return 1; }
	my $contents; { local $/ = undef; $contents = <$fh> } close($fh);

	# replace all newline escape (\<newline>) from the Makefile with a single space. It's much easier for our regexes
	$contents =~ s/\\\n/ /gm;

	# now we have all the Makefile flags in $o->{ILSM}->{MAKEFILE_PL}
	# e.g. in $o->{ILSM}->{MAKEFILE_PL}->{CCFLAGS}, etc.
	# (see all the keys in @Inline::CUDA::Utils::MAKEFILE_PL_FLAGS_KEYS)

	# replace each of the cflags with '-Xcompiler \"...\"' and ditto for -Xlinker
	# this is dirty
	_do_compile_options_to_be_modified($o, \$contents);
	# find the C_FILES
	$contents =~ s/^(C_FILES\s*=\s*)(.+?)$/$1##__rep1__##\n/m; my $c_files = $2;
	if( ! defined $c_files ){ print STDERR "_fix_make_cuda() : error, Makefile '$makefile' most  likely has an error because 'C_FILES' is empty.\n"; return 1 }

	# replace the extension (.c) of each in the C_FILES with .cu
	my @c_files_substitutions = map {
		my $x = $_; $x =~ s/\.c/.cu/;
		[$_ => $x]
	} split(/\s+/, $c_files);

	my @cu_files = map { $_->[1] } @c_files_substitutions;
	$contents =~ s/##__rep1__##/@{cu_files}/s;
	# and now C_FILES = a.cu
	for (@c_files_substitutions){
		my ($from, $to) = @$_;
		$contents =~ s/\b${from}\b/${to}/g;
	}

	# add .cu extension to .SUFFIXES
	$contents =~ s/\.SUFFIXES\s*:\s*/.SUFFIXES : .cu /gs;

	# change the .xs.c: rule to .xs.cu with relevant .c->.cu changes
	# assumes that the rule ends in double-newline!
	$contents =~ s/^(\.xs\.c.+?)\n\n/##__rep1__##\n\n/ms;
	my $xscurule = $1;
	$xscurule =~ s/\.c/.cu/gs;
	$contents =~ s/##__rep1__##/${xscurule}/s;

	# change the '.xs$(OBJ_EXT) :' rule .c -> .cu
	$contents =~ s/^(.xs\$\(OBJ_EXT\)\s*:\s*)(.+?)\n\n/$1##__rep1__##\n\n/ms;
	$xscurule = $2;
	$xscurule =~ s/\.c\b/.cu/gs;
	$contents =~ s/##__rep1__##/${xscurule}/s;

	# rename original Makefile to Makefile.ori
	File::Copy::move($makefile, $makefile.'.ori');

	# write makefile out
	if( ! open($fh, '>', $makefile) ){ print STDERR "_fix_make_cuda() : error, failed to open '$makefile' for writing, $!"; return 1; }
	print $fh $contents;
	close($fh);

	return 0; # success
}

############
# Below are the subs which we do not inject from Inline::C
# they have been copied from Inline::C v0.86 and modified with indication BLIAKO
# If you add or remove any from below you must remove/add it from the my @IMPORT_SUBS_FROM_INLINE_C
############

# This is originally implemented in Inline::C but we need to add some
# modifications to it for CUDA. So we copy this VERBATIM from Inline::C
# and insert our additions. This is a headache!!! Each time Inline::C is
# updated and in case this sub changes, you need to patch it here.
# Our additions are enclosed in # BLIAKO: addition tags.
# TODO: I guess using patch is the way to go but ...
# Purpuse: Run Makefile.PL
sub makefile_pl {
    my ($o) = @_;
    my $perl;
    -f ($perl = $Config::Config{perlpath})
        or ($perl = $^X)
        or croak "Can't locate your perl binary";
    $perl = qq{"$perl"} if $perl =~ m/\s/;

# BLIAKO: addition
	my $build_dir = $o->{API}->{build_dir};
	my $flags = Inline::CUDA::Utils::enquire_Makefile_PL(
		$build_dir,
		'Makefile.PL',
		$o->{CONFIG}->{BUILD_NOISY},
		$o->{CONFIG}{CLEAN_AFTER_BUILD}
	);
	die "call to Inline::CUDA::Utils::enquire_Makefile_PL() has failed." unless defined $flags;
	# creating an entry '{ILSM}->{MAKEFILE_PL}', will it break things? 
	$o->{ILSM}->{MAKEFILE_PL} = {%$flags};
	# now we have all the Makefile flags without parsing it (yikes!)
	# e.g. in $o->{ILSM}->{MAKEFILE_PL}->{CCFLAGS}, etc.
	# (see all the keys in @Inline::CUDA::Utils::MAKEFILE_PL_FLAGS_KEYS)
# BLIAKO: end addition

    $o->system_call("$perl Makefile.PL", 'out.Makefile_PL');
    $o->fix_make;
}

## make with the makefile
## this is copied verbatim from Inline::C v0.86 and modify some part.
## NOTE: Unfortunately it does not return anything, instead it dies/croaks
## on fail/warnings.
## So you need to try/catch it or anything that calls it.
sub make {
    my ($o) = @_;
    my $make = $o->{ILSM}{MAKE} || $Config::Config{make}
        or croak "Can't locate your make binary";
    local $ENV{MAKEFLAGS} = $ENV{MAKEFLAGS} =~ s/(--jobserver-fds=[\d,]+)//
        if $ENV{MAKEFLAGS};
    # BLIAKO: add the following
    # the Makefile is now in _Inline/build/.../
    # this sub reads and modifies its contents and then writes the new Makefile out, ready to be 'made'
    if( _fix_make_cuda({
	'o' => $o,
	'Makefile_base' => 'Makefile',
    }) ){ die "call to _fix_make_cuda() has failed."; }
    # BLIAKO: end addition

    $o->system_call("$make", 'out.make');
}

#==============================================================================
# Register this module as an Inline language support module
#==============================================================================
sub register {
    return {
        language => 'CUDA',
        aliases => ['CUDA','Cuda','cuda'],
        type => 'compiled',
        suffix => '.cu',
    };
}

#==============================================================================
# Validate the CUDA config options
#==============================================================================
## this is copied verbatim from Inline::C v0.86 and modify some part.
## you need to change this when Inline::C changes and then add BLIAKO's changes to it
sub validate {
    my $o = shift;

    print STDERR "validate Stage\n" if $o->{CONFIG}{BUILD_NOISY};

# BLIAKO modification: find the module dist dir, from which we get the configuration file
    my $dist_name = __PACKAGE__;
    $dist_name =~ s/::/-/g; # silly!
    my $module_dist_dir = File::ShareDir::dist_dir($dist_name);
    if( ! defined $module_dist_dir ){ die "error, call to ".'File::ShareDir::dist_dir()'." has failed for dist name '$dist_name', can not find where install dir is."; }
    my $system_configuration_file = Inline::CUDA::Utils::path_of_configuration_file_in_shared_dir($dist_name);
    if( ! defined $system_configuration_file ){ die "error, call to ".'Inline::CUDA::Utils::path_of_configuration_file_in_shared_dir()'." has failed for module dist dir '$module_dist_dir', can not find where the system-wide configuration file is."; }
    # if CONFIGURATION_FILE is specified in Config section, we set this and read it
    my $userspecified_configuration_file;
# BLIAKO end modification

    $o->{ILSM} ||= {};
    $o->{ILSM}{XS} ||= {};
    $o->{ILSM}{MAKEFILE} ||= {};
    if (not $o->UNTAINT) {
        require FindBin;
        if (not defined $o->{ILSM}{MAKEFILE}{INC}) {
            # detect Microsoft Windows OS, and either Microsoft Visual Studio compiler "cl.exe", "clarm.exe", or Intel C compiler "icl.exe"
            if (($Config{osname} eq 'MSWin32') and ($Config{cc} =~ /\b(cl\b|clarm|icl)/)) {
                warn "\n   Any header files specified relative to\n",
                     "   $FindBin::Bin\n",
                     "   will be included only if no file of the same relative path and\n",
                     "   name is found elsewhere in the search locations (including those\n",
                     "   specified in \$ENV{INCLUDE}).\n",
                     "   Otherwise, that header file \"found elsewhere\" will be included.\n";
                warn "  ";    # provide filename and line number.
                $ENV{INCLUDE} .= qq{;"$FindBin::Bin"};
            }
            # detect Oracle Solaris/SunOS OS, and Oracle Developer Studio compiler "cc" (and double check it is not GCC)
            elsif ((($Config{osname} eq 'solaris') or ($Config{osname} eq 'sunos')) and ($Config{cc} eq 'cc') and (not $Config{gccversion})) {
		# angle-bracket includes will NOT incorrectly search -I dirs given before -I-
                $o->{ILSM}{MAKEFILE}{INC} = " -I\"$FindBin::Bin\" ";
                warn q{NOTE: Oracle compiler detected, unable to utilize '-iquote' compiler option, falling back to '-I-' which should produce correct results for files included in angle brackets}, "\n";
            }
            else {
		# angle-bracket includes will NOT correctly search -iquote dirs
                $o->{ILSM}{MAKEFILE}{INC} .= " ".qq{-iquote"$FindBin::Bin"};
            }
        }
    }

    $o->{ILSM}{AUTOWRAP} = 0 if not defined $o->{ILSM}{AUTOWRAP};
    $o->{ILSM}{XSMODE} = 0 if not defined $o->{ILSM}{XSMODE};
    $o->{ILSM}{AUTO_INCLUDE} ||= <<END;
#include "EXTERN.h"
#include "perl.h"
#include "XSUB.h"
#include "INLINE.h"
END
    $o->{ILSM}{FILTERS} ||= [];
    $o->{STRUCT} ||= {
        '.macros' => '',
        '.xs' => '',
        '.any' => 0,
        '.all' => 0,
    };

    while (@_) {
        my ($key, $value) = (shift, shift);
        if ($key eq 'PRE_HEAD') {
            unless( -f $value) {
                $o->{ILSM}{AUTO_INCLUDE} = $value . "\n" .
                $o->{ILSM}{AUTO_INCLUDE};
            }
            else {
                my $insert;
                open RD, '<', $value
                    or die "Couldn't open $value for reading: $!";
                while (<RD>) {$insert .= $_}
                close RD
                    or die "Couldn't close $value after reading: $!";
                $o->{ILSM}{AUTO_INCLUDE} =
                    $insert . "\n" . $o->{ILSM}{AUTO_INCLUDE};
            }
            next;
        }
        if ($key eq 'MAKE' or
            $key eq 'AUTOWRAP' or
            $key eq 'XSMODE'
        ) {
            $o->{ILSM}{$key} = $value;
            next;
        }

       if ($key eq 'CC' or
            $key eq 'LD' or
	    # BLIAKO: addition, NVCC config key has been added:
	    $key eq 'NVCC' or
	    $key eq 'CXX' or
	    # BLIAKO: addition, NVLD config key has been added
	    $key eq 'NVLD'
        ) {
            $o->{ILSM}{MAKEFILE}{$key} = $value;
            next;
        }
        if ($key eq 'LIBS') {
            $o->add_list($o->{ILSM}{MAKEFILE}, $key, $value, []);
            next;
        }
        if ($key eq 'INC') {
            $o->add_string(
                $o->{ILSM}{MAKEFILE},
                $key,
                quote_space($value),
                '',
            );
            next;
        }
        if ($key eq 'MYEXTLIB' or
            $key eq 'OPTIMIZE' or
            $key eq 'CCFLAGS' or
            $key eq 'LDDLFLAGS'
# BLIAKO: addition, NVCCFLAGS config key has been added:
	 or $key eq 'NVCCFLAGS'
	 or $key eq 'NVLDFLAGS'
# BLIAKO: end addition
        ) {
            $o->add_string($o->{ILSM}{MAKEFILE}, $key, $value, '');
            next;
        }
# BLIAKO: addition to pass back the Inline object, $value is expected to be a ref to place in there our object $o
	if( $key eq 'PASS_ME_O') {
		if( (ref($value)eq'SCALAR') && (ref($$value)eq'') ){
			$$value = $o;
			next
		} else { croak "value is not a reference to scalar while processing option 'PASS_ME_O'." }
	}
# BLIAKO: end addition
# BLIAKO: addition to set underlying language of host-code to C or C++=CPP (case insensitive)
	if( $key eq 'HOST_CODE_LANGUAGE') {
		croak "Invalid value for 'HOST_CODE_LANGUAGE' option '$value', it must be 'C' or 'C++' or 'CPP', case-insensitive."
		  unless $value =~ /^C|C\+\+|CPP$/i;
		$o->{ILSM}{MAKEFILE}{$key} = $value;
		next;
	}
# BLIAKO: end addition
# BLIAKO: addition to read user-specified configuration file
	if( $key eq 'CONFIGURATION_FILE') {
		$userspecified_configuration_file = $value;
		next;
	}
# BLIAKO: end addition
# BLIAKO: addition (hey why not a 'ldflagsex' as well???? Let's do this)
        if ($key eq 'LDDLFLAGSEX') {
            $o->add_string(
                $o->{ILSM}{MAKEFILE},
                'LDDLFLAGS',
                $Config{lddlflags} . ' ' . $value, '',
            );
            next;
        }
        if ($key eq 'LDFLAGSEX') {
            $o->add_string(
                $o->{ILSM}{MAKEFILE},
                'LDFLAGS',
                $Config{ldflags} . ' ' . $value, '',
            );
            next;
        }
# BLIAKO: end addition
        if ($key eq 'CCFLAGSEX') {
            $o->add_string(
                $o->{ILSM}{MAKEFILE},
                'CCFLAGS',
                $Config{ccflags} . ' ' . $value, '',
            );
            next;
        }
        if ($key eq 'TYPEMAPS') {
            unless(ref($value) eq 'ARRAY') {
                croak "TYPEMAPS file '$value' not found"
                    unless -f $value;
                $value = File::Spec->rel2abs($value);
            }
            else {
                for (my $i = 0; $i < scalar(@$value); $i++) {
                    croak "TYPEMAPS file '${$value}[$i]' not found"
                        unless -f ${$value}[$i];
                    ${$value}[$i] = File::Spec->rel2abs(${$value}[$i]);
                }
            }
            $o->add_list($o->{ILSM}{MAKEFILE}, $key, $value, []);
            next;
        }
        if ($key eq 'AUTO_INCLUDE') {
            $o->add_text($o->{ILSM}, $key, $value, '');
            next;
        }
        if ($key eq 'BOOT') {
            $o->add_text($o->{ILSM}{XS}, $key, $value, '');
            next;
        }
        if ($key eq 'PREFIX') {
            croak "Invalid value for 'PREFIX' option"
              unless ($value =~ /^\w*$/ and
                      $value !~ /\n/);
            $o->{ILSM}{XS}{PREFIX} = $value;
            next;
        }
        if ($key eq 'FILTERS') {
            next if $value eq '1' or $value eq '0'; # ignore ENABLE, DISABLE
            $value = [$value] unless ref($value) eq 'ARRAY';
            my %filters;
            for my $val (@$value) {
                if (ref($val) eq 'CODE') {
                    $o->add_list($o->{ILSM}, $key, $val, []);
                }
                elsif (ref($val) eq 'ARRAY') {
                    my ($filter_plugin, @args) = @$val;
 
                    croak "Bad format for filter plugin name: '$filter_plugin'"
                        unless $filter_plugin =~ m/^[\w:]+$/;
 
                    eval "require Inline::Filters::${filter_plugin}";
                    croak "Filter plugin Inline::Filters::$filter_plugin not installed"
                        if $@;
 
                    croak "No Inline::Filters::${filter_plugin}::filter sub found"
                        unless defined &{"Inline::Filters::${filter_plugin}::filter"};
 
                    my $filter_factory = \&{"Inline::Filters::${filter_plugin}::filter"};
 
                    $o->add_list($o->{ILSM}, $key, $filter_factory->(@args), []);
                }
                else {
                    eval { require Inline::Filters };
                    croak "'FILTERS' option requires Inline::Filters to be installed."
                        if $@;
                    %filters = Inline::Filters::get_filters($o->{API}{language})
                        unless keys %filters;
                    if (defined $filters{$val}) {
                        my $filter = Inline::Filters->new(
                            $val, $filters{$val}
			);
                        $o->add_list($o->{ILSM}, $key, $filter, []);
                    }
                    else {
                        croak "Invalid filter $val specified.";
                    }
                }
            }
            next;
        }
        if ($key eq 'STRUCTS') {
            # A list of struct names
            if (ref($value) eq 'ARRAY') {
                for my $val (@$value) {
                    croak "Invalid value for 'STRUCTS' option"
                        unless ($val =~ /^[_a-z][_0-9a-z]*$/i);
                    $o->{STRUCT}{$val}++;
                }
            }
            # Enable or disable
            elsif ($value =~ /^\d+$/) {
                $o->{STRUCT}{'.any'} = $value;
            }
            # A single struct name
            else {
                croak "Invalid value for 'STRUCTS' option"
                    unless ($value =~ /^[_a-z][_0-9a-z]*$/i);
                $o->{STRUCT}{$value}++;
            }
            eval { require Inline::Struct };
            croak "'STRUCTS' option requires Inline::Struct to be installed."
                if $@;
            $o->{STRUCT}{'.any'} = 1;
            next;
        }
        if ($key eq 'PROTOTYPES') {
            $o->{CONFIG}{PROTOTYPES} = $value;
            next if $value eq 'ENABLE';
            next if $value eq 'DISABLE';
            die "PROTOTYPES can be only either 'ENABLE' or 'DISABLE' - not $value";
        }
        if ($key eq 'PROTOTYPE') {
            die "PROTOTYPE configure arg must specify a hash reference"
                unless ref($value) eq 'HASH';
            $o->{CONFIG}{PROTOTYPE} = $value;
            next;
        }
        if ($key eq 'CPPFLAGS') {
            # C preprocessor flags, used by Inline::Filters::Preprocess()
            next;
        }
 
        my $class = ref $o; # handles subclasses correctly.
        croak "'$key' is not a valid config option for $class\n";
    }

   # BLIAKO: modification: add --compiler-bindir <gcc> to nvcc
   # care should be taken for spaces and quotes!

    my $userconf;
    if( defined($userspecified_configuration_file)
     && ($userspecified_configuration_file!~/^\s*$/)
    ){
        if( ! -f $userspecified_configuration_file ){ die "user-specified file '".$userspecified_configuration_file."' does not exist.\n"; }
	print STDOUT "validate() : reading user-specified configuration file '".$userspecified_configuration_file."' ...\n" if $o->{CONFIG}{BUILD_NOISY};
	$userconf = Inline::CUDA::Utils::read_configuration($userspecified_configuration_file);
	if( ! defined $userconf ){ die "failed to read configuration from user-specified file '".$userspecified_configuration_file."'.\n"; }
	print STDOUT perl2dump($userconf)."\nvalidate() : read above configuration from user-specified file '".$userspecified_configuration_file."'.\n" if $o->{CONFIG}{BUILD_NOISY};
    }

    print STDOUT "validate() : reading system-wide configuration from file '".Inline::CUDA::Utils::path_of_configuration_file_in_shared_dir()."' ...\n";
    my $conf = Inline::CUDA::Utils::read_configuration_file_in_shared_dir();
    if( ! defined $conf ){ die "call to Inline::CUDA::Utils::read_configuration_file_in_shared_dir() has failed, configuration file is '".Inline::CUDA::Utils::path_of_configuration_file_in_shared_dir()."'." }

#    for my $ak (qw/nvcc nvlink cc cxx ld host_compiler_bindir/){
    for my $ak (qw/nvcc nvlink cc cxx ld host_compiler_bindir/){
    	my $AK = uc $ak;
	my $val;
    	if( ! exists($o->{ILSM}{MAKEFILE}{$AK}) || ! defined($o->{ILSM}{MAKEFILE}{$AK}) ){
    		if( defined($userconf)
		 && (
		      (exists($userconf->{$ak}) && defined($val=$userconf->{$ak}))
		   || (exists($userconf->{$AK}) && defined($val=$userconf->{$AK}))
		    )
		){ $o->{ILSM}{MAKEFILE}{$AK} = _quoteme($val); }
		elsif( (exists($conf->{$ak}) && defined($val=$conf->{$ak}))
		    || (exists($conf->{$AK}) && defined($val=$conf->{$AK}))
		){ $o->{ILSM}{MAKEFILE}{$AK} = _quoteme($val); }
		else { die "Inline=>CUDA=>Config=>$AK => '...' was not specified in source file and site configuration file (${system_configuration_file}) ".(defined($userconf)?"nor user-specified configuration (".$userconf.")":"")." does contain entry '$ak', can not continue" }
    	}
    	print STDERR "validate() : found config entry '${AK}' => '".$o->{ILSM}{MAKEFILE}{$AK}."'.\n" if $o->{CONFIG}{BUILD_NOISY};
    }
    # if they are not defined already set them to an empty string, it saves us a lot of checks later
    for my $ak (qw/ccflags nvccflags lddlflags nvldflags/){
    	my $AK = uc $ak;
    	if( ! exists($o->{ILSM}{MAKEFILE}{$AK}) || ! defined($o->{ILSM}{MAKEFILE}{$AK}) ){
		$o->{ILSM}{MAKEFILE}{$AK} = ''
	}
    }

    # set default code language to be C++
    # nvcc expects C++ code and will at random cases will resort to c++
    # even if code is C
    if( ! exists($o->{ILSM}{MAKEFILE}{HOST_CODE_LANGUAGE})
     || ! defined($o->{ILSM}{MAKEFILE}{HOST_CODE_LANGUAGE})
    ){ $o->{ILSM}{MAKEFILE}{HOST_CODE_LANGUAGE} = 'C++' }

    # AND occassionally it looks for g++ as default linker
    # a solution to all these is to just specify the dir to all the host compilers and tools
    my $host_compiler_cmd = ($o->{ILSM}{MAKEFILE}{HOST_CODE_LANGUAGE} eq 'C')
		? $o->{ILSM}{MAKEFILE}{CC} : $o->{ILSM}{MAKEFILE}{CXX}
    ;
    my $host_linker_cmd = ($o->{ILSM}{MAKEFILE}{HOST_CODE_LANGUAGE} eq 'C')
		? $o->{ILSM}{MAKEFILE}{CC} : $o->{ILSM}{MAKEFILE}{CXX}
    ;

    # we need to check what's the earliest nvcc version which supports -forward-unknown-to-host-compiler
    # 20.2 until we know this works...	
    if( version->parse($conf->{'nvcc-release'}) > version->parse('20.2') ){
        # nvcc versions > 10.2 forward all unknown flags to host compiler (e.g. gcc), otherwise
        # we need to append each compiler option (in @compile_options_to_be_passed_to_the_host_compiler) with -Xcompiler
        @compile_options_to_be_passed_to_the_host_compiler = ();
        @compile_options_to_be_passed_to_the_host_linker = ();
        # add both *forward* to linker and compiler
	$o->{ILSM}{MAKEFILE}{NVCCFLAGS} .= ' -forward-unknown-to-host-compiler -forward-unknown-to-host-linker';
	# it seems there is a problem with forwarding to linker, e.g. nvcc fatal   : Unknown option '-Wl,-z,relro'
	#$o->{ILSM}{MAKEFILE}{LDDLFLAGS} .= ' -forward-unknown-to-host-compiler -forward-unknown-to-host-linker';
	$o->{ILSM}{MAKEFILE}{LDDLFLAGS} .= ' -forward-unknown-to-host-compiler';
    } elsif( $o->{ILSM}{MAKEFILE}{NVCCFLAGS} =~ /(-forward-unknown-to-host-(?:compiler|linker))/ ){ print STDERR "validate() : error, ".$conf->{'nvcc'}." option '$1' is only supported for higher versions and not current (".$conf->{'nvcc-release'}.").\n"; croak "can not continue..."; }
      elsif( $o->{ILSM}{MAKEFILE}{NVLDFLAGS} =~ /(-forward-unknown-to-host-(?:compiler|linker))/ ){ print STDERR "validate() : error, ".$conf->{'nvlink'}." option '$1' is only supported for higher versions and not current (".$conf->{'nvcc-release'}.").\n"; croak "can not continue..."; }

    # when you fix this, then check commented code above
    if( $o->{ILSM}{MAKEFILE}{NVLDFLAGS} =~ /(-forward-unknown-to-host-linker)/ ){ print STDERR "validate() : error, ".$conf->{'nvlink'}." option '$1' should not be used until we figure out why it throws: nvcc fatal   : Unknown option '-Wl,-z,relro'.\n"; croak "can not continue..."; }

    # scan the CCFLAGS, LDFLAGS for illegal options or options that nvcc can not undertand 
    # and need to be forwarded to the host compiler (e.g. gcc)
    # and also perform replacements from the @compile_options_to_be_modified array
    # any options in the Makefile contents will be modified or removed
#    if( $o->{ILSM}{MAKEFILE}{CCFLAGS} !~ /^\s*$/ ){
#       _do_compile_options_to_be_modified($o, \$o->{ILSM}{MAKEFILE}{CCFLAGS});
#    }
#    if( $o->{ILSM}{MAKEFILE}{LDDLFLAGS} !~ /^\s*$/ ){
#        _do_compile_options_to_be_modified($o, \$o->{ILSM}{MAKEFILE}{LDDLFLAGS});
#    }

    $o->{ILSM}{MAKEFILE}{CC} =
	# NVCC must be quoted already (see above)	
	$o->{ILSM}{MAKEFILE}{NVCC}
        . ' --compiler-bindir '
	# this must already be quoted (see above)
        . $host_compiler_cmd
    ;
    if( exists($o->{ILSM}{MAKEFILE}{NVCCFLAGS}) && defined($o->{ILSM}{MAKEFILE}{NVCCFLAGS}) ){
        $o->{ILSM}{MAKEFILE}{CC} .= " ".$o->{ILSM}{MAKEFILE}{NVCCFLAGS};
    }
    #NVLINK: better use NVCC for linking and be done with this crap
    #$o->{ILSM}{MAKEFILE}{LD} = $o->{ILSM}{MAKEFILE}{NVLINK} . ' --compiler-bindir "'. $o->{ILSM}{MAKEFILE}{LD}.'"';
    $o->{ILSM}{MAKEFILE}{LD} =
	# NVCC must be quoted already (see above)
	$o->{ILSM}{MAKEFILE}{NVCC}
        . ' --compiler-bindir '
	  # this must already be quoted (see above)
        . $host_linker_cmd
	# this will avoid the error
	#    undefined symbol: __gxx_personality_v0
	# it must be last
	. ' -lstdc++'
    ;
    if( exists($o->{ILSM}{MAKEFILE}{NVLDFLAGS}) && defined($o->{ILSM}{MAKEFILE}{NVLDFLAGS}) ){
        $o->{ILSM}{MAKEFILE}{LD} .= " ".$o->{ILSM}{MAKEFILE}{NVLDFLAGS};
    }

# BLIAKO addition: find the C/inlinecuda dir in share-dir and add it to INC (c's inc that is)
    my $added_inc = File::Spec->catdir($module_dist_dir, 'C');
    if( ! -d $added_inc ){ die "error, module dist dir '$module_dist_dir' does not contain subdir 'C'."; }
    $added_inc = Cwd::abs_path($added_inc);
    if( ! defined $added_inc ){ die "error, call to ".'Cwd::abs_path()'." has failed for module dist dir."; }
    $o->{ILSM}{MAKEFILE}{INC} .= " ".qq{-iquote"${added_inc}"};
    print STDERR "validate() : appended '$added_inc' to includes and now is '".$o->{ILSM}{MAKEFILE}{INC}."'.\n";

    print STDERR "validate() : modified for host-code-language '".$o->{ILSM}{MAKEFILE}{HOST_CODE_LANGUAGE}."':\n  CC='".$o->{ILSM}{MAKEFILE}{CC}."'\n  LD='".$o->{ILSM}{MAKEFILE}{LD}."'\n" if $o->{CONFIG}{BUILD_NOISY};
    # delete keys unknown to ExtUtils::MakeMaker because
    # they will go into Makefile.PL (%options) and MM complains (warnings)
    delete $o->{ILSM}{MAKEFILE}{NVCC};
    delete $o->{ILSM}{MAKEFILE}{NVCCFLAGS};
    delete $o->{ILSM}{MAKEFILE}{NVLDFLAGS};
    delete $o->{ILSM}{MAKEFILE}{CXX};
    delete $o->{ILSM}{MAKEFILE}{HOST_COMPILER_BINDIR};
    delete $o->{ILSM}{MAKEFILE}{NVLINK};
    delete $o->{ILSM}{MAKEFILE}{HOST_CODE_LANGUAGE};
    print STDERR "validate() : here is the whole CONFIG:\n".perl2dump($o->{CONFIG})."\n" if $o->{CONFIG}{BUILD_NOISY};
# BLIAKO: end addition
}

# this quotes a string and exists here to make our life easier for checking different quotation schemes
sub _quoteme { return Inline::CUDA::Utils::quoteme($_[0]) }

##################
### ONLY POD BELOW:
##################
=pod

=encoding UTF-8

=head1 NAME

Inline::CUDA - Inline NVIDIA's CUDA code and GPU processing from within any Perl script.

=head1 VERSION

Version 0.16

=head1 SYNOPSIS

B<WARNING:> see section L</INSTALLATION> for how to install this package.

B<WARNING:> prior to installation, please install L<https://github.com/hadjiprocopis/perl-nvidia2-ml>

C<Inline::CUDA> is a module that allows you to write Perl subroutines in C
or C++ with CUDA extensions.

Similarly to L<Inline::C>,
C<Inline::CUDA> is not meant to be used directly but rather in this way:

	#Firstly, specify some configuration options:

	use Inline CUDA => Config =>
	# optionally specify some options,
	# you don't have to
	# if they are already stored in a configuration file
	# which is consulted before running any CUDA program
	#    host_compiler_bindir => '/usr/local/gcc82/bin',
	#    cc => '/usr/local/gcc82/bin/gcc82',
	#    cxx => '/usr/local/gcc82/bin/g++82',
	#    ld => '/usr/local/gcc82/bin/gcc82',
	#    nvcc => '/usr/local/cuda/bin/nvcc',
	#    nvld => '/usr/local/cuda/bin/nvcc',
	# pass options to nvcc:
	#  this is how to deal with unknown compiler flags passed on to nvcc: pass them all to gcc
	#  only supported in nvcc versions 11+
	#    nvccflags => '--forward-unknown-to-host-compiler',
	#  do not check compiler version, use whatever the user wants
	#    nvccflags => '--allow-unsupported-compiler',
	# this will use CC or CXX depending on the language specified here
	# you can use C++ in your CUDA code, and there are tests in t/*
	# which check if c or c++ and show how to do this:
	    host_code_language => 'c', # or 'c++' or 'cpp', case insensitive, see also cxx =>
	# optional extra Include and Lib dirs
	    #inc => '-I...',
	    #libs => '-L... -l...',
	# for debugging
  	    BUILD_NOISY => 1,
	# code will be left in ./_Inline/build/ after successful build
	    clean_after_build => 0,
	    warnings => 10,
	  ;

	# and then, suck in code from __DATA__ and run it at runtime
	# notice that Inline->use(CUDA => <<'EOCUDA') is run at compiletime
	my $codestr;
	{ local $/ = undef; $codestr = <DATA> }
	Inline->bind( CUDA => $codestr );

	if( do_add() ){ die "error running do_add()..." }

	1;
	__DATA__
	/* this is C code with CUDA extensions */
	#include <stdio.h>
	
	#define N 1000
	
	/* This is the CUDA Kernel which nvcc compiles: */
	__global__
	void add(int *a, int *b) {
		int i = blockIdx.x;
		if (i<N) b[i] = a[i]+b[i];
	}
	/* this function can be called from Perl.
	   It returns 0 on success or 1 on failure.
	   This simple code does not support passing parameters in,
	   which is covered elsewhere.
	*/
	int do_add() {
		cudaError_t err;
	
		// Create int arrays on the CPU.
		// ('h' stands for "host".)
		int ha[N], hb[N];
	
		// Create corresponding int arrays on the GPU.
		// ('d' stands for "device".)
		int *da, *db;
		if( (err=cudaMalloc((void **)&da, N*sizeof(int))) != cudaSuccess ){
			fprintf(stderr, "do_add(): error, call to cudaMalloc() has failed for %zu bytes for da: %s\n",
				N*sizeof(int), cudaGetErrorString(err)
			);
			return 1;
		}
		if( (err=cudaMalloc((void **)&db, N*sizeof(int))) != cudaSuccess ){
			fprintf(stderr, "do_add(): error, call to cudaMalloc() has failed for %zu bytes for db: %s\n",
				N*sizeof(int), cudaGetErrorString(err)
			);
			return 1;
		}
	
		// Initialise the input data on the CPU.
		for (int i = 0; i<N; ++i) ha[i] = i;
	
		// Copy input data to array on GPU.
		if( (err=cudaMemcpy(da, ha, N*sizeof(int), cudaMemcpyHostToDevice)) != cudaSuccess ){
			fprintf(stderr, "do_add(): error, call to cudaMemcpy(cudaMemcpyHostToDevice) has failed for %zu bytes for ha->da: %s\n",
				N*sizeof(int), cudaGetErrorString(err)
			);
			return 1;
		}
	
		// Launch GPU code with N threads, one per array element.
		add<<<N, 1>>>(da, db);
		if( (err=cudaGetLastError()) != cudaSuccess ){
			fprintf(stderr, "do_add(): error, failed to launch the kernel into the device: %s\n",
				cudaGetErrorString(err)
			);
			return 1;
		}
	
		// Copy output array from GPU back to CPU.
		if( (err=cudaMemcpy(hb, db, N*sizeof(int), cudaMemcpyDeviceToHost)) != cudaSuccess ){
			fprintf(stderr, "do_add(): error, call to cudaMemcpy(cudaMemcpyDeviceToHost) has failed for %zu bytes for db->ha: %s\n",
				N*sizeof(int), cudaGetErrorString(err)
			);
			return 1;
		}
	
		//for (int i = 0; i<N; ++i) printf("%d\n", hb[i]); // print results
	
		// Free up the arrays on the GPU.
		if( (err=cudaFree(da)) != cudaSuccess ){
			fprintf(stderr, "do_add(): error, call to cudaFree() has failed for da: %s\n",
				cudaGetErrorString(err)
			);
			return 1;
		}
		if( (err=cudaFree(db)) != cudaSuccess ){
			fprintf(stderr, "do_add(): error, call to cudaFree() has failed for db: %s\n",
				cudaGetErrorString(err)
			);
			return 1;
		}
	
		return 0;
	}

The statement: C<use Inline::CUDA =E<gt> ...;> is executed at
compile-time. Often this is not desirable because you may
want to read code from file, modify code at runtime or
even auto-generate the inlined code at runtime.
In these situations L<Inline> provides C<bind()>.

Here is how to inline code read at runtime from a file
called F<my_cruncher.cu>, whose contents are exactly the same
as the C<__DATA__> section in the previous example,

	use Inline;
	use File::Slurp;

	my $data = read_file('my_cruncher.cu');
	Inline->bind(CUDA => $data);
  
Using C<Inline-E<gt>use(CUDA =E<gt> "DATA")> seems to have a problem
when C<__DATA__> section contains identifiers enclosed
in double underscores, e.g. C<__global__> (this is a CUDA reserved keyword)
one workaround is to declare C<#define CUDA_GLOBAL __global__>
and then replace all C<__global__> with C<CUDA_GLOBAL>.

Sometimes, it is more convenient to configure L<Inline::CUDA>
not in a C<use> statement (as above) but in a C<require> statement.
The latter is executed during the runtime of your script as opposed
to loading the file during compile time for the former. This has
certain benefits as you can enclose it in a conditional,
eval or try/catch blocks. This is how
(thank you L<Corion@PerlMonks.org|https://perlmonks.org/?node_id=11159977>):

    require Inline;
    # configuration:
    Inline->import(
      CUDA => Config =>
        ccflagsex => '...'
    );
    # compile your code:
    Inline->import(
      CUDA => $my_code
    );

=head1 CUDA

The somewhat old news, at least since 2007, is that a Graphics Processing Unit (GPU)
has found uses beyond its traditional role in calculating and displaying graphics
to our computer monitor. This stems from the fact that a GPU is a highly parallel
computing machinery. Similar to the operating system sending data and instructions
to that GPU frame-after-frame from the time it is booted in order to display
windows, widgets, transparent menus, spinning animations, video games and
visual effects, a developer can now send data and instructions to the GPU
for doing any sort of arithmetic calculation in a highly parallel manner.
Case in point is matrix multiplication where thousands of GPU computing elements
are processing the matrices' elements in parallel. True parallelism, that is.
As opposed to the emulated or limited, by the number of cores, 2, 4, 8 for cheap desktops,
CPU's parallelism. It goes without saying that GPU processing is very powerful
and opens up to a new world of nunber-crunching possibilities without
the need for expensive super-computer capabilities.

NVIDIA's CUDA is "a parallel computing platform and programming
model that makes using a GPU for general purpose computing simple
and elegant"
(from L<NVIDIA's site|https://blogs.nvidia.com/blog/2012/09/10/what-is-cuda-2/>).
In short, we use CUDA to dispatch number-crunching code
to a Graphics Processing Unit (GPU) and then get the results back.

NVIDIA's CUDA comprises of a few keywords which can be inserted in C, C++, Fortran,
etc. code. In effect, developers still write programs in their preferred language (C, C++ etc.)
and whenever they need to access the GPU they use the CUDA
extensions. For more information check
L<CUDA Programming Guide|https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html> .

A CUDA program is, therefore, a C or C++ program with a few CUDA keywords added.
Generally, compiling such a program is done by a CUDA compiler, namely nvcc (nvidia cuda compiler)
which, simplistically put, splits the code in two parts, the CUDA part and the C part.
The C part is delegated to a C compiler, like gcc, and the CUDA part is handled by
nvcc. Finally nvcc links these components into an ordinary standalone executable.
For more information read
L<CUDA Toolkit Documentation|https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#cuda-source>

Notice that in NVIDIA jargon, a "device" is (one of) the GPU
and "host" is the CPU and the OS.

=head1 CAVEATS

In practice there are huge caveats which their conquering can be surprisingly easy
with some CLI magic. This is fine in Linux or even OSX but for poor M$-windows
victims, the same process can be painfully tortuous and possibly ending
to a mental breaker. As I don't belong to that category I will not be able to
help you with very specific requests regarding the so-called OS.

And on to the caveats.

=head3 Does your GPU support CUDA?

First of all, not all GPUs support CUDA. But new NVIDIA ones usually do and
at a price of less or around 100 euros.

=head3 Different CUDA SDK exists for different hardware

Secondly, different GPUs have different "compute capability" requiring different
versions of the CUDA SDK, which provides the nvcc and friends. For example my C<GeForce GTX 650>
has a compute capability of C<3.0> and that requires a SDK version of C<10.2>.
That's the last SDK to support a C<3.x> capability GPU. Currently, the SDK has reached
version 11.4 and supports compute capabilities of C<3.5> to C<8.6>. See the
L<Wikipedia article on CUDA|https://en.wikipedia.org/wiki/CUDA#GPUs_supported>
for what GPUs are supported and by what CUDA SDK version.

=head3 CUDA compiler requires specific compiler version

Thirdly and most importantly, C<nvcc> has specific and strict
requirements regarding the version of the "host compiler", for example,
C<gcc/g++>, C<clang>, C<cl.exe>. See which compilers are supported at

=over 2

=item L<Linux|https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>, 

=item L<mac-OSX|https://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html>,

=item L<Windows|https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html>

=back

For example, my GPU's compute capability (C<3.0>) requires CUDA SDK version C<10.2>
which requires gcc version less or equal to C<8>. Find out what compiler
your CUDA SDK supports in this
L<ax3l's gist|https://gist.github.com/ax3l/9489132>

There is a hack to stop C<nvcc> checking compiler version and using whatever
compiler it is specified by the user. Simply pass C<--allow-unsupported-compiler> to C<nvcc>
and hope for the best. According to
L<CUDA Toolkit Documentation|https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html>,
this flag has no effect in MacOS.

F<xt/30-check-basic-with-system-compiler.t> shows how to tell C<Inline::CUDA>
to use the system compiler and also tell C<nvcc> to not check
compiler version. This test can fail in particular OS/versions. It seems
to have worked for my particular setting. With this option you
are at least safe from getting into trouble because of
L</Perl and XS objects with mismatched compiler versions>.

=head3 GPU Programming: memory transfers overheads

Additionally, general GPU programming, in practice, has quite some caveats of
its own that the potential GPU programmer must be aware of. To start with,
there are some quite large overheads associated with sending data to the
GPU and receiving it back. Because the memory generally accessible to any program
running on the CPU (e.g. the C-part of the CUDA code) is not available to the
GPU in the simple and elegant manner C programmers take for granted
when presented with a memory pointer
and read the memory space it points to. And vice versa. Memory in the C-part of the
code must be C<cudaMemcpy()>'ed (the equivalent of C<memcpy()> for
host-to-device and device-to-host data transfers) to the GPU. And the results calculated
in the GPU remain there until are transfered back to host using another C<cudaMemcpy()> call.

Add to this the overhead of copying the value of each item of a Perl array into
a C array which C<cudaMemcpy()> understands and expects and you get quite
a significant overhead and a lot of paper-pushing for finally getting the same block
of data onto the GPU. And the same applies in doing the reverse.

Here is a rough sketch of what memory transfers are required
for calling an C<Inline::CUDA> function from Perl and doing GPU processing:

	my @array = (1..5); # memory allocated for Perl array
	inline_cuda_function(\@array, $result);
	...
	// now inside a Inline::CUDA code block
	int inline_cuda_function(SV *in, SV *out){
		// allocate memory for copying Perl array (in) to C
		h_in = malloc(5*sizeof(int));
		// allocate memory for holding the results on host
		h_out = malloc(5*sizeof(int));
		// allocate memory on the GPU for this same data
		cudaMalloc((void **)&d_in, 5*sizeof(int));
		// allocate memory on the GPU for the result
		cudaMalloc((void **)&d_out, 5*sizeof(int));
		// transfer Perl data onto host's C-array
		AV *anAV = (AV *)SvRV(in);
		for(int i=0;i<5;i++){
			SV *anSV = *av_fetch((AV *)SvRV(anAV), i, FALSE);
			h_in[i] = SvNV(anSV);
		}
		// and now transfer host's C-array onto the GPU
		cudaMemcpy(d_in, h_in, 5*sizeof(int), cudaMemcpyHostToDevice);
		// launch the kernel and do the processing onto the GPU
		...
		// extract results from the GPU onto host memory
		cudaMemcpy(h_out, d_in, 5*sizeof(int), cudaMemcpyDeviceToHost);
		// and now from host memory (the C array) onto Perl
		// we have been passed a scalar, we create a new arrayref
		// and place it to its RV slot
		anAV = newAV();
		av_extend(anAV, 5); // resize the Perl array to fit the result
		// sv_setrv() is a macro created by LeoNerd, see above
		// it places the new array we created onto the passed scalar (out)
		sv_setrv(SvRV(out), (SV *)av);
		for(int i=0;i<5;i++){
			av_store(av, i, newSVnv(h_out[i]));
		}
		free(h_in); free(h_out);
		cudaFree(d_in); cudaFree(d_out);
		return 0; // success
	}

There are some benchmarks in F<xt/benchmarks/*.b> which compare the
performance of a C<small> (size ~10x10), C<medium> (size ~100x100) and
C<large> (size ~1000x1000) data scenario for
doing matrix multiplication (run them with C<make benchmark>).
In my computer at least the pure-C,
CPU-hosted outperforms the GPU for the C<small>, C<medium> scenaria
exactly because of these overheads. But the GPU is a clear
winner for C<large> data scenario.

See for example this particular benchmark: F<xt/benchmarks/30-matrix-multiply.b>

=head3 Perl and XS objects with mismatched compiler versions

Finally, there is an issue with compiling XS code, which is essentially what C<Inline::CUDA> does,
with a compiler which is different to the compiler current Perl is built with. This is
the case when a special host compiler had to be installed because of
the CUDA SDK version. if that's true then you are essentially loading XS code
compiled with C<gcc82> (as per the example in section L</INSTALLATION>) with
a perl executable which was compiled with system compiler, for example C<gcc11>.
If that is really an issue then it will be insurmountable and the only
solution will be to L<perlbrew|https://perlbrew.pl/> a new Perl built
with the special host compiler, e.g. C<gcc82>.

The manual on
L<installing Perl|https://metacpan.org/dist/perl/view/INSTALL#C-compiler> states
that specifying the compiler is as simple as C<sh Configure -Dcc=/usr/local/gcc82/bin/gcc82>

If you want to compile and install a new Perl using L<perlbrew|https://perlbrew.pl/>
then this will do it (thank you L<Fletch@Perlmonks.org|https://perlmonks.org/?node_id=11159958>:

   PERLBREW_CONFIGURE_FLAGS='-d -Dcc=/usr/local/gcc82/bin/gcc' perlbrew install 5.38.2 --as 5.38.2.gcc82

The C<-d> is for not being asked trivial questions about the compilation options
and use sane defaults. The C<--as 5.38.2.gcc82> tells L<perlbrew|https://perlbrew.pl/>
to rename the new installed perl in case there is already one with the same name.


=head1 INSTALLATION

Installation of C<Inline::CUDA> is a nightmare because it depends on external
dependencies. It needs NVIDIA's CUDA SDK (providing C<nvcc> (the nvidia cuda compiler)
which requires specific host compiler versions. Which means that it is
very likely that you will also need to install in your system
an older compiler compatible with C<nvcc> version. Even if your
GPU supports the latest CUDA SDK version (at C<11.4> as of July 2021),
the maximum C<gcc> version allowed with that is C<10.21>.
Currently, C<gcc> is at version C<11.2> and upgrades monthly.

Installing a "private" compiler, in Linux, can be easy or hard depending
whether the package manager allows it. Mine does not. See L</how-to-install-compiler>
for instructions on how to do that on Linux and label the new compiler with
its own name so that one can have system compiler and older compiler
living in parallel and not disturbing each other.

B<That said>, there is a workaround: add this to pass
the C<--allow-unsupported-compiler> flag to C<nvcc>.
This can be achieved via the C<use Inline => Config => ...>, as below:

	use Inline => Config =>
		nvccflags => '--allow-unsupported-compiler',
		... # other config options
	;
	... # Inline::CUDA etc.

The long and proper way of installing C<Inline::CUDA> is described below.

So, if all goes Merfy you will have to install
C<nvcc> and an additional host compiler C<gcc>. The latter is not
the most pleasant of experiences in Linux. I don't know what's the situation
with Windows. I can only imagine the horror.

Here is a rough sketch of what one should do.

=over 2

=item Find the NVIDIA GPU name+version you have installed on your hardware kit. For example,
C<GeForce GTX 650>. This can be easy
or hard. 

=over 2

=item If you already have the executable C<nvidia-smi> installed or want to install it
(e.g. in Fedora CLI do C<dnf provides nvidia-smi> and make sure you have repo C<rpmfusion-nonfree>
enabled, somehow).

=item Install L<nvidia::ml> and run the script I provide with C<Inline::CUDA> at F<scripts/nvidia-ml-test.pl>

=back

=item With the NVIDIA GPU name+version available search this
L<Wikipedia article|https://en.wikipedia.org/wiki/CUDA#GPUs_supported>
for the "compute capability" of the GPU. For example this is C<3.0> for C<GeForce GTX 650>.

=item Use the "compute capability" of the GPU in order to find the CUDA SDK version you must install in the
same
L<Wikipedia article|https://en.wikipedia.org/wiki/CUDA#GPUs_supported> . For example, for the GPU
C<GeForce GTX 650>, one should download and install CUDA SDK 10.2.

=item Download, but not yet install, the specific version of the CUDA SDK from the
L<CUDA Toolkit Archive|https://developer.nvidia.com/cuda-toolkit-archive>

=item If you are lucky, your system's C compiler will be compatible with the CUDA SDK version
you downloaded and installing the above archive will be successful. it is worth to
give it a try, i.e. try to install and see if it will complain about incompatible host
compiler version. If it doesn't then you are good to go.

=item If installing the above archive yields errors about incompatible host
compiler then you must install a supported host compiler at a private path
(so as not to interfere with your actual system compiler) and provide that
path during installation (see below) of the CUDA SDK and also during
installation of C<Inline::CUDA> (see below).

=item Find the maximum host compiler version supported by your CUDA SDK you
downloaded. For example, CUDA SDK 10.2 in Linux is documented at
L<https://docs.nvidia.com/cuda/archive/10.2/cuda-installation-guide-linux/>. It states
that the maximum gcc version is C<8.2.1> for C<RHEL 8.1>.
I suspect that it is the compiler's major version, e.g. C<8>, that matters.
I can confirm that C<gcc 8.4.0> works fine
for C<Linux, Fedora 34, kernel 5.12, perl v5.32, GeForce GTX 650>.

=item Once you decide on the compiler version, download it and install it to a private
path so as not to interfere with the system compiler. Note that path for later use.

=item X<how-to-install-compiler> I have instructions on how to do the above, in Linux for C<gcc>.
Download specific
C<gcc> version from: L<ftp://ftp.fu-berlin.de/unix/languages/gcc/releases/>
(other mirrors exist here L<https://gcc.gnu.org/mirrors.html>). Compile
the compiler and make sure you give it a C<prefix> and a C<suffix>. You must also
download packages L<https://ftp.gnu.org/gnu/mpfr>, L<https://ftp.gnu.org/gnu/mpc/>
and L<https://ftp.gnu.org/gnu/gmp/>, choosing versions compatible with the gcc version
you have already downloaded. The crucial line in the configuration stage of
compiling gcc is C<configure --prefix=/usr/local/gcc82 --program-suffix=82 --enable-languages=c,c++ --disable-multilib --disable-libstdcxx-pch> .
Here is a gist from L<https://stackoverflow.com/questions/58859081/how-to-install-an-older-version-of-gcc-on-fedora>:

	tar xvf gcc-8.2.0.tar.xz 
	cd gcc-8.2.0/
	tar xvf mpfr-4.0.2.tar.xz && mv -v mpfr-4.0.2 mpfr
	tar xvf gmp-6.1.2.tar.xz && mv -v gmp-6.1.2 gmp
	tar xvf mpc-1.1.0.tar.gz && mv -v mpc-1.1.0 mpc
	cd ../
	mkdir build-gcc820
	cd build-gcc820/
	../gcc-8.2.0/configure --prefix=/usr/local/gcc82 --program-suffix=82 --enable-languages=c,c++,fortran --disable-multilib --disable-libstdcxx-pch
	make && make install

From now on, I will be using C</usr/local/gcc82/bin/gcc82> and C</usr/local/gcc82/bin/g++82> as my
host compilers.

=item Now you have our special compiler at C</usr/local/gcc82> under the name C</usr/local/gcc82/bin/gcc82>
and also C</usr/local/gcc82/bin/g++82>. We need to install the CUDA SDK and tell it to skip
checking host compiler compatibility (I don't think there is a way to point it to the correct compiler to use).
In Linux, this is like C<sh cuda_10.2.89_440.33.01_linux.run --override>. After a successful installation
you should be able to see C</usr/local/cuda/bin/nvcc>. Optionally add this to your PATH,
C<export PATH="${PATH}:/usr/local/cuda/bin">

=item In general, compiling CUDA code, for example L<this one|https://gist.github.com/dpiponi/1502434>,
is as simple as:

	nvcc --compiler-bindir /usr/local/gcc82/bin simple.cu && a.out

B<Notice the cuda program extension C<.cu>>. Without this extension,
in the best case C<nvcc> will not know what to do with that file
and with the C<C> extension, C<nvcc> will bomb.
It is important to keep the CUDA compiler happy.
Also note that if your CUDA SDK does not require installing an older
version of a compiler but instead it is happy with your system compiler,
then you can omit this: C<--compiler-bindir /usr/local/gcc82/bin>

=item If you did compile the simple cuda program and managed to run it, then
you are ready to install C<Inline::CUDA>. If your system compiler is acceptable
by CUDA SDK, then it is as simple as running

	perl Makefile.PL
	make
	make install

But if you need to declare a special host compiler (re: F</usr/local/gcc82/bin/gcc82>)
because your system compiler is not accepted by CUDA SDK then you need to
specify that to the installation process via one of the following two methods:

=over 2

=item The first method is more permanent but assumes that you
can (re-)install the module.
During installation, specify the following environment variables,
assuming a bash-based terminal, then this should do it:

	CC=/usr/local/gcc82/bin/gcc82 \
	CXX=/usr/local/gcc82/bin/g++82 \
	LD=/usr/local/gcc82/bin/g++82 \
	perl Makefile.PL
	make
	make install

=item The second method assumes you can edit C<Inline::CUDA>'s
configuration file located to a place like:
F</usr/local/share/perl5/5.32/auto/share/dist/Inline-CUDA/Inline-CUDA.conf>
(different systems will have a slightly different path),
and modify the entries for 'cc', 'cxx' and 'ld'.

=item The third and most versatile method is to specify
a custom configuration file
at the C<Config> section of L<Inline::CUDA>, like so:

  use Inline CUDA => Config =>
    ...
    CONFIGURATION_FILE => 'abc/mycuda.conf',
    ...
  ;

with only those entries you want to modify. Any entry
not specified in your custom configuration file will
be read from the system-wide configuration.

=back

=item Whatever the host compiler was, the configuration will be saved in
a file called C<Inline-CUDA.conf>. This file will be saved
in a C<share-dir> relative to your current Perl installation path. As an
example mine is at F</usr/local/share/perl5/5.32/auto/share/dist/Inline-CUDA/Inline-CUDA.conf>

This configuration file will be consulted every time you use C<Inline::CUDA> and will
know where the special host compiler resides.

=item Finally, C<make test> will
run a suite of test scripts and if all goes well all will succeed.
Additionally, C<make benchmark> will run a matrix multiplication benchmark
which will reveal if you can indeed get any benefits using GPGPU on your
specific hardware for this specific problem. Feel free to extend benchmarks
for your use-case.

=item At this stage I would urge people installing the code to run
also C<make author-test> and report back errors.

=back

=head1 DEMO

The folder C<demos/> in the base dir of the current distribution
contains self-contained C<Inline::CUDA> demo(s). One of which
produces the Mandelbrot Fractal on the GPU using Cuda code
copied from L<marioroy|https://perlmonks.org/?node=marioroy>'s
excellent work at L<https://github.com/marioroy/mandelbrot-python>,
see also PerlMonks post at L<https://perlmonks.org/?node_id=11139880>.
The demo is not complete, it just plugs L<marioroy|https://perlmonks.org/?node=marioroy>'s
Cuda code into C<Inline::CUDA>.

From the base dir of the current distribution run:

    make demo

=head1 CAVEATS

In your CUDA code do not implement C<main()>!
Place your CUDA code in your own functions which
you call from Perl. If you get segmentation faults
check the above first.

=head1 CONTRIBUTIONS BY OTHERS

This is a module which stands on the shoulders of Giants.

Literally!

To start with, CUDA and C<nvidia cuda compiler> are
two NVIDIA projects which offer general programming on the GPU
to the masses opening a new world of computational
capabilities as an alternative to the traditional CPU model.
A big thank you to NVIDIA.

Then there is Perl's L<Inline> module created by L<Ingy döt Net|https://metacpan.org/author/INGY>.
This module makes it easy to inline a lot of computer languages and call
them within a Perl script, passing Perl data structures and obtaining
results back.

This module is the key to opening many doors for Perl scripts.

A big thank you to L<Ingy döt Net|https://metacpan.org/author/INGY>.

Then there is Perl's L<Inline::C> module created/co-created/maintained
by L<Ingy döt Net|https://metacpan.org/author/INGY>,
L<Sisyphus|https://metacpan.org/author/SISYPHUS> and L<Tina Müller|https://metacpan.org/author/TINITA>.

The current C<Inline::CUDA> module relies heavily on L<Inline::C>. Because
the underlying CUDA language is C, I decided that instead of copying what
L<Inline::C> does and modifying the section where the Makefile is written,
I decided to inject all L<Inline::C>'s subs into C<Inline::CUDA> except
some sections which require special treatment, like when writing the Makefile
and also allowing some special C<Config> keywords. The sub injection happens
every time the module is called, and that definetely adds a tiny overhead
which, in my opinion, is compensated by the huge advantage of not
copy-pasting code from L<Inline::C> into  C<Inline::CUDA> and then
incorporating my changes every time L<Inline::C> updates.
A big thank you to L<Ingy döt Net|https://metacpan.org/author/INGY> (again!),
L<Sisyphus|https://metacpan.org/author/SISYPHUS> and L<Tina Müller|https://metacpan.org/author/TINITA>.

For writing test cases and benchmarks I had to descend into C and become
acquainted with L<perlguts>,
e.g. what is an L<SV|https://perldoc.perl.org/perlguts#Working-with-SVs>.
In this process I had to ask for the wisdom of L<PerlMonks.org> and
L<#perl|https://web.libera.chat/#perl>. A particular question was
how to pass in a C function an arrayref, a scalar or a scalarref,
store the results of the computation in there, in a call-by-reference manner.
Fortunately LeoNerd at L<#perl|https://kiwiirc.com/nextclient/#irc://irc.perl.org/#perl>
created the following C<sv_setrv()> macro which saved the day. Big thank you LeoNerd.

	/************************************************************/
	/* MONKEYPATCH by LeoNerd to set an arrayref into a scalarref
	   As posted on https://kiwiirc.com/nextclient/#irc://irc.perl.org/#perl
	   at 10:50 23/07/2021
	   A BIG THANK YOU LeoNerd
	*/
	#define HAVE_PERL_VERSION(R, V, S) \
	    (PERL_REVISION > (R) || (PERL_REVISION == (R) && (PERL_VERSION > (V) || (PERL_VERSION == (V) && (PERL_SUBVERSION >= (S))))))

	#define sv_setrv(s, r)  S_sv_setrv(aTHX_ s, r)
	static void S_sv_setrv(pTHX_ SV *sv, SV *rv)
	{
	  sv_setiv(sv, (IV)rv);
	#if !HAVE_PERL_VERSION(5, 24, 0)
	  SvIOK_off(sv);
	#endif
	  SvROK_on(sv);
	}

I copied numerical recipes (as C code, Cuda kernels, etc.) from the repository of
L<Zhengchun Liu|https://github.com/lzhengchun> this code resides in 'C/inlinecuda'
of the current distribution and offers shortcuts to GPU-based matrix multiplication,
for example.

The idea of this project came to me when L<kcott|https://www.perlmonks.org/?node=kcott>
asked whether there are L<https://www.perlmonks.org/?node_id=11134476|https://www.perlmonks.org/?node_id=11134476>
which I responded with the L<preliminary idea|https://www.perlmonks.org/?node_id=11134582> for what is
now C<Inline::CUDA>. A big thank you to L<kcott|https://www.perlmonks.org/?node=kcott>.

I got helpful comments, advice and the odd smiley from
C<LeoNerd>, C<mst>, C<Bojte>, C<shlomif> at L<#perl|https://kiwiirc.com/nextclient/#irc://irc.perl.org/#perl>,
thank you.

I got helpful comments and advice in this L<PerlMonks.org post|https://perlmonks.org/?node_id=11135324>
from L<syphilis|https://perlmonks.org/?node=syphilis> and L<perlfan|https://perlmonks.org/?node=perlfan>,
although the problem was cracked by LeoNerd L<#perl|https://kiwiirc.com/nextclient/#irc://irc.perl.org/#perl>.

I also got helpful comments and advice from L<Ed J|https://metacpan.org/author/ETJ> when I filed a bug over
at L<ExtUtils::MakeMaker> (see L<https://rt.cpan.org/Ticket/Display.html?id=138022> and
L<https://rt.cpan.org/Ticket/Display.html?id=137912>).


=head1 AUTHOR

Andreas Hadjiprocopis,
C<< <bliako at cpan dot org> >>,
C<< <andreashad2 at gmail dot com> >>,
L<https://perlmonks.org/?node=bliako>


=head1 DEDICATIONS

!Almaz!


=head1 BUGS

Please report any bugs or feature requests to C<bug-inline-cuda at rt.cpan.org>, or through
the web interface at L<https://rt.cpan.org/NoAuth/ReportBug.html?Queue=Inline-CUDA>.  I will be notified, and then you'll
automatically be notified of progress on your bug as I make changes.

NOTE: this project is not yet on CPAN so report bugs by email to the author. I am not
very comfortable with github so cloning and merging and pushing and pulling are beyond me.


=head1 SUPPORT

You can find documentation for this module with the perldoc command.

    perldoc Inline::CUDA


You can also look for information at:

=over 4

=item * RT: CPAN's request tracker (report bugs here)

L<https://rt.cpan.org/NoAuth/Bugs.html?Dist=Inline-CUDA>

=item * PerlMonks.org : a great forum to find Perl support and wisdom

The main side is this L<https://perlmonks.org> where you can post
questions. The author's page is this L<https://perlmonks.org/?node=bliako>

=item * CPAN Ratings

L<https://cpanratings.perl.org/d/Inline-CUDA>

=item * Search CPAN

L<https://metacpan.org/release/Inline-CUDA>

=back


=head1 ACKNOWLEDGEMENTS

This module stands on the shoulders of giants, namely
the authors of L<Inline> and L<Inline::C>. I wish
to thank them here and pass most credit to them. I will keep 1%.

A big thank you to NVIDIA for providing tools
and support for doing numerical programming on their GPU.

All mentioned above provided keys to many doors, all free and
open source. Thank you!


=head1 LICENSE AND COPYRIGHT

This software is Copyright (c) 2021 by Andreas Hadjiprocopis.

This is free software, licensed under:

  The Artistic License 2.0 (GPL Compatible)


=cut

1; # End of Inline::CUDA
