use 5.006;
use strict;
use warnings;

use File::Temp;
use File::Copy;

our $VERSION = 0.16;

use Test::More;

use Inline::CUDA::Utils;

my $conf = {
	'XX' => 'a/b/c',
	'YY' => '1/2/3',
};
my ($tmpfh, $tmpfilename) = File::Temp::tempfile(SUFFIX => '.conf');
ok(defined($tmpfh) && (-f $tmpfilename), "File::Temp::tempfile() : called/1.");

my $conf_file = $tmpfilename;

# save some dummy configuration in user-specified location
ok(0==Inline::CUDA::Utils::save_configuration($conf, $conf_file), "Inline::CUDA::Utils::save_configuration() : called.");
ok(-f $conf_file, "Inline::CUDA::Utils::save_configuration(): conf-file exists: '$conf_file'.");

# read it back and check if same contents
my $newconf = Inline::CUDA::Utils::read_configuration($conf_file);
ok(defined($newconf), "Configuration read from file '$conf_file'.");

# same contents?
is_deeply($conf, $newconf, "Configurations written and read are identical.");
close $tmpfh; unlink($tmpfilename);

######
# now with a proper conf file in shared-dir which must exist there
# after a successful "make" (not necessarily a "make install")
# we call it with no conf_file
######
my $default_conf_file = Inline::CUDA::Utils::path_of_configuration_file_in_shared_dir();
ok(defined($default_conf_file), "Inline::CUDA::Utils::path_of_configuration_file_in_shared_dir() : called.");

# load from default file, which is found just like above
$conf = Inline::CUDA::Utils::read_configuration();
ok(defined($conf), "Inline::CUDA::Utils::read_configuration() : called for default shared-dir conf file: '$default_conf_file'.");

# get a tmp filename
($tmpfh, $tmpfilename) = File::Temp::tempfile(SUFFIX => '.conf');
ok(defined($tmpfh) && (-f $tmpfilename), "File::Temp::tempfile() : called/1.");

# now save conf to a tmp location
ok(0==Inline::CUDA::Utils::save_configuration($conf, $tmpfilename), "Inline::CUDA::Utils::save_configuration(): called for '$tmpfilename'.");
close($tmpfh);

# load what we just saved
$newconf = Inline::CUDA::Utils::read_configuration($tmpfilename);
ok(defined($conf), "Inline::CUDA::Utils::read_configuration() : called for '$tmpfilename'.");

is_deeply($conf, $newconf, "Configurations written and read are identical, '$tmpfilename'");
unlink($tmpfilename);

if( -W $default_conf_file ){
	# now save it to the default location, only if it is writable by us
	# WARNING: it overwrites the previous one! but we made a backup
	ok(File::Copy::copy($default_conf_file, $tmpfilename), "backed-up configuration '$default_conf_file' => '$tmpfilename'.");
	ok(0==Inline::CUDA::Utils::save_configuration($conf), "Inline::CUDA::Utils::save_configuration(): called for default location.");
	
	# load what we just saved
	$newconf = Inline::CUDA::Utils::read_configuration();
	ok(defined($conf), "Inline::CUDA::Utils::read_configuration() : called for default-location configuration file '$default_conf_file'.");
	
	# are they the same?
	is_deeply($conf, $newconf, "Configurations written and read are identical, '$default_conf_file'");
	
	# copy back original configuration
	ok(File::Copy::copy($tmpfilename, $default_conf_file), "backed-up configuration restored.");
	
	unlink($tmpfilename);
}	
done_testing();
