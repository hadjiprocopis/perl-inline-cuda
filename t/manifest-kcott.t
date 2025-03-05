#!perl

use strict;
use warnings;

our $VERSION = 0.16;

use Test::More;

BEGIN {
    if (! $ENV{RELEASE_TESTING}) {
        plan skip_all => 'Author test: $ENV{RELEASE_TESTING} false.';
    }
}

use ExtUtils::Manifest qw{manicheck filecheck};

plan tests => 2;

is_deeply([manicheck()], [], 'Check files in "MANIFEST" exist.');
is_deeply([filecheck()], [], 'Check for files not in "MANIFEST" or "MANIFEST.SKIP".');
