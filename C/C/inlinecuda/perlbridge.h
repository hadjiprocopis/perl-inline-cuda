#ifndef _PERLBRIDGE_H
#define _PERLBRIDGE_H

/* our $VERSION = 0.05; */

/************************************************************/

/* Monkeypath by LeoNerd to set an arrayref into a scalarref
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
/************************************************************/

int is_array_ref(
	SV *array,
	size_t *array_sz
);
int array_numelts_2D(
	SV *array,
	size_t *_Nd1,
	size_t **_Nd2
);
int array_of_unsigned_int_into_AV(
	size_t *src,
	size_t src_sz,
	SV *dst
);
int array_of_int_into_AV(
	int *src,
	size_t src_sz,
	SV *dst
);
#endif
