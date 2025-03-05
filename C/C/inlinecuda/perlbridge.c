#ifndef _PERLBRIDGE_C
#define _PERLBRIDGE_C

/* our $VERSION = 0.05; */

#include <inlinecuda/utils.h>

#include <inlinecuda/perlbridge.h>

/* Check if it is a Perl arrayref and return its size if yes
   (via the placeholder pointer)
   returns 0 on FAILURE <<<< (not 1 as the others)
	   1 on SUCCESS
   Ripped from the Inline::C::Cookbook
*/
int is_array_ref(
	SV *array,
	size_t *array_sz
){
	if( ! SvROK(array) ){
		//fprintf(stderr, "is_array_ref() : warning, input '%p' is not a reference.\n", array);
		return 0;
	}
	if( SvTYPE(SvRV(array)) != SVt_PVAV ){
		//fprintf(stderr, "is_array_ref() : warning, input ref '%p' is not an ARRAY reference.\n", array);
		return 0;
	}
	// it's an array, cast it to AV to get its len via av_len();
	// yes, av_len needs to be bumped up
	int asz = 1+av_len((AV *)SvRV(array));
	if( asz < 0 ){ fprintf(stderr, "is_array_ref() : error, input array ref '%p' has negative size!\n", array); return 0; }
	*array_sz = (size_t )asz;
	return 1; // success, it is an array and size returned by ref, above
}
/* Find the shape (i.e size) of a Perl 1D array-ref (A)
   The results are placed in placeholders supplied via pointer B
   returns 0 on success,
	   1 on failure.
*/
#define array_numelts_1D(A,B) (!is_array_ref(A,B))

/* Finds the shape of a Perl 2D array-ref,
   The results are placed in placeholders supplied via pointers _Nd1, _Nd2
   returns 0 on success,
	   1 on failure.
*/
int array_numelts_2D(
	SV *array,
	size_t *_Nd1,
	size_t **_Nd2
){
	size_t anN, anN2, *Nd2 = NULL;

	if( ! is_array_ref(array, &anN) ){
		//fprintf(stderr, "is_array_ref_2D() : error, call to is_array_ref() has failed for array '%p'.\n", array);
		return 1;
	}

	if( *_Nd2 == NULL ){
		if( (Nd2=(size_t *)malloc(anN*sizeof(size_t))) == NULL ){ fprintf(stderr, "array_numelts_2D() : error, failed to allocate %zu bytes for %zu items for Nd2.\n", anN*sizeof(size_t), anN); return 1; }
	} else Nd2 = *_Nd2;
	AV *anAV = (AV *)SvRV(array);
	size_t *pNd2 = &(Nd2[0]);
	for(size_t i=0;i<anN;i++,pNd2++){
		SV *subarray = *av_fetch(anAV, i, FALSE);
		if( ! is_array_ref(subarray, &anN2) ){ fprintf(stderr, "is_array_ref_2D() : error, call to is_array_ref() has failed for [%p][%p], item %zu.\n", array, subarray, i); if(*_Nd2==NULL) free(Nd2); return 1; }
		*pNd2 = anN2;
	}
	if( *_Nd2 == NULL ) *_Nd2 = Nd2;
	*_Nd1 = anN;
	return 0; // success
}

/* Insert a C array of size_t's (unsigned int) into a Perl 1D array
   Returns 0 on success
	   1 on failure
*/
int array_of_unsigned_int_into_AV(
	size_t *src,
	size_t src_sz,
	SV *dst
){
	size_t dst_sz;	
	if( ! is_array_ref(dst, &dst_sz) ){
		//fprintf(stderr, "array_of_unsigned_int_into_AV() : error, call to is_array_ref() has failed.\n");
		return 1;
	}
	AV *dstAV = (AV *)SvRV(dst);
	for(size_t i=0;i<src_sz;i++){
		av_push(dstAV, newSViv(src[i]));
	}
	return 0; // success
}

/* Insert a C array of integers into a Perl 1D array
   Returns 0 on success
	   1 on failure
*/
int array_of_int_into_AV(
	int *src,
	size_t src_sz,
	SV *dst
){
	size_t dst_sz;	
	if( ! is_array_ref(dst, &dst_sz) ){
		//fprintf(stderr, "array_of_int_into_AV() : error, call to is_array_ref() has failed.\n");
		return 1;
	}
	AV *dstAV = (AV *)SvRV(dst);
	for(size_t i=0;i<src_sz;i++){
		av_push(dstAV, newSViv(src[i]));
	}
	return 0; // success
}
#endif
