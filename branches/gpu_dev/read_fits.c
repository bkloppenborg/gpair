/* Example code for FITS-based optical interferometry exchange format

$Id: read_fits.c,v 1.8 2006/03/22 14:51:55 jsy1001 Exp $
Code to read FITS file and write to data structures in memory


Release 8  22 March 2006

John Young <jsy1001@cam.ac.uk>

*/

#include <stdlib.h>
#include <string.h>

#include "exchange.h"
#include "fitsio.h"


/**
* Read OI_ARRAY fits binary table
*
*   @param fptr    see cfitsio documentation
*   @param arrname read table with this value for ARRNAME
*   @param array   ptr to array data struct, see exchange.h
*   @param status  pointer to status variable
*
*   @return On error, returns non-zero cfitsio error code (also assigned to
*           *status). Contents of array data struct are undefined
*/
int read_oi_array(fitsfile *fptr, char *arrname, oi_array *array, int *status) 
{
	
	const char function[] = "read_oi_array";
	char comment[FLEN_COMMENT], extname[FLEN_VALUE], name[FLEN_VALUE];
	char *p;
	char nullstring[] = "NULL";
	int nullint = 0;
	float nullfloat = 0.0F;
	double nulldouble = 0.0;
	const int revision = 1;
	int irow, colnum, anynull, nhdu, ihdu, hdutype;
	long repeat;
	
	if (*status) return *status; /* error flag set - do nothing */
	
	/* Move to correct HDU - don't assume anything about EXTVERs */
	fits_get_num_hdus(fptr, &nhdu, status);
	for (ihdu=2; ihdu<=nhdu; ihdu++) {
		fits_movabs_hdu(fptr, ihdu, &hdutype, status);
		if (hdutype == BINARY_TBL) {
			fits_read_key(fptr, TSTRING, "EXTNAME", extname, comment, status);
			fits_read_key(fptr, TSTRING, "ARRNAME", name, comment, status);
			if (*status) {
				*status = 0;
				continue; /* next HDU */
			}
			if (strcmp(extname, "OI_ARRAY") != 0 || strcmp(name, arrname) != 0)
				continue; /* next HDU */
		}
		break; /* current HDU matches */
	}
	if (ihdu > nhdu) {
		/* no matching HDU */
		*status = BAD_HDU_NUM;
		return *status;
	}
	
	/* Read table */
	fits_read_key(fptr, TINT, "OI_REVN", &array->revision, comment, status);
	if (array->revision != revision) {
		printf("WARNING! Expecting value %d for OI_REVN keyword in OI_ARRAY table. Got %d\n", revision, array->revision);
	}
	strcpy(array->arrname, name);
	fits_read_key(fptr, TSTRING, "FRAME", array->frame, comment, status);
	fits_read_key(fptr, TDOUBLE, "ARRAYX", &array->arrayx, comment, status);
	fits_read_key(fptr, TDOUBLE, "ARRAYY", &array->arrayy, comment, status);
	fits_read_key(fptr, TDOUBLE, "ARRAYZ", &array->arrayz, comment, status);
	/* get number of rows */
	fits_get_num_rows(fptr, &repeat, status);
	array->nelement = repeat;
	array->elem = malloc(array->nelement*sizeof(element));
	/* read rows */
	for (irow=1; irow<=array->nelement; irow++) {
		fits_get_colnum(fptr, CASEINSEN, "TEL_NAME", &colnum, status);
		p = array->elem[irow-1].tel_name;
		fits_read_col(fptr, TSTRING, colnum, irow, 1, 1, nullstring, &p, &anynull,
		status);
		fits_get_colnum(fptr, CASEINSEN, "STA_NAME", &colnum, status);
		p = array->elem[irow-1].sta_name;
		fits_read_col(fptr, TSTRING, colnum, irow, 1, 1, nullstring, &p, &anynull,
		status);
		fits_get_colnum(fptr, CASEINSEN, "STA_INDEX", &colnum, status);
		fits_read_col(fptr, TINT, colnum, irow, 1, 1, &nullint,
		&array->elem[irow-1].sta_index, &anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "DIAMETER", &colnum, status);
		fits_read_col(fptr, TFLOAT, colnum, irow, 1, 1, &nullfloat,
		&array->elem[irow-1].diameter, &anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "STAXYZ", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, 3, &nulldouble,
		&array->elem[irow-1].staxyz, &anynull, status);
		
	}
	if (*status) {
		fprintf(stderr, "CFITSIO error in %s:\n", function);
		fits_report_error(stderr, *status);
	}
	return *status;
}


/**
* Read OI_TARGET fits binary table
*
*   @param fptr    see cfitsio documentation
*   @param targets ptr to targets data struct, see exchange.h
*   @param status  pointer to status variable
*
*   @return On error, returns non-zero cfitsio error code (also assigned to
*           *status). Contents of targets data struct are undefined
*/
int read_oi_target(fitsfile *fptr, oi_target *targets, int *status) 
{
	
	const char function[] = "read_oi_target";
	char comment[FLEN_COMMENT];
	char *p;
	char nullstring[] = "NULL";
	int nullint = 0;
	float nullfloat = 0.0F;
	double nulldouble = 0.0;
	const int revision = 1;
	int irow, colnum, anynull;
	long repeat;
	
	if (*status) return *status; /* error flag set - do nothing */
	
	fits_movnam_hdu(fptr, BINARY_TBL, "OI_TARGET", 0, status);
	fits_read_key(fptr, TINT, "OI_REVN", &targets->revision, comment, status);
	if (targets->revision != revision) {
		printf("WARNING! Expecting value %d for OI_REVN keyword in OI_TARGETS table. Got %d\n", revision, targets->revision);
	}
	/* get number of rows */
	fits_get_num_rows(fptr, &repeat, status);
	targets->ntarget = repeat;
	targets->targ = malloc(targets->ntarget*sizeof(target));
	/* read rows */
	for (irow=1; irow<=targets->ntarget; irow++) {
		fits_get_colnum(fptr, CASEINSEN, "TARGET_ID", &colnum, status);
		fits_read_col(fptr, TINT, colnum, irow, 1, 1, &nullint,
		&targets->targ[irow-1].target_id, &anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "TARGET", &colnum, status);
		p = targets->targ[irow-1].target;
		fits_read_col(fptr, TSTRING, colnum, irow, 1, 1, nullstring, &p,
		&anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "RAEP0", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, 1, &nulldouble,
		&targets->targ[irow-1].raep0, &anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "DECEP0", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, 1, &nulldouble,
		&targets->targ[irow-1].decep0, &anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "EQUINOX", &colnum, status);
		fits_read_col(fptr, TFLOAT, colnum, irow, 1, 1, &nullfloat,
		&targets->targ[irow-1].equinox, &anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "RA_ERR", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, 1, &nulldouble,
		&targets->targ[irow-1].ra_err, &anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "DEC_ERR", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, 1, &nulldouble,
		&targets->targ[irow-1].dec_err, &anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "SYSVEL", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, 1, &nulldouble,
		&targets->targ[irow-1].sysvel, &anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "VELTYP", &colnum, status);
		p = targets->targ[irow-1].veltyp;
		fits_read_col(fptr, TSTRING, colnum, irow, 1, 1, nullstring, &p,
		&anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "VELDEF", &colnum, status);
		p = targets->targ[irow-1].veldef;
		fits_read_col(fptr, TSTRING, colnum, irow, 1, 1, nullstring, &p,
		&anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "PMRA", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, 1, &nulldouble,
		&targets->targ[irow-1].pmra, &anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "PMDEC", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, 1, &nulldouble,
		&targets->targ[irow-1].pmdec, &anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "PMRA_ERR", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, 1, &nulldouble,
		&targets->targ[irow-1].pmra_err, &anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "PMDEC_ERR", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, 1, &nulldouble,
		&targets->targ[irow-1].pmdec_err, &anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "PARALLAX", &colnum, status);
		fits_read_col(fptr, TFLOAT, colnum, irow, 1, 1, &nullfloat,
		&targets->targ[irow-1].parallax, &anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "PARA_ERR", &colnum, status);
		fits_read_col(fptr, TFLOAT, colnum, irow, 1, 1, &nullfloat,
		&targets->targ[irow-1].para_err, &anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "SPECTYP", &colnum, status);
		p = targets->targ[irow-1].spectyp;
		fits_read_col(fptr, TSTRING, colnum, irow, 1, 1, nullstring, &p,
		&anynull, status);
		/*printf("%16s  %10f %10f  %8s\n",
		targets->targ[irow-1].target,
		targets->targ[irow-1].raep0, targets->targ[irow-1].decep0,
		targets->targ[irow-1].spectyp);*/
	}
	if (*status) {
		fprintf(stderr, "CFITSIO error in %s:\n", function);
		fits_report_error(stderr, *status);
	}
	return *status;
}


/**
* Read OI_WAVELENGTH fits binary table
*
*   @param fptr    see cfitsio documentation
*   @param insname read table with this value for INSNAME
*   @param wave    ptr to wavelength data struct, see exchange.h
*   @param status  pointer to status variable
*
*   @return On error, returns non-zero cfitsio error code (also assigned to
*           *status). Contents of wavelength data struct are undefined
*/
int read_oi_wavelength(fitsfile *fptr, char *insname, oi_wavelength *wave,
int *status) {
	
	const char function[] = "read_oi_wavelength";
	char comment[FLEN_COMMENT], extname[FLEN_VALUE], name[FLEN_VALUE];
	float nullfloat = 0.0F;
	const int revision = 1;
	int colnum, anynull, nhdu, ihdu, hdutype;
	long repeat;
	
	if (*status) return *status; /* error flag set - do nothing */
	
	/* Move to correct HDU - don't assume anything about EXTVERs */
	fits_get_num_hdus(fptr, &nhdu, status);
	for (ihdu=2; ihdu<=nhdu; ihdu++) {
		fits_movabs_hdu(fptr, ihdu, &hdutype, status);
		if (hdutype == BINARY_TBL) {
			fits_read_key(fptr, TSTRING, "EXTNAME", extname, comment, status);
			fits_read_key(fptr, TSTRING, "INSNAME", name, comment, status);
			if (*status) {
				*status = 0;
				continue; /* next HDU */
			}
			if (strcmp(extname, "OI_WAVELENGTH") != 0 || strcmp(name, insname) != 0)
				continue; /* next HDU */
		}
		break; /* current HDU matches */
	}
	if (ihdu > nhdu) {
		/* no matching HDU */
		*status = BAD_HDU_NUM;
		return *status;
	}
	
	/* Read table */
	fits_read_key(fptr, TINT, "OI_REVN", &wave->revision, comment, status);
	if (wave->revision != revision) {
		printf("WARNING! Expecting value %d for OI_REVN keyword in OI_WAVELENGTH table. Got %d\n", revision, wave->revision);
	}
	strcpy(wave->insname, name);
	
	/* get number of rows */
	fits_get_num_rows(fptr, &repeat, status);
	wave->nwave = repeat;
	wave->eff_wave = malloc(wave->nwave*sizeof(float));
	wave->eff_band = malloc(wave->nwave*sizeof(float));
	/* read columns */
	fits_get_colnum(fptr, CASEINSEN, "EFF_WAVE", &colnum, status);
	fits_read_col(fptr, TFLOAT, colnum, 1, 1, wave->nwave, &nullfloat,
	wave->eff_wave, &anynull, status);
	fits_get_colnum(fptr, CASEINSEN, "EFF_BAND", &colnum, status);
	fits_read_col(fptr, TFLOAT, colnum, 1, 1, wave->nwave, &nullfloat,
	wave->eff_band, &anynull, status);
	if (*status) {
		fprintf(stderr, "CFITSIO error in %s:\n", function);
		fits_report_error(stderr, *status);
	}
	return *status;
}


/**
* Read next OI_VIS fits binary table
*
*   @param fptr   see cfitsio documentation
*   @param vis    ptr to data struct, see exchange.h
*   @param status pointer to status variable
*
*   @return On error, returns non-zero cfitsio error code (also assigned to
*           *status). Contents of data struct are undefined
*/
int read_next_oi_vis(fitsfile *fptr, oi_vis *vis, int *status) {
	
	const char function[] = "read_oi_vis";
	char comment[FLEN_COMMENT], extname[FLEN_VALUE];
	char nullchar = 0;
	int nullint = 0;
	double nulldouble = 0.0;
	const int revision = 1;
	int irow, colnum, anynull, hdutype;
	long repeat;
	
	if (*status) return *status; /* error flag set - do nothing */
	
	/* Move to correct HDU - don't assume anything about EXTVERs */
	while (1==1) {
		fits_movrel_hdu(fptr, 1, &hdutype, status);
		if (*status) return *status; /* no more HDUs */
		fits_read_key(fptr, TSTRING, "EXTNAME", extname, comment, status);
		if (strcmp(extname, "OI_VIS") == 0) break; /* current HDU matches */
	}
	
	/* Read table */
	fits_read_key(fptr, TINT, "OI_REVN", &vis->revision, comment, status);
	if (vis->revision != revision) {
		printf("WARNING! Expecting value %d for OI_REVN keyword in OI_VIS table. Got %d\n", revision, vis->revision);
	}
	fits_read_key(fptr, TSTRING, "DATE-OBS", vis->date_obs, comment, status);
	fits_read_key(fptr, TSTRING, "ARRNAME", vis->arrname, comment, status);
	if (*status == KEY_NO_EXIST) { /* ARRNAME is optional */
		vis->arrname[0] = '\0';
		*status = 0;
	}
	fits_read_key(fptr, TSTRING, "INSNAME", vis->insname, comment, status);
	/* get number of rows */
	fits_get_num_rows(fptr, &vis->numrec, status);
	vis->record = malloc(vis->numrec*sizeof(oi_vis_record));
	/* get value for nwave */
	/* format specifies same repeat count for VIS* columns */
	fits_get_colnum(fptr, CASEINSEN, "VISAMP", &colnum, status);
	fits_get_coltype(fptr, colnum, NULL, &repeat, NULL, status);
	vis->nwave = repeat;
	/* read rows */
	for (irow=1; irow<=vis->numrec; irow++) {
		fits_get_colnum(fptr, CASEINSEN, "TARGET_ID", &colnum, status);
		fits_read_col(fptr, TINT, colnum, irow, 1, 1, &nullint,
		&vis->record[irow-1].target_id, &anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "TIME", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, 1, &nulldouble,
		&vis->record[irow-1].time, &anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "MJD", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, 1, &nulldouble,
		&vis->record[irow-1].mjd, &anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "INT_TIME", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, 1, &nulldouble,
		&vis->record[irow-1].int_time, &anynull, status);
		vis->record[irow-1].visamp = malloc(vis->nwave*sizeof(double));
		vis->record[irow-1].visamperr = malloc(vis->nwave*sizeof(double));
		vis->record[irow-1].visphi = malloc(vis->nwave*sizeof(double));
		vis->record[irow-1].visphierr = malloc(vis->nwave*sizeof(double));
		vis->record[irow-1].flag = malloc(vis->nwave*sizeof(char));
		fits_get_colnum(fptr, CASEINSEN, "VISAMP", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, vis->nwave,
		&nulldouble, vis->record[irow-1].visamp, &anynull,
		status);
		fits_get_colnum(fptr, CASEINSEN, "VISAMPERR", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, vis->nwave,
		&nulldouble, vis->record[irow-1].visamperr, &anynull,
		status);
		fits_get_colnum(fptr, CASEINSEN, "VISPHI", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, vis->nwave,
		&nulldouble, vis->record[irow-1].visphi, &anynull,
		status);
		fits_get_colnum(fptr, CASEINSEN, "VISPHIERR", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, vis->nwave,
		&nulldouble, vis->record[irow-1].visphierr, &anynull,
		status);
		fits_get_colnum(fptr, CASEINSEN, "UCOORD", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, 1, &nulldouble,
		&vis->record[irow-1].ucoord, &anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "VCOORD", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, 1, &nulldouble,
		&vis->record[irow-1].vcoord, &anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "STA_INDEX", &colnum, status);
		fits_read_col(fptr, TINT, colnum, irow, 1, 2, &nullint,
		vis->record[irow-1].sta_index, &anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "FLAG", &colnum, status);
		fits_read_col(fptr, TLOGICAL, colnum, irow, 1, vis->nwave, &nullchar,
		vis->record[irow-1].flag, &anynull, status);
	}
	if (*status) {
		fprintf(stderr, "CFITSIO error in %s:\n", function);
		fits_report_error(stderr, *status);
	}
	return *status;
}


/**
* Read next OI_VIS2 fits binary table
*
*   @param fptr   see cfitsio documentation
*   @param vis2   ptr to data struct, see exchange.h
*   @param status pointer to status variable
*
*   @return On error, returns non-zero cfitsio error code (also assigned to
*           *status). Contents of data struct are undefined
*/
int read_next_oi_vis2(fitsfile *fptr, oi_vis2 *vis2, int *status) {
	
	const char function[] = "read_oi_vis2";
	char comment[FLEN_COMMENT], extname[FLEN_VALUE];
	char nullchar = 0;
	int nullint = 0;
	double nulldouble = 0.0;
	const int revision = 1;
	int irow, colnum, anynull, hdutype;
	long repeat;
	
	if (*status) return *status; /* error flag set - do nothing */
	
	/* Move to correct HDU - don't assume anything about EXTVERs */
	while (1==1) {
		fits_movrel_hdu(fptr, 1, &hdutype, status);
		if (*status) return *status; /* no more HDUs */
		fits_read_key(fptr, TSTRING, "EXTNAME", extname, comment, status);
		if (strcmp(extname, "OI_VIS2") == 0) break; /* current HDU matches */
	}
	
	/* Read table */
	fits_read_key(fptr, TINT, "OI_REVN", &vis2->revision, comment, status);
	if (vis2->revision != revision) {
		printf("WARNING! Expecting value %d for OI_REVN keyword in OI_VIS2 table. Got %d\n", revision, vis2->revision);
	}
	fits_read_key(fptr, TSTRING, "DATE-OBS", vis2->date_obs, comment, status);
	fits_read_key(fptr, TSTRING, "ARRNAME", vis2->arrname, comment, status);
	if (*status == KEY_NO_EXIST) { /* ARRNAME is optional */
		vis2->arrname[0] = '\0';
		*status = 0;
	}
	fits_read_key(fptr, TSTRING, "INSNAME", vis2->insname, comment, status);
	/* get number of rows */
	fits_get_num_rows(fptr, &vis2->numrec, status);
	vis2->record = malloc(vis2->numrec*sizeof(oi_vis2_record));
	/* get value for nwave */
	/* format specifies same repeat count for VIS2DATA & VIS2ERR columns */
	fits_get_colnum(fptr, CASEINSEN, "VIS2DATA", &colnum, status);
	fits_get_coltype(fptr, colnum, NULL, &repeat, NULL, status);
	vis2->nwave = repeat;
	/* read rows */
	for (irow=1; irow<=vis2->numrec; irow++) {
		fits_get_colnum(fptr, CASEINSEN, "TARGET_ID", &colnum, status);
		fits_read_col(fptr, TINT, colnum, irow, 1, 1, &nullint,
		&vis2->record[irow-1].target_id, &anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "TIME", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, 1, &nulldouble,
		&vis2->record[irow-1].time, &anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "MJD", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, 1, &nulldouble,
		&vis2->record[irow-1].mjd, &anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "INT_TIME", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, 1, &nulldouble,
		&vis2->record[irow-1].int_time, &anynull, status);
		vis2->record[irow-1].vis2data = malloc(vis2->nwave*sizeof(double));
		vis2->record[irow-1].vis2err = malloc(vis2->nwave*sizeof(double));
		vis2->record[irow-1].flag = malloc(vis2->nwave*sizeof(char));
		fits_get_colnum(fptr, CASEINSEN, "VIS2DATA", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, vis2->nwave,
		&nulldouble, vis2->record[irow-1].vis2data, &anynull,
		status);
		fits_get_colnum(fptr, CASEINSEN, "VIS2ERR", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, vis2->nwave,
		&nulldouble, vis2->record[irow-1].vis2err, &anynull,
		status);
		fits_get_colnum(fptr, CASEINSEN, "UCOORD", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, 1, &nulldouble,
		&vis2->record[irow-1].ucoord, &anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "VCOORD", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, 1, &nulldouble,
		&vis2->record[irow-1].vcoord, &anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "STA_INDEX", &colnum, status);
		fits_read_col(fptr, TINT, colnum, irow, 1, 2, &nullint,
		vis2->record[irow-1].sta_index, &anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "FLAG", &colnum, status);
		fits_read_col(fptr, TLOGICAL, colnum, irow, 1, vis2->nwave, &nullchar,
		vis2->record[irow-1].flag, &anynull, status);
	}
	if (*status) {
		fprintf(stderr, "CFITSIO error in %s:\n", function);
		fits_report_error(stderr, *status);
	}
	return *status;
}


/**
* Read next OI_T3 fits binary table
*
*   @param fptr    see cfitsio documentation
*   @param t3      ptr to data struct, see exchange.h
*   @param status  pointer to status variable
*
*   @return On error, returns non-zero cfitsio error code (also assigned to
*           *status). Contents of data struct are undefined
*/
int read_next_oi_t3(fitsfile *fptr, oi_t3 *t3, int *status) {
	
	const char function[] = "read_oi_t3";
	char comment[FLEN_COMMENT], extname[FLEN_VALUE];
	char nullchar = 0;
	int nullint = 0;
	double nulldouble = 0.0;
	const int revision = 1;
	int irow, colnum, anynull, hdutype;
	long repeat;
	
	if (*status) return *status; /* error flag set - do nothing */
	
	/* Move to correct HDU - don't assume anything about EXTVERs */
	while (1==1) {
		fits_movrel_hdu(fptr, 1, &hdutype, status);
		if (*status) return *status; /* no more HDUs */
		fits_read_key(fptr, TSTRING, "EXTNAME", extname, comment, status);
		if (strcmp(extname, "OI_T3") == 0) break; /* current HDU matches */
	}
	
	/* Read table */
	fits_read_key(fptr, TINT, "OI_REVN", &t3->revision, comment, status);
	if (t3->revision != revision) {
		printf("WARNING! Expecting value %d for OI_REVN keyword in OI_T3 table. Got %d\n", revision, t3->revision);
	}
	fits_read_key(fptr, TSTRING, "DATE-OBS", t3->date_obs, comment, status);
	fits_read_key(fptr, TSTRING, "ARRNAME", t3->arrname, comment, status);
	if (*status == KEY_NO_EXIST) { /* ARRNAME is optional */
		t3->arrname[0] = '\0';
		*status = 0;
	}
	fits_read_key(fptr, TSTRING, "INSNAME", t3->insname, comment, status);
	/* get number of rows & allocate storage */
	fits_get_num_rows(fptr, &t3->numrec, status);
	t3->record = malloc(t3->numrec*sizeof(oi_t3_record));
	/* get value for nwave */
	/* format specifies same repeat count for VIS2DATA & VIS2ERR columns */
	fits_get_colnum(fptr, CASEINSEN, "T3AMP", &colnum, status);
	fits_get_coltype(fptr, colnum, NULL, &repeat, NULL, status);
	t3->nwave = repeat;
	/* read rows */
	for (irow=1; irow<=t3->numrec; irow++) {
		fits_get_colnum(fptr, CASEINSEN, "TARGET_ID", &colnum, status);
		fits_read_col(fptr, TINT, colnum, irow, 1, 1, &nullint,
		&t3->record[irow-1].target_id, &anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "TIME", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, 1, &nulldouble,
		&t3->record[irow-1].time, &anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "MJD", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, 1, &nulldouble,
		&t3->record[irow-1].mjd, &anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "INT_TIME", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, 1, &nulldouble,
		&t3->record[irow-1].int_time, &anynull, status);
		t3->record[irow-1].t3amp = malloc(t3->nwave*sizeof(double));
		t3->record[irow-1].t3amperr = malloc(t3->nwave*sizeof(double));
		t3->record[irow-1].t3phi = malloc(t3->nwave*sizeof(double));
		t3->record[irow-1].t3phierr = malloc(t3->nwave*sizeof(double));
		t3->record[irow-1].flag = malloc(t3->nwave*sizeof(char));
		fits_get_colnum(fptr, CASEINSEN, "T3AMP", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, t3->nwave,
		&nulldouble, t3->record[irow-1].t3amp, &anynull,
		status);
		fits_get_colnum(fptr, CASEINSEN, "T3AMPERR", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, t3->nwave,
		&nulldouble, t3->record[irow-1].t3amperr, &anynull,
		status);
		fits_get_colnum(fptr, CASEINSEN, "T3PHI", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, t3->nwave,
		&nulldouble, t3->record[irow-1].t3phi, &anynull,
		status);
		fits_get_colnum(fptr, CASEINSEN, "T3PHIERR", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, t3->nwave,
		&nulldouble, t3->record[irow-1].t3phierr, &anynull,
		status);
		fits_get_colnum(fptr, CASEINSEN, "U1COORD", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, 1,
		&nulldouble, &t3->record[irow-1].u1coord, &anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "V1COORD", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, 1, &nulldouble,
		&t3->record[irow-1].v1coord, &anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "U2COORD", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, 1, &nulldouble,
		&t3->record[irow-1].u2coord, &anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "V2COORD", &colnum, status);
		fits_read_col(fptr, TDOUBLE, colnum, irow, 1, 1, &nulldouble,
		&t3->record[irow-1].v2coord, &anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "STA_INDEX", &colnum, status);
		fits_read_col(fptr, TINT, colnum, irow, 1, 3, &nullint,
		t3->record[irow-1].sta_index, &anynull, status);
		fits_get_colnum(fptr, CASEINSEN, "FLAG", &colnum, status);
		fits_read_col(fptr, TLOGICAL, colnum, irow, 1, t3->nwave, &nullchar,
		t3->record[irow-1].flag, &anynull, status);
	}
	if (*status) {
		fprintf(stderr, "CFITSIO error in %s:\n", function);
		fits_report_error(stderr, *status);
	}
	return *status;
}
