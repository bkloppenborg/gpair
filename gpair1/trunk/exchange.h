/* Example code for FITS-based optical interferometry exchange format

   $Id: exchange.h,v 1.8 2006/03/22 14:51:55 jsy1001 Exp $
   Definitions of data structures, function prototypes


   Release 8  22 March 2006

   John Young <jsy1001@cam.ac.uk>

*/

#include "fitsio.h"


/* Data structures */
/* NB must allow for final null when dimensioning character arrays */

/** Array element. Corresponds to one row of an OI_ARRAY FITS table. */
typedef struct _element {
  char tel_name[17];
  char sta_name[17];
  int sta_index;
  float diameter;
  double staxyz[3];
} element;

/** Data for OI_ARRAY FITS table */
typedef struct _oi_array {
  int revision;
  char arrname[FLEN_VALUE];
  char frame[FLEN_VALUE];
  double arrayx, arrayy, arrayz;
  int nelement;
  element *elem;
} oi_array;

/** Info on an observing target.
 *
 * Corresponds to one row of an OI_TARGET FITS table.
 */
typedef struct _target {
  int target_id;
  char target[17];
  double raep0, decep0;
  float equinox;
  double ra_err, dec_err;
  double sysvel;
  char veltyp[9], veldef[9];
  double pmra, pmdec;
  double pmra_err, pmdec_err;
  float parallax, para_err;
  char spectyp[17];
} target;

/** Data for OI_TARGET FITS table */
typedef struct _oi_target {
  int revision;
  int ntarget;
  target *targ;
} oi_target;

/** Data for OI_WAVELENGTH FITS table */
typedef struct _oi_wavelength {
  int revision;
  char insname[FLEN_VALUE];
  int nwave;
  float *eff_wave;
  float *eff_band;
} oi_wavelength;

/** Complex visibility record. Corresponds to one row of an OI_VIS FITS table. */
typedef struct _oi_vis_record {
  int target_id;
  double time;
  double mjd;
  double int_time;
  double *visamp, *visamperr;
  double *visphi, *visphierr;
  double ucoord, vcoord;
  int sta_index[2];
  char *flag;
} oi_vis_record;

/** Data for OI_VIS FITS table */
typedef struct _oi_vis {
  int revision;
  char date_obs[FLEN_VALUE];
  char arrname[FLEN_VALUE]; /* empty string "" means not specified */
  char insname[FLEN_VALUE];
  long numrec;
  int nwave;
  oi_vis_record *record;
} oi_vis;

/** Visibility squared record. Corresponds to one row of an OI_VIS2 FITS table. */
typedef struct _oi_vis2_record {
  int target_id;
  double time;
  double mjd;
  double int_time;
  double *vis2data, *vis2err;
  double ucoord, vcoord;
  int sta_index[2];
  char *flag;
} oi_vis2_record;

/** Data for OI_VIS2 FITS table */
typedef struct _oi_vis2 {
  int revision;
  char date_obs[FLEN_VALUE];
  char arrname[FLEN_VALUE]; /* empty string "" means not specified */
  char insname[FLEN_VALUE];
  long numrec;
  int nwave;
  oi_vis2_record *record;
} oi_vis2;

/** Triple product record. Corresponds to one row of an OI_T3 FITS table. */
typedef struct _oi_t3_record {
  int target_id;
  double time;
  double mjd;
  double int_time;
  double *t3amp, *t3amperr;
  double *t3phi, *t3phierr;
  double u1coord, v1coord, u2coord, v2coord;
  int sta_index[3];
  char *flag;
} oi_t3_record;

/** Data for OI_T3 FITS table */
typedef struct _oi_t3 {
  int revision;
  char date_obs[FLEN_VALUE];
  char arrname[FLEN_VALUE]; /* empty string "" means not specified */
  char insname[FLEN_VALUE];
  long numrec;
  int nwave;
  oi_t3_record *record;
} oi_t3;


/* Function prototypes */

/* Functions from write_oi_fits.c */
int write_oi_array(fitsfile *fptr, oi_array array, int extver, int *status);
int write_oi_target(fitsfile *fptr, oi_target targets, int *status);
int write_oi_wavelength(fitsfile *fptr, oi_wavelength wave, int extver, int *status);
int write_oi_vis(fitsfile *fptr, oi_vis vis, int extver, int *status);
int write_oi_vis2(fitsfile *fptr, oi_vis2 vis2, int extver, int *status);
int write_oi_t3(fitsfile *fptr, oi_t3 t3, int extver, int *status);
/* Functions from read_oi_fits.c */
int read_oi_array(fitsfile *fptr, char *arrname, oi_array *array, int *status);
int read_oi_target(fitsfile *fptr, oi_target *targets, int *status);
int read_oi_wavelength(fitsfile *fptr, char *insname, oi_wavelength *wave, int *status);
int read_next_oi_vis(fitsfile *fptr, oi_vis *vis, int *status);
int read_next_oi_vis2(fitsfile *fptr, oi_vis2 *vis2, int *status);
int read_next_oi_t3(fitsfile *fptr, oi_t3 *t3, int *status);
