/* Header for oifits routines.
 * This package uses the Oifits Exchange routines by John Young to view,
 * select and extract oi data.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <string.h>
#include "exchange.h"
#include "fitsio.h"

#define maxins 50
#define billion (1.0e9)
#define PARAM_NUMBER 8

//#include <nfft3.h>

typedef struct
{
  int itype;
  double init[ PARAM_NUMBER ]; // backup parameters
  double parameters[ PARAM_NUMBER ];
  double priors[ PARAM_NUMBER ];
  float *img; // bitmap fits image - float type needed for most fits files
} MODL; // defines one component of a MODL

typedef struct _usersel
{
  char file[FLEN_FILENAME];
  int target_id;
  char target[FLEN_VALUE];
  long numvis2;
  long numt3;
  double wavel;
  char insname[maxins][FLEN_VALUE];
  long numins;
  double minband;
  double maxband;
  int ntelescopes;
}oi_usersel;

typedef struct _uvpnt
{
  short sign;
  long uvpnt;
}oi_uvpnt;

typedef struct _bsref
{
  /* Structure for bisp to uv coord table referencing.
   * Negative uv table number means the particular baseline is conjugated.
   */
  oi_uvpnt ab;
  oi_uvpnt bc;
  oi_uvpnt ca;
}oi_bsref;

typedef struct _uv
{
  double u;
  double v;
}oi_uv;

typedef struct _data
{
  double *pow;
  double *powerr;
  double *bisamp;
  double *bisphs;
  double *bisamperr;
  double *bisphserr;
  double *time;
  oi_uv *uv;
  oi_bsref *bsref;
  int npow;
  int nbis;
  int nuv;
}oi_data;

typedef struct _dataonly
{
  double *pow;
  double complex *t3;
  int nbis;
  int npow;
}data_only;

typedef struct{  /* This user-defined context contains application specific information. */ 
  // MEMConverge mc;  /* Stopping parameters and statistics */
	int Ndata;
	double complex *DFT_table;    // I  DFT table
  // double *model; // prior image
	int npixels;
	double xyint;
  // oi_usersel usersel;
	oi_data data;
  // nfft_plan p;
} User;

/* Constants */
#define PI 3.14159265358979323
#define RPMAS (3.14159265358979323/180.0)/3600000.0
#define NMOD 20

/* Function declarations */
int get_oi_fits_selection(oi_usersel *usersel, int* status);
int get_oi_fits_data(oi_usersel usersel, oi_data *data, int* status);
int compare_uv(oi_uv uv, oi_uv withuv, double thresh);
void free_oi_target(oi_target *targets);
void free_oi_wavelength(oi_wavelength *wave);
void free_oi_vis2(oi_vis2 *vis2);
void free_oi_t3(oi_t3 *t3);
void free_oi_data(oi_data *data);
int count_redundant_bsuv(oi_bsref *bsref, int nbs);
double bsuv_coverage_quality(oi_bsref *bsref, int nbs, oi_uv *uv, int nuv);
void write_fits_image( double* image , int* status);
void read_fits_image(char* fname, double* img, int* n, int* status);
