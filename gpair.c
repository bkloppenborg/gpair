#include "gpair.h"
#include "getoifits.h"

// Global variables
int model_image_size;
float model_image_pixellation;
int status;
oi_usersel usersel;
oi_data oifits_info; // stores all the info from oifits files

float *mock; // stores the mock current pseudo-data derived from the image
float *data; // stores the quantities derived from the data
float *err; // stores the error bars on the data
float complex *bisphasor; // bispectrum rotation precomputed value

float complex *visi; // current visibilities
float *current_image;

// DFT precomputed coefficient tables
float complex* DFT_tablex;
float complex* DFT_tabley;

int npow;
int nbis;
int nuv;

float square( float number )
{
    return number*number;
}

int main(int argc, char *argv[])
{   
  register int ii, uu;
  float chi2;
 
  // read OIFITS and visibilities
  read_oifits(&oifits_info);
  
  // setup visibility/current data matrices

  npow = oifits_info.npow;
  nbis = oifits_info.nbis;
  nuv = oifits_info.nuv;

  visi = malloc( nuv * sizeof( float complex));
  mock = malloc( (npow + 2 * nbis) * sizeof( float )); // 0:npow-1 == powerspectrum data  npow:npow+2*nbis-1 == bispectrum real/imag
  data = malloc( (npow + 2 * nbis) * sizeof( float ));
  err =  malloc( (npow + 2 * nbis) * sizeof( float ));
  bisphasor = malloc( nbis * sizeof( float complex ));

  for(ii=0; ii < npow; ii++)
    {
      data[ii] = oifits_info.pow[ii];
      err[ii]=  oifits_info.powerr[ii];
    }

  for(ii = 0; ii < nbis; ii++)
    {
      bisphasor[ii] = cexp( - I * oifits_info.bisphs[ii] );
      data[npow + 2* ii] = oifits_info.bisamp[ii];
      data[npow + 2* ii + 1 ] = 0.;
      err[npow + 2* ii]=  oifits_info.bisamperr[ii];
      err[npow + 2* ii + 1] = oifits_info.bisamp[ii] * oifits_info.bisphserr[ii]  ;      
    }

  // setup initial image as 128x128 pixel, centered Dirac of flux = 1.0, pixellation of 1.0 mas/pixel
  int npix = 128;
  float model_image_pixellation = 0.15 ;
  current_image = (float*)malloc(npix*npix*sizeof(float));
  for(ii=0; ii< (npix * npix - 1); ii++)
    {
      current_image[ ii ]= 0. ;
    }  
  current_image[(npix * (npix + 1 ) )/ 2 ] = 1.0;

  
  // setup precomputed DFT table
  DFT_tablex = malloc( nuv * model_image_size * sizeof(float complex));
  DFT_tabley = malloc( nuv * model_image_size * sizeof(float complex));
  for(uu=0 ; uu < nuv; uu++)
    {
      for(ii=0; ii < model_image_size; ii++)
	{
	  DFT_tablex[ model_image_size * uu + ii ] =  
	    cexp( - 2.0 * I * PI * RPMAS * model_image_pixellation * oifits_info.uv[uu].u * (float)ii )  ;
	  DFT_tabley[ model_image_size * uu + ii ] =  
	    cexp( - 2.0 * I * PI * RPMAS * model_image_pixellation * oifits_info.uv[uu].v * (float)ii )  ;
	}
    }
  
  //compute complex visibilities 
  image2vis();

  // compute mock data, powerspectra + bispectra
  vis2data( );
  
  // compute reduced chi2
  chi2 = data2chi2( )/(float)( npow + 2 * nbis);
  printf("Reduced chi2 = %f\n", chi2);

  return 1;

}

int read_oifits()
{
  // Read the image
  strcpy(usersel.file, "./2004contest1.oifits" );
  get_oi_fits_selection( &usersel , &status );
  get_oi_fits_data( usersel , &oifits_info , &status );
  printf("OIFITS File read\n");  
  printf("There are %d data : %d powerspectrum and %d bispectrum\n", npow + 2 * nbis, npow, nbis);
  return 1;
}

void image2vis( )
{	
  // DFT
  int ii, jj, uu;	
  float v0 = 0.; // zeroflux 
  
  for(ii=0 ; ii < model_image_size * model_image_size ; ii++) 
    v0 += current_image[ii];

  for(uu=0 ; uu < nuv; uu++)
    {
      visi[uu] = 0.0 + I * 0.0;
      for(ii=0; ii < model_image_size; ii++)
	  for(jj=0; jj < model_image_size; jj++)
	    visi[uu] += current_image[ ii + model_image_size * jj ] *  DFT_tablex[ model_image_size * uu +  ii] * DFT_tablex[ model_image_size * uu +  jj];
      if (v0 > 0.) visi[uu] /= v0;
    }
  
  printf("Check - visi 0 %f %f\n", creal(visi[0]), cimag(visi[0]));
}

void vis2data(  )
{
  int ii;
  float complex vab, vbc, vca, t3;
  
  for( ii = 0; ii< npow; ii++)
    {
    mock[ ii ] = square ( cabs( visi[ii] ) );
    }

  for( ii = 0; ii< nbis; ii++)
    {
      vab = visi[ oifits_info.bsref[ii].ab.uvpnt ];
      vbc = visi[ oifits_info.bsref[ii].bc.uvpnt ];
      vca = visi[ oifits_info.bsref[ii].ca.uvpnt ];	
      if( oifits_info.bsref[ii].ab.sign < 0) vab = conj(vab);
      if( oifits_info.bsref[ii].bc.sign < 0) vbc = conj(vbc);
      if( oifits_info.bsref[ii].ca.sign < 0) vca = conj(vca);
      t3 =  ( vab * vbc * vca ) * bisphasor[ii] ;   
      mock[ npow + 2 * ii ] = creal(t3) ;
      mock[ npow + 2 * ii + 1] = cimag(t3) ;
    } 

}

float data2chi2( )
{
  float chi2 = 0.;
  register int ii;  
  for(ii=0; ii< npow + 2 * nbis; ii++)
    {
      chi2 += square( ( mock[ii] - data[ii] ) / err[ii] ) ;
    }

  return chi2;
}


void write_fits_image( float* image , int* status )
{
  fitsfile *fptr;
  int i;
  long fpixel = 1, naxis = 2, nelements;
  long naxes[ 2 ];
  char fitsimage[ 100 ];
  for (i = 0; i < 100; i++)
    fitsimage[ i ] = '\0';
  /*Initialise storage*/
  naxes[ 0 ] = (long) model_image_size;
  naxes[ 1 ] = (long) model_image_size;
  nelements = naxes[ 0 ] * naxes[ 1 ];
  strcpy(fitsimage, "!output.fits");

  /*Create new file*/
  if (*status == 0)
    fits_create_file(&fptr, fitsimage, status);

  /*Create primary array image*/
  if (*status == 0)
    fits_create_img(fptr, FLOAT_IMG, naxis, naxes, status); 
  /*Write a keywords (datafile, target, image pixelation) */
  if (*status == 0)
    fits_update_key(fptr, TSTRING, "DATAFILE", "dummy", "Data File Name", status); 
  if (*status == 0)
    fits_update_key(fptr, TSTRING, "TARGET", "dummy", "Target Name", status); 
  if (*status == 0)
    fits_update_key(fptr, TFLOAT, "PIXSIZE", &model_image_pixellation, "Pixelation (mas)", status);
  if (*status == 0)
    fits_update_key(fptr, TINT, "WIDTH", &model_image_size, "Size (pixels)", status); 
 
  /*Write image*/
  if (*status == 0)
    fits_write_img(fptr, TFLOAT, fpixel, nelements, &image[ 0 ], status); 

  /*Close file*/
  if (*status == 0)
    fits_close_file(fptr, status);

  /*Report any errors*/
  fits_report_error(stderr, *status);

}
