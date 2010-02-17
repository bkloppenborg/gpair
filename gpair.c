#include "gpair.h"
#include "getoifits.h"

// Global variables
int model_image_size;
double model_image_pixellation;
oi_data data;
int status;
oi_usersel usersel;
oi_data data; // stores the data points from oifits files
data_only current; // stores the powerspectrum and bispectrum derived from the current model
double complex *visi; // current visibilities
double *current_image;
double complex* DFT_table;


double square( double number )
{
    return number*number;
}

int main(int argc, char *argv[])
{   
  register int ii, jj, uu;
  double chi2;
 
  // read OIFITS and visibilities
  read_oifits(&data);
  
  // setup visibility/current data matrices
  visi = malloc( data.nuv * sizeof( double complex));
  current.pow = malloc( data.npow * sizeof( double ));
  current.t3 = malloc( data.nbis * sizeof( double complex));   

  // setup initial image as 128x128 pixel, centered Dirac of flux = 1.0, pixellation of 1.0 mas/pixel
  int npix = 128;
  double model_image_pixellation = 0.15 ;
  current_image = (double*)malloc(npix*npix*sizeof(double));
  for(ii=0; ii< (npix * npix - 1); ii++)
    {
      current_image[ ii ]= 0. ;
    }  
  current_image[(npix * (npix + 1 ) )/ 2 ] = 1.0;

  
  // setup precomputed DFT table
  DFT_table = malloc( data.nuv * model_image_size * model_image_size * sizeof(double complex));
  for(uu=0 ; uu < data.nuv; uu++)
    for(ii=0; ii < model_image_size; ii++)
      for(jj=0; jj < model_image_size; jj++)
	    DFT_table[ model_image_size * model_image_size * uu + model_image_size * ii + jj ] =  
	  cexp( - 2.0 * I * PI * RPMAS * model_image_pixellation * ( data.uv[uu].u * (double) ii + data.uv[uu].v * (double)( jj ) ) )  ;

  //compute complex visibilities 
  image2vis();

  // compute mock data, powerspectra + bispectra
  vis2data( );
  
  // compute reduced chi2
  chi2 = data2chi2( )/(double)( data.npow + 2.*data.nbis);
  printf("Reduced chi2 = %f\n", chi2);

  return 1;

}

int read_oifits()
{
  // Read the image
  strcpy(usersel.file, "./2004contest1.oifits" );
  get_oi_fits_selection( &usersel , &status );
  get_oi_fits_data( usersel , &data , &status );
  printf("OIFITS File read\n");  
  printf("There are %d data : %d powerspectrum and %d bispectrum\n", data.npow + 2 * data.nbis, data.npow, data.nbis);
  return 1;
}

void image2vis( )
{	
  // DFT
  int ii, jj, uu;	
  double v0 = 0.; // zeroflux 
  
  for(ii=0 ; ii < model_image_size * model_image_size ; ii++) 
    v0 += current_image[ii];
  printf("v0 == %f\n", v0);
  printf("%f %f %f\n", data.uv[0].u, data.uv[0].v, model_image_pixellation);
  for(uu=0 ; uu < data.nuv; uu++)
    {
      visi[uu] = 0.0 + I * 0.0;
      for(ii=0; ii < model_image_size; ii++)
	  for(jj=0; jj < model_image_size; jj++)
		   visi[uu] += current_image[ ii + model_image_size * jj ] *  DFT_table[ model_image_size * model_image_size * uu + model_image_size * ii + jj ]
      if (v0 > 0.) visi[uu] /= v0;
    }
  
  printf("Check - visi 0 %f %f\n", creal(visi[0]), cimag(visi[0]));
}

void vis2data(  )
{
  int ii;
  double complex vab, vbc, vca;
  
  for( ii = 0; ii< data.npow; ii++)
    {
    current.pow[ ii ] = square ( cabs( visi[ii] ) );
    }


  for( ii = 0; ii< data.nbis; ii++)
    {
      vab = visi[ data.bsref[ii].ab.uvpnt ];
      vbc = visi[ data.bsref[ii].bc.uvpnt ];
      vca = visi[ data.bsref[ii].ca.uvpnt ];	
      if( data.bsref[ii].ab.sign < 0) vab = conj(vab);
      if( data.bsref[ii].bc.sign < 0) vbc = conj(vbc);
      if( data.bsref[ii].ca.sign < 0) vca = conj(vca);
      current.t3[ ii ] = vab * vbc * vca ;
    } 
}

double data2chi2( )
{
  double chi2 = 0., chi2v2 = 0., chi2bs = 0. ;
  double complex t3;
  //  double t3r, t3i;
  register int ii;
  
  for(ii=0; ii< data.npow ; ii++)
    {
      chi2v2 += square( ( current.pow[ii] - data.pow[ii] ) / data.powerr[ii] ) ;
    }
 
  for(ii =0; ii < data.nbis ; ii++)
    {
      
      t3 = ( current.t3[ ii ] * cexp( - I * data.bisphs[ii] )   - data.bisamp[ii] ) ;
      t3 = creal(t3) / data.bisamperr[ii] + I * cimag(t3) / ( data.bisamp[ii] * data.bisphserr[ii] );
      chi2bs += square( cabs(t3 ) );
      // alternative
      // data.t3ierr[ii] = data.bisamp[ii] * data.bisphserr[ii];
      // data.t3rerr[ii] = data.bisamperr[ii];
      //  data.t3phasor[ii] = cexp( - I * data.bisphs[ii] )
      // t3= ( current.t3[ ii ] * data.t3phasor[ ii ]   - data.bisamp[ii] ) ;
      // t3r = creal(t3) / data.realerr[ii];
      // t3i = cimag(t3) / data.imagerr[ii];
      // chi2bs += t3r * t3r + t3i * t3i;
    } 
  
  chi2 = chi2v2 + chi2bs;
  return chi2/(double)( data.npow + 2.*data.nbis);
}

void write_fits_image( double* image , int* status )
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
    fits_create_img(fptr, DOUBLE_IMG, naxis, naxes, status); 
  /*Write a keywords (datafile, target, image pixelation) */
  if (*status == 0)
    fits_update_key(fptr, TSTRING, "DATAFILE", "dummy", "Data File Name", status); 
  if (*status == 0)
    fits_update_key(fptr, TSTRING, "TARGET", "dummy", "Target Name", status); 
  if (*status == 0)
    fits_update_key(fptr, TDOUBLE, "PIXSIZE", &model_image_pixellation, "Pixelation (mas)", status);
  if (*status == 0)
    fits_update_key(fptr, TINT, "WIDTH", &model_image_size, "Size (pixels)", status); 
 
  /*Write image*/
  if (*status == 0)
    fits_write_img(fptr, TDOUBLE, fpixel, nelements, &image[ 0 ], status); 

  /*Close file*/
  if (*status == 0)
    fits_close_file(fptr, status);

  /*Report any errors*/
  fits_report_error(stderr, *status);

}
