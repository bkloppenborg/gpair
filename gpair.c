#include "gpair.h"
#include "getoifits.h"
#include <time.h>
#include "gpu.h"

// Global variables
int model_image_size;
float model_image_pixellation;
int status;
oi_usersel usersel;
oi_data oifits_info; // stores all the info from oifits files

float *mock_data; // stores the mock current pseudo-data derived from the image
float *data; // stores the quantities derived from the data
float *data_err; // stores the error bars on the data
float complex *data_bis; // bispectrum rotation precomputed value

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
    
    // First thing we do is make all data occupy memory elements that are powers of two.  This makes
    // the GPU code much easier to speed up.
    int data_size = npow + 2 * nbis;
    int data_alloc = pow(2, ceil(log(data_size) / log(2)));    // Arrays are allocated to be powers of 2
    int data_alloc_uv = pow(2, ceil(log(nuv) / log(2)));
    int data_alloc_bis = pow(2, ceil(log(nbis) / log(2)));   
   
    // TODO: Only output if we are in a verbose mode.
    printf("Data Size: %i , Data Allocation: %i \n", data_size, data_alloc);
    printf("UV Data Size: %i , Allocation: %i \n", nuv, data_alloc_uv);
    printf("bis Size: %i , Allocation: %i \n", nbis, data_alloc_bis);

    // Allocate memory for the data, error, and mock arrays:
    data = malloc(data_alloc * sizeof( float ));
    data_err =  malloc(data_alloc * sizeof( float ));
    visi = malloc(data_alloc_uv * sizeof( float complex));   
    data_bis = malloc(data_alloc_bis * sizeof( float complex ));
    mock_data = malloc(data_alloc * sizeof( float ));

         
    // Set elements [0, npow - 1] equal to the power spectrum
    for(ii=0; ii < npow; ii++)
    {
        data[ii] = oifits_info.pow[ii];
        data_err[ii]=  1 / oifits_info.powerr[ii];
        mock_data[ii] = 0;
    }

    // Let j = npow, set elements [j, j + nbis - 1] to the powerspectrum data.
    for(ii = 0; ii < nbis; ii++)
    {
        data_bis[ii] = cexp( - I * oifits_info.bisphs[ii] );
        data[npow + 2* ii] = oifits_info.bisamp[ii];
        data[npow + 2* ii + 1 ] = 0.;
        data_err[npow + 2* ii]=  1 / oifits_info.bisamperr[ii];
        data_err[npow + 2* ii + 1] = 1 / (oifits_info.bisamp[ii] * oifits_info.bisphserr[ii]);
        mock_data[npow + 2* ii] = 0;
        mock_data[npow + 2* ii + 1] = 0;
    }
    
    // Pad the arrays with zeros and ones after this
    for(ii = data_size; ii < data_alloc; ii++)
    {
        data[ii] = 0;
        data_err[ii] = 0;
        mock_data[ii] = 0;
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

    // TODO: Remove after testing
    int iterations = 10000;

    // compute mock data, powerspectra + bispectra
    clock_t tick, tock;

    vis2data( );

    
    tick = clock();
    // TODO: Remove after testing
    //for(ii=0; ii < iterations; ii++)
        chi2 = data2chi2( );
        
    tock=clock();
    float cpu_time_chi2 = (float)(tock - tick) / (float)CLOCKS_PER_SEC;

    // GPU Code:  
    
    // Convert visi over to a cl_float2 in format <real, imaginary>
    cl_float2 * gpu_visi;
    gpu_visi = malloc(data_alloc_uv * sizeof(cl_float2));
    int i;
    for(i = 0; i < nuv; i++)
    {
        gpu_visi[i][0] = __real__ visi[i];
        gpu_visi[i][1] = __imag__ visi[i];
    }
    // Pad the remainder
    for(i = nuv; i < data_alloc_uv; i++)
    {
        gpu_visi[i][0] = 0;
        gpu_visi[i][1] = 0;
    }    
    
    // Convert the biphasor over to a cl_float2 in format <real, imaginary>    
    cl_float2 * gpu_bis;
    gpu_bis = malloc(data_alloc_bis * sizeof(cl_float2));
    for(i = 0; i < nbis; i++)
    {
        gpu_bis[i][0] = __real__ data_bis[i];
        gpu_bis[i][1] = __imag__ data_bis[i];
    }
    // Pad the remainder
    for(i = nbis; i < data_alloc_bis; i++)
    {
        gpu_bis[i][0] = 0;
        gpu_bis[i][1] = 0;
    }
    
    // We will also need the uvpnt and sign information for bisepctrum computations.
    cl_long * gpu_bsref_uvpnt;
    cl_short * gpu_bsref_sign;
    int data_alloc_bsref = 3 * data_alloc_bis;
    gpu_bsref_uvpnt = malloc(data_alloc_bsref * sizeof(cl_long));
    gpu_bsref_sign = malloc(data_alloc_bsref * sizeof(cl_short));
    for(i = 0; i < nbis; i++)
    {
        gpu_bsref_uvpnt[3*i] = oifits_info.bsref[i].ab.uvpnt;
        gpu_bsref_uvpnt[3*i+1] = oifits_info.bsref[i].bc.uvpnt;
        gpu_bsref_uvpnt[3*i+2] = oifits_info.bsref[i].ca.uvpnt;

        gpu_bsref_sign[3*i] = oifits_info.bsref[i].ab.sign;
        gpu_bsref_sign[3*i+1] = oifits_info.bsref[i].bc.sign;
        gpu_bsref_sign[3*i+2] = oifits_info.bsref[i].ca.sign;
    }        
        
    
    float chi2_gpu = 0;
    gpu_init();
    gpu_copy_data(data, data_err, data_alloc, gpu_bis, data_alloc_bis, gpu_bsref_uvpnt, gpu_bsref_sign, data_alloc_bsref);  
    gpu_build_kernels();
    gpu_build_reduce_kernels(data_alloc);   

    gpu_vis2data(gpu_visi, nuv, npow, nbis);
    
    tick = clock();
    // TODO: Remove after testing
    //for(ii=0; ii < iterations; ii++)
        gpu_data2chi2(data_alloc);
    
    tock = clock();
    float gpu_time_chi2 = (float)(tock - tick) / (float)CLOCKS_PER_SEC;
    printf("-----------------------------------------------------------\n");
    printf("Chi2 Computations:");
    printf("Chi2 (CPU): %f \n", chi2);
    printf("CPU time (s): = %E\n", cpu_time_chi2);
    printf("GPU time (s): = %E\n", gpu_time_chi2);
    
    
    gpu_cleanup();
    
    return 0;

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
        mock_data[ ii ] = square ( cabs( visi[ii] ) );
    }

    for( ii = 0; ii< nbis; ii++)
    {
        vab = visi[ oifits_info.bsref[ii].ab.uvpnt ];
        vbc = visi[ oifits_info.bsref[ii].bc.uvpnt ];
        vca = visi[ oifits_info.bsref[ii].ca.uvpnt ];	
        if( oifits_info.bsref[ii].ab.sign < 0) 
            vab = conj(vab);
        if( oifits_info.bsref[ii].bc.sign < 0) 
            vbc = conj(vbc);
        if( oifits_info.bsref[ii].ca.sign < 0) 
            vca = conj(vca);
            
        t3 =  ( vab * vbc * vca ) * data_bis[ii] ;   
        mock_data[ npow + 2 * ii ] = creal(t3) ;
        mock_data[ npow + 2 * ii + 1] = cimag(t3) ;
    } 
    
    // Uncomment to see the mock data array.
/*    int count = npow + 2 * nbis;*/
/*    for(ii = 0; ii < count; ii++)     */
/*        printf("%i %f \n", ii, mock[ii]);*/
/*        */
/*    printf("\n");*/

}

float data2chi2( )
{
    float chi2 = 0.;
    register int ii;  
    for(ii=0; ii< npow + 2 * nbis; ii++)
    {
        chi2 += square( ( mock_data[ii] - data[ii] ) * data_err[ii] ) ;
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
