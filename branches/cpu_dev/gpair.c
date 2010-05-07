#include "gpair.h"
#include <time.h>
#include "cpu.h"

#ifndef GETOIFITS_H
#include "getoifits.h"
#endif

#ifndef DATA_TYPES_H
#include "data_types.h"
#endif

// Preprocessor directive for the GPU:
#ifdef USE_GPU
#include "gpu.h"
#endif

#define SEP "-----------------------------------------------------------\n"

// Global variables
int image_width = 0;
float image_pixellation = 0;
int status = 0;
oi_usersel usersel;
oi_data oifits_info; // stores all the info from oifits files


float * data = NULL; // stores the quantities derived from the data
float * data_err = NULL; // stores the error bars on the data
float complex *data_phasor = NULL; // bispectrum rotation precomputed value
float * data_gradient = NULL;

// DFT precomputed coefficient tables
float complex* DFT_tablex = NULL;
float complex* DFT_tabley = NULL;

int npow = 0;
int nbis = 0;
int nuv = 0;

int main(int argc, char *argv[])
{   
    // TODO: GPU Memory Allocations simply round to the nearest power of two, make this more intelligent
    // by rounding to the nearest sum of powers of 2 (if it makes sense).
    register int ii = 0;
    register int uu = 0;
    float chi2 = 0;

    // read OIFITS and visibilities
    read_oifits(&oifits_info);

    // setup visibility/current data matrices

    npow = oifits_info.npow;
    nbis = oifits_info.nbis;
    nuv = oifits_info.nuv;
    printf("There are %d data : %d powerspectrum and %d bispectrum\n", npow + 2 * nbis, npow, nbis);

    // First thing we do is make all data occupy memory elements that are powers of two.  This makes
    // the GPU code much easier to speed up.
    int data_size = npow + 2 * nbis;
    int data_alloc = pow(2, ceil(log(data_size) / log(2)));    // Arrays are allocated to be powers of 2
    int data_alloc_uv = pow(2, ceil(log(nuv) / log(2)));
    int data_alloc_phasor = pow(2, ceil(log(nbis) / log(2)));   

    // Allocate memory for the data, error, and mock arrays:
    data = malloc(data_alloc * sizeof( float ));
    data_err =  malloc(data_alloc * sizeof( float ));
    data_phasor = malloc(data_alloc_phasor * sizeof( float complex ));
    
    float complex* visi = malloc(data_alloc_uv * sizeof( float complex)); // current visibilities 
    float complex * new_visi= malloc(data_alloc_uv * sizeof( float complex)); // tentative visibilities  
    float* mock = malloc(data_alloc * sizeof( float )); // stores the mock current pseudo-data derived from the image

         
    // Set elements [0, npow - 1] equal to the power spectrum
    for(ii=0; ii < npow; ii++)
    {
        data[ii] = oifits_info.pow[ii];
        data_err[ii]=  1 / oifits_info.powerr[ii];
        mock[ii] = 0;
    }

    // Let j = npow, set elements [j, j + nbis - 1] to the powerspectrum data.
    for(ii = 0; ii < nbis; ii++)
    {
        data_phasor[ii] = cexp( - I * oifits_info.bisphs[ii] );
        data[npow + 2* ii] = oifits_info.bisamp[ii];
        data[npow + 2* ii + 1 ] = 0.;
        data_err[npow + 2* ii] =  1 / oifits_info.bisamperr[ii];
        data_err[npow + 2* ii + 1] = 1 / (oifits_info.bisamp[ii] * oifits_info.bisphserr[ii]);
        mock[npow + 2* ii] = 0;
        mock[npow + 2* ii + 1] = 0;
    }
    
    // Pad the arrays with zeros and ones after this
    for(ii = data_size; ii < data_alloc; ii++)
    {
        data[ii] = 0;
        data_err[ii] = 0;
        mock[ii] = 0;
    }

    // setup initial image as 128x128 pixel, centered Dirac of flux = 1.0, pixellation of 1.0 mas/pixel
    image_width = 128;
    float image_pixellation = 0.15 ;
    int image_size = image_width * image_width;
    printf("Image Buffer Size %i \n", image_size);
    float* current_image = malloc(image_size * sizeof(float)); 
    memset(current_image, 0, image_size);
    current_image[(image_width * (image_width + 1 ) )/ 2 ] = 2.0;
    float * data_gradient = malloc(image_size * sizeof(float));


    // setup precomputed DFT table
    int dft_size = nuv * image_width;
    int dft_alloc = pow(2, ceil(log(dft_size) / log(2)));   // Amount of space to allocate on the GPU for each axis of the DFT table. 
    DFT_tablex = malloc( dft_size * sizeof(float complex));
    memset(DFT_tablex, 0, dft_size);
    DFT_tabley = malloc( dft_size * sizeof(float complex));
    memset(DFT_tabley, 0, dft_size);
    for(uu=0 ; uu < nuv; uu++)
    {
        for(ii=0; ii < image_width; ii++)
        {
            DFT_tablex[ image_width * uu + ii ] =  
                cexp( - 2.0 * I * PI * RPMAS * image_pixellation * oifits_info.uv[uu].u * (float)ii )  ;
            DFT_tabley[ image_width * uu + ii ] =  
                cexp( - 2.0 * I * PI * RPMAS * image_pixellation * oifits_info.uv[uu].v * (float)ii )  ;
        }
    }

    // TODO: Only output if we are in a verbose mode.
    printf("Data Size: %i, Data Allocation: %i \n", data_size, data_alloc);
    printf("UV Size: %i, Allocation: %i \n", nuv, data_alloc_uv);
    printf("POW Size: %i \n", npow);
    printf("BIS Size: %i, Allocation: %i \n", nbis, data_alloc_phasor);
    printf("DFT Size: %i , DFT Allocation: %i \n", dft_size, dft_alloc);

   
    // TODO: Remove after testing
    int iterations = 1;

    // #########
    // CPU Code:
    // #########

    chi2_info i2v_info;
    i2v_info.npow = npow;
    i2v_info.nbis = nbis;
    i2v_info.nuv = nuv;
    i2v_info.data = data;
    i2v_info.data_err = data_err;
    i2v_info.data_phasor = data_phasor;
    i2v_info.oifits_info = &oifits_info;
    i2v_info.dft_x = DFT_tablex;
    i2v_info.dft_y = DFT_tabley;
    i2v_info.visi = visi;
    i2v_info.mock = mock;
    i2v_info.image_width = image_width;
    i2v_info.image = current_image;

    // Test 1 : compute mock data, powerspectra + bispectra from scratch
    clock_t tick = clock();
    clock_t tock = 0;
    for(ii=0; ii < iterations; ii++)
    {
        chi2 = image2chi2(&i2v_info);
        
    }        
    tock=clock();
    float time_chi2 = (float)(tock - tick) / (float)CLOCKS_PER_SEC;
    printf(SEP);
    printf("Full DFT Calculation (CPU)\n");
    printf(SEP);
    printf("CPU time (s): = %f\n", time_chi2);
    printf("CPU Chi2: %f (CPU only)\n", chi2);
    
    // Test 2 : recompute mock data, powerspectra + bispectra when changing only the flux of one pixel
    tick = clock();
    float total_flux = compute_flux(image_width, current_image);
    float inc = 1.1;
    int x_changed = 64;
    int y_changed = 4;

    // Disabled for now, disagreement between CPU and GPU values.
/*    for(ii=0; ii < iterations; ii++)*/
/*    {*/
/*        //compute complex visibilities and the chi2*/
/*	    update_vis_fluxchange(x_changed, y_changed, current_image[ x_changed + y_changed * image_width ],  current_image[ x_changed + y_changed * image_width ] + inc, visi, new_visi ) ;*/
/*	    current_image[ x_changed + y_changed * image_width ] += inc;*/
/*	    total_flux += inc;*/
/*	    vis2data(new_visi, mock);*/
/*	    chi2 = data2chi2( mock );*/
/*    }       */
/*    tock=clock();*/
/*    time_chi2 = (float)(tock - tick) / (float)CLOCKS_PER_SEC;*/
/*    printf(SEP);*/
/*    printf("Atomic change (CPU)\n");*/
/*    printf(SEP);*/
/*    printf("CPU time (s): = %f\n", time_chi2);*/
/*    printf("CPU Chi2: %f (CPU only)\n", chi2);*/

    // Compute the gradient of the mock data.  Note, vis2data should have been caled before this call.
    compute_data_gradient(image_width, npow, nbis, &oifits_info, 
        data, data_err, data_phasor, 
        DFT_tablex, DFT_tabley, 
        visi, mock, current_image, data_gradient);

#ifdef USE_GPU
    // #########
    // GPU Code:  
    // #########
        
    // Convert visi over to a cl_float2 in format <real, imaginary>
    cl_float2 * gpu_visi = NULL;
    gpu_visi = malloc(data_alloc_uv * sizeof(cl_float2));
    int i;
    for(i = 0; i < nuv; i++)
    {
        gpu_visi[i].s0 = __real__ visi[i];
        gpu_visi[i].s1 = __imag__ visi[i];
    }
    // Pad the remainder
    for(i = nuv; i < data_alloc_uv; i++)
    {
        gpu_visi[i].s0 = 0;
        gpu_visi[i].s1 = 0;
    }    
    
    // Convert the biphasor over to a cl_float2 in format <real, imaginary>    
    cl_float2 * gpu_phasor = NULL;
    gpu_phasor = malloc(data_alloc_phasor * sizeof(cl_float2));
    for(i = 0; i < nbis; i++)
    {
        gpu_phasor[i].s0 = creal(data_phasor[i]);
        gpu_phasor[i].s1 = cimag(data_phasor[i]);
    }
    // Pad the remainder
    for(i = nbis; i < data_alloc_phasor; i++)
    {
        gpu_phasor[i].s0 = 0;
        gpu_phasor[i].s1 = 0;
    }
    
    // We will also need the uvpnt and sign information for bisepctrum computations.
    // Although we waste a little space, we use cl_long4 and cl_float4 so that we may have
    // coalesced loads on the GPU.
    cl_long4 * gpu_bsref_uvpnt = NULL;
    cl_short4 * gpu_bsref_sign = NULL;
    int data_alloc_bsref = data_alloc_phasor; // TODO: Update code below for this.
    gpu_bsref_uvpnt = malloc(data_alloc_bsref * sizeof(cl_long4));
    gpu_bsref_sign = malloc(data_alloc_bsref * sizeof(cl_short4));
    for(i = 0; i < nbis; i++)
    {
        gpu_bsref_uvpnt[i].s0 = oifits_info.bsref[i].ab.uvpnt;
        gpu_bsref_uvpnt[i].s1 = oifits_info.bsref[i].bc.uvpnt;
        gpu_bsref_uvpnt[i].s2 = oifits_info.bsref[i].ca.uvpnt;
        gpu_bsref_uvpnt[i].s3 = 0;

        gpu_bsref_sign[i].s0 = oifits_info.bsref[i].ab.sign;
        gpu_bsref_sign[i].s1 = oifits_info.bsref[i].bc.sign;
        gpu_bsref_sign[i].s2 = oifits_info.bsref[i].ca.sign;
        gpu_bsref_sign[i].s3 = 0;
    }    
    
    // Copy the DFT table over to a GPU-friendly format:
    cl_float2 * gpu_dft_x = NULL;
    cl_float2 * gpu_dft_y = NULL;
    gpu_dft_x = malloc(dft_alloc * sizeof(cl_float2));
    gpu_dft_y = malloc(dft_alloc * sizeof(cl_float2));
    for(uu=0 ; uu < nuv; uu++)
    {
        for(ii=0; ii < image_width; ii++)
        {
            i = image_width * uu + ii;
            gpu_dft_x[i].s0 = __real__ DFT_tablex[i];
            gpu_dft_x[i].s1 = __imag__ DFT_tablex[i];
            gpu_dft_y[i].s0 = __real__ DFT_tabley[i];
            gpu_dft_y[i].s1 = __imag__ DFT_tabley[i];
        }
    }

    // Pad out the remainder of the array with zeros:
    for(i = nuv * image_width; i < dft_alloc; i++)
    {
        gpu_dft_x[i].s0 = 0;
        gpu_dft_x[i].s1 = 0;
        gpu_dft_y[i].s0 = 0;
        gpu_dft_y[i].s1 = 0;
    }    
    
    // Initalize the GPU, copy data, and build the kernels.
    gpu_init();

    gpu_copy_data(data, data_err, data_alloc, data_alloc_uv, gpu_phasor, data_alloc_phasor,
        npow, gpu_bsref_uvpnt, gpu_bsref_sign, data_alloc_bsref, image_size,
        image_width);    
         
    gpu_build_kernels(data_alloc, image_size);
    gpu_copy_dft(gpu_dft_x, gpu_dft_y, dft_alloc);
    
    // Free variables used to store values pepared for the GPU
    free(gpu_visi);
    free(gpu_phasor);
    free(gpu_bsref_uvpnt);
    free(gpu_bsref_sign);
    free(gpu_dft_x);
    free(gpu_dft_y);
    
    // Do the full DFT calculation:
    tick = clock();
    for(ii=0; ii < iterations; ii++)
    {
        // In the final version of the code, the following lines will be iterated.
        gpu_copy_image(current_image, image_width, image_width);
        gpu_image2chi2(nuv, npow, nbis, data_alloc, data_alloc_uv);
    }
    tock = clock();
    time_chi2 = (float)(tock - tick) / (float)CLOCKS_PER_SEC;
    printf(SEP);
    printf("Full DFT (GPU)\n");
    printf(SEP);
    printf("GPU time (s): = %f\n", time_chi2);
    
    gpu_compute_data_gradient(npow, nbis, image_width);

    // Disabled for now, there be a bug between GPU and CPU values.
/*    // Now do the Atomic change to visi*/
/*    tick = clock();*/
/*    for(ii=0; ii < iterations; ii++)*/
/*    {*/
/*        gpu_update_vis_fluxchange(x_changed, y_changed, inc, image_width, nuv, data_alloc_uv);*/
/*        gpu_new_chi2(nuv, npow, nbis, data_alloc); */
/*    }       */
/*    tock=clock();*/
/*    time_chi2 = (float)(tock - tick) / (float)CLOCKS_PER_SEC;*/
/*    printf(SEP);*/
/*    printf("Atomic change (GPU)\n");*/
/*    printf(SEP);*/
/*    printf("GPU time (s): = %f\n", time_chi2);*/
    
    // Enable for debugging purposes.
    gpu_check_data(&chi2, nuv, visi, data_alloc, mock, image_size, data_gradient);
    
    // Cleanup, shutdown, were're done.
    gpu_cleanup();
    
#endif  // End of ifdef USE_GPU
    
    // Free CPU-based Memory
    free(mock);
    free(data);
    free(data_err);
    free(data_phasor);
    free(visi);
    free(current_image);
    free(DFT_tablex);
    free(DFT_tabley);
    
    return 0;

}

int read_oifits()
{
  // Read the image
  strcpy(usersel.file, "./2004contest1.oifits" );
  get_oi_fits_selection( &usersel , &status );
  get_oi_fits_data( usersel , &oifits_info , &status );
  printf("OIFITS File read\n");  
  return 1;
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
  naxes[ 0 ] = (long) image_width;
  naxes[ 1 ] = (long) image_width;
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
    fits_update_key(fptr, TFLOAT, "PIXSIZE", &image_pixellation, "Pixelation (mas)", status);
  if (*status == 0)
    fits_update_key(fptr, TINT, "WIDTH", &image_width, "Size (pixels)", status); 
 
  /*Write image*/
  if (*status == 0)
    fits_write_img(fptr, TFLOAT, fpixel, nelements, &image[ 0 ], status); 

  /*Close file*/
  if (*status == 0)
    fits_close_file(fptr, status);

  /*Report any errors*/
  fits_report_error(stderr, *status);

}

