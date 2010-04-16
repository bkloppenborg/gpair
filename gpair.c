#include "gpair.h"
#include "getoifits.h"
#include <time.h>
#include "gpu.h"

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

// DFT precomputed coefficient tables
float complex* DFT_tablex = NULL;
float complex* DFT_tabley = NULL;

int npow = 0;
int nbis = 0;
int nuv = 0;

float square( float number )
{
    return number*number;
}

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

    // Test 1 : compute mock data, powerspectra + bispectra from scratch
    clock_t tick = clock();
    clock_t tock = 0;
    for(ii=0; ii < iterations; ii++)
    {
        //compute complex visibilities and the chi2
      image2vis( current_image, visi );
      vis2data(visi, mock);
      chi2 = data2chi2( mock);
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
    float total_flux = compute_flux( current_image );
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
    cl_long * gpu_bsref_uvpnt = NULL;
    cl_short * gpu_bsref_sign = NULL;
    int data_alloc_bsref = 3 * data_alloc_phasor;
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
    //gpu_check_data(&chi2, nuv, visi, data_alloc, mock);
    
    // Cleanup, shutdown, were're done.
    gpu_cleanup();
    
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
void image2vis(float* image, float complex* visi ) // DFT implementation
{	 
  register int ii = 0, jj = 0, uu = 0;
  float zeroflux = 0.; // zeroflux 
  
  for(ii=0 ; ii < image_width * image_width ; ii++) 
    zeroflux += image[ii];
    
  for(uu=0 ; uu < nuv; uu++)
    {
      visi[uu] = 0.0 + I * 0.0;
      for(ii=0; ii < image_width; ii++)
        {
	  for(jj=0; jj < image_width; jj++)
            {
	      visi[uu] += image[ ii + image_width * jj ] *  DFT_tablex[ image_width * uu +  ii] * DFT_tabley[ image_width * uu +  jj];
            }
        }
      if (zeroflux > 0.) visi[uu] /= zeroflux;
    }
}

void update_vis_fluxchange(int x, int y, float flux_old, float flux_new, float complex* visi_old, float complex* visi_new) 
// finite difference routine giving visi when changing the flux of an element in (x,y)
{	
  // Note : two effects, one due to the flux change in the pixel, the other to the change in total flux
  // the total flux should be updated outside this loop
  register int uu;
  float flux_ratio =  flux_old / flux_new;
  for(uu=0 ; uu < nuv; uu++)
      visi_new[uu] = visi_old[uu] *  flux_ratio + ( 1.0 - flux_ratio ) *  DFT_tablex[ image_width * uu +  x] * DFT_tabley[ image_width * uu +  y] ;
}

void update_vis_positionchange(int x_old, int y_old, int x_new, int y_new, float flux, float complex* visi_old, float complex* visi_new) 
// finite difference routine giving visi when moving one element from (x_old,y_old) to (x_new, y_new)
{ // Note : no change in total flux in this case
  register int uu;
  for(uu=0 ; uu < nuv; uu++)
      visi_new[uu] = visi_old[uu] 
	+ flux  * ( DFT_tablex[ image_width * uu +  x_new] * DFT_tabley[ image_width * uu +  y_new] 
		    -  DFT_tablex[ image_width * uu +  x_old] * DFT_tabley[ image_width * uu +  y_old]) ;
}


// TBD -- Check the expression used in this function
void update_vis_fluxpositionchange(int x_old, int y_old, int x_new, int y_new, float flux_old, float flux_new, float complex* visi_old, float complex* visi_new)
{
  register int uu;
  for(uu=0 ; uu < nuv; uu++)
    visi_new[uu] =  visi_old[uu] * flux_old / flux_new 
      +  DFT_tablex[ image_width * uu +  x_new ] * DFT_tabley[ image_width * uu +  y_new ] 
      -  flux_old / flux_new * DFT_tablex[ image_width * uu +  x_old ] * DFT_tabley[ image_width * uu +  y_old ] ;
}


void vis2data( float complex* visi, float* mock )
{
    int ii = 0;
    float complex vab = 0;
    float complex vbc = 0;
    float complex vca = 0;
    float complex t3 = 0;

    for( ii = 0; ii< npow; ii++)
    {
        mock[ ii ] = square ( cabs( visi[ii] ) );
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
            
        t3 =  ( vab * vbc * vca ) * data_phasor[ii] ;   
        mock[ npow + 2 * ii ] = creal(t3) ;
        mock[ npow + 2 * ii + 1] = cimag(t3) ;
    } 
    
    // Uncomment to see the mock data array.
/*    int count = npow + 2 * nbis;*/
/*    for(ii = 0; ii < count; ii++)     */
/*        printf("%i %f \n", ii, mock[ii]);*/
/*        */
/*    printf("\n");*/

}

float data2chi2( float* mock )
{
    float chi2 = 0.;
    register int ii = 0;  
    for(ii=0; ii< npow + 2 * nbis; ii++)
    {
        chi2 += square( ( mock[ii] - data[ii] ) * data_err[ii] ) ;
    }

    return chi2;
}

float compute_flux( float* image )
{
  register int ii;
  float total=0.;
  for(ii=0; ii < image_width * image_width; ii++)
    total += image[ii];
  return total;
}

void compute_data_gradient(double complex* visi, double* mock, double* image, double* data_gradient) // need to call vis2data before this
{

    register int ii, jj, kk;
    double complex vab, vbc, vca, vabder, vbcder, vcader, t3der;

    double flux = 0.; // if the flux has already been computed, we could use its value 
    for(ii = 0 ; ii < image_width * image_width ; ii++)
        flux +=  image[ ii ];

    double invflux = 1. / flux;

    for(ii=0; ii < image_width; ii++)
    {
        for(jj=0; jj < image_width; jj++)
        {
            data_gradient[ii + jj * image_width] = 0.;

            // Add gradient of chi2v2
            for(kk = 0 ; kk < npow; kk++)
            {
                data_gradient[ii + jj * image_width] += 4. * data_err[ kk ] * data_err[ kk ] * invflux 
                *  ( mock[ kk ] - data[ kk ] ) 
                * creal( conj( visi[ kk ] ) *  ( DFT_tablex[ image_width * kk +  ii ] * DFT_tabley[ image_width * kk +  jj ] - visi[ kk ] ) );
            }

            // Add gradient of chi2bs
            for(kk = 0 ; kk < nbis; kk++)
            {
                vab = visi[oifits_info.bsref[kk].ab.uvpnt];
                vbc = visi[oifits_info.bsref[kk].bc.uvpnt];
                vca = visi[oifits_info.bsref[kk].ca.uvpnt];

                vabder =  DFT_tablex[ oifits_info.bsref[kk].ab.uvpnt * image_width + ii  ] * DFT_tabley[ oifits_info.bsref[kk].ab.uvpnt * image_width + jj  ]  ;
                vbcder =  DFT_tablex[ oifits_info.bsref[kk].bc.uvpnt * image_width + ii  ] * DFT_tabley[ oifits_info.bsref[kk].bc.uvpnt * image_width + jj  ]  ;
                vcader =  DFT_tablex[ oifits_info.bsref[kk].ca.uvpnt * image_width + ii  ] * DFT_tabley[ oifits_info.bsref[kk].ca.uvpnt * image_width + jj  ]  ;

                if(oifits_info.bsref[kk].ab.sign < 0) { vab = conj(vab);} 
                if(oifits_info.bsref[kk].bc.sign < 0) { vbc = conj(vbc);}
                if(oifits_info.bsref[kk].ca.sign < 0) { vca = conj(vca);}
                if(oifits_info.bsref[kk].ab.sign < 0) { vabder = conj(vabder);} 
                if(oifits_info.bsref[kk].bc.sign < 0) { vbcder = conj(vbcder);}
                if(oifits_info.bsref[kk].ca.sign < 0) { vabder = conj(vcader);}

                t3der = ( (vabder - vab) * vbc * vca + vab * (vbcder - vbc) * vca + vab * vbc * (vcader - vca) ) * data_bisphasor[kk] * invflux ;

                // gradient from real part
                data_gradient[ii + jj * image_width] += 2. * data_err[2 * kk] * data_err[2 * kk]  * ( mock[ npow + 2 * kk] - data[npow + 2 * kk] ) * creal( t3der );  

                // gradient from imaginary part
                data_gradient[ii + jj * image_width] += 2. * data_err[2 * kk + 1] * data_err[2 * kk + 1] * mock[ npow + 2 * kk + 1]  * cimag( t3der );			
            }
        }
    }
}	



float GullSkilling_entropy(float *image, float *default_model)
{
  register int ii;
  float S = 0.;
  for(ii=0 ; ii < image_width * image_width; ii++)
    {
      if ( (image[ii] > 0.) && (default_model[ii] > 0.) )
	{
	  S += image[ii] - default_model[ii] - image[ii] * log( image[ii] / default_model[ii] );
	}   
      else 
	S += - default_model[ii];
    }
  return S;
}

void GullSkilling_entropy_gradient(float *image, float *default_model, float* gradient)
{
 register int ii;
 for(ii=0 ; ii < image_width * image_width; ii++)
   {
     if( (image[ii] > 0.) && (default_model[ii] > 0.) )
       {
	 gradient[ ii ] = - log( image[ii] / default_model[ii] );
       }
   }
}

float GullSkilling_entropy_diff( int x_old, int y_old, int x_new, int y_new , float old_flux, float new_flux, float *default_model)
{
  int position_old = x_old + y_old * image_width ;
  int position_new = x_new + y_new * image_width;
  float S_old, S_new;
  if( ( old_flux > 0.) && (default_model[position_old] > 0.) )
    S_old = old_flux - default_model[ position_old ] - old_flux * log( old_flux / default_model[ position_old ] );
  else 
    S_old = - default_model[ position_old ];
  
  if( ( new_flux > 0.) && (default_model[position_new] > 0.) )
     S_new = new_flux - default_model[ position_new ] - new_flux * log( new_flux / default_model[ position_new ] );
  else 
    S_new = - default_model[ position_new ];
  
  return S_new - S_old;
}


float L2_entropy(float *image, float *default_model)
{
  register int ii;
  float S = 0.;
  for(ii=0 ; ii < image_width * image_width; ii++)
    {
      if( default_model[ii] > 0.)
	{
	  S += - image[ii] * image[ii]  / ( 2. * default_model[ii] );
	}   
    }
  return S;
}

float L2_entropy_gradient(float *image, float *default_model)
{
  register int ii;
  float S = 0.;
  for(ii=0 ; ii < image_width * image_width; ii++)
    {
      if( default_model[ii] > 0.)
	{
	  S += - image[ii] / default_model[ii] ;
	}   
    }
  return S;
}

float L2_diff( int x_old, int y_old, int x_new, int y_new, float *image, float *default_model )
{
  int position_old = x_old + y_old * image_width ;
  int position_new = x_new + y_new * image_width;
  float S_old = 0., S_new = 0.;
   if( default_model[ position_old ] > 0. )
     S_old = - image[ position_old ] * image[ position_old ] / ( 2. * default_model[ position_old ] );

   if( default_model[ position_new ] > 0. )
     S_new = - image[ position_new ] * image[ position_new ] / ( 2. * default_model[ position_new ] );
  
   return S_new - S_old;
}



// Prior image

void set_model(int modeltype, float modelwidth, float modelflux, float* default_model )
{
  float flux = 0.0;
  int i, j;

  switch (modeltype)
    {
      case 0:
      {
        // Flat prior
	printf("Flat prior, total flux: %8.3f\n", modelflux);
	float norm = ((float) image_width * (float) image_width);
        for (i = 0; i < image_width * image_width; i++)
	  default_model[ i ] = modelflux / norm;

        break;
      }

      case 1:
      {
        // Centered Dirac (negligible flux around)
	printf("Dirac, flux: %8.3f\n", modelflux);
        for (i = 0; i < image_width * image_width ; i++)
            default_model[ i ] = 1e-8;
        default_model[ (image_width * (image_width + 1)) / 2 ] = modelflux;

        break;
      }

      case 2:
      {
        // Centered Uniform disk
	printf("Uniform disk, Radius:%f mas, flux:%f\n", modelwidth, modelflux);
        flux = 0.0;
        float rsq;
        for (i = 0; i < image_width; i++)
        {
          for (j = 0; j < image_width; j++)
	    {
	      rsq = square((float) (image_width / 2 - i)) + square((float) (image_width / 2 - j));
	      
	      if (sqrt(rsq) <= (modelwidth / image_pixellation))
		default_model[ j * image_width + i ] = 1.;
	      else
		default_model[ j * image_width + i ] = 1e-8;
	      flux += default_model[ j * image_width + i ];
	    }
        }
	
        for (i = 0; i < image_width * image_width; i++)
          default_model[ i ] *= modelflux / flux;
	
        break;
      }
      
    case 3:
      {
        // Centered Gaussian
        float sigma = modelwidth / (2. * sqrt(2. * log(2.)));
          printf("Gaussian, FWHM:%f mas, sigma:%f mas, flux:%f\n", modelwidth, sigma, modelflux);
        flux = 0.0;
        for (i = 0; i < image_width; i++)
	  {
	    for (j = 0; j < image_width; j++)
	      {
		default_model[ j * image_width + i ] = exp(-(square((float) image_width / 2 - i) + square((float) image_width / 2 - j)) / (2. * square(sigma
																	      / image_pixellation)));
		// Fix problem with support at zero
		//if ( default_model[ j*image_width+i ]<1e-8 ) default_model[ j*image_width+i ]=1e-8;
		flux += default_model[ j * image_width + i ];
	      }
        }
        for (i = 0; i < image_width * image_width; i++)
          default_model[ i ] *= modelflux / flux;
        break;
      }

    case 4:
      {
        //Centered Lorentzian
        float sigma = modelwidth;
	printf("Lorentzian, FWHM:%f mas, sigma:%f mas, flux:%f\n", modelwidth, sigma, modelflux);
        flux = 0.0;
        for (i = 0; i < image_width; i++)
	  {
	    for (j = 0; j < image_width; j++)
	      {
		default_model[ j * image_width + i ] = sigma / (square(sigma) + square((float) image_width / 2 - i) + square((float) image_width / 2 - j));
		flux += default_model[ j * image_width + i ];
          }
	  }
        for (i = 0; i < image_width * image_width; i++)
          default_model[ i ] *= modelflux / flux;
	
        break;
      }

  }
}


