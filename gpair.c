#include "gpair.h"
#include "getoifits.h"
#include <time.h>
#include "gpu.h"
#include <complex.h>

#define SEP "-----------------------------------------------------------\n"

// Global variables
int model_image_size = 0;
float model_image_pixellation = 0;
int status = 0;
oi_usersel usersel;
oi_data oifits_info; // stores all the info from oifits files

float * mock = NULL; // stores the mock current pseudo-data derived from the image
float * data = NULL; // stores the quantities derived from the data
float * data_err = NULL; // stores the error bars on the data
float complex *data_phasor = NULL; // bispectrum rotation precomputed value

float complex * visi = NULL; // current visibilities
float * current_image = NULL; // current image
float * data_gradient = NULL; // gradient on the likelihood

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
    register int ii ;
    register int uu ;
    float chi2 ;

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
    int data_alloc_bis = pow(2, ceil(log(nbis) / log(2)));   

    // Allocate memory for the data, error, and mock arrays:
    data = malloc(data_alloc * sizeof( float ));
    data_err =  malloc(data_alloc * sizeof( float ));
    visi = malloc(data_alloc_uv * sizeof( float complex));   
    data_phasor = malloc(data_alloc_bis * sizeof( float complex ));
    mock = malloc(data_alloc * sizeof( float ));

         
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
    model_image_size = 128;
    float model_image_pixellation = 0.15 ;
    int image_size = model_image_size * model_image_size;
    printf("Image Buffer Size %i \n", image_size);
    current_image = malloc(image_size * sizeof(float));
    memset(current_image, 0, image_size);
    current_image[(model_image_size * (model_image_size + 1 ) )/ 2 ] = 1.0;


    // setup precomputed DFT table
    int dft_size = nuv * model_image_size;
    int dft_alloc = pow(2, ceil(log(dft_size) / log(2)));   // Amount of space to allocate on the GPU for each axis of the DFT table. 
    DFT_tablex = malloc( dft_size * sizeof(float complex));
    memset(DFT_tablex, 0, dft_size);
    DFT_tabley = malloc( dft_size * sizeof(float complex));
    memset(DFT_tabley, 0, dft_size);
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

    // TODO: Only output if we are in a verbose mode.
    printf("Data Size: %i, Data Allocation: %i \n", data_size, data_alloc);
    printf("UV Size: %i, Allocation: %i \n", nuv, data_alloc_uv);
    printf("POW Size: %i \n", npow);
    printf("BIS Size: %i, Allocation: %i \n", nbis, data_alloc_bis);
    printf("DFT Size: %i , DFT Allocation: %i \n", dft_size, dft_alloc);

   
    // TODO: Remove after testing
    int iterations = 1;

    // #########
    // CPU Code:
    // #########

    // compute mock data, powerspectra + bispectra
    clock_t tick = clock();
    clock_t tock = 0;
    for(ii=0; ii < iterations; ii++)
    {
        //compute complex visibilities and the chi2
        image2vis(current_image, model_image_size, visi);
        vis2data();
        chi2 = data2chi2();
    }
        
    tock=clock();
    float cpu_time_chi2 = (float)(tock - tick) / (float)CLOCKS_PER_SEC;
    printf(SEP);
    printf("CPU Chi2: %f (CPU only)\n", chi2);
    printf(SEP);
    
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
    cl_float2 * gpu_bis = NULL;
    gpu_bis = malloc(data_alloc_bis * sizeof(cl_float2));
    for(i = 0; i < nbis; i++)
    {
        gpu_bis[i][0] = creal(data_phasor[i]);
        gpu_bis[i][1] = cimag(data_phasor[i]);
        gpu_bis[i].s0 = creal(data_bis[i]);
        gpu_bis[i].s1 = cimag(data_bis[i]);
    }
    // Pad the remainder
    for(i = nbis; i < data_alloc_bis; i++)
    {
        gpu_bis[i].s0 = 0;
        gpu_bis[i].s1 = 0;
    }
    
    // We will also need the uvpnt and sign information for bisepctrum computations.
    cl_long * gpu_bsref_uvpnt = NULL;
    cl_short * gpu_bsref_sign = NULL;
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
    
    // Copy the DFT table over to a GPU-friendly format:
    cl_float2 * gpu_dft_x = NULL;
    cl_float2 * gpu_dft_y = NULL;
    gpu_dft_x = malloc(dft_alloc * sizeof(cl_float2));
    gpu_dft_y = malloc(dft_alloc * sizeof(cl_float2));
    for(uu=0 ; uu < nuv; uu++)
    {
        for(ii=0; ii < model_image_size; ii++)
        {
            i = model_image_size * uu + ii;
            gpu_dft_x[i].s0 = __real__ DFT_tablex[i];
            gpu_dft_x[i].s1 = __imag__ DFT_tablex[i];
            gpu_dft_y[i].s0 = __real__ DFT_tabley[i];
            gpu_dft_y[i].s1 = __imag__ DFT_tabley[i];
        }
    }
    
    // Pad out the remainder of the array with zeros:
    for(i = nuv * model_image_size; i < dft_alloc; i++)
    {
        gpu_dft_x[i].s0 = 0;
        gpu_dft_x[i].s1 = 0;
        gpu_dft_y[i].s0 = 0;
        gpu_dft_y[i].s1 = 0;
    }
          
    
    // Initalize the GPU, copy data, and build the kernels.
    gpu_init();

    gpu_copy_data(data, data_err, data_alloc, data_alloc_uv, gpu_bis, data_alloc_bis, 
        gpu_bsref_uvpnt, gpu_bsref_sign, data_alloc_bsref, image_size, model_image_size); 
         
    gpu_build_kernels(data_alloc, image_size);
    gpu_copy_dft(gpu_dft_x, gpu_dft_y, dft_alloc);
    
    // Free variables used to store values pepared for the GPU
    free(gpu_visi);
    free(gpu_bis);
    free(gpu_bsref_uvpnt);
    free(gpu_bsref_sign);
    free(gpu_dft_x);
    free(gpu_dft_y);

    tick = clock();
    for(ii=0; ii < iterations; ii++)
    {
        // In the final version of the code, the following lines will be iterated.
        gpu_copy_image(current_image, model_image_size, model_image_size);
        gpu_image2vis(data_alloc_uv);
        gpu_vis2data(gpu_visi, nuv, npow, nbis);

        gpu_data2chi2(data_alloc);
        
        // Read back the necessary values
    }
    tock = clock();
    float gpu_time_chi2 = (float)(tock - tick) / (float)CLOCKS_PER_SEC;
    printf("-----------------------------------------------------------\n");   
    printf("CPU time (s): = %f\n", cpu_time_chi2);
    printf("GPU time (s): = %f\n", gpu_time_chi2);
    
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

void image2vis(float * image, int image_width, float complex * visi)
{	
    // DFT
  register int ii, jj, uu;
  float v0 = 0.; // zeroflux 

    for(ii=0 ; ii < image_width * image_width ; ii++) 
        v0 += image[ii];
    
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
        // TODO: Re-enable normalization, implement on the GPU too.
        if (v0 > 0.) visi[uu] /= v0;

    }
  //printf("Check - visi 0 %f %f\n", creal(visi[0]), cimag(visi[0]));
}

void vis2data(  )
{
    register int ii;
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

float data2chi2( )
{
    float chi2 = 0.;
    register int ii = 0;  
    for(ii=0; ii< npow + 2 * nbis; ii++)
    {
        chi2 += square( ( mock[ii] - data[ii] ) * data_err[ii] ) ;
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

float flux( )
{
  register int ii;
  float total=0.;
  for(ii=0; ii < model_image_size * model_image_size; ii++)
    total += current_image[ii];
  return total;
}

void compute_data_gradient() // need to call to vis2data before this
{

  register int ii, jj, kk;
  float complex vab, vbc, vca, vabder, vbcder, vcader, t3der;

  for(ii=0; ii < model_image_size; ii++)
    {
      for(jj=0; jj < model_image_size; jj++)
	{
	  data_gradient[ii + jj * model_image_size] = 0.;
	  
	  // Add gradient of chi2v2
	  for(kk = 0 ; kk < npow; kk++)
	    {
	      data_gradient[ii + jj * model_image_size] += 4. / ( data_err[ kk ] * data_err[ kk ] ) 
		*  ( mock[ kk ] - data[ kk ] ) * creal( conj( visi[ kk ] ) *  DFT_tablex[ model_image_size * kk +  ii ] * DFT_tabley[ model_image_size * kk +  jj ] );
	    }
	  
	  // Add gradient of chi2bs
	  for(kk = 0 ; kk < nbis; kk++)
	    {
	      vab = visi[oifits_info.bsref[kk].ab.uvpnt];
	      vbc = visi[oifits_info.bsref[kk].bc.uvpnt];
	      vca = visi[oifits_info.bsref[kk].ca.uvpnt];
	      if(oifits_info.bsref[kk].ab.sign < 0) { vab = conj(vab);} 
	      if(oifits_info.bsref[kk].bc.sign < 0) { vbc = conj(vbc);}
	      if(oifits_info.bsref[kk].ca.sign < 0) { vca = conj(vca);}

	      vabder = DFT_tablex[ oifits_info.bsref[kk].ab.uvpnt * model_image_size + ii  ] * DFT_tabley[ oifits_info.bsref[kk].ab.uvpnt * model_image_size + jj  ];
	      vbcder = DFT_tablex[ oifits_info.bsref[kk].bc.uvpnt * model_image_size + ii  ] * DFT_tabley[ oifits_info.bsref[kk].bc.uvpnt * model_image_size + jj  ];
	      vcader = DFT_tablex[ oifits_info.bsref[kk].ca.uvpnt * model_image_size + ii  ] * DFT_tabley[ oifits_info.bsref[kk].ca.uvpnt * model_image_size + jj  ];

	      if(oifits_info.bsref[kk].ab.sign < 0) { vabder = conj(vabder);} 
	      if(oifits_info.bsref[kk].bc.sign < 0) { vbcder = conj(vbcder);}
	      if(oifits_info.bsref[kk].ca.sign < 0) { vabder = conj(vcader);}
	      
	      t3der = ( ( vabder -  vab ) * vbc * vca + vab * ( vbcder -  vbc ) * vca + vab * vbc * ( vcader -  vca ) ) ;
	      t3der *= data_phasor[kk];
	      
	      // gradient from real part
	      data_gradient[ii + jj * model_image_size] += 2. / ( data_err[2 * kk] * data_err[2 * kk] ) * ( mock[ npow + 2 * kk] - data[npow + 2 * kk] ) * creal( t3der );  
	      
	      // gradient from imaginary part
	      data_gradient[ii + jj * model_image_size] += 2. / ( data_err[2 * kk + 1] * data_err[2 * kk + 1] )  * mock[ npow + 2 * kk + 1]  * cimag( t3der );			
	    }
	}
    }

  // the current gradient values correspond is with respect to the normalized image
  // Here we now compute the gradient with respect to unnormalized pixel intensities
  
  float grad_correction = 0.;
  float normalization = 0.; // if the flux has already been computed, we could use this value 
  for(ii = 0 ; ii < model_image_size * model_image_size ; ii++)
    {
    grad_correction += current_image[ ii ] * data_gradient[ ii ];
    normalization +=  current_image[ ii ];
    }
  for(ii = 0 ; ii < model_image_size * model_image_size ; ii++)
    data_gradient[ ii ]  = ( data_gradient[ ii ] - grad_correction / normalization ) / normalization ;

}	

int read_fits_image(char* fname, float* image, int* n, int* status)
{
  fitsfile *fptr;       // pointer to the FITS file, defined in fitsio.h
  int  nfound, anynull;
  long naxes[2], npixels; 
  long fpixel = 1 ;
  float nullval = 0 ;
  
  if (*status==0)fits_open_file(&fptr, fname, READONLY, status);
  if (*status==0)fits_read_keys_lng(fptr, "NAXIS", 1, 2, naxes, &nfound, status);
  //  if (*status==0)fits_read_key_str(fptr, "DATAFILE", datafile, comment, status);
  //  if (*status==0)fits_read_key_str(fptr, "TARGET", target, comment, status);
  //  if (*status==0)fits_read_key_flt(fptr, "PIXELATION", xyint, comment, status);
  npixels  = naxes[0] * naxes[1];         /* number of pixels in the image */
  fpixel   = 1;
  nullval  = 0;                // don't check for null values in the image
  
  if(naxes[0] != naxes[1])
    {
      printf("Image dimension are not square.\n");
      if(*status==0)fits_close_file(fptr, status);
      return *status;
    }
  *n = naxes[0];
  // Note: you need to allocate enough memory outside of this routine 
  if(*status==0)fits_read_img(fptr, TFLOAT, fpixel, npixels, &nullval, image, &anynull, status);
  if(*status==0)fits_close_file(fptr, status);
  return *status;
}
