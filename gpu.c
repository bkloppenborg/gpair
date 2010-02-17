#include "gpu.h"
#include "cl.h" // OpenCL header file

// A quick way to output an error from an OpenCL function:
void print_opencl_error(char* error_message, int error_code)
{
    printf("%s \n", error_message);
    printf("OpenCL Error %i \n", error_code);
    exit(0);
}

// Compute the chi2 of the data using a GPU
double data2chi2_gpu()
{
    double chi2 = 0., chi2v2 = 0., chi2bs = 0. ;
    double complex t3;
    //  double t3r, t3i;
    register int ii;
    
    // Variables used in the loop below:
    // current.t3
    // data.bisphs
    // data.bisamp
    // data.bisamperr
    
    // Functions used below:
    // cimag
    // square

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

// Computes the chi2 for the visibilities on the GPU
double chi2v2_gpu()
{
    // Variables used in the loop below:
    // current.pow
    // data.pow
    // data.powerr
    // Functions used below:
    // square
    for(ii=0; ii< data.npow ; ii++)
    {
      chi2v2 += square( ( current.pow[ii] - data.pow[ii] ) / data.powerr[ii] ) ;
    }

}
