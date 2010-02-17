#include "gpu.h"
#include "cl.h" // OpenCL header file

// A quick way to output an error from an OpenCL function:
void print_opencl_error(char* error_message, int error_code)
{
    printf("%s \n", error_message);
    printf("OpenCL Error %i \n", error_code);
    exit(0);
}

// A function to setup the OpenCL device with all of our kernels
void setup_device()
{
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_kernel kernel;                 // kernel
}


// Compute the chi2 of the data using a GPU
double data2chi2_gpu()
{
    // TODO: Pass pointers into this function.
    // Pointer names: data, model
    // TODO: Move this code out of this function and just keep the data on the GPU:
    // Move the necessary data over to the GPU
    
    // Create the input and output arrays in device memory for our calculation
    gpu_data = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) *count, NULL, NULL);
    gpu_model = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) *count, NULL, NULL);
    gpu_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) *count, NULL, NULL);
    if (!input || !output)
        print_opencl_error("clCreateBuffer", 0);

    // Write our data set into the input array in device memory          [11]
    err = clEnqueueWriteBuffer(queue, gpu_data, CL_TRUE, 0, sizeof(float) *count, data, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, gpu_model, CL_TRUE, 0, sizeof(float) *count, model, 0, NULL, NULL);
    if (err != CL_SUCCESS)
        print_opencl_error("clEnqueueWriteBuffer", err);  
    
    // First compute the individual array elements:
    //  current.pow[ii] - data.pow[ii] ) / data.powerr[ii]
    
    // Now square them:
    
    // Copy the result back to the host and sum them up
    
    // Now compute the chi2 for the V2
    double chi2v2 = chi2v2_gpu();
    
    // And now the chi2 for the bispectrum:
    double chi2bs = chi2bs_gpu();

    return (chi2v2 + chi2bs) / (double)( data.npow + 2.*data.nbis);
}

// Computes the chi2 for the visibilities on the GPU
double chi2v2_gpu()
{
    // TODO: Pass pointers into this function.
    // TODO: Remove 
    
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

// Computes the chi2 for the bispectrum on the GPU
double chi2bs_gpu()
{
    double complex;
    double t3;
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
}
