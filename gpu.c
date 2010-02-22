#include "gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

// TODO: Move the kernels elsewhere.

// Load the OCL kernel:

static char * LoadProgramSourceFromFile(const char *filename)
{
    struct stat statbuf;
    FILE        *fh;
    char        *source;

    fh = fopen(filename, "r");
    if (fh == 0)
        return 0;

    stat(filename, &statbuf);
    source = (char *) malloc(statbuf.st_size + 1);
    fread(source, statbuf.st_size, 1, fh);
    source[statbuf.st_size] = '\0';

    return source;
}


/*void vis2data(  )*/
/*{*/
/*    int ii;*/
/*    float complex vab, vbc, vca, t3;*/

/*    for( ii = 0; ii< npow; ii++)*/
/*    {*/
/*        mock[ ii ] = square ( cabs( visi[ii] ) );*/
/*    }*/

/*    for( ii = 0; ii< nbis; ii++)*/
/*    {*/
/*        vab = visi[ oifits_info.bsref[ii].ab.uvpnt ];*/
/*        vbc = visi[ oifits_info.bsref[ii].bc.uvpnt ];*/
/*        vca = visi[ oifits_info.bsref[ii].ca.uvpnt ];	*/
/*        if( oifits_info.bsref[ii].ab.sign < 0) */
/*            vab = conj(vab);*/
/*        if( oifits_info.bsref[ii].bc.sign < 0) */
/*            vbc = conj(vbc);*/
/*        if( oifits_info.bsref[ii].ca.sign < 0) */
/*            vca = conj(vca);*/
/*            */
/*        t3 =  ( vab * vbc * vca ) * bisphasor[ii] ;   */
/*        mock[ npow + 2 * ii ] = creal(t3) ;*/
/*        mock[ npow + 2 * ii + 1] = cimag(t3) ;*/
/*    } */

/*}*/

// Global variables
cl_device_id * pDevice_id = NULL;           // device ID
cl_context * pContext = NULL;               // context
cl_command_queue * pQueue = NULL;           // command queue

// Pointers for programs and kernels:
cl_kernel * pKernel_chi2 = NULL;
cl_kernel * pKernel_powspec = NULL;
cl_program * pPro_chi2 = NULL;
cl_program * pPro_powspec = NULL;

// Pointers for data stored on the GPU
cl_mem * pGpu_data = NULL;             // OIFITS Data
cl_mem * pGpu_data_err = NULL;         // OIFITS Data Error
cl_mem * pGpu_data_bip = NULL;         // OIFITS Data Biphasor
cl_mem * pGpu_data_uvpnt = NULL;       // OIFITS UV Point indicies for bispectrum data 
cl_mem * pGpu_data_sign = NULL;        // OIFITS UV Point signs.
cl_mem * pGpu_mock_data = NULL;        // Mock data (current "image")

// A quick way to output an error from an OpenCL function:
void print_opencl_error(char* error_message, int error_code)
{
    // Something bad happened.  Clean up memory first.
    gpu_cleanup();
    
    printf("%s \n", error_message);
    printf("OpenCL Error %i \n", error_code);
    exit(0);
}

void gpu_build_kernel(cl_program * program, cl_kernel * kernel, char * kernel_name, char * filename)
{   
    int err = 0;
    
    // Load the kernel source:
    char * kernel_source = LoadProgramSourceFromFile(filename);
    // Create the program
    *program = clCreateProgramWithSource(*pContext, 1, (const char **) & kernel_source, NULL, &err);     
    if (err != CL_SUCCESS)   
        print_opencl_error("clCreateProgramWithSource", err);    
        
    // Build the program executable for the start of the chi2 computation.
    err = clBuildProgram(*program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable for %s \n", kernel_name);          
        clGetProgramBuildInfo(*program, *pDevice_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }
   
    // Create the compute kernel in the program we wish to run           
    *kernel = clCreateKernel(*program, kernel_name, &err);
    if (!kernel || err != CL_SUCCESS)
        print_opencl_error("clCreateKernel", err); 
}

void gpu_build_kernels()
{
    static cl_program pro_chi2;
    static cl_kernel kern_chi2;
    
    // Kernel and program for computing chi2:
    gpu_build_kernel(&pro_chi2, &kern_chi2, "compute_chi2", "./kernel_chi2.cl");
    pPro_chi2 = &pro_chi2;
    pKernel_chi2 = &kern_chi2;
    
    static cl_program pro_powspec;
    static cl_kernel kern_powspec;
    // Kernel and program for computing the power spectrum
    gpu_build_kernel(&pro_powspec, &kern_powspec, "compute_pow_spec", "./kernel_pow_spec.cl");
    pPro_powspec = &pro_powspec;
    pKernel_powspec = &kern_powspec;   
}

void gpu_cleanup()
{
    // Release program and kernel objects:
    if(pPro_chi2 != NULL)
        clReleaseProgram(*pPro_chi2);
    if(pPro_powspec != NULL)
        clReleaseProgram(*pPro_powspec);
    
    if(pKernel_chi2 != NULL)
        clReleaseKernel(*pKernel_chi2);
    if(pKernel_powspec != NULL)
        clReleaseKernel(*pKernel_powspec);

    // Releate Memory objects:
    if(pGpu_data != NULL)
        clReleaseMemObject(*pGpu_data);
    if(pGpu_data_err != NULL)
        clReleaseMemObject(*pGpu_data_err);
    if(pGpu_data_bip != NULL)
        clReleaseMemObject(*pGpu_data_bip);
    if(pGpu_data_uvpnt != NULL)
        clReleaseMemObject(*pGpu_data_uvpnt);
    if(pGpu_data_sign != NULL)
        clReleaseMemObject(*pGpu_data_sign);
    if(pGpu_mock_data != NULL)
        clReleaseMemObject(*pGpu_mock_data);

    // Release the command queue and context:
    if(pQueue != NULL)
        clReleaseCommandQueue(*pQueue);
    if(pContext != NULL)
        clReleaseContext(*pContext);
    
}

void gpu_copy_data(float *data, float *data_err, \
                    cl_float2 * data_bip, \
                    long * data_uvpnt, short * data_sign, \
                    int npow, int nbis)
{
    int count = npow+2*nbis;
    int err = 0;

    static cl_mem gpu_data;         // Data
    static cl_mem gpu_data_err;     // Data Error
    static cl_mem gpu_data_bip;     // Biphasor
    static cl_mem gpu_data_uvpnt;   // UV Points for the bispectrum
    static cl_mem gpu_data_sign;    // Signs for the bispectrum.
    static cl_mem gpu_mock_data;    // Mock Data

    // Init some mock data (to allow resumes in the future I suppose...)
    int i = 0;
    float * mock_data;
    mock_data = malloc(count * sizeof(float));
    for(i = 0; i < count; i++)
        mock_data[i] = 0; 
    
    // Create buffers on the device:    
    gpu_data = clCreateBuffer(*pContext,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, NULL);
    gpu_data_err = clCreateBuffer(*pContext,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, NULL); 
    gpu_data_bip = clCreateBuffer(*pContext,  CL_MEM_READ_ONLY,  sizeof(float) * nbis, NULL, NULL); 
    gpu_data_uvpnt = clCreateBuffer(*pContext, CL_MEM_READ_ONLY, sizeof(long) * nbis, NULL, NULL);
    gpu_data_sign = clCreateBuffer(*pContext, CL_MEM_READ_ONLY, sizeof(short) * nbis, NULL, NULL);
    gpu_mock_data = clCreateBuffer(*pContext, CL_MEM_READ_WRITE, sizeof(float) * count, NULL, NULL);
    if (!gpu_data || !gpu_data_err)
        print_opencl_error("Error: gpu_copy_data.  Create Buffer.", 0);

    // Copy the data over to the device:
    err = clEnqueueWriteBuffer(*pQueue, gpu_data, CL_TRUE, 0, sizeof(float) * count, data, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_data_err, CL_TRUE, 0, sizeof(float) * count, data_err, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_data_bip, CL_TRUE, 0, sizeof(float) * nbis, data_bip, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_data_uvpnt, CL_TRUE, 0, sizeof(long) * nbis, data_uvpnt, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_data_sign, CL_TRUE, 0, sizeof(short) * nbis, data_sign, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_mock_data, CL_TRUE, 0, sizeof(float) * count, mock_data, 0, NULL, NULL);    
    if (err != CL_SUCCESS)
        print_opencl_error("Error: gpu_copy_data. Write Buffer", err);     
        
    pGpu_data = &gpu_data;
    pGpu_data_err = &gpu_data_err; 
    pGpu_data_bip = &gpu_data_bip;
    pGpu_data_uvpnt = &gpu_data_uvpnt;
    pGpu_data_sign = &gpu_data_sign;
    pGpu_mock_data = &gpu_mock_data;
}


// Compute the chi2 of the data using a GPU
double gpu_data2chi2(float *mock, int npow, int nbis)
{

    // TODO: Move this elsewhere later.
    int err;                          // error code returned from api calls

    size_t global;                    // global domain size for our calculation
    size_t local;                     // local domain size for our calculation

    cl_mem gpu_model;
    cl_mem gpu_result;            // device memory used for the output array
    int count = npow+2*nbis;               // the length of the data
    float results[count]; 
          
    
    // Create the input and output arrays in device memory for our calculation 
    gpu_model = clCreateBuffer(*pContext,  CL_MEM_READ_ONLY,  sizeof(float) *count, NULL, NULL);
    gpu_result = clCreateBuffer(*pContext, CL_MEM_WRITE_ONLY, sizeof(float) *count, NULL, NULL);
    if (!gpu_model || !gpu_result)
        print_opencl_error("clCreateBuffer", 0);

    // Write our data set into the input array in device memory          
    err = clEnqueueWriteBuffer(*pQueue, gpu_model, CL_TRUE, 0, sizeof(float) *count, mock, 0, NULL, NULL);
    if (err != CL_SUCCESS)
        print_opencl_error("clEnqueueWriteBuffer", err);  
    

    // Set the arguments to our compute kernel                         
    err = 0;
    err  = clSetKernelArg(*pKernel_chi2, 0, sizeof(cl_mem), pGpu_data);
    err |= clSetKernelArg(*pKernel_chi2, 1, sizeof(cl_mem), pGpu_data_err);
    err |= clSetKernelArg(*pKernel_chi2, 2, sizeof(cl_mem), &gpu_model);
    err |= clSetKernelArg(*pKernel_chi2, 3, sizeof(cl_mem), &gpu_result);
    err |= clSetKernelArg(*pKernel_chi2, 4, sizeof(unsigned int), &count);
    if (err != CL_SUCCESS)
        print_opencl_error("clSetKernelArg", err);

    // Get the maximum work-group size for executing the kernel on the device
    err = clGetKernelWorkGroupInfo(*pKernel_chi2, *pDevice_id, CL_KERNEL_WORK_GROUP_SIZE , sizeof(size_t), &local, NULL);
    if (err != CL_SUCCESS)
        print_opencl_error("clGetKernelWorkGroupInfo", err);


    // Execute the kernel over the entire range of the data set        
    global = count;
    err = clEnqueueNDRangeKernel(*pQueue, *pKernel_chi2, 1, NULL, &global, NULL, 0, NULL, NULL);
    if (err)
        print_opencl_error("clEnqueueNDRangeKernel chi2", err);

    
    // Wait for the command queue to get serviced before reading back results
    clFinish(*pQueue);

    // Read the results from the device                                  
    err = clEnqueueReadBuffer(*pQueue, gpu_result, CL_TRUE, 0, sizeof(float) *count, results, 0, NULL, NULL );
    if (err != CL_SUCCESS)
        print_opencl_error("clEnqueueReadBuffer", err);
        
    float chi2_temp = 0;
    int i;
    for(i = 0; i < count; i++)
        chi2_temp += results[i];
    
    // Shut down and clean up
    clReleaseMemObject(gpu_model);
    clReleaseMemObject(gpu_result);
        
    return chi2_temp/(double)(count);  
}


void gpu_init()
{
    // Init a few variables.  Static so they won't go out of scope.
    static cl_device_id device_id;           // device ID
    static cl_context context;               // context
    static cl_command_queue queue;           // command queue
    
    int err = 0;
    
    // Get an ID for the device                                   a
    err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS)   //      [3]
        print_opencl_error("Unable to get GPU Device", err);                                      

    // Create a context                                           
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if(err != CL_SUCCESS)
        print_opencl_error("Unable to create OpenCL context", err);      

    // Create a command queue                                          
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    if (err != CL_SUCCESS)   
        print_opencl_error("Unable to create command queue", err); 
    
    // Set the pointers.    
    pDevice_id = &device_id;
    pContext = &context;
    pQueue = &queue;
}

void gpu_vis2data(cl_float2 *vis, int nuv, int npow, int nbis)
{
    // Begin by copying vis over to the GPU
    cl_mem gpu_vis;
    int count = npow+2*nbis;               // the length of the data
    int err = 0;

    size_t global;                    // global domain size for our calculation
    size_t local;                     // local domain size for our calculation
    
    gpu_vis = clCreateBuffer(*pContext,  CL_MEM_READ_ONLY,  sizeof(cl_float2) * nuv, NULL, NULL);
    if (!gpu_vis)
        print_opencl_error("clCreateBuffer", 0);
    
    err |= clEnqueueWriteBuffer(*pQueue, gpu_vis, CL_TRUE, 0, sizeof(cl_float2) * nuv, vis, 0, NULL, NULL);
    if (err != CL_SUCCESS)
        print_opencl_error("clEnqueueWriteBuffer", err);     


    
    err  = clSetKernelArg(*pKernel_powspec, 0, sizeof(cl_mem), &gpu_vis);
    err |= clSetKernelArg(*pKernel_powspec, 1, sizeof(cl_mem), pGpu_mock_data);    // Output is stored on the GPU.
    err |= clSetKernelArg(*pKernel_powspec, 2, sizeof(unsigned int), &nuv);
    if (err != CL_SUCCESS)
        print_opencl_error("clSetKernelArg", err);    
 

 
   // Get the maximum work-group size for executing the kernel on the device
    err = clGetKernelWorkGroupInfo(*pKernel_powspec, *pDevice_id, CL_KERNEL_WORK_GROUP_SIZE , sizeof(size_t), &local, NULL);
    if (err != CL_SUCCESS)
        print_opencl_error("clGetKernelWorkGroupInfo", err);

    // Execute the kernel over the entire range of the data set        
    global = count;
    err = clEnqueueNDRangeKernel(*pQueue, *pKernel_powspec, 1, NULL, &global, NULL, 0, NULL, NULL);
    if (err)
        print_opencl_error("clEnqueueNDRangeKernel vis", err);   
}

