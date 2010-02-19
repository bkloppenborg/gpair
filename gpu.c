#include "gpu.h"
#include <stdio.h>
#include <stdlib.h>

// TODO: Move the kernels elsewhere.

// Load the OCL kernel:
const char *ocl_kernel_chi2 = "\n" \
"__kernel void chi2(                                                    \n" \
"    __global float* data,                                              \n" \
"    __global float* data_err,                                          \n" \
"    __global float* curr_model,                                        \n" \
"    __global float* output,                                            \n" \
"    const unsigned int count)                                          \n" \
"{                                                                      \n" \
"    int i = get_global_id(0);                                          \n" \
"    if(i < count)                                                      \n" \
"        output[i] = (curr_model[i] - data[i]) / data_err[i];           \n" \
"        output[i] *= output[i];                                        \n" \
"}                                                                      \n" \
"\n";       

const char *ocl_kernel_vis = "\n" \
"__kernel void vis(                                                     \n" \
"    __global float2 * input,                                           \n" \
"    __global float * output,                                           \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"    int i = get_global_id(0);                                          \n" \
"    if(i < count)                                                      \n" \
"        output[i] = input[i][0] * input[i][0] + input[i][1] * input[i][1];  \n" \
"        output[i] *= output[i];                                         \n" \
"}                                                                      \n" \
"\n"; 

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
cl_device_id * pDevice_id;           // device ID
cl_context * pContext;               // context
cl_command_queue * pQueue;           // command queue

// Pointers for programs and kernels:
cl_kernel * pKernel_chi2;
cl_kernel * pKernel_vis;
cl_program * pPro_chi2;
cl_program * pPro_vis;

// Pointers for data stored on the GPU
cl_mem * pGpu_data;             // Pointer to GPU Memory location of OIFITS Data
cl_mem * pGpu_data_err;         // Pointer to GPU Memory location of OIFITS Data Error
cl_mem * pGpu_data_bsp;         // Pointer to GPU Memory location of OIFITS Data Biphasor
cl_mem * pGpu_mock_data;        // Pointer to GPU Memory location of Mock data


// A quick way to output an error from an OpenCL function:
void print_opencl_error(char* error_message, int error_code)
{
    printf("%s \n", error_message);
    printf("OpenCL Error %i \n", error_code);
    exit(0);
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

void gpu_build_kernels()
{
    // Static Program variables:
    static cl_program pro_chi2; 
    static cl_program pro_vis;
    
    // Static Kernels
    static cl_kernel kernel_chi2;
    static cl_kernel kernel_vis;
    
    int err = 0;
    
    // First build the chi2 kernel
    // ########
     // Create the compute program from the source buffer                 
    pro_chi2 = clCreateProgramWithSource(*pContext, 1, (const char **) & ocl_kernel_chi2, NULL, &err);     
    if (err != CL_SUCCESS)   
        print_opencl_error("clCreateProgramWithSource", err);    
        
    // Build the program executable for the start of the chi2 computation.
    err = clBuildProgram(pro_chi2, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable\n");          
        clGetProgramBuildInfo(pro_chi2, *pDevice_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }
   
    // Create the compute kernel in the program we wish to run           
    kernel_chi2 = clCreateKernel(pro_chi2, "chi2", &err);
    if (!kernel_chi2 || err != CL_SUCCESS)
        print_opencl_error("clCreateKernel", err); 
        
    pKernel_chi2 = &kernel_chi2;
    pPro_chi2 = &pro_chi2;
    
    // Now build the vis kernel:
    // ########
    pro_vis = clCreateProgramWithSource(*pContext, 1, (const char **) & ocl_kernel_vis, NULL, &err);     
    if (err != CL_SUCCESS)   
        print_opencl_error("clCreateProgramWithSource", err);    
        
    // Build the program executable for the start of the chi2 computation.
    err = clBuildProgram(pro_vis, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable\n");          
        clGetProgramBuildInfo(pro_vis, *pDevice_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }
   
    // Create the compute kernel in the program we wish to run           
    kernel_vis = clCreateKernel(pro_vis, "vis", &err);
    if (!kernel_chi2 || err != CL_SUCCESS)
        print_opencl_error("clCreateKernel", err); 
        
    pKernel_vis = &kernel_vis;
    pPro_vis = &pro_vis;
}

void gpu_copy_data(float *data, float *data_err, cl_float2 * bisphasor, int npow, int nbis)
{
    int count = npow+2*nbis;
    int err = 0;

    static cl_mem gpu_data;
    static cl_mem gpu_data_err;  
    static cl_mem gpu_data_bsp;
    static cl_mem gpu_mock_data;
    
    gpu_data = clCreateBuffer(*pContext,  CL_MEM_READ_ONLY,  sizeof(float) *count, NULL, NULL);
    gpu_data_err = clCreateBuffer(*pContext,  CL_MEM_READ_ONLY,  sizeof(float) *count, NULL, NULL); 
    gpu_data_bsp = clCreateBuffer(*pContext,  CL_MEM_READ_ONLY,  sizeof(float) *nbis, NULL, NULL); 
    gpu_mock_data = clCreateBuffer(*pContext, CL_MEM_READ_WRITE, sizeof(float) *count, NULL, NULL);
    if (!gpu_data || !gpu_data_err)
        print_opencl_error("Error: gpu_copy_data.  Create Buffer.", 0);
    
    err = clEnqueueWriteBuffer(*pQueue, gpu_data, CL_TRUE, 0, sizeof(float) *count, data, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_data_err, CL_TRUE, 0, sizeof(float) *count, data_err, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_data_bsp, CL_TRUE, 0, sizeof(float) * nbis, bisphasor, 0, NULL, NULL);
    if (err != CL_SUCCESS)
        print_opencl_error("Error: gpu_copy_data. Write Buffer", err);     
        
    pGpu_data = &gpu_data;
    pGpu_data_err = &gpu_data_err; 
    pGpu_data_bsp = &gpu_data_bsp;
    pGpu_mock_data = &gpu_mock_data;
}

void gpu_cleanup()
{
    // Release program and kernel objects:
    clReleaseProgram(*pPro_chi2);
    clReleaseProgram(*pPro_vis);
    
    clReleaseKernel(*pKernel_vis);
    clReleaseKernel(*pKernel_chi2);

    // Releate Memory objects:
    clReleaseMemObject(*pGpu_data);
    clReleaseMemObject(*pGpu_data_err);
    clReleaseMemObject(*pGpu_data_bsp);
    clReleaseMemObject(*pGpu_mock_data);
    
    // Release the command queue and context:
    clReleaseCommandQueue(*pQueue);
    clReleaseContext(*pContext);
    
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
    
    err  = clSetKernelArg(*pKernel_vis, 0, sizeof(cl_mem), &gpu_vis);
    err |= clSetKernelArg(*pKernel_vis, 1, sizeof(cl_mem), pGpu_mock_data);    // Output is stored on the GPU.
    err |= clSetKernelArg(*pKernel_vis, 2, sizeof(unsigned int), &nuv);
    if (err != CL_SUCCESS)
        print_opencl_error("clSetKernelArg", err);    
 
   // Get the maximum work-group size for executing the kernel on the device
    err = clGetKernelWorkGroupInfo(*pKernel_vis, *pDevice_id, CL_KERNEL_WORK_GROUP_SIZE , sizeof(size_t), &local, NULL);
    if (err != CL_SUCCESS)
        print_opencl_error("clGetKernelWorkGroupInfo", err);

    // Execute the kernel over the entire range of the data set        
    global = count;
    err = clEnqueueNDRangeKernel(*pQueue, *pKernel_vis, 1, NULL, &global, NULL, 0, NULL, NULL);
    if (err)
        print_opencl_error("clEnqueueNDRangeKernel vis", err);   
}

