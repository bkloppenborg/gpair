#include "gpu.h"
#include "cl.h" // OpenCL header file
#include <stdio.h>
#include <stdlib.h>

// TODO: Move the kernels elsewhere.

// Load the OCL kernel:
const char *ocl_kernel_chi2 = "\n" \
"__kernel void chi2(                                                \n" \
"    __global float* data,                                          \n" \
"    __global float* data_err,                                      \n" \
"    __global float* curr_model,                                    \n" \
"    __global float* output,                                        \n" \
"    const unsigned int count)                                      \n" \
"{                                                                  \n" \
"    int i = get_global_id(0);                                      \n" \
"    if(i < count)                                                  \n" \
"        output[i] = (curr_model[i] - data[i]) / data_err[i];       \n" \
"        output[i] = pown(output[i], 2);                            \n" \
"}                                                                  \n" \
"\n";       




// A quick way to output an error from an OpenCL function:
void print_opencl_error(char* error_message, int error_code)
{
    printf("%s \n", error_message);
    printf("OpenCL Error %i \n", error_code);
    exit(0);
}

/*// A function to setup the OpenCL device with all of our kernels*/
/*void setup_device()*/
/*{*/
/*    cl_device_id device_id;           // device ID*/
/*    cl_context context;               // context*/
/*    cl_command_queue queue;           // command queue*/
/*    cl_program program;               // program*/
/*    cl_kernel kernel;                 // kernel*/
/*}*/


// Compute the chi2 of the data using a GPU
double data2chi2_gpu(float *data, float *data_err, float *mock, int npow, int nbis)
{

    // TODO: Move this elsewhere later.
    int err;                          // error code returned from api calls

    size_t global;                    // global domain size for our calculation
    size_t local;                     // local domain size for our calculation

    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program pro_chi2;               // program
    cl_kernel kernel_chi2;                 // kernel

    cl_mem gpu_data;
    cl_mem gpu_data_err;    
    cl_mem gpu_model;
    cl_mem gpu_result;            // device memory used for the output array
    int count = npow+2*nbis;               // the length of the data
    float results[count];

    // Get an ID for the device                                   
    int gpu = 1;
    err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS)   //      [3]
        print_opencl_error("clGetDeviceIDs", err);                                      

    // Create a context                                           
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if(err != CL_SUCCESS)
        print_opencl_error("clCreateContext", err);      

    // Create a command queue                                          

    queue = clCreateCommandQueue(context, device_id, 0, &err);
    if (err != CL_SUCCESS)   
        print_opencl_error("clCreateCommandQueue", err);  
        
    // TODO: Pass pointers into this function.
    // Pointer names: data, model
    // TODO: Move this code out of this function and just keep the data on the GPU:
    // Move the necessary data over to the GPU
 
     // Create the compute program from the source buffer                 
    pro_chi2 = clCreateProgramWithSource(context, 1, (const char **) & ocl_kernel_chi2, NULL, &err);     
    if (err != CL_SUCCESS)   
        print_opencl_error("clCreateProgramWithSource", err);    
        
    // Build the program executable for the start of the chi2 computation.
    err = clBuildProgram(pro_chi2, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable\n");          
        clGetProgramBuildInfo(pro_chi2, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }
   

    // Create the compute kernel in the program we wish to run           
    kernel_chi2 = clCreateKernel(pro_chi2, "chi2", &err);
    if (!kernel_chi2 || err != CL_SUCCESS)
        print_opencl_error("clCreateKernel", err);   
    
    // Create the input and output arrays in device memory for our calculation
    gpu_data = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) *count, NULL, NULL);
    gpu_data_err = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) *count, NULL, NULL);    
    gpu_model = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) *count, NULL, NULL);
    gpu_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) *count, NULL, NULL);
    if (!gpu_data || !gpu_model || !gpu_result)
        print_opencl_error("clCreateBuffer", 0);

    // Write our data set into the input array in device memory        
    err = clEnqueueWriteBuffer(queue, gpu_data, CL_TRUE, 0, sizeof(float) *count, data, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, gpu_data_err, CL_TRUE, 0, sizeof(float) *count, data_err, 0, NULL, NULL);    
    err |= clEnqueueWriteBuffer(queue, gpu_model, CL_TRUE, 0, sizeof(float) *count, mock, 0, NULL, NULL);
    if (err != CL_SUCCESS)
        print_opencl_error("clEnqueueWriteBuffer", err);  
    

    // Set the arguments to our compute kernel                         
    err = 0;
    err  = clSetKernelArg(kernel_chi2, 0, sizeof(cl_mem), &gpu_data);
    err |= clSetKernelArg(kernel_chi2, 1, sizeof(cl_mem), &gpu_data_err);
    err |= clSetKernelArg(kernel_chi2, 2, sizeof(cl_mem), &gpu_model);
    err |= clSetKernelArg(kernel_chi2, 3, sizeof(cl_mem), &gpu_result);
    err |= clSetKernelArg(kernel_chi2, 4, sizeof(unsigned int), &count);
    if (err != CL_SUCCESS)
        print_opencl_error("clSetKernelArg", err);

    // Get the maximum work-group size for executing the kernel on the device
    err = clGetKernelWorkGroupInfo(kernel_chi2, device_id, CL_KERNEL_WORK_GROUP_SIZE , sizeof(size_t), &local, NULL);
    if (err != CL_SUCCESS)
        print_opencl_error("clGetKernelWorkGroupInfo", err);

    // Execute the kernel over the entire range of the data set        
    global = count;
    err = clEnqueueNDRangeKernel(queue, kernel_chi2, 1, NULL, &global, NULL, 0, NULL, NULL);
    if (err)
        print_opencl_error("clEnqueueNDRangeKernel chi2", err);

    
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);

    // Read the results from the device                                  
    err = clEnqueueReadBuffer(queue, gpu_result, CL_TRUE, 0, sizeof(float) *count, results, 0, NULL, NULL );
    if (err != CL_SUCCESS)
        print_opencl_error("clEnqueueReadBuffer", err);
        
    float chi2_temp = 0;
    int i;
    for(i = 0; i < count; i++)
        chi2_temp += results[i];
    
    // Shut down and clean up
    clReleaseMemObject(gpu_data);
    clReleaseMemObject(gpu_data_err);
    clReleaseMemObject(gpu_model);
    clReleaseMemObject(gpu_result);
    clReleaseProgram(pro_chi2);
    clReleaseKernel(kernel_chi2);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
        
    return chi2_temp/(double)(count);  
}


