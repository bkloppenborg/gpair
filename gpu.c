#include "gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

// Global variable to enable/disable debugging output:
int gpu_enable_verbose = 1;

// Global variables
cl_device_id * pDevice_id = NULL;           // device ID
cl_context * pContext = NULL;               // context
cl_command_queue * pQueue = NULL;           // command queue

// Pointers for programs and kernels:
cl_program * pPro_chi2 = NULL;
cl_kernel * pKernel_chi2 = NULL;
cl_program * pPro_powspec = NULL;
cl_kernel * pKernel_powspec = NULL;
cl_program * pPro_bispec  = NULL;
cl_kernel * pKernel_bispec  = NULL;

// Pointers for data stored on the GPU
cl_mem * pGpu_data = NULL;             // OIFITS Data
cl_mem * pGpu_data_err = NULL;         // OIFITS Data Error
cl_mem * pGpu_data_bip = NULL;         // OIFITS Data Biphasor
cl_mem * pGpu_data_uvpnt = NULL;       // OIFITS UV Point indicies for bispectrum data 
cl_mem * pGpu_data_sign = NULL;        // OIFITS UV Point signs.
cl_mem * pGpu_mock_data = NULL;        // Mock data (current "image")

#define SEP printf("-----------------------------------------------------------\n")


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
    if(gpu_enable_verbose)
        printf("Loading and compiling program '%s'\n\n", filename);
    
    // Load the kernel source:
    char * kernel_source = LoadProgramSourceFromFile(filename);
    // Create the program
    *program = clCreateProgramWithSource(*pContext, 1, (const char **) & kernel_source, NULL, &err);     
    if (err != CL_SUCCESS)   
        print_opencl_error("clCreateProgramWithSource", err);    
        
    // Build the program executable
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
    // Kernel and program for computing chi2:
    static cl_program pro_chi2;
    static cl_kernel kern_chi2;
    gpu_build_kernel(&pro_chi2, &kern_chi2, "compute_chi2", "./kernel_chi2.cl");
    pPro_chi2 = &pro_chi2;
    pKernel_chi2 = &kern_chi2;
    
    // Kernel and program for computing the power spectrum
    static cl_program pro_powspec;
    static cl_kernel kern_powspec;
    gpu_build_kernel(&pro_powspec, &kern_powspec, "compute_pow_spec", "./kernel_pow_spec.cl");
    pPro_powspec = &pro_powspec;
    pKernel_powspec = &kern_powspec;   
    
    // Kernel and program for computing the bispectrum
    static cl_program pro_bispec;
    static cl_kernel kern_bispec;
    gpu_build_kernel(&pro_bispec, &kern_bispec, "compute_bispec", "./kernel_bispec.cl");
    pPro_bispec = &pro_bispec;
    pKernel_bispec = &kern_bispec;
}

void gpu_cleanup()
{
    if(gpu_enable_verbose)
        printf("Freeing program, kernel, and device objects. \n");
        
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
    
    // Output some additional information if we are in verbose mode
    if(gpu_enable_verbose)
        printf("Creating buffers on the device. \n");
    
    // Create buffers on the device:    
    gpu_data = clCreateBuffer(*pContext,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, NULL);
    gpu_data_err = clCreateBuffer(*pContext,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, NULL); 
    gpu_data_bip = clCreateBuffer(*pContext,  CL_MEM_READ_ONLY,  sizeof(float) * nbis, NULL, NULL); 
    gpu_data_uvpnt = clCreateBuffer(*pContext, CL_MEM_READ_ONLY, sizeof(long) * 3 * nbis, NULL, NULL);
    gpu_data_sign = clCreateBuffer(*pContext, CL_MEM_READ_ONLY, sizeof(short) * 3 * nbis, NULL, NULL);
    gpu_mock_data = clCreateBuffer(*pContext, CL_MEM_READ_WRITE, sizeof(float) * count, NULL, NULL);
    if (!gpu_data || !gpu_data_err)
        print_opencl_error("Error: gpu_copy_data.  Create Buffer.", 0);

    if(gpu_enable_verbose)
        printf("Copying data to device. \n");

    // Copy the data over to the device:
    err = clEnqueueWriteBuffer(*pQueue, gpu_data, CL_TRUE, 0, sizeof(float) * count, data, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_data_err, CL_TRUE, 0, sizeof(float) * count, data_err, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_data_bip, CL_TRUE, 0, sizeof(float) * nbis, data_bip, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_data_uvpnt, CL_TRUE, 0, sizeof(long) * 3 * nbis, data_uvpnt, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_data_sign, CL_TRUE, 0, sizeof(short) * 3 * nbis, data_sign, 0, NULL, NULL);
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
double gpu_data2chi2(int npow, int nbis)
{
    int err;                          // error code returned from api calls

    size_t global;                    // global domain size for our calculation
    size_t local;                     // local domain size for our calculation

    //cl_mem gpu_model;
    cl_mem gpu_result;            // device memory used for the output array
    int count = npow+2*nbis;               // the length of the data
    
    // TODO: Remove this later.
    float * results;
    results = malloc(count * sizeof(float));
          
    
    // Create the input and output arrays in device memory for our calculation 
    //gpu_model = clCreateBuffer(*pContext,  CL_MEM_READ_ONLY,  sizeof(float) *count, NULL, NULL);
    gpu_result = clCreateBuffer(*pContext, CL_MEM_WRITE_ONLY, sizeof(float) *count, NULL, NULL);
    if (!gpu_result)
        print_opencl_error("clCreateBuffer", 0);

    // Write our data set into the input array in device memory          
    //err = clEnqueueWriteBuffer(*pQueue, gpu_model, CL_TRUE, 0, sizeof(float) *count, mock, 0, NULL, NULL);
    //if (err != CL_SUCCESS)
    //    print_opencl_error("clEnqueueWriteBuffer gpu_model", err);  
    
    int local_size = 16;    // TODO: Figure this out dynamically.

    // Set the arguments to our compute kernel                         
    err = 0;
    err  = clSetKernelArg(*pKernel_chi2, 0, sizeof(cl_mem), pGpu_data);
    err |= clSetKernelArg(*pKernel_chi2, 1, sizeof(cl_mem), pGpu_data_err);
    err |= clSetKernelArg(*pKernel_chi2, 2, sizeof(cl_mem), pGpu_mock_data);
    err |= clSetKernelArg(*pKernel_chi2, 3, sizeof(cl_mem), &gpu_result);
    err |= clSetKernelArg(*pKernel_chi2, 4, sizeof(float) * 3 * local_size, NULL);
    err |= clSetKernelArg(*pKernel_chi2, 5, sizeof(unsigned int), &count);
    if (err != CL_SUCCESS)
        print_opencl_error("clSetKernelArg", err);

    // Get the maximum work-group size for executing the kernel on the device
    //err = clGetKernelWorkGroupInfo(*pKernel_chi2, *pDevice_id, CL_KERNEL_WORK_GROUP_SIZE , sizeof(size_t), &local, NULL);
    //if (err != CL_SUCCESS)
    //    print_opencl_error("clGetKernelWorkGroupInfo", err);


    // Execute the kernel over the entire range of the data set        
    global = local_size;
    err = clEnqueueNDRangeKernel(*pQueue, *pKernel_chi2, 1, NULL, &global, NULL, 0, NULL, NULL);
    if (err)
        print_opencl_error("clEnqueueNDRangeKernel chi2", err);

    
    // Wait for the command queue to get serviced before reading back results
    clFinish(*pQueue);

    // Read the results from the device                                  
    err = clEnqueueReadBuffer(*pQueue, gpu_result, CL_TRUE, 0, sizeof(float) *count, results, 0, NULL, NULL );
    if (err != CL_SUCCESS)
        print_opencl_error("clEnqueueReadBuffer gpu_result", err);
        
    float chi2_temp = 0;
    int i;
    for(i = 0; i < count; i++)
        chi2_temp += results[i];
    
    // Shut down and clean up
    //clReleaseMemObject(gpu_model);
    clReleaseMemObject(gpu_result);
        
    return chi2_temp/(double)(count);  
}

int gpu_device_stats(cl_device_id device_id)
{	
	int err;
	int i;
	size_t j;
	size_t returned_size;
	
	printf("\n");
	// Report the device vendor and device name
    // 
    cl_char vendor_name[1024] = {0};
    cl_char device_name[1024] = {0};
	cl_char device_profile[1024] = {0};
	cl_char device_extensions[1024] = {0};
	cl_device_local_mem_type local_mem_type;
	
    cl_ulong global_mem_size, global_mem_cache_size;
	cl_ulong max_mem_alloc_size;
	
	cl_uint clock_frequency, vector_width, max_compute_units;
	
	size_t max_work_item_dims,max_work_group_size, max_work_item_sizes[3];
	
	cl_uint vector_types[] = {CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE}; 
	char *vector_type_names[] = {"char","short","int","long","float","double"};
	
	err = clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(vendor_name), vendor_name, &returned_size);
    err|= clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), device_name, &returned_size);
	err|= clGetDeviceInfo(device_id, CL_DEVICE_PROFILE, sizeof(device_profile), device_profile, &returned_size);
	err|= clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, sizeof(device_extensions), device_extensions, &returned_size);
	err|= clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(local_mem_type), &local_mem_type, &returned_size);
	
	err|= clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, &returned_size);
	err|= clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(global_mem_cache_size), &global_mem_cache_size, &returned_size);
	err|= clGetDeviceInfo(device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_mem_alloc_size), &max_mem_alloc_size, &returned_size);
	
	err|= clGetDeviceInfo(device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, &returned_size);
	
	err|= clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, &returned_size);
	
	err|= clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(max_work_item_dims), &max_work_item_dims, &returned_size);
	
	err|= clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_item_sizes), max_work_item_sizes, &returned_size);
	
	err|= clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_compute_units), &max_compute_units, &returned_size);
	
	printf("Vendor: %s\n", vendor_name);
	printf("Device Name: %s\n", device_name);
	printf("Profile: %s\n", device_profile);
	printf("Supported Extensions: %s\n\n", device_extensions);
	
	printf("Local Mem Type (Local=1, Global=2): %i\n",(int)local_mem_type);
	printf("Global Mem Size (MB): %i\n",(int)global_mem_size/(1024*1024));
	printf("Global Mem Cache Size (Bytes): %i\n",(int)global_mem_cache_size);
	printf("Max Mem Alloc Size (MB): %ld\n",(long int)max_mem_alloc_size/(1024*1024));
	
	printf("Clock Frequency (MHz): %i\n\n",clock_frequency);
	
	for(i = 0; i < 6; i++)
	{
		err|= clGetDeviceInfo(device_id, vector_types[i], sizeof(clock_frequency), &vector_width, &returned_size);
		printf("Vector type width for: %s = %i\n",vector_type_names[i],vector_width);
	}
	
	printf("\nMax Work Group Size: %lu\n",max_work_group_size);
	printf("Max Work Item Dims: %lu\n",max_work_item_dims);
	for(j = 0; j < max_work_item_dims; j++) 
		printf("Max Work Items in Dim %lu: %lu\n",(long unsigned)(j+1),(long unsigned)max_work_item_sizes[j]);
	
	printf("Max Compute Units: %i\n",max_compute_units);
	printf("\n");
	
	return CL_SUCCESS;
}

void gpu_init()
{
    // Init a few variables.  Static so they won't go out of scope.
    static cl_device_id device_id;           // device ID
    static cl_context context;               // context
    static cl_command_queue queue;           // command queue
    
    int err = 0;
    
    // Get an ID for the device
    err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS)   //      [3]
        print_opencl_error("Unable to get GPU Device", err);    
    
    // Output some information about the card if we are in debug mode.    
    if(gpu_enable_verbose)
        gpu_device_stats(device_id);                                  

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
    int err = 0;

    size_t global;                    // global domain size for our calculation
    size_t local;                     // local domain size for our calculation
    
    // ############
    // First we run a kernel to compute the powerspectrum:
    // ############
    gpu_vis = clCreateBuffer(*pContext,  CL_MEM_READ_ONLY,  sizeof(cl_float2) * nuv, NULL, NULL);
    if (!gpu_vis)
        print_opencl_error("clCreateBuffer", 0);
    
    err |= clEnqueueWriteBuffer(*pQueue, gpu_vis, CL_TRUE, 0, sizeof(cl_float2) * nuv, vis, 0, NULL, NULL);
    if (err != CL_SUCCESS)
        print_opencl_error("clEnqueueWriteBuffer gpu_vis", err);     

    err  = clSetKernelArg(*pKernel_powspec, 0, sizeof(cl_mem), &gpu_vis);
    err |= clSetKernelArg(*pKernel_powspec, 1, sizeof(cl_mem), pGpu_mock_data);    // Output is stored on the GPU.
    if (err != CL_SUCCESS)
        print_opencl_error("clSetKernelArg", err);    
 

   // Get the maximum work-group size for executing the kernel on the device
    err = clGetKernelWorkGroupInfo(*pKernel_powspec, *pDevice_id, CL_KERNEL_WORK_GROUP_SIZE , sizeof(size_t), &local, NULL);
    if (err != CL_SUCCESS)
        print_opencl_error("clGetKernelWorkGroupInfo", err);

    // Execute the kernel over the entire range of the data set        
    global = npow;
    err = clEnqueueNDRangeKernel(*pQueue, *pKernel_powspec, 1, NULL, &global, NULL, 0, NULL, NULL);
    if (err)
        print_opencl_error("clEnqueueNDRangeKernel vis", err);   
        
    clFinish(*pQueue);
        
    // ############
    // Now we run a kernel to compute the bispectrum:
    // ############
    err  = clSetKernelArg(*pKernel_bispec, 0, sizeof(cl_mem), &gpu_vis);
    err |= clSetKernelArg(*pKernel_bispec, 1, sizeof(cl_mem), pGpu_data_bip);
    err |= clSetKernelArg(*pKernel_bispec, 2, sizeof(cl_mem), pGpu_data_uvpnt);
    err |= clSetKernelArg(*pKernel_bispec, 3, sizeof(cl_mem), pGpu_data_sign);
    err |= clSetKernelArg(*pKernel_bispec, 4, sizeof(cl_mem), pGpu_mock_data);
    err |= clSetKernelArg(*pKernel_bispec, 5, sizeof(int), &npow);    // Output is stored on the GPU.
    if (err != CL_SUCCESS)
        print_opencl_error("clSetKernelArg", err); 
 
   // Get the maximum work-group size for executing the kernel on the device
    err = clGetKernelWorkGroupInfo(*pKernel_bispec, *pDevice_id, CL_KERNEL_WORK_GROUP_SIZE , sizeof(size_t), &local, NULL);
    if (err != CL_SUCCESS)
        print_opencl_error("clGetKernelWorkGroupInfo", err);

    // Execute the kernel over the entire range of the data set        
    global = nbis;
    err = clEnqueueNDRangeKernel(*pQueue, *pKernel_bispec, 1, NULL, &global, NULL, 0, NULL, NULL);
    if (err)
        print_opencl_error("clEnqueueNDRangeKernel vis", err);   
        
    clReleaseMemObject(gpu_vis);
    clFinish(*pQueue);
}

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
