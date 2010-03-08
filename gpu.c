#include "gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#define MAX_GROUPS      (64)
#define MAX_WORK_ITEMS  (64)
#define SEP "-----------------------------------------------------------\n"

// Global variable to enable/disable debugging output:
int gpu_enable_verbose = 0;     // Turns on verbose output from GPU messages.
int gpu_enable_debug = 0;       // Turns on debugging output, slows stuff down considerably.

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
cl_program * pGpu_chi2_programs = NULL;
cl_kernel * pGpu_chi2_kernels = NULL;
cl_program * pGpu_flux_programs = NULL;
cl_kernel * pGpu_flux_kernels = NULL;

// Pointers for data stored on the GPU
cl_mem * pGpu_data = NULL;             // OIFITS Data
cl_mem * pGpu_data_err = NULL;         // OIFITS Data Error
cl_mem * pGpu_data_bip = NULL;         // OIFITS Data Biphasor
cl_mem * pGpu_data_uvpnt = NULL;       // OIFITS UV Point indicies for bispectrum data 
cl_mem * pGpu_data_sign = NULL;        // OIFITS UV Point signs.
cl_mem * pGpu_mock_data = NULL;        // Mock data (current "image")

cl_mem * pGpu_chi2 = NULL;             // Buffer for storing the (single summed) chi2 value.
cl_mem * pGpu_chi2_buffer0 = NULL;       // Used as input buffer
cl_mem * pGpu_chi2_buffer1 = NULL;       // Used as output buffer
cl_mem * pGpu_chi2_buffer2 = NULL;       // Used as partial sum buffer

cl_mem * pGpu_dft_x = NULL;         // Pointer to Memory for x-DFT table
cl_mem * pGpu_dft_y = NULL;         // Pointer to Memory for y-DFT table
cl_mem * pGpu_image = NULL;

cl_mem * pGpu_flux = NULL;          // Buffer storing the (single, summed) flux value.
cl_mem * pGpu_flux_buffer0 = NULL;  // Used as input buffer
cl_mem * pGpu_flux_buffer1 = NULL;  // Used as output buffer
cl_mem * pGpu_flux_buffer2 = NULL;  // Used as partial sum buffer

// Variables for the parallel sum in the chi2 (again, globals... urgh).
int Chi2_pass_count = 0;
size_t * Chi2_group_counts = NULL;
size_t * Chi2_work_item_counts = NULL;
int * Chi2_operation_counts = NULL;
int * Chi2_entry_counts = NULL;

int Flux_pass_count = 0;
size_t * Flux_group_counts = NULL;
size_t * Flux_work_item_counts = NULL;
int * Flux_operation_counts = NULL;
int * Flux_entry_counts = NULL;

void create_reduction_pass_counts(
    int count, 
    int max_group_size,    
    int max_groups,
    int max_work_items, 
    int *pass_count, 
    size_t **group_counts, 
    size_t **work_item_counts,
    int **operation_counts,
    int **entry_counts)
{
    int work_items = (count < max_work_items * 2) ? count / 2 : max_work_items;
    if(count < 1)
        work_items = 1;
        
    int groups = count / (work_items * 2);
    groups = max_groups < groups ? max_groups : groups;

    int max_levels = 1;
    int s = groups;

    while(s > 1) 
    {
        int work_items = (s < max_work_items * 2) ? s / 2 : max_work_items;
        s = s / (work_items*2);
        max_levels++;
    }
 
    *group_counts = (size_t*)malloc(max_levels * sizeof(size_t));
    *work_item_counts = (size_t*)malloc(max_levels * sizeof(size_t));
    *operation_counts = (int*)malloc(max_levels * sizeof(int));
    *entry_counts = (int*)malloc(max_levels * sizeof(int));

    (*pass_count) = max_levels;
    (*group_counts)[0] = groups;
    (*work_item_counts)[0] = work_items;
    (*operation_counts)[0] = 1;
    (*entry_counts)[0] = count;
    if(max_group_size < work_items)
    {
        (*operation_counts)[0] = work_items;
        (*work_item_counts)[0] = max_group_size;
    }
    
    s = groups;
    int level = 1;
   
    while(s > 1) 
    {
        int work_items = (s < max_work_items * 2) ? s / 2 : max_work_items;
        int groups = s / (work_items * 2);
        groups = (max_groups < groups) ? max_groups : groups;

        (*group_counts)[level] = groups;
        (*work_item_counts)[level] = work_items;
        (*operation_counts)[level] = 1;
        (*entry_counts)[level] = s;
        if(max_group_size < work_items)
        {
            (*operation_counts)[level] = work_items;
            (*work_item_counts)[level] = max_group_size;
        }
        
        s = s / (work_items*2);
        level++;
    }
}

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
        gpu_cleanup();
        exit(1);
    }
   
    // Create the compute kernel in the program we wish to run           
    *kernel = clCreateKernel(*program, kernel_name, &err);
    if (!kernel || err != CL_SUCCESS)
        print_opencl_error("clCreateKernel", err); 
}

void gpu_build_kernels(int data_size, int image_size)
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
    
    // Now build the reduction kernels
    gpu_build_reduction_kernels(data_size, &pGpu_chi2_programs, &pGpu_chi2_kernels, &Chi2_pass_count, &Chi2_group_counts, &Chi2_work_item_counts, &Chi2_operation_counts, &Chi2_entry_counts);
    gpu_build_reduction_kernels(image_size, &pGpu_flux_programs, &pGpu_flux_kernels, &Flux_pass_count, &Flux_group_counts, &Flux_work_item_counts, &Flux_operation_counts, &Flux_entry_counts);
}

void gpu_build_reduction_kernels(int data_size, cl_program ** pPrograms, cl_kernel ** pKernels, 
    int * pass_count, size_t ** group_counts, size_t ** work_item_counts, 
    int ** operation_counts, int ** entry_counts)
{
    // Init a few variables:
    int err = 0;
    int i;
    char * source = LoadProgramSourceFromFile("./kernel_reduce_float.cl");

    size_t returned_size = 0;
    size_t max_workgroup_size = 0;
    err = clGetDeviceInfo(*pDevice_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_workgroup_size, &returned_size);
    if(err != CL_SUCCESS)
        print_opencl_error("Couldn't get maximum work group size from the device.", err); 
    
    // Determine the reduction pass configuration for each level in the pyramid
    create_reduction_pass_counts(data_size, max_workgroup_size, MAX_GROUPS, MAX_WORK_ITEMS, pass_count, group_counts, work_item_counts, operation_counts, entry_counts);

    // Create specialized programs and kernels for each level of the reduction
    cl_program * programs = (cl_program*)malloc((*pass_count) * sizeof(cl_program));
    memset(programs, 0, (*pass_count) * sizeof(cl_program));

    cl_kernel * kernels = (cl_kernel*)malloc((*pass_count) * sizeof(cl_kernel));
    memset(kernels, 0, (*pass_count) * sizeof(cl_kernel));

    for(i = 0; i < (*pass_count); i++)
    {
        char *block_source = malloc(strlen(source) + 1024);
        size_t source_length = strlen(source) + 1024;
        memset(block_source, 0, source_length);
        
        // Insert macro definitions to specialize the kernel to a particular group size
        //
        const char group_size_macro[] = "#define GROUP_SIZE";
        const char operations_macro[] = "#define OPERATIONS";
        sprintf(block_source, "%s (%d) \n%s (%d)\n\n%s\n", 
            group_size_macro, (int)(*group_counts)[i], 
            operations_macro, (int)(*operation_counts)[i], 
            source);
        
        // Create the compute program from the source buffer
        //
        programs[i] = clCreateProgramWithSource(*pContext, 1, (const char **) & block_source, NULL, &err);
        if (!programs[i] || err != CL_SUCCESS)
        {
            printf("%s\n", block_source);
            printf("Error: Failed to create compute program!\n");
            gpu_cleanup();
            exit(1);
        }
    
        // Build the program executable
        //
        err = clBuildProgram(programs[i], 0, NULL, NULL, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            size_t length;
            char build_log[2048];
            printf("%s\n", block_source);
            printf("Error: Failed to build program executable!\n");
            clGetProgramBuildInfo(programs[i], *pDevice_id, CL_PROGRAM_BUILD_LOG, sizeof(build_log), build_log, &length);
            printf("%s\n", build_log);
            gpu_cleanup();
            exit(1);
        }
    
        // Create the compute kernel from within the program
        //
        kernels[i] = clCreateKernel(programs[i], "reduce", &err);
        if (!kernels[i] || err != CL_SUCCESS)
            print_opencl_error("Failed to create parallel sum kernels.", err); 

        free(block_source);
    }
    
    (*pPrograms) = programs;
    (*pKernels) = kernels;
}

void gpu_cleanup()
{
    int i;
    if(gpu_enable_verbose)
        printf("Freeing program, kernel, and device objects. \n");
        
    // Release program and kernel objects:
    if(pPro_chi2 != NULL)
        clReleaseProgram(*pPro_chi2);
    if(pPro_powspec != NULL)
        clReleaseProgram(*pPro_powspec);
    if(pGpu_chi2_programs != NULL)
    {
        for(i = 0; i < Chi2_pass_count; i++)
            clReleaseProgram(pGpu_chi2_programs[i]);
    }
    if(pGpu_flux_programs != NULL)
    {
        for(i = 0; i < Flux_pass_count; i++)
            clReleaseProgram(pGpu_flux_programs[i]);
    }
    
    if(pKernel_chi2 != NULL)
        clReleaseKernel(*pKernel_chi2);
    if(pKernel_powspec != NULL)
        clReleaseKernel(*pKernel_powspec);
    if(pGpu_chi2_kernels != NULL)
    {
        for(i = 0; i < Chi2_pass_count; i++)
            clReleaseKernel(pGpu_chi2_kernels[i]);
    }
    if(pGpu_flux_kernels != NULL)
    {
        for(i = 0; i < Flux_pass_count; i++)
            clReleaseKernel(pGpu_flux_kernels[i]);
    }

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

    if(pGpu_dft_x != NULL)
        clReleaseMemObject(*pGpu_dft_x);
    if(pGpu_dft_y != NULL)
        clReleaseMemObject(*pGpu_dft_y);
    if(pGpu_image != NULL)
        clReleaseMemObject(*pGpu_image);

    // Release chi2 memory objects:
    if(pGpu_chi2 != NULL)
        clReleaseMemObject(*pGpu_chi2);
    if(pGpu_chi2_buffer0 != NULL)
        clReleaseMemObject(*pGpu_chi2_buffer0);
    if(pGpu_chi2_buffer1 != NULL)
        clReleaseMemObject(*pGpu_chi2_buffer1);
    if(pGpu_chi2_buffer2 != NULL)
        clReleaseMemObject(*pGpu_chi2_buffer2);

    // Release flux memory objects:
    if(pGpu_flux != NULL)
        clReleaseMemObject(*pGpu_flux);
    if(pGpu_flux_buffer0 != NULL)
        clReleaseMemObject(*pGpu_flux_buffer0);
    if(pGpu_flux_buffer1 != NULL)
        clReleaseMemObject(*pGpu_flux_buffer1);
    if(pGpu_flux_buffer2 != NULL)
        clReleaseMemObject(*pGpu_flux_buffer2);



    // Release the command queue and context:
    if(pQueue != NULL)
        clReleaseCommandQueue(*pQueue);
    if(pContext != NULL)
        clReleaseContext(*pContext);
    
}

void gpu_compute_sum(cl_mem * input_buffer, cl_mem * output_buffer, cl_mem * partial_sum_buffer, cl_mem * final_buffer, 
    cl_kernel * pKernels, 
    int pass_count, size_t * group_counts, size_t * work_item_counts, 
    int * operation_counts, int * entry_counts)
{
    int i;
    int err;
    // Do the reduction for each level  
    //
    cl_mem pass_swap;
    cl_mem pass_input = *output_buffer;
    cl_mem pass_output = *input_buffer;
    cl_mem partials_buffer = *partial_sum_buffer; // Partial sum buffer

    for(i = 0; i < pass_count; i++)
    {
        size_t global = group_counts[i] * work_item_counts[i];        
        size_t local = work_item_counts[i];
        unsigned int operations = operation_counts[i];
        unsigned int entries = entry_counts[i];
        size_t shared_size = sizeof(float) * local * operations;

        if(gpu_enable_debug && gpu_enable_verbose)
        {
            printf("Pass[%4d] Global[%4d] Local[%4d] Groups[%4d] WorkItems[%4d] Operations[%d] Entries[%d]\n",  i, 
                (int)global, (int)local, (int)group_counts[i], (int)work_item_counts[i], operations, entries);
        }

        // Swap the inputs and outputs for each pass
        //
        pass_swap = pass_input;
        pass_input = pass_output;
        pass_output = pass_swap;
        
        err = CL_SUCCESS;
        err |= clSetKernelArg(pKernels[i],  0, sizeof(cl_mem), &pass_output);  
        err |= clSetKernelArg(pKernels[i],  1, sizeof(cl_mem), &pass_input);
        err |= clSetKernelArg(pKernels[i],  2, shared_size,    NULL);
        err |= clSetKernelArg(pKernels[i],  3, sizeof(int),    &entries);
        if (err != CL_SUCCESS)
            print_opencl_error("Failed to set partial sum kernel arguments.", err); 
        
        // After the first pass, use the partial sums for the next input values
        //
        if(pass_input == *input_buffer)
            pass_input = partials_buffer;
            
        err = CL_SUCCESS;
        err |= clEnqueueNDRangeKernel(*pQueue, pGpu_chi2_kernels[i], 1, NULL, &global, &local, 0, NULL, NULL);
        if (err != CL_SUCCESS)
            print_opencl_error("Failed to enqueue parallel sum kernels.", err); 
    }

    // Let the queue complete.
    clFinish(*pQueue);

    // Copy the new chi2 value over to it's final place in GPU memory.
    err = clEnqueueCopyBuffer(*pQueue, pass_output, *final_buffer, 0, 0, sizeof(float), 0, NULL, NULL);
    if(err != CL_SUCCESS)
        print_opencl_error("Could not copy summed value to/from buffers on the GPU.", err);
        
    if(gpu_enable_debug)
    {
        float chi2 = 0;
        err = clEnqueueReadBuffer(*pQueue, pass_output, CL_TRUE, 0, sizeof(float), &chi2, 0, NULL, NULL );
        if(err != CL_SUCCESS)
            print_opencl_error("Could not read back summed GPU value.", err);
        
        printf("GPU Sum: %f \n", chi2);

        chi2 = 0;
        err = clEnqueueReadBuffer(*pQueue, *final_buffer, CL_TRUE, 0, sizeof(float), &chi2, 0, NULL, NULL );
        if(err != CL_SUCCESS)
            print_opencl_error("Could not read back GPU chi2 value.", err);
        
        printf("GPU Copied Value: %f \n", chi2);
    }        
}

// Init memory locations and copy data over to the GPU.
void gpu_copy_data(float *data, float *data_err, int data_size,\
                    cl_float2 * data_bis, int bis_size,\
                    long * gpu_bsref_uvpnt, short * gpu_bsref_sign, int bsref_size,
                    int image_size)
{
    int err = 0;

    static cl_mem gpu_data;         // Data
    static cl_mem gpu_data_err;     // Data Error
    static cl_mem gpu_data_bip;     // Biphasor
    static cl_mem gpu_data_uvpnt;   // UV Points for the bispectrum
    static cl_mem gpu_data_sign;    // Signs for the bispectrum.
    static cl_mem gpu_mock_data;    // Mock Data
    
    static cl_mem gpu_chi2;
    static cl_mem gpu_chi2_buffer0;       // Temporary storage for the chi2 computation.
    static cl_mem gpu_chi2_buffer1;       // Temporary storage for the chi2 computation.
    static cl_mem gpu_chi2_buffer2;     // Temporary storage for the chi2 computation.
    
    static cl_mem gpu_flux;
    static cl_mem gpu_flux_buffer0;       // Temporary storage for the chi2 computation.
    static cl_mem gpu_flux_buffer1;       // Temporary storage for the chi2 computation.
    static cl_mem gpu_flux_buffer2;     // Temporary storage for the chi2 computation.    

    // Init some mock data (to allow resumes in the future I suppose...)
    int i = 0;
    float * mock_data;
    float * temp;
    float zero = 0;
    mock_data = malloc(data_size * sizeof(float));
    temp = malloc(data_size * sizeof(float));   
    for(i = 0; i < data_size; i++)
        mock_data[i] = 0; 
        temp[i] = 0; 
 
    float * zero_flux;
    zero_flux = malloc(image_size * sizeof(float));
    for(i = 0; i < image_size; i++)
        zero_flux[i] = 0;        
    
    // Output some additional information if we are in verbose mode
    if(gpu_enable_verbose)
        printf("Creating buffers on the device. \n");
    
    // Create buffers on the device:    
    gpu_data = clCreateBuffer(*pContext,  CL_MEM_READ_ONLY,  sizeof(float) * data_size, NULL, NULL);
    gpu_data_err = clCreateBuffer(*pContext,  CL_MEM_READ_ONLY,  sizeof(float) * data_size, NULL, NULL); 
    gpu_data_bip = clCreateBuffer(*pContext,  CL_MEM_READ_ONLY,  sizeof(float) * bis_size, NULL, NULL); 
    gpu_data_uvpnt = clCreateBuffer(*pContext, CL_MEM_READ_ONLY, sizeof(long) * bsref_size, NULL, NULL);
    gpu_data_sign = clCreateBuffer(*pContext, CL_MEM_READ_ONLY, sizeof(short) * bsref_size, NULL, NULL);
    gpu_mock_data = clCreateBuffer(*pContext, CL_MEM_READ_WRITE, sizeof(float) * data_size, NULL, NULL);
    gpu_chi2 = clCreateBuffer(*pContext, CL_MEM_READ_WRITE, sizeof(float), NULL, NULL);
    gpu_chi2_buffer0 = clCreateBuffer(*pContext, CL_MEM_READ_WRITE, sizeof(float) * data_size, NULL, NULL);
    gpu_chi2_buffer1 = clCreateBuffer(*pContext, CL_MEM_READ_WRITE, sizeof(float) * data_size, NULL, NULL);
    gpu_chi2_buffer2 = clCreateBuffer(*pContext, CL_MEM_READ_WRITE, sizeof(float) * data_size, NULL, NULL);
    
    gpu_flux = clCreateBuffer(*pContext, CL_MEM_READ_WRITE, sizeof(float), NULL, &err);
    gpu_flux_buffer0 = clCreateBuffer(*pContext, CL_MEM_READ_WRITE, sizeof(float) * image_size, NULL, &err);
    gpu_flux_buffer1 = clCreateBuffer(*pContext, CL_MEM_READ_WRITE, sizeof(float) * image_size, NULL, &err);
    gpu_flux_buffer2 = clCreateBuffer(*pContext, CL_MEM_READ_WRITE, sizeof(float) * image_size, NULL, &err);
    
    if (err != CL_SUCCESS)
        print_opencl_error("Error: gpu_copy_data.  Create Buffer.", err);

    if(gpu_enable_verbose)
        printf("Copying data to device. \n");

    // Copy the data over to the device.  (note, non-blocking cals)
    err = clEnqueueWriteBuffer(*pQueue, gpu_data, CL_FALSE, 0, sizeof(float) * data_size, data, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_data_err, CL_FALSE, 0, sizeof(float) * data_size, data_err, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_data_bip, CL_FALSE, 0, sizeof(float) * bis_size, data_bis, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_data_uvpnt, CL_FALSE, 0, sizeof(long) * bsref_size, gpu_bsref_uvpnt, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_data_sign, CL_FALSE, 0, sizeof(short) * bsref_size, gpu_bsref_sign, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_mock_data, CL_FALSE, 0, sizeof(float) * data_size, mock_data, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_chi2, CL_FALSE, 0, sizeof(float), &zero, 0, NULL, NULL);        
    err |= clEnqueueWriteBuffer(*pQueue, gpu_chi2_buffer0, CL_FALSE, 0, sizeof(float) * data_size, temp, 0, NULL, NULL);  
    err |= clEnqueueWriteBuffer(*pQueue, gpu_chi2_buffer1, CL_FALSE, 0, sizeof(float) * data_size, temp, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_chi2_buffer2, CL_FALSE, 0, sizeof(float) * data_size, temp, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_flux, CL_FALSE, 0, sizeof(float), &zero, 0, NULL, NULL);        
    err |= clEnqueueWriteBuffer(*pQueue, gpu_flux_buffer0, CL_FALSE, 0, sizeof(float) * image_size, zero_flux, 0, NULL, NULL);  
    err |= clEnqueueWriteBuffer(*pQueue, gpu_flux_buffer1, CL_FALSE, 0, sizeof(float) * image_size, zero_flux, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_flux_buffer2, CL_FALSE, 0, sizeof(float) * image_size, zero_flux, 0, NULL, NULL);
    
    if (err != CL_SUCCESS)
        print_opencl_error("Error: gpu_copy_data. Write Buffer", err);    
 
    clFinish(*pQueue);
        
    pGpu_data = &gpu_data;
    pGpu_data_err = &gpu_data_err; 
    pGpu_data_bip = &gpu_data_bip;
    pGpu_data_uvpnt = &gpu_data_uvpnt;
    pGpu_data_sign = &gpu_data_sign;
    pGpu_mock_data = &gpu_mock_data;
    pGpu_chi2 = &gpu_chi2;
    pGpu_chi2_buffer0 = &gpu_chi2_buffer0;
    pGpu_chi2_buffer1 = &gpu_chi2_buffer1;
    pGpu_chi2_buffer2 = &gpu_chi2_buffer2;
    pGpu_flux = &gpu_flux;   
    pGpu_flux_buffer0 = &gpu_flux_buffer0;
    pGpu_flux_buffer1 = &gpu_flux_buffer1;
    pGpu_flux_buffer2 = &gpu_flux_buffer2;
}

// Copy the DFT tables over to GPU memory
void gpu_copy_dft(cl_float2 * dft_x, cl_float2 * dft_y, int dft_size)
{
    int err;
    static cl_mem dft_t_x;  // DFT Table x
    static cl_mem dft_t_y;  // DFT Table y
    
    // Allocate memory on the device:
    dft_t_x = clCreateBuffer(*pContext,  CL_MEM_READ_ONLY,  sizeof(cl_float2) * dft_size, NULL, NULL);
    dft_t_y = clCreateBuffer(*pContext,  CL_MEM_READ_ONLY,  sizeof(cl_float2) * dft_size, NULL, NULL);
    
    // Copy the data over
    err = clEnqueueWriteBuffer(*pQueue, dft_t_x, CL_FALSE, 0, sizeof(cl_float2) * dft_size, dft_x, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, dft_t_y, CL_FALSE, 0, sizeof(cl_float2) * dft_size, dft_y, 0, NULL, NULL);
    if (err != CL_SUCCESS)
        print_opencl_error("Error copying DFT table to the GPU", err);       
        
    clFinish(*pQueue);
    
    pGpu_dft_x = & dft_t_x;
    pGpu_dft_y = & dft_t_y;
}

// Copys an image over to the GPU for analysis
void gpu_copy_image(float * image, int x_size, int y_size)
{
    int size = x_size * y_size * sizeof(float);
    int err;
    
    if(pGpu_image == NULL)
    {
        static cl_mem gpu_image;
        gpu_image = clCreateBuffer(*pContext, CL_MEM_READ_ONLY, size, NULL, NULL);
        pGpu_image = &gpu_image;
    }
    
    // Copy the data over, do this as a blocking call.
    err = clEnqueueWriteBuffer(*pQueue, *pGpu_image, CL_TRUE, 0, size, image, 0, NULL, NULL);
    if(err != CL_SUCCESS)
        print_opencl_error("Error copying image to the GPU", err);
    
}

// Compute the chi2 of the data using a GPU
void gpu_data2chi2(int data_size)
{
    int err;                          // error code returned from api calls

    size_t global;                    // global domain size for our calculation
    size_t local;                     // local domain size for our calculation

    // Set the arguments to our compute kernel                         
    err = 0;
    err  = clSetKernelArg(*pKernel_chi2, 0, sizeof(cl_mem), pGpu_data);
    err |= clSetKernelArg(*pKernel_chi2, 1, sizeof(cl_mem), pGpu_data_err);
    err |= clSetKernelArg(*pKernel_chi2, 2, sizeof(cl_mem), pGpu_mock_data);
    err |= clSetKernelArg(*pKernel_chi2, 3, sizeof(cl_mem), pGpu_chi2_buffer0);
    if (err != CL_SUCCESS)
        print_opencl_error("clSetKernelArg", err);

    // Get the maximum work-group size for executing the kernel on the device
    err = clGetKernelWorkGroupInfo(*pKernel_chi2, *pDevice_id, CL_KERNEL_WORK_GROUP_SIZE , sizeof(size_t), &local, NULL);
    if (err != CL_SUCCESS)
        print_opencl_error("clGetKernelWorkGroupInfo", err);

    // Execute the kernel over the entire range of the data set        
    global = data_size;
    err = clEnqueueNDRangeKernel(*pQueue, *pKernel_chi2, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
        print_opencl_error("clEnqueueNDRangeKernel chi2", err);
    
    // Wait for the command queue to finish
    clFinish(*pQueue);

    // If we have debugging turned on, read in the individual chi2 values and output the summed chi2.
    if(gpu_enable_debug)
    {
        int i;
        float * results;
        results = malloc(data_size * sizeof(float));
        err = clEnqueueReadBuffer(*pQueue, *pGpu_chi2_buffer0, CL_TRUE, 0, sizeof(float) * data_size, results, 0, NULL, NULL );
            if (err != CL_SUCCESS)
                print_opencl_error("Could not read back GPU chi2 array elements.", err);
        
        float chi2 = 0;
        for(i = 0; i < data_size; i++)
        {
              chi2 += results[i];
              //printf("%f ", results[i]);  // Enable if you want to see the elements of the results array.
        }

        printf("\n");
        printf(SEP);  
        printf("GPU Chi2: %f (summed on the CPU)\n", chi2);
    }
    
    // Now start up the partial sum kernel:
    gpu_compute_sum(pGpu_chi2_buffer0, pGpu_chi2_buffer1, pGpu_chi2_buffer2, pGpu_chi2, pGpu_chi2_kernels, Chi2_pass_count, Chi2_group_counts, Chi2_work_item_counts, Chi2_operation_counts, Chi2_entry_counts);

}

void gpu_device_stats(cl_device_id device_id)
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

void gpu_image2vis()
{ 
    // First, compute the total flux.  Store the result in the GPU buffer, pGpu_flux
    gpu_compute_sum(pGpu_image, pGpu_flux_buffer1, pGpu_flux_buffer2, pGpu_flux, pGpu_flux_kernels, Flux_pass_count, Flux_group_counts, Flux_work_item_counts, Flux_operation_counts, Flux_entry_counts);

    // DFT
/*    int ii, jj, uu;	*/
/*    float v0 = 0.; // zeroflux */

/*    for(ii=0 ; ii < model_image_size * model_image_size ; ii++) */
/*        v0 += current_image[ii];*/

/*    for(uu=0 ; uu < nuv; uu++)*/
/*    {*/
/*        visi[uu] = 0.0 + I * 0.0;*/
/*        for(ii=0; ii < model_image_size; ii++)*/
/*            for(jj=0; jj < model_image_size; jj++)*/
/*                visi[uu] += current_image[ ii + model_image_size * jj ] *  DFT_tablex[ model_image_size * uu +  ii] * DFT_tablex[ model_image_size * uu +  jj];*/
/*        if (v0 > 0.) visi[uu] /= v0;*/
/*    }*/
/*  */
/*  printf("Check - visi 0 %f %f\n", creal(visi[0]), cimag(visi[0]));*/

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
