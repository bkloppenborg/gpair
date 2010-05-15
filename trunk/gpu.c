#include "gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>
#include <complex.h>

#ifndef GPAIR
#include "gpair.h"
#endif

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
cl_program * pPro_visi = NULL;
cl_kernel * pKernel_visi = NULL;
cl_program * pPro_u_vis_flux = NULL;
cl_kernel * pKernel_u_vis_flux = NULL;
cl_program * pPro_grad_pow = NULL;
cl_kernel * pKernel_grad_pow = NULL;
cl_program * pPro_grad_bis = NULL;
cl_kernel * pKernel_grad_bis = NULL;
cl_program * pPro_entropy = NULL;
cl_kernel * pKernel_entropy = NULL;
cl_program * pPro_entropy_grad = NULL;
cl_kernel * pKernel_entropy_grad = NULL;
cl_program * pPro_criterion_grad = NULL;
cl_kernel * pKernel_criterion_grad = NULL;
cl_program * pPro_criterion_step = NULL;
cl_kernel * pKernel_criterion_step = NULL;
cl_program * pPro_descent_dir = NULL;
cl_kernel * pKernel_descent_dir = NULL;
cl_program * pPro_update_image = NULL;
cl_kernel * pKernel_update_image = NULL;
cl_program * pPro_update_tempimage = NULL;
cl_kernel * pKernel_update_tempimage = NULL;
cl_program * pPro_scalarprod = NULL;
cl_kernel * pKernel_scalarprod = NULL;

// Pointers for data stored on the GPU
cl_mem * pGpu_data = NULL;             // OIFITS Data
cl_mem * pGpu_data_err = NULL;         // OIFITS Data Error
cl_mem * pGpu_data_phasor = NULL;         // OIFITS Data Biphasor
cl_mem * pGpu_pow_size = NULL;
cl_mem * pGpu_data_uvpnt = NULL;       // OIFITS UV Point indicies for bispectrum data 
cl_mem * pGpu_data_sign = NULL;        // OIFITS UV Point signs.
cl_mem * pGpu_mock_data = NULL;        // Mock data (current "image")

cl_mem * pGpu_entropy = NULL;           // Buffer for storing the (summed) entropy
cl_mem * pGpu_scaprod = NULL;           // Buffer for storing the (summed) scalar product
cl_mem * pGpu_data_grad = NULL;

cl_mem * pGpu_entropy_image = NULL;     // Buffer to store the entropy of each pixel in the image
cl_mem * pGpu_entropy_grad = NULL;
cl_mem * pGpu_full_grad = NULL;         // Buffer to store the full gradient of the image
cl_mem * pGpu_full_grad_new = NULL;
cl_mem * pGpu_grad_temp = NULL;
cl_mem * pGpu_descent_dir = NULL;

cl_mem * pGpu_default_model = NULL;
cl_mem * pGpu_image = NULL;
cl_mem * pGpu_image_temp = NULL;

cl_mem * pGpu_chi2 = NULL;             // Buffer for storing the (single summed) chi2 value.
cl_mem * pGpu_chi2_buffer0 = NULL;       // Used as input buffer
cl_mem * pGpu_chi2_buffer1 = NULL;       // Used as output buffer
cl_mem * pGpu_chi2_buffer2 = NULL;       // Used as partial sum buffer

cl_mem * pGpu_dft_x = NULL;         // Pointer to Memory for x-DFT table
cl_mem * pGpu_dft_y = NULL;         // Pointer to Memory for y-DFT table

cl_mem * pGpu_flux0 = NULL;         // Buffer storing the (single, summed) flux value.
cl_mem * pGpu_flux1 = NULL;         // Buffer for storing 1/flux
cl_mem * pGpu_flux_buffer0 = NULL;  // Used as input buffer
cl_mem * pGpu_flux_buffer1 = NULL;  // Used as output buffer
cl_mem * pGpu_flux_buffer2 = NULL;  // Used as partial sum buffer

cl_mem * pGpu_visi0 = NULL;          // Used to store on-gpu visibilities.
cl_mem * pGpu_visi1 = NULL;
cl_mem * pGpu_image_width = NULL;   // Stores the size of the image.

// Variables for the parallel sum in the chi2 (again, globals... urgh).
int Chi2_pass_count = 0;
size_t * Chi2_group_counts = NULL;
size_t * Chi2_work_item_counts = NULL;
int * Chi2_operation_counts = NULL;
int * Chi2_entry_counts = NULL;
cl_program * pGpu_chi2_programs = NULL;
cl_kernel * pGpu_chi2_kernels = NULL;

int Flux_pass_count = 0;
size_t * Flux_group_counts = NULL;
size_t * Flux_work_item_counts = NULL;
int * Flux_operation_counts = NULL;
int * Flux_entry_counts = NULL;
cl_program * pGpu_flux_programs = NULL;
cl_kernel * pGpu_flux_kernels = NULL;

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
    // Something bad happened.
    printf(SEP);
    printf("Error Detected\n");
    printf(SEP);
    
    // Clean up memory so we can exit somewhat nicely.
    gpu_cleanup();
    
    char * error_string = print_cl_errstring(error_code);
    
    printf("%s \n", error_message);
    printf("OpenCL Error: %s\n", error_string);
    printf(SEP);
    exit(0);
}

char * print_cl_errstring(cl_int err) 
{
    switch (err) {
        case CL_SUCCESS:                          return strdup("Success!");
        case CL_DEVICE_NOT_FOUND:                 return strdup("Device not found.");
        case CL_DEVICE_NOT_AVAILABLE:             return strdup("Device not available");
        case CL_COMPILER_NOT_AVAILABLE:           return strdup("Compiler not available");
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:    return strdup("Memory object allocation failure");
        case CL_OUT_OF_RESOURCES:                 return strdup("Out of resources");
        case CL_OUT_OF_HOST_MEMORY:               return strdup("Out of host memory");
        case CL_PROFILING_INFO_NOT_AVAILABLE:     return strdup("Profiling information not available");
        case CL_MEM_COPY_OVERLAP:                 return strdup("Memory copy overlap");
        case CL_IMAGE_FORMAT_MISMATCH:            return strdup("Image format mismatch");
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:       return strdup("Image format not supported");
        case CL_BUILD_PROGRAM_FAILURE:            return strdup("Program build failure");
        case CL_MAP_FAILURE:                      return strdup("Map failure");
        case CL_INVALID_VALUE:                    return strdup("Invalid value");
        case CL_INVALID_DEVICE_TYPE:              return strdup("Invalid device type");
        case CL_INVALID_PLATFORM:                 return strdup("Invalid platform");
        case CL_INVALID_DEVICE:                   return strdup("Invalid device");
        case CL_INVALID_CONTEXT:                  return strdup("Invalid context");
        case CL_INVALID_QUEUE_PROPERTIES:         return strdup("Invalid queue properties");
        case CL_INVALID_COMMAND_QUEUE:            return strdup("Invalid command queue");
        case CL_INVALID_HOST_PTR:                 return strdup("Invalid host pointer");
        case CL_INVALID_MEM_OBJECT:               return strdup("Invalid memory object");
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  return strdup("Invalid image format descriptor");
        case CL_INVALID_IMAGE_SIZE:               return strdup("Invalid image size");
        case CL_INVALID_SAMPLER:                  return strdup("Invalid sampler");
        case CL_INVALID_BINARY:                   return strdup("Invalid binary");
        case CL_INVALID_BUILD_OPTIONS:            return strdup("Invalid build options");
        case CL_INVALID_PROGRAM:                  return strdup("Invalid program");
        case CL_INVALID_PROGRAM_EXECUTABLE:       return strdup("Invalid program executable");
        case CL_INVALID_KERNEL_NAME:              return strdup("Invalid kernel name");
        case CL_INVALID_KERNEL_DEFINITION:        return strdup("Invalid kernel definition");
        case CL_INVALID_KERNEL:                   return strdup("Invalid kernel");
        case CL_INVALID_ARG_INDEX:                return strdup("Invalid argument index");
        case CL_INVALID_ARG_VALUE:                return strdup("Invalid argument value");
        case CL_INVALID_ARG_SIZE:                 return strdup("Invalid argument size");
        case CL_INVALID_KERNEL_ARGS:              return strdup("Invalid kernel arguments");
        case CL_INVALID_WORK_DIMENSION:           return strdup("Invalid work dimension");
        case CL_INVALID_WORK_GROUP_SIZE:          return strdup("Invalid work group size");
        case CL_INVALID_WORK_ITEM_SIZE:           return strdup("Invalid work item size");
        case CL_INVALID_GLOBAL_OFFSET:            return strdup("Invalid global offset");
        case CL_INVALID_EVENT_WAIT_LIST:          return strdup("Invalid event wait list");
        case CL_INVALID_EVENT:                    return strdup("Invalid event");
        case CL_INVALID_OPERATION:                return strdup("Invalid operation");
        case CL_INVALID_GL_OBJECT:                return strdup("Invalid OpenGL object");
        case CL_INVALID_BUFFER_SIZE:              return strdup("Invalid buffer size");
        case CL_INVALID_MIP_LEVEL:                return strdup("Invalid mip-map level");
        default:                                  return strdup("Unknown");
    }
} 

void gpu_backup_gradient(int data_size, cl_mem * input, cl_mem * output)
{
    if(gpu_enable_verbose || gpu_enable_debug)
        printf("Backing up gradient kernel.\n");
 
    if(input == NULL || output == NULL)
        print_opencl_error("Pointer to input or output is NULL in backup_gradient!", 0);
        
    // Enqueue a copy operation, wait for it to complete before returning.
    clEnqueueCopyBuffer(*pQueue, *input, *output, 0, 0, sizeof(float) * data_size, 0, NULL, NULL);
    clFinish(*pQueue);
}

void gpu_build_kernel(cl_program * program, cl_kernel * kernel, char * kernel_name, char * filename)
{   
    int err = 0;
    if(gpu_enable_verbose || gpu_enable_debug)
        printf("\nLoading and compiling program '%s'\n", filename);
    
    // Load the kernel source:
    char * kernel_source = LoadProgramSourceFromFile(filename);
    // Create the program
    *program = clCreateProgramWithSource(*pContext, 1, (const char **) & kernel_source, NULL, &err);     
    if (err != CL_SUCCESS)   
        print_opencl_error("clCreateProgramWithSource with kernel", err);    
        
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
        
    if(gpu_enable_debug || gpu_enable_verbose)    
        gpu_kernel_workgroup_info(kernel, kernel_name);
}

void gpu_build_kernels(int data_size, int image_width, int image_size)
{
    if(gpu_enable_verbose || gpu_enable_debug)
        printf("Building Kernels\n");
        
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

    // Build the DFT (visi) kernel
    static cl_program pro_visi;
    static cl_kernel kern_visi;
    gpu_build_kernel(&pro_visi, &kern_visi, "visi", "./kernel_visi.cl");
    pPro_visi = &pro_visi;
    pKernel_visi = &kern_visi;
    
    // Build the update vis fluxchange kernel.
/*    static cl_program pro_u_vis_flux;*/
/*    static cl_kernel kern_u_vis_flux;*/
/*    gpu_build_kernel(&pro_u_vis_flux, &kern_u_vis_flux, "update_vis_fluxchange", "./kernel_vis_update_fluxchange.cl");*/
/*    pPro_u_vis_flux = &pro_u_vis_flux;*/
/*    pKernel_u_vis_flux = &kern_u_vis_flux;*/
    
    // Build the gradient kernel for the powerspectrum.
    static cl_program pro_grad_pow;
    static cl_kernel kern_grad_pow;
    gpu_build_kernel(&pro_grad_pow, &kern_grad_pow, "grad_pow", "./kernel_grad_pow.cl");
    pPro_grad_pow = &pro_grad_pow;
    pKernel_grad_pow = &kern_grad_pow;

    // Build the gradient kernel for the bispectrum
    static cl_program pro_grad_bis;
    static cl_kernel kern_grad_bis;
    gpu_build_kernel(&pro_grad_bis, &kern_grad_bis, "grad_bis", "./kernel_grad_bis.cl");
    pPro_grad_bis = &pro_grad_bis;
    pKernel_grad_bis = &kern_grad_bis;
    
    static cl_program pro_entropy;
    static cl_kernel kern_entropy;
    gpu_build_kernel(&pro_entropy, &kern_entropy, "entropy_gs", "./kernel_entropy_gs.cl");
    pPro_entropy = &pro_entropy;
    pKernel_entropy = &kern_entropy;
    
    static cl_program pro_entropy_grad;
    static cl_kernel kern_entropy_grad;
    gpu_build_kernel(&pro_entropy_grad, &kern_entropy_grad, "entropy_gs_grad", "./kernel_entropy_gs_grad.cl");
    pPro_entropy_grad = &pro_entropy_grad;
    pKernel_entropy_grad = &kern_entropy_grad;   
    
    static cl_program pro_criterion_grad;
    static cl_kernel kern_criterion_grad;
    gpu_build_kernel(&pro_criterion_grad, &kern_criterion_grad, "criterion_grad", "./kernel_criterion_grad.cl");
    pPro_criterion_grad = &pro_criterion_grad;
    pKernel_criterion_grad = &kern_criterion_grad; 

    static cl_program pro_criterion_step;
    static cl_kernel kern_criterion_step;
    gpu_build_kernel(&pro_criterion_step, &kern_criterion_step, "criterion_step", "./kernel_criterion_step.cl");
    pPro_criterion_step = &pro_criterion_step;
    pKernel_criterion_step = &kern_criterion_step; 

    static cl_program pro_descent_dir;
    static cl_kernel kern_descent_dir;
    gpu_build_kernel(&pro_descent_dir, &kern_descent_dir, "descent_dir", "./kernel_descent_dir.cl");
    pPro_descent_dir = &pro_descent_dir;
    pKernel_descent_dir = &kern_descent_dir;     
 
    static cl_program pro_update_image;
    static cl_kernel kern_update_image;
    gpu_build_kernel(&pro_update_image, &kern_update_image, "update_image", "./kernel_update_image.cl");
    pPro_update_image = &pro_update_image;
    pKernel_update_image = &kern_update_image; 
 
    static cl_program pro_update_tempimage;
    static cl_kernel kern_update_tempimage;
    gpu_build_kernel(&pro_update_tempimage, &kern_update_tempimage, "update_tempimage", "./kernel_update_tempimage.cl");
    pPro_update_tempimage = &pro_update_tempimage;
    pKernel_update_tempimage = &kern_update_tempimage; 
    
    static cl_program pro_scalarprod;
    static cl_kernel kern_scalarprod;
    gpu_build_kernel(&pro_scalarprod, &kern_scalarprod, "scalarprod", "./kernel_scalarprod.cl");
    pPro_scalarprod = &pro_scalarprod;
    pKernel_scalarprod = &kern_scalarprod;
}

void gpu_build_reduction_kernels(int data_size, cl_program ** pPrograms, cl_kernel ** pKernels, 
    int * pass_count, size_t ** group_counts, size_t ** work_item_counts, 
    int ** operation_counts, int ** entry_counts)
{
    // Init a few variables:
    int err = 0;
    int i;
    
    if(gpu_enable_debug && gpu_enable_verbose)
        printf("Loading and Compiling program ./kernel_reduce_float.cl \n");
        
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
            size_t len;
            char buffer[2048];
            printf("%s\n", block_source);
            printf("Error: Failed to create compute program!\n");
            clGetProgramBuildInfo(programs[i], *pDevice_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
            printf("%s\n", buffer);
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

// A function to double-check computations between the GPU and CPU.
void gpu_check_data(float * cpu_chi2, 
    int nuv, float complex * visi, 
    int data_size, float * mock_data,
    int image_size, float * data_grad)
{
    if(gpu_enable_verbose || gpu_enable_debug)
        printf("Checking data.\n");
        
    printf(SEP);
    printf("Comparing CPU and GPU Visiblity values:\n");
    gpu_compare_complex_data(nuv, visi, pGpu_visi0);

    printf(SEP);    
    printf("Comparing CPU and GPU Mock Data Values:\n");
    gpu_compare_data(data_size, mock_data, pGpu_mock_data);   

    printf(SEP);    
    printf("Comparing CPU and GPU chi2 values:\n");
    //gpu_compare_data(1, cpu_chi2, pGpu_chi2);
    
    printf(SEP);
    printf("Comparing CPU and GPU gradient values:\n");
    //gpu_compare_data(image_size, data_grad, pGpu_data_grad);
}

// Compare floating point data on the CPU and GPU.  Displays the percent error.
void gpu_compare_data(int size, float * cpu_data, cl_mem * pGpu_data)
{
    // TODO: It might be worthwile to replace this with something from the following page:
    //  http://www.cygnus-software.com/papers/comparingfloats/comparingfloats.htm
    int err = 0;
    
    // Init a temporary variable for storing the data:
    float * gpu_data;
    gpu_data = malloc(sizeof(float) * size);
    memset(gpu_data, 0, size);
    
    err = clEnqueueReadBuffer(*pQueue, *pGpu_data, CL_TRUE, 0, sizeof(float) * size, gpu_data, 0, NULL, NULL );
    if (err != CL_SUCCESS)
        print_opencl_error("Could not read back GPU Data for comparision!", err);    
        
    int i;
    float max_err = 0;
    float error = 0;
    float perr = 0;
    for(i = 0; i < size; i++)
    {
        // Check to see if they are identicle first
        if(gpu_data[i] == cpu_data[i])
        {
            error = 0;
            perr = 0;
        }
        else if(cpu_data[i] == 0)
        {
            // Oh no, possible divide by zero here.  Need exceptions!
            printf("[%i] CPU value is zero, did not compute percent error.\n", i);
        }
        else
        {
            error = gpu_data[i] - cpu_data[i];
            perr = fabs(error / cpu_data[i]);
        }
        
        //if(perr > 0.01) // Greater than 1%
            printf("[%i] %e %e %e %e\n", i, cpu_data[i], gpu_data[i], error, perr);
        
        if(perr > max_err)
            max_err = perr;    
    }
        
    // Print out the maximum percent difference in the data.  Multiply by 100 so it is indeed a percent.    
    printf("Max Difference: %f %%\n", 100 * max_err);
    
    free(gpu_data);
}

void gpu_compare_complex_data(int size, float complex * cpu_data, cl_mem * pGpu_data)
{
    // TODO: Add in percent difference calculation.
    int err = 0;
    
    // Init a temporary variable for storing the data:
    cl_float2 * gpu_data;
    gpu_data = malloc(sizeof(cl_float2) * size);
    
    err = clEnqueueReadBuffer(*pQueue, *pGpu_data, CL_TRUE, 0, sizeof(cl_float2) * size, gpu_data, 0, NULL, NULL );
    if (err != CL_SUCCESS)
        print_opencl_error("Could not read back GPU Data for comparision!", err);    
        
    int i;
    float real, imag;
    float error = 0;

    float err_sum = 0;
    for(i = 0; i < size; i++)
    {
        error = 0;
        real = gpu_data[i].s0 - creal(cpu_data[i]);
        imag = gpu_data[i].s1 - cimag(cpu_data[i]);
        error = sqrt(real * real + imag * imag);
        
        if(error > 0.01)
            printf("[%i] %f %f R(A) %f R(B) %f I(A) %f I(B) %f Err: %f\n", i, real, imag, creal(cpu_data[i]), gpu_data[i].s0, cimag(cpu_data[i]), gpu_data[i].s1, error);
            
        err_sum += error;
    }   
        
    printf("Total Difference: %f \n", err_sum);
    
    free(gpu_data);
}

// Compute the gradient of the entropy for the image, gpu_image.  This gets stored in pGpu_entropy_grad
void gpu_compute_criterion_gradient(int image_width, float hyperparameter_entropy, cl_mem * gradient_buffer)
{
    if(gpu_enable_verbose || gpu_enable_debug)
        printf("%sComputing Criterion Gradient.\n%s", SEP, SEP);
        
    int err = 0;
    // TODO: Figure out how to determine the size of local dynamically.
    size_t * local;
    local = malloc(2 * sizeof(size_t));
    local[0] = local[1] = (size_t) 16;
    
    size_t * global;
    global = malloc(2 * sizeof(size_t));
    global[0] = global[1] = (size_t) image_width;
    
    // Set the arguments for the entropy kernel:
    err  = clSetKernelArg(*pKernel_criterion_grad, 0, sizeof(cl_mem), pGpu_data_grad);
    err |= clSetKernelArg(*pKernel_criterion_grad, 1, sizeof(cl_mem), pGpu_entropy_grad);
    err |= clSetKernelArg(*pKernel_criterion_grad, 2, sizeof(float), &hyperparameter_entropy);
    err |= clSetKernelArg(*pKernel_criterion_grad, 3, sizeof(cl_mem), pGpu_image_width);  
    err |= clSetKernelArg(*pKernel_criterion_grad, 4, sizeof(cl_mem), gradient_buffer); 

/*   // Get the maximum work-group size for executing the kernel on the device*/
/*    err = clGetKernelWorkGroupInfo(*pKernel_u_vis_flux, *pDevice_id, CL_KERNEL_WORK_GROUP_SIZE , sizeof(size_t), &local, NULL);*/
/*    if (err != CL_SUCCESS)*/
/*        print_opencl_error("clGetKernelWorkGroupInfo", err);*/

/*    // Round down to the nearest power of two.*/
/*    local = pow(2, floor(log(npow) / log(2)));*/
        
    err = clEnqueueNDRangeKernel(*pQueue, *pKernel_criterion_grad, 2, 0, global, local, 0, NULL, NULL);
    if (err)
        print_opencl_error("Cannot enqueue Criterion Gradient kernel.", err); 
    
    clFinish(*pQueue);   
}

void gpu_compute_criterion_gradient_curr(int image_width, float hyperparameter_entropy)
{
    if(gpu_enable_verbose || gpu_enable_debug)
        printf("%sComputing Criterion Gradient on Current Image.\n%s", SEP, SEP);
        
    gpu_compute_criterion_gradient(image_width,hyperparameter_entropy, pGpu_full_grad_new);
}

void gpu_compute_criterion_gradient_temp(int image_width, float hyperparameter_entropy)
{
    if(gpu_enable_verbose || gpu_enable_debug)
        printf("%sComputing Criterion Gradient on Temp Image.\n%s", SEP, SEP);
        
    gpu_compute_criterion_gradient(image_width,hyperparameter_entropy, pGpu_grad_temp);
}

void gpu_compute_criterion_step(int image_width, float steplength, float minvalue)
{
    if(gpu_enable_verbose || gpu_enable_debug)
        printf("%sComputing Criterion Step.\n%s", SEP, SEP);
        
    int err = 0;
    // TODO: Figure out how to determine the size of local dynamically.
    size_t * local;
    local = malloc(2 * sizeof(size_t));
    local[0] = local[1] = (size_t) 16;
    
    size_t * global;
    global = malloc(2 * sizeof(size_t));
    global[0] = global[1] = (size_t) image_width;
    
    // Set the arguments for the entropy kernel:
    err  = clSetKernelArg(*pKernel_criterion_step, 0, sizeof(cl_mem), pGpu_image);
    err |= clSetKernelArg(*pKernel_criterion_step, 1, sizeof(cl_mem), pGpu_descent_dir);
    err |= clSetKernelArg(*pKernel_criterion_step, 2, sizeof(float), &steplength);
    err |= clSetKernelArg(*pKernel_criterion_step, 3, sizeof(cl_mem), pGpu_image_width);  
    err |= clSetKernelArg(*pKernel_criterion_step, 4, sizeof(cl_mem), pGpu_image_temp); 

/*   // Get the maximum work-group size for executing the kernel on the device*/
/*    err = clGetKernelWorkGroupInfo(*pKernel_u_vis_flux, *pDevice_id, CL_KERNEL_WORK_GROUP_SIZE , sizeof(size_t), &local, NULL);*/
/*    if (err != CL_SUCCESS)*/
/*        print_opencl_error("clGetKernelWorkGroupInfo", err);*/

/*    // Round down to the nearest power of two.*/
/*    local = pow(2, floor(log(npow) / log(2)));*/
        
    err = clEnqueueNDRangeKernel(*pQueue, *pKernel_criterion_step, 2, 0, global, local, 0, NULL, NULL);
    if (err)
        print_opencl_error("Cannot enqueue criterion step kernel.", err); 
    
    clFinish(*pQueue);   
}

void gpu_compute_descent_dir(int image_width, float beta)
{
    if(gpu_enable_verbose || gpu_enable_debug)
        printf("%sComputing Descent Direction.\n%s", SEP, SEP);
        
    int err = 0;
    // TODO: Figure out how to determine the size of local dynamically.
    size_t * local;
    local = malloc(2 * sizeof(size_t));
    local[0] = local[1] = (size_t) 16;
    
    size_t * global;
    global = malloc(2 * sizeof(size_t));
    global[0] = global[1] = (size_t) image_width;
    
    // Set the arguments for the entropy kernel:
    err  = clSetKernelArg(*pKernel_descent_dir, 0, sizeof(cl_mem), pGpu_descent_dir);
    err |= clSetKernelArg(*pKernel_descent_dir, 1, sizeof(cl_mem), pGpu_full_grad_new);
    err |= clSetKernelArg(*pKernel_descent_dir, 2, sizeof(float), &beta);
    err |= clSetKernelArg(*pKernel_descent_dir, 3, sizeof(cl_mem), pGpu_image_width);   

/*   // Get the maximum work-group size for executing the kernel on the device*/
/*    err = clGetKernelWorkGroupInfo(*pKernel_u_vis_flux, *pDevice_id, CL_KERNEL_WORK_GROUP_SIZE , sizeof(size_t), &local, NULL);*/
/*    if (err != CL_SUCCESS)*/
/*        print_opencl_error("clGetKernelWorkGroupInfo", err);*/

/*    // Round down to the nearest power of two.*/
/*    local = pow(2, floor(log(npow) / log(2)));*/
        
    err = clEnqueueNDRangeKernel(*pQueue, *pKernel_descent_dir, 2, 0, global, local, 0, NULL, NULL);
    if (err)
        print_opencl_error("Descent Direction Kernel.", err); 
    
    clFinish(*pQueue);   
}

// Compute the sum of the entropy, stores it in the location specified by entropy_storage.
void gpu_compute_entropy(int image_width, cl_mem * gpu_image, cl_mem * entropy_storage)
{
    if(gpu_enable_verbose || gpu_enable_debug)
        printf("Computing Entropy.\n%s", SEP);
        
    if(gpu_image == NULL || entropy_storage == NULL)
        print_opencl_error("A pointer to gpu_compute_entropy is NULL!", 0);
        
    int err = 0;
    // TODO: Figure out how to determine the size of local dynamically.
    size_t * local;
    local = malloc(2 * sizeof(size_t));
    local[0] = local[1] = (size_t) 16;
    size_t l_suggested = 0;
    
    size_t * global;
    global = malloc(2 * sizeof(size_t));
    global[0] = global[1] = (size_t) image_width;
    
    // Set the arguments for the entropy kernel:
    err  = clSetKernelArg(*pKernel_entropy, 0, sizeof(cl_mem), pGpu_image_width);
    err |= clSetKernelArg(*pKernel_entropy, 1, sizeof(cl_mem), gpu_image);
    err |= clSetKernelArg(*pKernel_entropy, 2, sizeof(cl_mem), pGpu_default_model);
    err |= clSetKernelArg(*pKernel_entropy, 3, sizeof(cl_mem), pGpu_entropy_image);  

   // Get the maximum work-group size for executing the kernel on the device
    err = clGetKernelWorkGroupInfo(*pKernel_entropy, *pDevice_id, CL_KERNEL_WORK_GROUP_SIZE , sizeof(size_t), &l_suggested, NULL);
    if (err != CL_SUCCESS)
        print_opencl_error("Cannot", err);

    if(l_suggested < local[0] * local[1])
    {
        printf("Warning: Decreasing Kernel size for Entropy kernel.\n");
        // Round down to the nearest power of two.
        l_suggested = floor(log(l_suggested) / log(2));
        local[0] = l_suggested;
        local[1] = l_suggested;
    }

    if(gpu_enable_debug || gpu_enable_verbose)
        printf("Entropy Kernel: Global: %i, %i Local %i, %i\n", (int)global[0], (int)global[1], (int)local[0], (int)local[1]);
    
    err = clEnqueueNDRangeKernel(*pQueue, *pKernel_entropy, 2, 0, global, local, 0, NULL, NULL);
    if (err)
        print_opencl_error("Cannot enqueue entropy kernel.", err); 
    

    // Now compute the sum of the entropy, store it in entropy_storage.
    gpu_compute_sum(pGpu_entropy_image, pGpu_flux_buffer1, pGpu_flux_buffer2, entropy_storage, pGpu_flux_kernels, Flux_pass_count, Flux_group_counts, Flux_work_item_counts, Flux_operation_counts, Flux_entry_counts);
    clFinish(*pQueue);
      
}

// Compute the gradient of the entropy for the image, gpu_image.  This gets stored in pGpu_entropy_grad
void gpu_compute_entropy_gradient(int image_width, cl_mem * gpu_image)
{  
    if(gpu_image == NULL)
        print_opencl_error("Pointer to compute_entropy_gradient is NULL!", 0);
        
    int err = 0;
    // TODO: Figure out how to determine the size of local dynamically.
    size_t * local;
    local = malloc(2 * sizeof(size_t));
    local[0] = local[1] = (size_t) 16;
    int l_suggested = 0;
    
    size_t * global;
    global = malloc(2 * sizeof(size_t));
    global[0] = global[1] = (size_t) image_width;
    
    // Set the arguments for the entropy kernel:
    err  = clSetKernelArg(*pKernel_entropy_grad, 0, sizeof(cl_mem), pGpu_image_width);
    err |= clSetKernelArg(*pKernel_entropy_grad, 1, sizeof(cl_mem), gpu_image);
    err |= clSetKernelArg(*pKernel_entropy_grad, 2, sizeof(cl_mem), pGpu_default_model);
    err |= clSetKernelArg(*pKernel_entropy_grad, 3, sizeof(cl_mem), pGpu_entropy_grad);  

   // Get the maximum work-group size for executing the kernel on the device
    err = clGetKernelWorkGroupInfo(*pKernel_entropy_grad, *pDevice_id, CL_KERNEL_WORK_GROUP_SIZE , sizeof(size_t), &l_suggested, NULL);
    if (err != CL_SUCCESS)
        print_opencl_error("Cannot determine local size for entropy gradient kernel.", err);

    if(l_suggested < local[0] * local[1])
    {
        printf("Warning: Decreasing Kernel size for Entropy kernel.\n");
        // Round down to the nearest power of two.
        l_suggested = floor(log(l_suggested) / log(2));
        local[0] = l_suggested;
        local[1] = l_suggested;
    }

    if(gpu_enable_debug || gpu_enable_verbose)
        printf("Entropy Gradient Kernel: Global: %i, %i Local %i, %i\n", (int)global[0], (int)global[1], (int)local[0], (int)local[1]);
    
        
    err = clEnqueueNDRangeKernel(*pQueue, *pKernel_entropy_grad, 2, 0, global, local, 0, NULL, NULL);
    if (err)
        print_opencl_error("Cannot enqueue entropy gradient kernel.", err); 
    
    clFinish(*pQueue);   
}

void gpu_compute_entropy_gradient_curr(int image_width)
{
    if(gpu_enable_verbose || gpu_enable_debug)
        printf("%sComputing Entropy Gradient on current image.\n%s", SEP, SEP);

    gpu_compute_entropy_gradient(image_width, pGpu_image);
}

void gpu_compute_entropy_gradient_temp(int image_width)
{
    if(gpu_enable_verbose || gpu_enable_debug)
        printf("%sComputing Entropy Gradient on temporary image.\n%s", SEP, SEP);

    gpu_compute_entropy_gradient(image_width, pGpu_image_temp);
}

// Computes the flux of the image located in pGpu_image.
void gpu_compute_flux(cl_mem * gpu_image, cl_mem * flux_storage, cl_mem * flux_inverse_storage)
{
    if(gpu_enable_verbose || gpu_enable_debug)
        printf("%sComputing Flux.\n%s", SEP, SEP);
        
    if(flux_storage == NULL || flux_inverse_storage == NULL)
        print_opencl_error("Pointer input to gpu_compute_flux is NULL!", 0);
        
    // Computes the sum of the image array, stores the result in the flux_storage buffer
    gpu_compute_sum(gpu_image, pGpu_flux_buffer1, pGpu_flux_buffer2, flux_storage, pGpu_flux_kernels, Flux_pass_count, Flux_group_counts, Flux_work_item_counts, Flux_operation_counts, Flux_entry_counts);

    int err = 0;
    // First copy the normalization value back to the CPU (blocking call)
    float value = 0;
    err = clEnqueueReadBuffer(*pQueue, *flux_storage, CL_TRUE, 0, sizeof(float), &value, 0, NULL, NULL );
    if(err != CL_SUCCESS)
        print_opencl_error("Could not read back the flux value.", err);
        
    // Invert the value then pass it back to the GPU, storing it in the flux_inverse_storage location.
    // Do this as a blocking call to ensure the value is on the GPU when we exit this function.
    value = 1 / value;  
    err = clEnqueueWriteBuffer(*pQueue, *flux_inverse_storage, CL_TRUE, 0, sizeof(float), & value, 0, NULL, NULL);
    if(err != CL_SUCCESS)
        print_opencl_error("Could not read back 1/flux.", err);
}

void gpu_cleanup()
{
    int i = 0;
    int err = 0;
    if(gpu_enable_verbose || gpu_enable_debug)
        printf("Freeing program, kernel, and device objects. \n");
        
    // Release programs
    if(pPro_chi2 != NULL)
        err |= clReleaseProgram(*pPro_chi2);
    if(pPro_powspec != NULL)
        err |= clReleaseProgram(*pPro_powspec);
    if(pPro_bispec != NULL)
        err |= clReleaseProgram(*pPro_bispec);   
    if(pGpu_chi2_programs != NULL)
    {
        for(i = 0; i < Chi2_pass_count; i++)
            err |= clReleaseProgram(pGpu_chi2_programs[i]);
    }
    if(pGpu_flux_programs != NULL)
    {
        for(i = 0; i < Flux_pass_count; i++)
            err |= clReleaseProgram(pGpu_flux_programs[i]);
    }
    if(pPro_visi != NULL)
        err |= clReleaseProgram(*pPro_visi);
    if(pPro_u_vis_flux != NULL)
        err |= clReleaseProgram(*pPro_u_vis_flux);
    if(pPro_grad_pow != NULL)
        err |= clReleaseProgram(*pPro_grad_pow);
    if(pPro_grad_bis != NULL)
        err |= clReleaseProgram(*pPro_grad_bis);
    if(pPro_entropy != NULL)
        err |= clReleaseProgram(*pPro_entropy);
    if(pPro_entropy_grad != NULL)
        err |= clReleaseProgram(*pPro_entropy_grad);
    if(pPro_criterion_grad != NULL)
        err |= clReleaseProgram(*pPro_criterion_grad);
    if(pPro_criterion_step != NULL)
        err |= clReleaseProgram(*pPro_criterion_step);
    if(pPro_descent_dir != NULL)
        err |= clReleaseProgram(*pPro_descent_dir);
    if(pPro_update_image != NULL)
        err |= clReleaseProgram(*pPro_update_image);
    if(pPro_update_tempimage != NULL)
        err |= clReleaseProgram(*pPro_update_tempimage);
    if(pPro_scalarprod != NULL)
        err |= clReleaseProgram(*pPro_scalarprod);

    if(err != CL_SUCCESS)
        printf("Failed to Free GPU Program Memory.\n");
    
    err = 0;
    // Release Kernels
    if(pKernel_chi2 != NULL)
        err |= clReleaseKernel(*pKernel_chi2);
    if(pKernel_powspec != NULL)
        err |= clReleaseKernel(*pKernel_powspec);
    if(pKernel_bispec != NULL)
        err |= clReleaseKernel(*pKernel_bispec);
    if(pGpu_chi2_kernels != NULL)
    {
        for(i = 0; i < Chi2_pass_count; i++)
            err |= clReleaseKernel(pGpu_chi2_kernels[i]);
    }
    if(pGpu_flux_kernels != NULL)
    {
        for(i = 0; i < Flux_pass_count; i++)
            err |= clReleaseKernel(pGpu_flux_kernels[i]);
    }
    if(pKernel_visi != NULL)
        err |= clReleaseKernel(*pKernel_visi);
    if(pKernel_u_vis_flux != NULL)
    	err |= clReleaseKernel(*pKernel_u_vis_flux);
    if(pKernel_grad_pow != NULL)
        err |= clReleaseKernel(*pKernel_grad_pow);
    if(pKernel_grad_bis != NULL)
        err |= clReleaseKernel(*pKernel_grad_bis);
    if(pKernel_entropy != NULL)
        err |= clReleaseKernel(*pKernel_entropy);
    if(pKernel_entropy_grad != NULL)
        err |= clReleaseKernel(*pKernel_entropy_grad);
    if(pKernel_criterion_grad != NULL)
        err |= clReleaseKernel(*pKernel_criterion_grad);
    if(pKernel_criterion_step != NULL)
        err |= clReleaseKernel(*pKernel_criterion_step);
    if(pKernel_descent_dir != NULL)
        err |= clReleaseKernel(*pKernel_descent_dir);        
    if(pKernel_update_image != NULL)
        err |= clReleaseKernel(*pKernel_update_image); 
    if(pKernel_update_tempimage != NULL)
        err |= clReleaseKernel(*pKernel_update_tempimage);
    if(pKernel_scalarprod != NULL)
        err |= clReleaseKernel(*pKernel_scalarprod);

    if(err != CL_SUCCESS)
        printf("Failed to Free GPU Kernel Memory.\n");

    // Release Memory objects:
    err = 0;
    if(pGpu_data != NULL)
        err |= clReleaseMemObject(*pGpu_data);
    if(pGpu_data_err != NULL)
        err |= clReleaseMemObject(*pGpu_data_err);
    if(pGpu_data_phasor != NULL)
        err |= clReleaseMemObject(*pGpu_data_phasor);
    if(pGpu_pow_size != NULL)
        err |= clReleaseMemObject(*pGpu_pow_size);
    if(pGpu_data_uvpnt != NULL)
        err |= clReleaseMemObject(*pGpu_data_uvpnt);
    if(pGpu_data_sign != NULL)
        err |= clReleaseMemObject(*pGpu_data_sign);
    if(pGpu_mock_data != NULL)
        err |= clReleaseMemObject(*pGpu_mock_data);

    // Free up scalar product buffers:
    if(pGpu_scaprod != NULL)
        err |= clReleaseMemObject(*pGpu_scaprod);

    // Free up gradient buffers:
    if(pGpu_data_grad != NULL)
        err |= clReleaseMemObject(*pGpu_data_grad);
    if(pGpu_entropy_image != NULL)
        err |= clReleaseMemObject(*pGpu_entropy_image);
    if(pGpu_entropy_grad != NULL)
        err |= clReleaseMemObject(*pGpu_entropy_grad);
    if(pGpu_full_grad != NULL)
        err |= clReleaseMemObject(*pGpu_full_grad);
    if(pGpu_full_grad_new != NULL)
        err |= clReleaseMemObject(*pGpu_full_grad_new);
    if(pGpu_grad_temp != NULL)
        err |= clReleaseMemObject(*pGpu_grad_temp);
    if(pGpu_descent_dir != NULL)
        err |= clReleaseMemObject(*pGpu_descent_dir);           
  
    // Release DFT buffers, image buffers:
    if(pGpu_dft_x != NULL)
        err |= clReleaseMemObject(*pGpu_dft_x);
    if(pGpu_dft_y != NULL)
        err |= clReleaseMemObject(*pGpu_dft_y);
    if(pGpu_image != NULL)
        err |= clReleaseMemObject(*pGpu_image);
    if(pGpu_image_temp != NULL)
        err |= clReleaseMemObject(*pGpu_image_temp);
    if(pGpu_default_model != NULL)
        err |= clReleaseMemObject(*pGpu_default_model);
        

    // Release chi2 memory objects:
    if(pGpu_entropy != NULL)
        err |= clReleaseMemObject(*pGpu_entropy);
    if(pGpu_chi2 != NULL)
        err |= clReleaseMemObject(*pGpu_chi2);
    if(pGpu_chi2_buffer0 != NULL)
        err |= clReleaseMemObject(*pGpu_chi2_buffer0);
    if(pGpu_chi2_buffer1 != NULL)
        err |= clReleaseMemObject(*pGpu_chi2_buffer1);
    if(pGpu_chi2_buffer2 != NULL)
        err |= clReleaseMemObject(*pGpu_chi2_buffer2);

    // Release flux memory objects:
    if(pGpu_flux0 != NULL)
        err |= clReleaseMemObject(*pGpu_flux0);
    if(pGpu_flux1 != NULL)
        err |= clReleaseMemObject(*pGpu_flux1);
    if(pGpu_flux_buffer0 != NULL)
        err |= clReleaseMemObject(*pGpu_flux_buffer0);
    if(pGpu_flux_buffer1 != NULL)
        err |= clReleaseMemObject(*pGpu_flux_buffer1);
    if(pGpu_flux_buffer2 != NULL)
        err |= clReleaseMemObject(*pGpu_flux_buffer2);
    if(pGpu_image_width != NULL)
        err |= clReleaseMemObject(*pGpu_image_width);        
    if(pGpu_visi0 != NULL)
        err |= clReleaseMemObject(*pGpu_visi0);
    if(pGpu_visi1 != NULL)
        err |= clReleaseMemObject(*pGpu_visi1);

        
    if(err != CL_SUCCESS)
        printf("Failed to free GPU Memory Object(s).\n");


    err = 0;
    // Release the command queue and context:
    if(pQueue != NULL)
        err |= clReleaseCommandQueue(*pQueue);
    if(pContext != NULL)
        err |= clReleaseContext(*pContext);

    if(err != CL_SUCCESS)
        printf("Failed to free GPU Queue or Context.\n");    
    
    // Now free global variables    
    free(Chi2_group_counts);
    free(Chi2_work_item_counts);
    free(Chi2_operation_counts);
    free(Chi2_entry_counts);

    free(Flux_group_counts);
    free(Flux_work_item_counts);
    free(Flux_operation_counts);
    free(Flux_entry_counts);
}

void gpu_compute_sum(cl_mem * input_buffer, cl_mem * output_buffer, cl_mem * partial_sum_buffer, cl_mem * final_buffer, 
    cl_kernel * pKernels, 
    int pass_count, size_t * group_counts, size_t * work_item_counts, 
    int * operation_counts, int * entry_counts)
{
    if(gpu_enable_verbose || gpu_enable_debug)
        printf("Computing Parallel Sum. \n");  
        
    if(input_buffer == NULL || output_buffer == NULL || partial_sum_buffer == NULL || final_buffer == NULL)
        print_opencl_error("Input to gpu_compute_sum is NULL!", 0);
        
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
        err |= clEnqueueNDRangeKernel(*pQueue, pKernels[i], 1, NULL, &global, &local, 0, NULL, NULL);
        if (err != CL_SUCCESS)
            print_opencl_error("Failed to enqueue parallel sum kernels.", err); 
    }

    // Let the queue complete.
    clFinish(*pQueue);

    // Copy the new chi2 value over to it's final place in GPU memory.
    err = clEnqueueCopyBuffer(*pQueue, pass_output, *final_buffer, 0, 0, sizeof(float), 0, NULL, NULL);
    if(err != CL_SUCCESS)
        print_opencl_error("Could not copy summed value to/from buffers on the GPU.", err);
        
    if(gpu_enable_debug && gpu_enable_verbose)
    {
        float sum = 0;
        err = clEnqueueReadBuffer(*pQueue, *final_buffer, CL_TRUE, 0, sizeof(float), &sum, 0, NULL, NULL );
        if(err != CL_SUCCESS)
            print_opencl_error("Could not read back GPU SUM value.", err);
            
        if(isnan(sum))
            print_opencl_error("Error: Calculation yielded NAN, aborting!", 0);
    
        printf("Sum: %f (copied value on GPU)\n", sum);
    }    
    
    clFinish(*pQueue);    
}

// Init memory locations and copy data over to the GPU.
void gpu_copy_data(float * data, float * data_err, int data_size, int data_size_uv,\
                    cl_float2 * data_phasor, int phasor_size, int pow_size, \
                    cl_long4 * gpu_bsref_uvpnt, cl_short4 * gpu_bsref_sign, int bsref_size,
                    float * default_model,
                    int image_size, int image_width)
{
    int err = 0;
    int i;
    
    int data_impact = 0;            // For internal use, keeps track of number of bytes required.
    int fsize = sizeof(float);

    static cl_mem gpu_data;         // Data
    static cl_mem gpu_data_err;     // Data Error
    static cl_mem gpu_data_phasor;     // Biphasor
    static cl_mem gpu_pow_size;
    static cl_mem gpu_data_uvpnt;   // UV Points for the bispectrum
    static cl_mem gpu_data_sign;    // Signs for the bispectrum.
    static cl_mem gpu_mock_data;    // Mock Data
    
    static cl_mem gpu_scaprod;          // Buffer to store the (summed) scalar product

    // Buffers for Gradients, entropy
    static cl_mem gpu_data_grad;
    static cl_mem gpu_entropy_image;    // Buffer to store the entropy of each pixel in the image
    static cl_mem gpu_entropy_grad;
    static cl_mem gpu_full_grad;        // Buffer to store the full gradient of the image
    static cl_mem gpu_full_grad_new;
    static cl_mem gpu_grad_temp;
    static cl_mem gpu_descent_dir;
    
    // Image and model pointers:
    static cl_mem gpu_default_model;
    static cl_mem gpu_image;
    static cl_mem gpu_image_temp;
    
    static cl_mem gpu_chi2;
    static cl_mem gpu_chi2_buffer0;       // Temporary storage for the chi2 computation.
    static cl_mem gpu_chi2_buffer1;       // Temporary storage for the chi2 computation.
    static cl_mem gpu_chi2_buffer2;     // Temporary storage for the chi2 computation.
    
    static cl_mem gpu_entropy;          // Buffer to store the (summed) entropy
    static cl_mem gpu_flux0;
    static cl_mem gpu_flux1;
    static cl_mem gpu_flux_buffer0;  
    static cl_mem gpu_flux_buffer1; 
    static cl_mem gpu_flux_buffer2;  
    
    static cl_mem gpu_visi0;         // Used for storing the visibilities  
    static cl_mem gpu_visi1;        // Used for storing temporary visibilities.
    static cl_mem gpu_image_width; 
    

    // Init some mock data (to allow resumes in the future I suppose...)
    float zero = 0;
    float * mock_data;
    mock_data = malloc(data_size * sizeof(float));
    memset(mock_data, 0, data_size);
    
    float * temp;   // Filler for chi2 buffers
    temp = malloc(sizeof(float) * data_size);   
    memset(temp, 0, sizeof(float) * data_size);
 
    float * zero_flux;  // Filler for image flux buffers
    zero_flux = malloc(sizeof(float) * image_size);
    memset(zero_flux, 0, sizeof(float) * image_size);

    cl_float2 * visi;   // Filler for visi buffer
    visi = malloc(data_size_uv * sizeof(cl_float2));
    for(i = 0; i < data_size_uv; i++)
    {   
        visi[i].s0 = 0;
        visi[i].s1 = 0;
    }  
    
    // Output some additional information if we are in verbose mode
    if(gpu_enable_verbose || gpu_enable_debug)
        printf("Creating buffers on the device. \n");
    
    // Create buffers on the device:    
    gpu_data = clCreateBuffer(*pContext,  CL_MEM_READ_ONLY,  sizeof(float) * data_size, NULL, NULL);
    gpu_data_err = clCreateBuffer(*pContext,  CL_MEM_READ_ONLY,  sizeof(float) * data_size, NULL, NULL); 
    gpu_data_phasor = clCreateBuffer(*pContext,  CL_MEM_READ_ONLY,  sizeof(cl_float2) * phasor_size, NULL, NULL); 
    gpu_pow_size = clCreateBuffer(*pContext, CL_MEM_READ_ONLY, sizeof(int), NULL, NULL);
    gpu_data_uvpnt = clCreateBuffer(*pContext, CL_MEM_READ_ONLY, sizeof(cl_long4) * bsref_size, NULL, NULL);
    gpu_data_sign = clCreateBuffer(*pContext, CL_MEM_READ_ONLY, sizeof(cl_short4) * bsref_size, NULL, NULL);
    gpu_mock_data = clCreateBuffer(*pContext, CL_MEM_READ_WRITE, sizeof(float) * data_size, NULL, NULL);
    
    data_impact += 3 * data_size * fsize +  2 * phasor_size * fsize + sizeof(int) 
        + 4 * sizeof(long) * bsref_size + 4* sizeof(short) * bsref_size;
    
    // Buffers for scalar products of size image_width wide
    gpu_scaprod = clCreateBuffer(*pContext, CL_MEM_READ_WRITE, sizeof(float), NULL, NULL);
    data_impact += fsize;

    // Buffers for Gradients, entropy, for the line search.
    gpu_data_grad = clCreateBuffer(*pContext, CL_MEM_READ_WRITE, sizeof(float) * image_size, NULL, NULL);
    gpu_entropy_image = clCreateBuffer(*pContext, CL_MEM_READ_WRITE, sizeof(float) * image_size, NULL, NULL);
    gpu_entropy_grad = clCreateBuffer(*pContext, CL_MEM_READ_WRITE, sizeof(float) * image_size, NULL, NULL); 
    gpu_full_grad = clCreateBuffer(*pContext, CL_MEM_READ_WRITE, sizeof(float) * image_size, NULL, NULL); 
    gpu_full_grad_new = clCreateBuffer(*pContext, CL_MEM_READ_WRITE, sizeof(float) * image_size, NULL, NULL);
    gpu_grad_temp = clCreateBuffer(*pContext, CL_MEM_READ_WRITE, sizeof(float) * image_size, NULL, NULL);
    gpu_descent_dir = clCreateBuffer(*pContext, CL_MEM_READ_WRITE, sizeof(float) * image_size, NULL, NULL);
    
    data_impact += 6 * image_size * fsize;
    
    // Image and model buffers:
    gpu_default_model = clCreateBuffer(*pContext, CL_MEM_READ_ONLY, sizeof(float) * image_size, NULL, NULL);
    gpu_image = clCreateBuffer(*pContext, CL_MEM_READ_WRITE, sizeof(float) * image_size, NULL, NULL);      
    gpu_image_temp = clCreateBuffer(*pContext, CL_MEM_READ_WRITE, sizeof(float) * image_size, NULL, NULL);    

    data_impact += 3 * image_size * fsize;
    
    gpu_chi2 = clCreateBuffer(*pContext, CL_MEM_READ_WRITE, sizeof(float), NULL, NULL);
    gpu_chi2_buffer0 = clCreateBuffer(*pContext, CL_MEM_READ_WRITE, sizeof(float) * data_size, NULL, NULL);
    gpu_chi2_buffer1 = clCreateBuffer(*pContext, CL_MEM_READ_WRITE, sizeof(float) * data_size, NULL, NULL);
    gpu_chi2_buffer2 = clCreateBuffer(*pContext, CL_MEM_READ_WRITE, sizeof(float) * data_size, NULL, NULL);
    
    data_impact += 3 * data_size * fsize;
    
    // Buffers for the entropy, and partial sums of the entropy.  
    gpu_entropy = clCreateBuffer(*pContext, CL_MEM_READ_WRITE, sizeof(float), NULL, NULL);
    gpu_flux0 = clCreateBuffer(*pContext, CL_MEM_READ_WRITE, sizeof(float), NULL, NULL);
    gpu_flux1 = clCreateBuffer(*pContext, CL_MEM_READ_ONLY, sizeof(float), NULL, NULL);
    gpu_flux_buffer0 = clCreateBuffer(*pContext, CL_MEM_READ_WRITE, sizeof(float) * image_size, NULL, NULL);
    gpu_flux_buffer1 = clCreateBuffer(*pContext, CL_MEM_READ_WRITE, sizeof(float) * image_size, NULL, NULL);
    gpu_flux_buffer2 = clCreateBuffer(*pContext, CL_MEM_READ_WRITE, sizeof(float) * image_size, NULL, NULL);
    
    data_impact += 3 * fsize + 3 * image_size * fsize;
    
    gpu_visi0 = clCreateBuffer(*pContext, CL_MEM_READ_WRITE, sizeof(cl_float2) * data_size_uv, NULL, NULL);
    gpu_visi1 = clCreateBuffer(*pContext, CL_MEM_READ_WRITE, sizeof(cl_float2) * data_size_uv, NULL, NULL);
    gpu_image_width = clCreateBuffer(*pContext, CL_MEM_READ_ONLY, sizeof(int), NULL, NULL);
    
    if (err != CL_SUCCESS)
        print_opencl_error("Error: Cannot Create Buffer on the GPU.", err);

    if(gpu_enable_verbose || gpu_enable_debug)
        printf("Copying data to device. \n");
        
    printf("Total memory needed on GPU: %i bytes.\n", data_impact);
        

    // Copy the data over to the device.  (note, non-blocking cals)
    err  = clEnqueueWriteBuffer(*pQueue, gpu_data, CL_FALSE, 0, sizeof(float) * data_size, data, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_data_err, CL_FALSE, 0, sizeof(float) * data_size, data_err, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_data_phasor, CL_FALSE, 0, sizeof(cl_float2) * phasor_size, data_phasor, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_pow_size, CL_FALSE, 0, sizeof(int), &pow_size, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_data_uvpnt, CL_FALSE, 0, sizeof(cl_long4) * bsref_size, gpu_bsref_uvpnt, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_data_sign, CL_FALSE, 0, sizeof(cl_short4) * bsref_size, gpu_bsref_sign, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_mock_data, CL_FALSE, 0, sizeof(float) * data_size, mock_data, 0, NULL, NULL);
 
    // Scalar product buffers:    
    err |= clEnqueueWriteBuffer(*pQueue, gpu_scaprod, CL_FALSE, 0, sizeof(float), &zero, 0, NULL, NULL);

    // Gradient buffers
    err |= clEnqueueWriteBuffer(*pQueue, gpu_data_grad, CL_FALSE, 0, sizeof(float) * image_size, zero_flux, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_entropy_image, CL_FALSE, 0, sizeof(float) * image_size, zero_flux, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_entropy_grad, CL_FALSE, 0, sizeof(float) * image_size, zero_flux, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_full_grad, CL_FALSE, 0, sizeof(float) * image_size, zero_flux, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_full_grad_new, CL_FALSE, 0, sizeof(float) * image_size, zero_flux, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_grad_temp, CL_FALSE, 0, sizeof(float) * image_size, zero_flux, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_descent_dir, CL_FALSE, 0, sizeof(float) * image_size, zero_flux, 0, NULL, NULL);
    if (err != CL_SUCCESS)
        print_opencl_error("Cannot write Gradient data to the GPU.", err); 
    
    // Image buffers:
    err |= clEnqueueWriteBuffer(*pQueue, gpu_default_model, CL_FALSE, 0, sizeof(float) * image_size, default_model, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_image, CL_FALSE, 0, sizeof(float) * image_size, zero_flux, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_image_temp, CL_FALSE, 0, sizeof(float) * image_size, zero_flux, 0, NULL, NULL);
    if (err != CL_SUCCESS)
        print_opencl_error("Cannot write Image data to the GPU.", err); 
    
    // Chi2 buffers
    err |= clEnqueueWriteBuffer(*pQueue, gpu_chi2, CL_FALSE, 0, sizeof(float), &zero, 0, NULL, NULL);        
    err |= clEnqueueWriteBuffer(*pQueue, gpu_chi2_buffer0, CL_FALSE, 0, sizeof(float) * data_size, temp, 0, NULL, NULL);  
    err |= clEnqueueWriteBuffer(*pQueue, gpu_chi2_buffer1, CL_FALSE, 0, sizeof(float) * data_size, temp, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_chi2_buffer2, CL_FALSE, 0, sizeof(float) * data_size, temp, 0, NULL, NULL);
    
    // Flux buffers
    // TODO: Rename these more generically to show they are used in calculating the summed entropy.
    err |= clEnqueueWriteBuffer(*pQueue, gpu_entropy, CL_FALSE, 0, sizeof(float), &zero, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_flux0, CL_FALSE, 0, sizeof(float), &zero, 0, NULL, NULL);        
    err |= clEnqueueWriteBuffer(*pQueue, gpu_flux1, CL_FALSE, 0, sizeof(float), &zero, 0, NULL, NULL); 
    err |= clEnqueueWriteBuffer(*pQueue, gpu_flux_buffer0, CL_FALSE, 0, sizeof(float) * image_size, zero_flux, 0, NULL, NULL);  
    err |= clEnqueueWriteBuffer(*pQueue, gpu_flux_buffer1, CL_FALSE, 0, sizeof(float) * image_size, zero_flux, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_flux_buffer2, CL_FALSE, 0, sizeof(float) * image_size, zero_flux, 0, NULL, NULL);
    
    err |= clEnqueueWriteBuffer(*pQueue, gpu_visi0, CL_FALSE, 0, sizeof(cl_float2) * data_size_uv, visi, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_visi1, CL_FALSE, 0, sizeof(cl_float2) * data_size_uv, visi, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(*pQueue, gpu_image_width, CL_FALSE, 0, sizeof(int), &image_width, 0, NULL, NULL);
    
    if (err != CL_SUCCESS)
        print_opencl_error("Cannot write data to the GPU.", err);    
 
    clFinish(*pQueue); 
        
    pGpu_data = &gpu_data;
    pGpu_data_err = &gpu_data_err; 
    pGpu_data_phasor = &gpu_data_phasor;
    pGpu_pow_size = &gpu_pow_size;
    pGpu_data_uvpnt = &gpu_data_uvpnt;
    pGpu_data_sign = &gpu_data_sign;
    pGpu_mock_data = &gpu_mock_data;
    
    pGpu_scaprod = &gpu_scaprod;
    
    pGpu_data_grad = &gpu_data_grad;
    pGpu_entropy_image = &gpu_entropy_image;    // Buffer to store the entropy of each pixel in the image
    pGpu_entropy_grad = &gpu_entropy_grad;
    pGpu_full_grad = &gpu_full_grad;        // Buffer to store the full gradient of the image
    pGpu_full_grad_new = &gpu_full_grad_new;
    pGpu_grad_temp = &gpu_grad_temp;
    pGpu_descent_dir = &gpu_descent_dir;
    
    pGpu_default_model = &gpu_default_model;
    pGpu_image = &gpu_image;
    pGpu_image_temp = &gpu_image_temp;
    
    pGpu_chi2 = &gpu_chi2;
    pGpu_chi2_buffer0 = &gpu_chi2_buffer0;
    pGpu_chi2_buffer1 = &gpu_chi2_buffer1;
    pGpu_chi2_buffer2 = &gpu_chi2_buffer2;
    
    pGpu_entropy = &gpu_entropy;
    pGpu_flux0 = &gpu_flux0;   
    pGpu_flux1 = &gpu_flux1;   
    pGpu_flux_buffer0 = &gpu_flux_buffer0;
    pGpu_flux_buffer1 = &gpu_flux_buffer1;
    pGpu_flux_buffer2 = &gpu_flux_buffer2;
    
    pGpu_visi0 = &gpu_visi0;
    pGpu_visi1 = &gpu_visi1;
    pGpu_image_width = &gpu_image_width;
    
    // Free CPU-based memory:
    free(mock_data);
    free(temp);
    free(zero_flux);
    free(visi);
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

    if(gpu_enable_debug)
        printf("%sComputing Chi2 on the GPU.\n%s", SEP, SEP);
        
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

            // Enable the next four lines if you want to see the array elements.
/*            printf("%s Chi2 Array Elements \n %s", SEP, SEP);*/
/*            printf("%f ", results[i]);  */
/*            if(i % 10 == 0)*/
/*                printf("\n");*/
        }
 
        printf("GPU Sum: %f (summed on the CPU)\n", chi2);
    }
    
    // Now start up the partial sum kernel:
    gpu_compute_sum(pGpu_chi2_buffer0, pGpu_chi2_buffer1, pGpu_chi2_buffer2, pGpu_chi2, pGpu_chi2_kernels, Chi2_pass_count, Chi2_group_counts, Chi2_work_item_counts, Chi2_operation_counts, Chi2_entry_counts);
    clFinish(*pQueue);
}

int gpu_error_callback(char * error)
{
    printf("Callback function called!\n %s", error);
    return 0;
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
	
	size_t max_work_item_dims = 3;
	size_t max_work_group_size, max_work_item_sizes[3];
	
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
	
	// TODO: There be a bug here somewhere.  Upgrade to Nvidia driver 195 somehow screwed up this command.
	//err|= clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(max_work_item_dims), &max_work_item_dims, &returned_size);
	
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

void gpu_compute_data_gradient(cl_mem * gpu_image, int nuv, int npow, int nbis, int image_width)
{               
    if(gpu_image == NULL)
        print_opencl_error("Image input is NULL in gpu_compute_data_gradient", 0);

    if(gpu_enable_verbose || gpu_enable_debug)
        printf("Starting with flux.\n");
        
    // Compute the current flux
    gpu_compute_flux(gpu_image, pGpu_flux0, pGpu_flux1);
    
    // Now launch the kernels
    int err = 0;
    // TODO: Figure out how to determine the size of local dynamically.
    size_t * local;
    local = malloc(2 * sizeof(size_t));
    local[0] = local[1] = (size_t) 8;
    
    size_t * global;
    global = malloc(2 * sizeof(size_t));
    global[0] = global[1] = (size_t) image_width;

    if(gpu_enable_debug || gpu_enable_verbose)
        printf("Data Gradient Kernels: Global: %i, %i Local %i, %i\n", (int)global[0], (int)global[1], (int)local[0], (int)local[1]);
    
    // Start with the chi2
    err  = clSetKernelArg(*pKernel_grad_pow, 0, sizeof(cl_mem), pGpu_data);
    err |= clSetKernelArg(*pKernel_grad_pow, 1, sizeof(cl_mem), pGpu_data_err);
    err |= clSetKernelArg(*pKernel_grad_pow, 2, sizeof(cl_mem), pGpu_mock_data);
    err |= clSetKernelArg(*pKernel_grad_pow, 3, sizeof(cl_mem), pGpu_dft_x);
    err |= clSetKernelArg(*pKernel_grad_pow, 4, sizeof(cl_mem), pGpu_dft_y);
    err |= clSetKernelArg(*pKernel_grad_pow, 5, sizeof(cl_mem), pGpu_visi0);
    err |= clSetKernelArg(*pKernel_grad_pow, 6, sizeof(cl_mem), pGpu_flux1); 
    err |= clSetKernelArg(*pKernel_grad_pow, 7, sizeof(int), &nuv);
    err |= clSetKernelArg(*pKernel_grad_pow, 8, sizeof(int), &npow);
    err |= clSetKernelArg(*pKernel_grad_pow, 9, sizeof(int), &image_width);
    err |= clSetKernelArg(*pKernel_grad_pow, 10, sizeof(cl_mem), pGpu_data_grad);  

/*   // Get the maximum work-group size for executing the kernel on the device*/
/*    err = clGetKernelWorkGroupInfo(*pKernel_u_vis_flux, *pDevice_id, CL_KERNEL_WORK_GROUP_SIZE , sizeof(size_t), &local, NULL);*/
/*    if (err != CL_SUCCESS)*/
/*        print_opencl_error("clGetKernelWorkGroupInfo", err);*/

/*    // Round down to the nearest power of two.*/
/*    local = pow(2, floor(log(npow) / log(2)));*/
    
     // Execute the kernel over the entire range of the data set        
/*    global = data_alloc_uv;*/
/*    if(gpu_enable_debug && gpu_enable_verbose)*/
/*        printf("Powerspectrum Kernel: Global: %i Local %i \n", (int)global, (int)local);*/
        
    err = clEnqueueNDRangeKernel(*pQueue, *pKernel_grad_pow, 2, 0, global, local, 0, NULL, NULL);
    if (err)
        print_opencl_error("Cannot enqueue powerspectrum gradient kernel.", err);     
     
    err  = clSetKernelArg(*pKernel_grad_bis, 0, sizeof(cl_mem), pGpu_data);
    err |= clSetKernelArg(*pKernel_grad_bis, 1, sizeof(cl_mem), pGpu_data_err);
    err |= clSetKernelArg(*pKernel_grad_bis, 2, sizeof(cl_mem), pGpu_data_uvpnt);    
    err |= clSetKernelArg(*pKernel_grad_bis, 3, sizeof(cl_mem), pGpu_data_sign);
    err |= clSetKernelArg(*pKernel_grad_bis, 4, sizeof(cl_mem), pGpu_data_phasor);
    err |= clSetKernelArg(*pKernel_grad_bis, 5, sizeof(cl_mem), pGpu_mock_data);
    err |= clSetKernelArg(*pKernel_grad_bis, 6, sizeof(cl_mem), pGpu_dft_x);
    err |= clSetKernelArg(*pKernel_grad_bis, 7, sizeof(cl_mem), pGpu_dft_y);
    err |= clSetKernelArg(*pKernel_grad_bis, 8, sizeof(cl_mem), pGpu_visi0);
    err |= clSetKernelArg(*pKernel_grad_bis, 9, sizeof(cl_mem), pGpu_flux1); 
    err |= clSetKernelArg(*pKernel_grad_bis, 10, sizeof(int), &nuv);
    err |= clSetKernelArg(*pKernel_grad_bis, 11, sizeof(int), &nbis);
    err |= clSetKernelArg(*pKernel_grad_bis, 12, sizeof(int), &npow);
    err |= clSetKernelArg(*pKernel_grad_bis, 13, sizeof(int), &image_width);
    err |= clSetKernelArg(*pKernel_grad_bis, 14, sizeof(cl_mem), pGpu_data_grad);
      
    err = clEnqueueNDRangeKernel(*pQueue, *pKernel_grad_bis, 2, 0, global, local, 0, NULL, NULL);
    if (err)
        print_opencl_error("Cannot enqueue bispectrum gradient kernel.", err); 
        
    // Let the queue finish out
    clFinish(*pQueue);
}

void gpu_compute_data_gradient_curr(int nuv, int npow, int nbis, int image_width)
{
    if(gpu_enable_verbose || gpu_enable_debug)
        printf("%sComputing data gradient on current image.\n%s", SEP, SEP);
    
    gpu_compute_data_gradient(pGpu_image, nuv, npow, nbis, image_width);
}

void gpu_compute_data_gradient_temp(int nuv, int npow, int nbis, int image_width)
{
    if(gpu_enable_verbose || gpu_enable_debug)
        printf("%sComputing data gradient on temporary image.\n%s", SEP, SEP);
    
    gpu_compute_data_gradient(pGpu_image_temp, nuv, npow, nbis, image_width);
}

void gpu_init()
{
    // Init a few variables.  Static so they won't go out of scope.
    static cl_device_id device_id;           // device ID
    static cl_context context;               // context
    static cl_command_queue queue;           // command queue
    cl_platform_id cpPlatform;

    
    int err = 0;
   
    // Get a platform ID
    err = clGetPlatformIDs(1, &cpPlatform, NULL);
    if (err != CL_SUCCESS)   //      [3]
        print_opencl_error("Unable to get platform", err);  

    // Get an ID for the device
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS)   //      [3]
        print_opencl_error("Unable to get GPU Device", err);    
    
    // Output some information about the card if we are in debug mode.    
    if(gpu_enable_verbose)
        gpu_device_stats(device_id);                                  

    // Create a context                                           
    context = clCreateContext(0, 1, &device_id, (void*) (gpu_error_callback), NULL, &err);
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

float gpu_get_chi2_curr(int nuv, int npow, int nbis, int data_alloc, int data_alloc_uv)
{
    if(gpu_enable_verbose || gpu_enable_debug)
        printf("%sGetting chi2 value from the GPU on current image.\n%s", SEP, SEP);
        
    return gpu_get_chi2(nuv, npow, nbis, data_alloc, data_alloc_uv, pGpu_image);
}

float gpu_get_chi2_temp(int nuv, int npow, int nbis, int data_alloc, int data_alloc_uv)
{
    if(gpu_enable_verbose || gpu_enable_debug)
        printf("%sGetting chi2 value from the GPU on temporary image.\n%s", SEP, SEP);
        
    return gpu_get_chi2(nuv, npow, nbis, data_alloc, data_alloc_uv, pGpu_image_temp);
}

float gpu_get_chi2(int nuv, int npow, int nbis, int data_alloc, int data_alloc_uv, cl_mem * gpu_image)
{       
    gpu_image2chi2(nuv, npow, nbis, data_alloc, data_alloc_uv, gpu_image);
    clFinish(*pQueue);

    // Now read back the chi2.
    int err = 0;
    float chi2 = 0;
    err = clEnqueueReadBuffer(*pQueue, *pGpu_chi2, CL_TRUE, 0, sizeof(float), &chi2, 0, NULL, NULL );
    if (err != CL_SUCCESS)
        print_opencl_error("Could not read back chi2 from the GPU!", err);
        
    return chi2;   
}


// Get the entropy from the GPU
float gpu_get_entropy(int image_width, cl_mem * gpu_image)
{
    if(gpu_image == NULL)
        print_opencl_error("Image for computing entropy is NULL!", 0);
        
    // TODO: We may need to modify this function to just return the entropy, for now we compute it too.
    // Becuase we are, presumably, calling this from the CPU, first call all functions to generate the entropy
    gpu_compute_entropy(image_width, gpu_image, pGpu_entropy);
    
    clFinish(*pQueue);
    
    float entropy = 0;
    int err = 0;
    
    err = clEnqueueReadBuffer(*pQueue, *pGpu_entropy, CL_TRUE, 0, sizeof(float), &entropy, 0, NULL, NULL );
    if (err != CL_SUCCESS)
        print_opencl_error("Could not read back Entropy from the GPU!", err);
        
    return entropy;
}

float gpu_get_entropy_curr(int image_width)
{
    if(gpu_enable_verbose || gpu_enable_debug)
        printf("%sGetting current image entropy.\n", SEP);
        
    return gpu_get_entropy(image_width, pGpu_image);
}

float gpu_get_entropy_temp(int image_width)
{
    if(gpu_enable_verbose || gpu_enable_debug)
        printf("%sGetting temporary image entropy.\n", SEP);
        
    return gpu_get_entropy(image_width, pGpu_image_temp);
}

// Returns a pointer to an array of floats of size image_size from gpu buffer gpu_image.
// this is a blocking call
float * gpu_get_image(int size, float * cpu_buffer, cl_mem * gpu_image)
{
    if(gpu_enable_verbose || gpu_enable_debug)
        printf("Getting image buffer from GPU.\n");
        
    int err = 0;
    // zero out the cpu buffer (just in case)
    memset(cpu_buffer, -1, sizeof(float) * size);
    
    if(cpu_buffer == NULL)
    {
        printf("Warning: Allocating buffer for storing image read-back from GPU. Be sure to free memory!\n");
        float * cpu_buffer;
        cpu_buffer = malloc(sizeof(float) * size);
    }
    
    err = clEnqueueReadBuffer(*pQueue, *gpu_image, CL_TRUE, 0, sizeof(float) * size, cpu_buffer, 0, NULL, NULL );
    if (err != CL_SUCCESS)
        print_opencl_error("Could not read back image from the GPU!", err);   
    
    return cpu_buffer;
}

cl_mem * gpu_getp_ci()
{
    return pGpu_image;
}

cl_mem * gpu_getp_ti()
{
    return pGpu_image_temp;
}

// Get a pointer to pGpu_full_gradient_new
cl_mem * gpu_getp_fgn()
{
    return pGpu_full_grad_new;
}

// Get a pointer to pGpu_full_gradient
cl_mem * gpu_getp_fg()
{
    return pGpu_full_grad;
}

// Get a pointer to pGpu_descent_dir
cl_mem * gpu_getp_dd()
{
    return pGpu_descent_dir;
}

// Get a pointer to pGpu_grad_temp
cl_mem * gpu_getp_tg()
{
    return pGpu_grad_temp;
}

cl_mem * gpu_getp_eg()
{
    return pGpu_entropy_grad;
}

// Given the image copied onto the GPU's buffer
void gpu_image2chi2(int nuv, int npow, int nbis, int data_alloc, int data_alloc_uv, cl_mem * gpu_image)
{
    if(gpu_image == NULL)
        print_opencl_error("Image for computing image2chi2 is NULL!", 0);
        
    gpu_image2vis(nuv, data_alloc_uv, gpu_image);
    gpu_vis2data(pGpu_visi0, nuv, npow, nbis);
    gpu_data2chi2(data_alloc);
}

void gpu_image2vis(int nuv, int data_alloc_uv, cl_mem * gpu_image)
{ 
    // Do a quick error check, make sure gpu_image is not null
    if(gpu_image == NULL)
        print_opencl_error("Invalid Image, did you forget to initalize?", 0);

    int err = 0;
    size_t global;                    // global domain size for our calculation
    size_t local;                     // local domain size for our calculation
               
    // First, compute the total flux, storing it in the flux0 buffer, then normalize the image.
    gpu_compute_flux(gpu_image, pGpu_flux0, pGpu_flux1);
    
    if(gpu_enable_debug)
        printf("%sComputing DFT on the GPU.\n%s", SEP, SEP);

   // Get the maximum work-group size for executing the kernel on the device
    err = clGetKernelWorkGroupInfo(*pKernel_visi, *pDevice_id, CL_KERNEL_WORK_GROUP_SIZE , sizeof(size_t), &local, NULL);
    if (err != CL_SUCCESS)
        print_opencl_error("Cannot get work group size information for Visi kernel!", err);

    // Round down to the nearest power of two.
    local = 16; //pow(2, floor(log(local) / log(2)));

    // Now we compute the DFT
    err  = clSetKernelArg(*pKernel_visi, 0, sizeof(cl_mem), gpu_image);
    err |= clSetKernelArg(*pKernel_visi, 1, sizeof(cl_mem), pGpu_dft_x);
    err |= clSetKernelArg(*pKernel_visi, 2, sizeof(cl_mem), pGpu_dft_y);
    err |= clSetKernelArg(*pKernel_visi, 3, sizeof(int), &nuv);
    err |= clSetKernelArg(*pKernel_visi, 4, sizeof(cl_mem), pGpu_image_width);   
    err |= clSetKernelArg(*pKernel_visi, 5, sizeof(cl_mem), pGpu_flux1);    
    err |= clSetKernelArg(*pKernel_visi, 6, sizeof(cl_mem), pGpu_visi0);
    err |= clSetKernelArg(*pKernel_visi, 7, local * sizeof(cl_float2), NULL);
    err |= clSetKernelArg(*pKernel_visi, 8, local * sizeof(cl_float2), NULL);
    err |= clSetKernelArg(*pKernel_visi, 9, local * sizeof(cl_float2), NULL);
    
    // Execute the kernel over the entire range of the data set        
    global = (size_t) data_alloc_uv;
    if(gpu_enable_debug || gpu_enable_verbose)
        printf("Visi Kernel: Global: %i Local %i \n", (int)global, (int)local);
        
    err = clEnqueueNDRangeKernel(*pQueue, *pKernel_visi, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
        print_opencl_error("Cannot Enqueue visi kernel.", err);  
        
    clFinish(*pQueue);
}

void gpu_kernel_workgroup_info(cl_kernel * kernel, char * kernel_name)
{
    int err = 0;
    size_t param_value_size = 0;
    size_t wg_size = 0;
    size_t compile_wg_size[3];
    cl_ulong mem_size = 0;
    
    
    err = clGetKernelWorkGroupInfo(*kernel, *pDevice_id, CL_KERNEL_WORK_GROUP_SIZE, param_value_size, &wg_size, NULL);
    err = clGetKernelWorkGroupInfo(*kernel, *pDevice_id, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, param_value_size, &compile_wg_size, NULL);
    err = clGetKernelWorkGroupInfo(*kernel, *pDevice_id, CL_KERNEL_LOCAL_MEM_SIZE, param_value_size, &mem_size, NULL); 
 
    printf("%sKernel Requirements:\n%s", SEP, SEP);
    printf("%s\n", kernel_name);
    printf("Workgroup Size: %i\n", (int) wg_size);
    printf("Compiled WG Sizes: %i %i %i\n", (int) compile_wg_size[0], (int) compile_wg_size[1], (int) compile_wg_size[2]); 
    printf("Reqired Memory: %li (bytes)\n", mem_size);
}

float gpu_linesearch_zoom(
    int nuv, int npow, int nbis, int data_alloc, int data_alloc_uv, int image_width,
    float steplength_low, float steplength_high, 
    float criterion_steplength_low, float wolfe_product1, float criterion_init, 
	int * criterion_evals, int * grad_evals, 
	cl_mem * pDescent_direction, cl_mem * pTemp_gradient,
	float hyperparameter_entropy)
{
    if(gpu_enable_verbose || gpu_enable_debug)
        printf("Running GPU Linesearch Zoom.\n");
        
	float chi2, entropy;
	float steplength =0., selected_steplength = 0., criterion = 0., criterion_old = 0., steplength_old = 0., wolfe_product2;
	int counter = 0;
	float minvalue = 1e-8;
	float wolfe_param1 = 1e-4, wolfe_param2 = 0.1;
	
	// Enable for debugging purposes
	int image_size = image_width * image_width;
	float * temp_image = malloc(image_size * sizeof(float));

	//printf("Entering zoom algorithm \n");

	while( 1 )
	{

		// Interpolation - for the moment by bisection (simple for now)
		//steplength = ( steplength_high - steplength_low ) / 2. + steplength_low;

		if((counter > 0) && ( criterion_old - criterion_init - wolfe_product1 * steplength_old ) != 0.)
		 	steplength = fabs(wolfe_product1 * steplength_old * steplength_old
				  / (2. * ( criterion_old - criterion_init - wolfe_product1 * steplength_old ) ));

			if((counter == 0) || (steplength < steplength_low ) || ( steplength > steplength_high))
	       	steplength = ( steplength_high - steplength_low ) / 2. + steplength_low;

		if( fabs( steplength_high - steplength_low ) < 1e-14)
		{
			selected_steplength=steplength_low;
			break;
		}

		// Evaluate criterion(steplength)
		
		// Start by updating the image:
/*		temp_image = gpu_get_image(image_size, temp_image, pGpu_image_temp);*/
/*	    writefits(temp_image, "!ti0.fits");*/
/*	    printf("ti0: %i\n", pGpu_image_temp);*/
		
        gpu_update_tempimage(image_width, steplength, minvalue, pDescent_direction);
/*		temp_image = gpu_get_image(image_size, temp_image, pGpu_image_temp);*/
/*	    writefits(temp_image, "!ti1.fits");*/
/*        printf("ti1: %i\n", pGpu_image_temp);  */

		chi2 = gpu_get_chi2_temp(nuv, npow, nbis, data_alloc, data_alloc_uv);
		entropy = gpu_get_entropy_temp(image_width);
		criterion = chi2 - hyperparameter_entropy * entropy;
		*criterion_evals++;

/*		temp_image = gpu_get_image(image_size, temp_image, pGpu_image_temp);*/
/*	    writefits(temp_image, "!ti2.fits");*/
/*	    printf("ti2: %i\n", pGpu_image_temp);*/
/*	    printf("Waiting\n");*/
/*	    getchar();*/
/*		*/

        if(gpu_enable_verbose && gpu_enable_debug)
        {
            printf("LSZoom\t criterion %lf criterion_init %lf criterion_old %lf \n", criterion , criterion_init, criterion_old );
            printf("LSZoom\t chi2 %1f entropy %1f hyperparameter_entropy %1f\n", chi2, entropy, hyperparameter_entropy);
		    printf("Criterion %8.8e Steplength %8.8le Low %8.8le High %8.8le Counter %d -- Zoom \n", criterion, steplength, steplength_low, steplength_high, counter);
		    printf("Test 1\t criterion %lf criterion_init %lf second member wolfe_param1 %1f wolfe_product1 %1f\n", criterion , criterion_init,  wolfe_param1, wolfe_product1);
		}
		if ( (criterion > ( criterion_init + wolfe_param1 * steplength * wolfe_product1 ) ) || ( criterion >= criterion_steplength_low ) )
		{
			steplength_high = steplength;
		}
		else
		{

			// Evaluate wolfe product 2
			gpu_compute_data_gradient_temp(nuv, npow, nbis, image_width);
			gpu_compute_entropy_gradient_temp(image_width);
			
			gpu_compute_criterion_gradient_temp(image_width, hyperparameter_entropy);

			*grad_evals++;
			wolfe_product2 = gpu_get_scalprod(image_width, image_width, pDescent_direction, pTemp_gradient);
			
			// Enable for debugging:
/*			temp_image = gpu_get_image(image_size, temp_image, pDescent_direction);*/
/*			writefits(temp_image, "!dd.fits");*/
/*			temp_image = gpu_get_image(image_size, temp_image, pTemp_gradient);*/
/*          writefits(temp_image, "!tg.fits");*/

            if(gpu_enable_verbose && gpu_enable_debug)
	    		printf("Wolfe products: %le %le Second member wolfe2 %le \n", wolfe_product1, wolfe_product2, - wolfe_param2 * wolfe_product1);
	
			if( ( wolfe_product2 >= wolfe_param2 * wolfe_product1 ) || ( counter > 10 ))
			{
				selected_steplength = steplength;
				break;
			}

			if(wolfe_product2 * (steplength_high - steplength_low) >= 0. )
			    steplength_high = steplength_low;

			steplength_low = steplength;

		}

		steplength_old = steplength;
		criterion_old = criterion;

		counter++;
	}

	return selected_steplength;
}

void gpu_new_chi2(int nuv, int npow, int nbis, int data_alloc)
{
    gpu_vis2data(pGpu_visi1, nuv, npow, nbis);
    gpu_data2chi2(data_alloc);
}

void gpu_scalar_prod(int data_width, int data_height, cl_mem * array1, cl_mem * array2, cl_mem * final_output)
{       
    if(final_output == NULL || array1 == NULL || array2 == NULL)
        print_opencl_error("Pointer to gpu_sclar_prod is NULL!", 0);
        
    size_t global = data_width * data_height;
    size_t local = 0;
    int err = 0;

    // Now we compute the scalarprod product
    err  = clSetKernelArg(*pKernel_scalarprod, 0, sizeof(cl_mem), array1);
    err |= clSetKernelArg(*pKernel_scalarprod, 1, sizeof(cl_mem), array2);
    err |= clSetKernelArg(*pKernel_scalarprod, 2, sizeof(cl_mem), pGpu_flux_buffer0);    

   // Get the maximum work-group size for executing the kernel on the device
    err = clGetKernelWorkGroupInfo(*pKernel_scalarprod, *pDevice_id, CL_KERNEL_WORK_GROUP_SIZE , sizeof(size_t), &local, NULL);
    if (err != CL_SUCCESS)
        print_opencl_error("clGetKernelWorkGroupInfo", err);

    // Round down to the nearest power of two.
    // TODO: Un-fix this value:
    err = clGetKernelWorkGroupInfo(*pKernel_visi, *pDevice_id, CL_KERNEL_WORK_GROUP_SIZE , sizeof(size_t), &local, NULL);
    if (err != CL_SUCCESS)
        print_opencl_error("clGetKernelWorkGroupInfo", err);

    // Round down to the nearest power of two.
    local = pow(2, floor(log(local) / log(2)));
        
    err = clEnqueueNDRangeKernel(*pQueue, *pKernel_scalarprod, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
        print_opencl_error("Could not enqueue scalar product kernel.", err);  

    clFinish(*pQueue);
    
    // And now we need to run a parallel sum
    // Note, we just need to sum up the scalar product values, so the flux kernels work fine here.
    gpu_compute_sum(pGpu_flux_buffer0, pGpu_flux_buffer1, pGpu_flux_buffer2, final_output, pGpu_flux_kernels, Flux_pass_count, Flux_group_counts, Flux_work_item_counts, Flux_operation_counts, Flux_entry_counts);

    clFinish(*pQueue);
}

float gpu_get_scalprod(int data_width, int data_height, cl_mem * array1, cl_mem * array2)
{
    if(gpu_enable_verbose || gpu_enable_debug)
        printf("%sComputing Scalar Product from the GPU.\n%s", SEP, SEP);
        
    gpu_scalar_prod(data_width, data_height, array1, array2, pGpu_scaprod);
    
    // Read back the value from the GPU:
    int err = 0;
    float value = 0;
    err = clEnqueueReadBuffer(*pQueue, *pGpu_scaprod, CL_TRUE, 0, sizeof(float), &value, 0, NULL, NULL );
    if (err != CL_SUCCESS)
        print_opencl_error("Could not read back scalar product value from GPU!", err);
        
    return value;     
}

void gpu_shutdown()
{
    clFinish(*pQueue);
    gpu_cleanup();
}

void gpu_update_vis_fluxchange(int x, int y, float new_pixel_flux, int image_width, int nuv, int data_alloc_uv)
{
    int err = 0;
    size_t local = 0;
    size_t global = 0;
    int offset = x + y * image_width;
    
    // First read back the current flux:
    float curr_flux = 0;
    err = clEnqueueReadBuffer(*pQueue, *pGpu_image, CL_TRUE, offset, sizeof(float), &curr_flux, 0, NULL, NULL );
    if(err != CL_SUCCESS)
        print_opencl_error("Could not read back the current pixel's flux value.", err);    
    
    float flux_ratio = curr_flux / new_pixel_flux;
        
    // Set the kernel arguments:
    err  = clSetKernelArg(*pKernel_u_vis_flux, 0, sizeof(cl_mem), pGpu_visi0);
    err |= clSetKernelArg(*pKernel_u_vis_flux, 1, sizeof(cl_mem), pGpu_visi1);
    err |= clSetKernelArg(*pKernel_u_vis_flux, 2, sizeof(cl_mem), pGpu_dft_x);
    err |= clSetKernelArg(*pKernel_u_vis_flux, 3, sizeof(cl_mem), pGpu_dft_y);
    err |= clSetKernelArg(*pKernel_u_vis_flux, 4, sizeof(int), &x);
    err |= clSetKernelArg(*pKernel_u_vis_flux, 5, sizeof(int), &y);
    err |= clSetKernelArg(*pKernel_u_vis_flux, 6, sizeof(int), &image_width);
    err |= clSetKernelArg(*pKernel_u_vis_flux, 7, sizeof(float), &flux_ratio);    
    if (err != CL_SUCCESS)
        print_opencl_error("clSetKernelArg", err);   

   // Get the maximum work-group size for executing the kernel on the device
    err = clGetKernelWorkGroupInfo(*pKernel_u_vis_flux, *pDevice_id, CL_KERNEL_WORK_GROUP_SIZE , sizeof(size_t), &local, NULL);
    if (err != CL_SUCCESS)
        print_opencl_error("clGetKernelWorkGroupInfo", err);

    // Round down to the nearest power of two.
    local = pow(2, floor(log(local) / log(2)));
    
    // Execute the kernel over the entire range of the data set        
    global = data_alloc_uv;
    if(gpu_enable_debug && gpu_enable_verbose)
      printf("Updated Visi Kernel: Global: %i Local %i \n", (int)global, (int)local);
        
    err = clEnqueueNDRangeKernel(*pQueue, *pKernel_u_vis_flux, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
        print_opencl_error("clEnqueueNDRangeKernel u_vis_flux", err);  
        
    clFinish(*pQueue);    
} 

void gpu_vis2data(cl_mem * gpu_visi, int nuv, int npow, int nbis)
{
    // Begin by copying vis over to the GPU
    int err = 0;

    size_t global;                    // global domain size for our calculation
    size_t local;                     // local domain size for our calculation
    
    // ############
    // First we run a kernel to compute the powerspectrum:
    // ############  

    err  = clSetKernelArg(*pKernel_powspec, 0, sizeof(cl_mem), pGpu_visi0);
    err |= clSetKernelArg(*pKernel_powspec, 1, sizeof(cl_mem), pGpu_mock_data);    // Output is stored on the GPU.
    if (err != CL_SUCCESS)
        print_opencl_error("clSetKernelArg", err);    
 
    // Get the maximum work-group size for executing the kernel on the device
    global = (size_t) npow;
    err = clGetKernelWorkGroupInfo(*pKernel_powspec, *pDevice_id, CL_KERNEL_WORK_GROUP_SIZE , sizeof(size_t), &local, NULL);
    if (err != CL_SUCCESS)
        print_opencl_error("clGetKernelWorkGroupInfo", err);
    

    // Execute the kernel over the entire range of the data set        
    err = clEnqueueNDRangeKernel(*pQueue, *pKernel_powspec, 1, NULL, &global, NULL, 0, NULL, NULL);
    if (err)
        print_opencl_error("Could not enqueue powerspectrum visi kernel.", err);   
        
    clFinish(*pQueue);
        
    // ############
    // Now we run a kernel to compute the bispectrum:
    // ############
    err  = clSetKernelArg(*pKernel_bispec, 0, sizeof(cl_mem), pGpu_visi0);
    err |= clSetKernelArg(*pKernel_bispec, 1, sizeof(cl_mem), pGpu_data_phasor);
    err |= clSetKernelArg(*pKernel_bispec, 2, sizeof(cl_mem), pGpu_data_uvpnt);
    err |= clSetKernelArg(*pKernel_bispec, 3, sizeof(cl_mem), pGpu_data_sign);
    err |= clSetKernelArg(*pKernel_bispec, 4, sizeof(cl_mem), pGpu_mock_data);      // Output is stored on the GPU.
    err |= clSetKernelArg(*pKernel_bispec, 5, sizeof(cl_mem), pGpu_pow_size);    
    if (err != CL_SUCCESS)
        print_opencl_error("clSetKernelArg", err); 
 
   // Get the maximum work-group size for executing the kernel on the device
    err = clGetKernelWorkGroupInfo(*pKernel_bispec, *pDevice_id, CL_KERNEL_WORK_GROUP_SIZE , sizeof(size_t), &local, NULL);
    if (err != CL_SUCCESS)
        print_opencl_error("clGetKernelWorkGroupInfo", err);

    // Execute the kernel over the entire range of the data set        
    global = (size_t) nbis;
    err = clEnqueueNDRangeKernel(*pQueue, *pKernel_bispec, 1, NULL, &global, NULL, 0, NULL, NULL);
    if (err)
        print_opencl_error("Could not enqueue bispectrum visi kernel.", err);   

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
        print_opencl_error("Could not Load Source file!  Enable Verbosity to see more information.", 0);

    stat(filename, &statbuf);
    source = (char *) malloc(statbuf.st_size + 1);
    fread(source, statbuf.st_size, 1, fh);
    source[statbuf.st_size] = '\0';

    return source;
}

void gpu_update_image(int image_width, float steplength, float minval, cl_mem * descent_direction)
{
    if(gpu_enable_verbose || gpu_enable_debug)
        printf("%sUpdating an the current image.\n%s", SEP, SEP);
        
    if(descent_direction == NULL)
        print_opencl_error("Descent Direction to update_image is NULL!", 0);
        
    int err = 0;
    // TODO: Figure out how to determine the size of local dynamically.
    size_t * local;
    local = malloc(2 * sizeof(size_t));
    local[0] = local[1] = (size_t) 16;
    
    size_t * global;
    global = malloc(2 * sizeof(size_t));
    global[0] = global[1] = (size_t) image_width;
    
    // Set the arguments for the entropy kernel:
    err  = clSetKernelArg(*pKernel_update_image, 0, sizeof(cl_mem), pGpu_image);
    err |= clSetKernelArg(*pKernel_update_image, 1, sizeof(cl_mem), descent_direction);
    err |= clSetKernelArg(*pKernel_update_image, 2, sizeof(float), &steplength);
    err |= clSetKernelArg(*pKernel_update_image, 3, sizeof(float), &minval);  
    err |= clSetKernelArg(*pKernel_update_image, 4, sizeof(cl_mem), pGpu_image_width); 

/*   // Get the maximum work-group size for executing the kernel on the device*/
/*    err = clGetKernelWorkGroupInfo(*pKernel_u_vis_flux, *pDevice_id, CL_KERNEL_WORK_GROUP_SIZE , sizeof(size_t), &local, NULL);*/
/*    if (err != CL_SUCCESS)*/
/*        print_opencl_error("clGetKernelWorkGroupInfo", err);*/

/*    // Round down to the nearest power of two.*/
/*    local = pow(2, floor(log(npow) / log(2)));*/
        
    err = clEnqueueNDRangeKernel(*pQueue, *pKernel_update_image, 2, 0, global, local, 0, NULL, NULL);
    if (err)
        print_opencl_error("Cannot enqueue image update kernel.", err); 
    
    clFinish(*pQueue);   
}

void gpu_update_tempimage(int image_width, float steplength, float minval, cl_mem * descent_direction)
{
    if(gpu_enable_verbose || gpu_enable_debug)
        printf("Updating the temporary image.\n");

    if(descent_direction == NULL)
        print_opencl_error("Descent Direction to update_tempimage is NULL!", 0);
        
    int err = 0;
    // TODO: Figure out how to determine the size of local dynamically.
    size_t * local;
    local = malloc(2 * sizeof(size_t));
    local[0] = local[1] = (size_t) 16;
    
    size_t * global;
    global = malloc(2 * sizeof(size_t));
    global[0] = global[1] = (size_t) image_width;
    
    // Set the arguments for the entropy kernel:
    err  = clSetKernelArg(*pKernel_update_tempimage, 0, sizeof(cl_mem), pGpu_image);
    err |= clSetKernelArg(*pKernel_update_tempimage, 1, sizeof(cl_mem), descent_direction);
    err |= clSetKernelArg(*pKernel_update_tempimage, 2, sizeof(float), &steplength);
    err |= clSetKernelArg(*pKernel_update_tempimage, 3, sizeof(float), &minval);  
    err |= clSetKernelArg(*pKernel_update_tempimage, 4, sizeof(cl_mem), pGpu_image_width); 
    err |= clSetKernelArg(*pKernel_update_tempimage, 5, sizeof(cl_mem), pGpu_image_temp); 

/*   // Get the maximum work-group size for executing the kernel on the device*/
/*    err = clGetKernelWorkGroupInfo(*pKernel_u_vis_flux, *pDevice_id, CL_KERNEL_WORK_GROUP_SIZE , sizeof(size_t), &local, NULL);*/
/*    if (err != CL_SUCCESS)*/
/*        print_opencl_error("clGetKernelWorkGroupInfo", err);*/

/*    // Round down to the nearest power of two.*/
/*    local = pow(2, floor(log(npow) / log(2)));*/
        
    err = clEnqueueNDRangeKernel(*pQueue, *pKernel_update_tempimage, 2, 0, global, local, 0, NULL, NULL);
    if (err)
        print_opencl_error("Cannot temp image update kernel.", err); 
    
    clFinish(*pQueue);   
}

