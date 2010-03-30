#include <CL/cl.h> // OpenCL header file for ATI
// #include "cl.h" // OpenCL header file for nvidia
#include <complex.h>

// Funcion declairations
void print_opencl_error(char* error_message, int error_code);
char * print_cl_errstring(cl_int err);

void gpu_build_kernels(int data_size, int image_size);
void gpu_build_reduction_kernels(int data_size, cl_program ** pPrograms, cl_kernel ** pKernels, 
    int * pass_counts, size_t ** group_counts, size_t ** work_item_counts, 
    int ** operation_counts, int ** entry_counts);

//void gpu_check_data(float * cpu_chi2, int nuv, float complex * visi, int data_size, float * mock_data);

void gpu_compute_sum(cl_mem * input_buffer, cl_mem * output_buffer, cl_mem * partial_sum_buffer, cl_mem * final_buffer, 
    cl_kernel * pKernels, 
    int pass_count, size_t * group_counts, size_t * work_item_counts, 
    int * operation_counts, int * entry_counts);

void gpu_copy_data(float *data, float *data_err, int data_size, int data_size_uv,\
                    cl_float2 * data_bis, int phasor_size, int pow_size,\
                    long * gpu_bsref_uvpnt, short * gpu_bsref_sign, int bsref_size,
                    int image_size, int image_width);
                    
void gpu_copy_dft(cl_float2 * dft_x, cl_float2 * dft_y, int dft_size);

void gpu_copy_image(float * image, int x_size, int y_size);

void gpu_cleanup();

void gpu_data2chi2(int data_size);

void gpu_image2vis(int data_alloc_uv);

void gpu_init();

void gpu_vis2data(cl_float2 *vis, int nuv, int npow, int nbis);

static char * LoadProgramSourceFromFile(const char *filename);

void gpu_device_stats(cl_device_id device_id);
void gpu_compare_data(int size, float * cpu_data, cl_mem * pGpu_data);
void gpu_compare_complex_data(int size, float complex * cpu_data, cl_mem * pGpu_data);

