#include <CL/cl.h> // OpenCL header file
#include <complex.h>

// Funcion declairations
void print_opencl_error(char* error_message, int error_code);

char * print_cl_errstring(cl_int err);

void gpu_build_kernels(int data_size, int image_width, int image_size);

void gpu_build_reduction_kernels(int data_size, cl_program ** pPrograms, cl_kernel ** pKernels, 
    int * pass_counts, size_t ** group_counts, size_t ** work_item_counts, 
    int ** operation_counts, int ** entry_counts);

void gpu_check_data(float * cpu_chi2, 
    int nuv, float complex * visi, 
    int data_size, float * mock_data, 
    int image_size, float * data_grad);

void gpu_compare_data(int size, float * cpu_data, cl_mem * pGpu_data);

void gpu_compare_complex_data(int size, float complex * cpu_data, cl_mem * pGpu_data);

void gpu_compute_entropy(int image_width, cl_mem * gpu_image, cl_mem * entropy_storage);
void gpu_compute_entropy_gradient(int image_width, cl_mem * gpu_image);

void gpu_compute_flux(cl_mem * gpu_image, cl_mem * flux_storage, cl_mem * flux_inverse_storage);

void gpu_compute_sum(cl_mem * input_buffer, cl_mem * output_buffer, cl_mem * partial_sum_buffer, cl_mem * final_buffer, 
    cl_kernel * pKernels, 
    int pass_count, size_t * group_counts, size_t * work_item_counts, 
    int * operation_counts, int * entry_counts);

void gpu_compute_data_gradient(cl_mem * gpu_image, int npow, int nbis, int image_width);

void gpu_copy_data(float * data, float * data_err, int data_size, int data_size_uv,\
                    cl_float2 * data_phasor, int phasor_size, int pow_size, \
                    cl_long4 * gpu_bsref_uvpnt, cl_short4 * gpu_bsref_sign, int bsref_size,
                    float * default_model,
                    int image_size, int image_width);
                    
void gpu_copy_dft(cl_float2 * dft_x, cl_float2 * dft_y, int dft_size);

void gpu_copy_image(float * image, int x_size, int y_size);

void gpu_cleanup();

void gpu_data2chi2(int data_size);

float gpu_get_chi2_curr(int nuv, int npow, int nbis, int data_alloc, int data_alloc_uv);
float gpu_get_chi2_temp(int nuv, int npow, int nbis, int data_alloc, int data_alloc_uv);
float gpu_get_chi2(int nuv, int npow, int nbis, int data_alloc, int data_alloc_uv, cl_mem * gpu_image);

float gpu_get_entropy();

void gpu_image2chi2(int nuv, int npow, int nbis, int data_alloc, int data_alloc_uv, cl_mem * gpu_image);

void gpu_image2vis(int data_alloc_uv, cl_mem * gpu_image);

void gpu_init();

void gpu_new_chi2(int nuv, int npow, int nbis, int data_alloc);

void gpu_vis2data(cl_mem * gpu_vis, int nuv, int npow, int nbis);

static char * LoadProgramSourceFromFile(const char *filename);

void gpu_device_stats(cl_device_id device_id);

void gpu_compare_data(int size, float * cpu_data, cl_mem * pGpu_data);

void gpu_compare_complex_data(int size, float complex * cpu_data, cl_mem * pGpu_data);

void gpu_scalar_prod(int data_width, int data_height, cl_mem * array1, cl_mem * array2, cl_mem * output);

