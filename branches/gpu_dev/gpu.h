#include <CL/cl.h> // OpenCL header file
#include <complex.h>

// Funcion declairations
void print_opencl_error(char* error_message, int error_code);

char * print_cl_errstring(cl_int err);

void gpu_backup_gradient(int data_size, cl_mem * input, cl_mem * output);

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

void gpu_compute_criterion_gradient(int image_width, float hyperparameter_entropy, cl_mem * gradient_buffer);
void gpu_compute_criterion_gradient_curr(int image_width, float hyperparameter_entropy);
void gpu_compute_criterion_gradient_temp(int image_width, float hyperparameter_entropy);

void gpu_compute_criterion_step(int image_width, float steplength, float minvalue);

void gpu_compute_entropy(int image_width, cl_mem * gpu_image, cl_mem * entropy_storage);
void gpu_compute_entropy_gradient(int image_width, cl_mem * gpu_image);
void gpu_compute_entropy_gradient_curr(int image_width);
void gpu_compute_entropy_gradient_temp(int image_width);

void gpu_compute_flux(cl_mem * gpu_image, cl_mem * flux_storage, cl_mem * flux_inverse_storage);

void gpu_compute_sum(cl_mem * input_buffer, cl_mem * output_buffer, cl_mem * partial_sum_buffer, cl_mem * final_buffer, 
    cl_kernel * pKernels, 
    int pass_count, size_t * group_counts, size_t * work_item_counts, 
    int * operation_counts, int * entry_counts);

void gpu_compute_data_gradient(cl_mem * gpu_image, int nuv, int npow, int nbis, int image_width);
void gpu_compute_data_gradient_curr(int nuv, int npow, int nbis, int image_width);
void gpu_compute_data_gradient_temp(int nuv, int npow, int nbis, int image_width);

void gpu_compute_descent_dir(int image_width, float beta);

void gpu_copy_data(float * data, float * data_err, int data_size, int data_size_uv,\
                    cl_float2 * data_phasor, int phasor_size, int pow_size, \
                    cl_long4 * gpu_bsref_uvpnt, cl_short4 * gpu_bsref_sign, int bsref_size,
                    float * default_model,
                    int image_size, int image_width);
                    
void gpu_copy_dft(cl_float2 * dft_x, cl_float2 * dft_y, int dft_size);

void gpu_copy_dft_info(int nuv, cl_float2 * gpu_uv_info, float image_pixellation);

void gpu_copy_image(float * image, int x_size, int y_size);

void gpu_cleanup();

void gpu_data2chi2(int data_size);

float gpu_get_chi2_curr(int nuv, int npow, int nbis, int data_alloc, int data_alloc_uv);
float gpu_get_chi2_temp(int nuv, int npow, int nbis, int data_alloc, int data_alloc_uv);
float gpu_get_chi2(int nuv, int npow, int nbis, int data_alloc, int data_alloc_uv, cl_mem * gpu_image);

float gpu_get_entropy();
float gpu_get_entropy_curr(int image_width);
float gpu_get_entropy_temp(int image_width);

float * gpu_get_image(int size, float * cpu_buffer, cl_mem * gpu_image);

float gpu_get_scalprod(int data_width, int data_height, cl_mem * array1, cl_mem * array2);

cl_mem * gpu_getp_ci();
cl_mem * gpu_getp_ti();
cl_mem * gpu_getp_fgn();
cl_mem * gpu_getp_fg();
cl_mem * gpu_getp_dd();
cl_mem * gpu_getp_tg();
cl_mem * gpu_getp_eg();

void gpu_image2chi2(int nuv, int npow, int nbis, int data_alloc, int data_alloc_uv, cl_mem * gpu_image);

void gpu_image2vis(int nuv, int data_alloc_uv, cl_mem * gpu_image);

void gpu_init();

float gpu_linesearch_zoom(
    int nuv, int npow, int nbis, int data_alloc, int data_alloc_uv, int image_width,
    float steplength_low, float steplength_high, 
    float criterion_steplength_low, float wolfe_product1, float criterion_init, 
	int * criterion_evals, int * grad_evals, 
	cl_mem * pDescent_direction, cl_mem * pTemp_gradient,
	float hyperparameter_entropy);

void gpu_new_chi2(int nuv, int npow, int nbis, int data_alloc);

void gpu_vis2data(cl_mem * gpu_vis, int nuv, int npow, int nbis);

static char * LoadProgramSourceFromFile(const char *filename);

void gpu_device_stats(cl_device_id device_id);

void gpu_compare_data(int size, float * cpu_data, cl_mem * pGpu_data);

void gpu_compare_complex_data(int size, float complex * cpu_data, cl_mem * pGpu_data);

void gpu_scalar_prod(int data_width, int data_height, cl_mem * array1, cl_mem * array2, cl_mem * output);

void gpu_shutdown();

void gpu_update_image(int image_width, float steplength, float minval, cl_mem * descent_direction);
void gpu_update_tempimage(int image_width, float steplength, float minval, cl_mem * descent_direction);

