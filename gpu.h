#include "cl.h" // OpenCL header file
#include <complex.h>

// Funcion declairations
void print_opencl_error(char* error_message, int error_code);

int gpu_build_kernel(cl_program * program, cl_kernel * kernel, char * kernel_name, char * filename);
void gpu_build_kernels();
int gpu_build_reduce_kernels(int data_size);

void gpu_compare_data(int size, float * cpu_data, cl_mem * gpu_data);
void gpu_copy_data(float *data, float *data_err, int data_size,\
                    cl_float2 * bisphasor, int bip_size,\
                    long * gpu_bsref_uvpnt, short * gpu_bsref_sign, int bsref_size);

void gpu_cleanup();

void gpu_data2chi2(int data_size);

void gpu_init();

void gpu_vis2data(cl_float2 *vis, int nuv, int npow, int nbis);

static char * LoadProgramSourceFromFile(const char *filename);

void gpu_device_stats(cl_device_id device_id);

void gpu_reduction_chi2(cl_mem * input_buffer, cl_mem * output_buffer, cl_mem * partials_buffer, cl_mem ** final_output);

void gpu_reduction_pass_counts(int count, int max_group_size, int max_groups, int max_work_items, 
    int *pass_count, size_t **group_counts, size_t **work_item_counts, int **operation_counts, 
    int **entry_counts);
