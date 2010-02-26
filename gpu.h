#include "cl.h" // OpenCL header file
#include <complex.h>

// Funcion declairations
void print_opencl_error(char* error_message, int error_code);

void gpu_build_kernels();

void gpu_compare_data(int size, float * cpu_data, cl_mem * gpu_data);
void gpu_copy_data(float *data, float *data_err, int data_size,\
                    cl_float2 * bisphasor, int bip_size,\
                    long * gpu_bsref_uvpnt, short * gpu_bsref_sign, int bsref_size);

void gpu_cleanup();

double gpu_data2chi2(int data_size);

void gpu_init();

void gpu_vis2data(cl_float2 *vis, int nuv, int npow, int nbis);

static char * LoadProgramSourceFromFile(const char *filename);

void gpu_device_stats(cl_device_id device_id);
