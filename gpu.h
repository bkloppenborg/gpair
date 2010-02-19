#include "cl.h" // OpenCL header file
#include <complex.h>

// Funcion declairations
void print_opencl_error(char* error_message, int error_code);

void gpu_build_kernels();

void gpu_copy_data(float *data, float *data_err, cl_float2 * bisphasor, int npow, int nbis);
void gpu_cleanup();

double gpu_data2chi2(float *mock, int npow, int nbis);

void gpu_init();

void gpu_vis2data(cl_float2 *vis, int nuv, int npow, int nbis);
