

// Load the OCL kernel:
const char *ocl_kernel_square = "\n" \
"__kernel void square(                                                  \n" \
"   __global float* input,                                              \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       output[i] = input[i] * input[i];                                \n" \
"}                                                                      \n" \
"\n";

const char *ocl_kernel_chi2 = "\n" \
"__kernel void chi2(                                                \n" \
"    __global float* data,                                          \n" \
"    __global float* curr_model,                                    \n" \
"    __global float* output,                                        \n" \
"    const unsigned int count)                                      \n" \
"{                                                                  \n" \
"    int i = get_global_id();                                       \n" \
"    if(i < count)                                                  \n" \
"        output[i] = (curr_model[i] - data[i] ) / data_err[i]       \n" \
"}                                                                  \n" \
"\n";       
