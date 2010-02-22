__kernel void compute_chi2(
    __global float * data,
    __global float * data_err,
    __global float * curr_model,
    __global float * output,
    const unsigned int count)
{
    int i = get_global_id(0);
    if(i < count)
    output[i] = (curr_model[i] - data[i]) / data_err[i];
    output[i] *= output[i];
}
