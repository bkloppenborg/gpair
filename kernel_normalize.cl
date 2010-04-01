__kernel void compute_chi2(
    __global float * array,
    __local const float norm_fac)
{
    int i = get_global_id(0);

    array[i] *= norm_fac
}
