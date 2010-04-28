__kernel void arr_normalize(
    __global float * array,
    __global float * norm_fac)
{
    int i = get_global_id(0);

    array[i] *= norm_fac[0];
}
