// A dot product kernel, need to run parallel reduce afterward
__kernel void scalarprod(
    __global float * array1,
    __global float * array2,
    __global float * output)
{
    int i = get_global_id(0);
    output[i] = array1[i] * array2[i];
}
