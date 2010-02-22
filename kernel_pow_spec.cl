__kernel void compute_pow_spec(
    __global float2 * input,
    __global float * output)
{
    int i = get_global_id(0);
    output[i] = input[i][0] * input[i][0] + input[i][1] * input[i][1];
}
