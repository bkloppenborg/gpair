__kernel void compute_pow_spec(
    __global float2 * input,
    __global float * mock_data)
{
    int i = get_global_id(0);
    mock_data[i] = input[i].s0 * input[i].s0 + input[i].s1 * input[i].s1;
}
