__kernel void compute_pow_spec(
    __global float2 * input,
    __global float * mock_data)
{
    int i = get_global_id(0);
    mock_data[i] = input[i][0] * input[i][0] + input[i][1] * input[i][1];
}
