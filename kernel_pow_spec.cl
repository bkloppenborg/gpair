__kernel void compute_pow_spec(
    __global float2 * input,
    __global float * mock_data)
{
    int i = get_global_id(0);
    float2 temp = input[i];
    
    temp.s0 = temp.s0 * temp.s0 + temp.s1 * temp.s1;
    temp.s1 = 0;
    mock_data[i] = temp.s0;
}
