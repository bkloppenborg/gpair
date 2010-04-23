__kernel void compute_chi2(
    __global float2 * data,
    __global float2 * data_err,
    __global float2 * mock_data,
    __global float * output)
{
    int i = get_global_id(0);
					
    float2 temp = (data[i] - mock_data[i]) * data_err[i];
    output[i] = temp.s0 * temp.s0 + temp.s1 * temp.s1;
}
