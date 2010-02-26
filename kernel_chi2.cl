__kernel void compute_chi2(
    __global float * data,
    __global float * data_err,
    __global float * mock_data,
    __global float * output)
{
    int i = get_global_id(0);
					
    float temp = (data[i] - mock_data[i]) * data_err[i];
    output[i] = temp * temp;
}
