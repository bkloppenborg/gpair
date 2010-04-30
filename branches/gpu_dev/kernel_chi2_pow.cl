__kernel void compute_chi2_pow(
    __global float * data,
    __global float * data_err,
    __global float * mock_data,
    __global int * npow,
    __global float * output)
{
    int i = get_global_id(0);
    float temp;
    
    // Only compute values if we are one of the first npow elements.	
    temp = (data[i] - mock_data[i]) * data_err[i];
    temp = temp * temp;

    if(i < npow[0])    
        output[i] = temp;
    else
        output[i] = 0;
}
