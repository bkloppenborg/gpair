__kernel void compute_chi2_bis(
    __global float2 * data,
    __global float2 * data_err,
    __global float2 * mock_data,
    __global int * nbis,
    __global float * output)
{
    int i = get_global_id(0);
    
    float2 temp;
    temp.s0 = 0;
    temp.s1 = 0;
    
    // Only compute values if we are one of the first nbis elements.
    if(i < 1)
    {		
        temp = data_err[i]; //(mock_data[i] - data[i]) * data_err[i];
        //temp = temp * temp;
    }
    
    output[i] = temp.s0; // + temp.s1;
}
