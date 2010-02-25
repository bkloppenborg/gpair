__kernel void compute_chi2(
    __global float * data,
    __global float * data_err,
    __global float * mock_data,
    __global float * output,
    __local float * shared,
    const unsigned int count)
{
    int i = get_global_id(0);
    int lsize = get_local_size(0);

    shared[i] = mock_data[i];
    shared[i + lsize] = data[i];
    shared[i + 2*lsize] = data_err[i];

	barrier(CLK_LOCAL_MEM_FENCE);
						
    if(i < count)
    {
        output[i] = (shared[i] - shared[i + lsize]) / shared[i + 2*lsize];
        output[i] *= output[i];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
}
