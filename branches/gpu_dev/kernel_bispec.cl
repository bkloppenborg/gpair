// Multiply four complex numbers.
float2 MultComplex4(float2 A, float2 B, float2 C, float2 D)
{
    float2 temp;
    float a = A.s1 * B.s1;
    float b = C.s1 * D.s1;
    float c = A.s0 * B.s0;
    float d = A.s0 * B.s1;
    float e = C.s0 * D.s1;
    float f = A.s1 * B.s0;
    float g = C.s1 * D.s0;
    float h = C.s0 * D.s0;
  
    temp.s0 = a*b - c*b - d*e - f*e - d*g - f*g - a*h + c*h;
    temp.s1 = f*h + d*h + c*g - a*g + c*e - a*e - f*b - d*b;

    return temp;
}

// The actual kernel function.
__kernel void compute_bispec(
    __global float2 * vis,
    __global float2 * data_bip,
    __global long4 * data_uvpnt,
    __global short4 * data_sign,
    __global float2 * mock_data_bs)
{   
    int i = get_global_id(0);

    // Pull some data from global memory:
    long4 uvpnt = data_uvpnt[i];
    float2 vab = vis[uvpnt.s0];
    float2 vbc = vis[uvpnt.s1];
    float2 vca = vis[uvpnt.s2];
    
    short4 sign = data_sign[i];
    vab.s1 *= sign.s0;
    vbc.s1 *= sign.s1;
    vca.s1 *= sign.s2;
    
    // TODO: Convert mock_data_bs over to a float2 array.
    mock_data_bs[i] = MultComplex4(vab, vbc, vca, data_bip[i]);
    //mock_data_bs[2*i] = temp.s0;
    //mock_data_bs[2*i + 1] = temp.s1;
}
