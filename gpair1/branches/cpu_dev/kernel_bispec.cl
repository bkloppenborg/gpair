// Function prototypes:
float2 MultComplex2(float2 A, float2 B);
float2 MultComplex4(float2 A, float2 B, float2 C, float2 D);

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
    
/*    float2 temp;*/
/*    temp = MultComplex2(A, B);*/
/*    temp = MultComplex2(temp, C);*/
/*    temp = MultComplex2(temp, D);*/
/*    return temp;*/
}

// Multiply two complex numbers
float2 MultComplex2(float2 A, float2 B)
{
    // There is the obvious way to do this:
/*    float2 temp;*/
/*    temp.s0 = A.s0*B.s0 - A.s1*B.s1;*/
/*    temp.s1 = A.s0*B.s1 + A.s1*B.s0;  */
/*    */
/*    return temp;*/
    
    // We can trade off one multiplication for three additional additions
    float k1 = A.s0 * B.s0;
    float k2 = A.s1 * B.s1;
    float k3 = (A.s0 + A.s1) * (B.s0 + B.s1);
    
    float2 temp;
    temp.s0 = k1 - k2;
    temp.s1 = k3 - k1 - k2;
    return temp;
}

// The actual kernel function.
__kernel void compute_bispec(
    __global float2 * vis,
    __global float2 * data_bip,
    __global long4 * data_uvpnt,
    __global short4 * data_sign,
    __global float * mock_data_bs,
    __global int * array_offset)
{   
    int i = get_global_id(0);
    
    // TODO: Workaround for error introduced in Nvidia 195-series drivers.  Should be able to
    // pass in an integer to this function without needing it to be an array.
    int offset = array_offset[0];

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
    float2 temp = MultComplex4(vab, vbc, vca, data_bip[i]);
    mock_data_bs[offset + 2*i] = temp.s0;
    mock_data_bs[offset + 2*i + 1] = temp.s1;
}
