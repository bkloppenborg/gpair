// Multiply three complex numbers.
// NOTE: This function has been modified for the case where A.s1 is always zero
float2 MultComplex3Special(float2 A, float2 B, float2 C)
{
    float2 temp;
    //temp.s0 = -1*A.s0*B.s1*C.s1 - A.s1*B.s0*C.s1 - A.s1*B.s1*C.s0 + A.s0*B.s0*C.s0;
    //temp.s1 = -1*A.s1*B.s1*C.s1 + A.s0*B.s0*C.s1 + A.s0*B.s1*C.s0 + A.s1*B.s0*C.s0;
    
    float a = A.s0 * C.s1;
    float b = A.s0 * C.s0;
    
    temp.s0 = -1*B.s1*a + B.s0*b;
    temp.s1 =    B.s0*a + B.s1*b;

    return temp;
}


__kernel void visi(
    __global float * image,
    __global float2 * dft_x,
    __global float2 * dft_y,
    __global int * image_size,
    __global float2 * output,
    __local float2 * sA,
    __local float2 * sB,
    __local float2 * sC)
{
    float2 visi;
    visi.s0 = 0;
    visi.s1 = 0;
    
    int image_width = image_size[0];

    float2 A, B, C, temp;
    A.s1 = 0;
    
    int h = get_global_id(0);
    int i = 0;
    int j = 0;
    
    int offset = image_width * h;
 
    for(j=0; j < image_width; j++)
    {
        C = dft_y[offset +  j];
        
        for(i=0; i < image_width; i++)
        {
            B = dft_x[offset +  i];
            A.s0 = image[image_width * j + i];

            visi += MultComplex3Special(A, B, C);
        }
    }
    
    // Write the result to the output array
    output[h] = visi;
}


