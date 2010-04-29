// Multiply three complex numbers
// NOTE: This has been modified from the traditional form to take advantage of A.s1 always being zero.
float2 MultComplex3Special(float2 A, float2 B, float2 C)
{
    // The traditional approach:
    //  real = -1*a0*B.s1*C.s1 - a1*B.s0*C.s1 - a1*B.s1*C.s0 + a0*B.s0*C.s0;
    //  imag = -1*a1*B.s1*C.s1 + a0*B.s0*C.s1 + a0*B.s1*C.s0 + a1*B.s0*C.s0;
    
    // But we may explot A.s1 always being zero, thereby simplifying the math
    //  real = -1*A.s0*B.s1*C.s1 + A.s0*B.s0*C.s0;
    //  imag =    A.s0*B.s0*C.s1 + A.s0*B.s1*C.s0;
    // But there is a little more simplification that can eliminate two multiplications.
    // The tradeoff is that we require two more local variables:
    //  float k1 = A.s0 * C.s1;
    //  float k2 = A.s0 * C.s0;
    //  real = -1*k1*B.s1 + k2*B.s0;
    //  imag =    k1*B.s0 + k2*B.s1;
    // but it turns out the above method is no faster than doing all of the multiplications.
    
    float2 temp;
    temp.s0 = -1*A.s0*B.s1*C.s1 + A.s0*B.s0*C.s0;
    temp.s1 =    A.s0*B.s0*C.s1 + A.s0*B.s1*C.s0;
    return temp;
}

__kernel void visi(
    __global float * image,
    __global float2 * dft_x,
    __global float2 * dft_y,
    __global int * image_size,
    __global float * inv_flux,
    __global float2 * output)
{
    float2 visi;
    visi.s0 = 0;
    visi.s1 = 0;
    
    int image_width = image_size[0];
    
    float2 A;
    float2 B;
    float2 C;
    
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
            A.s1 = 0;
            
            visi += MultComplex3Special(A, B, C);
        }
    }
    
    // Write the result to the output array
    output[h] = visi * inv_flux[0];
}


