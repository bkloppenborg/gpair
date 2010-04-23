// Multiply four complex numbers.
float2 MultComplex3(float2 A, float2 B, float2 C)
{
    float2 temp;

    float a = B.s1 * C.s1;
    float b = C.s0 * C.s1;
    float c = B.s1 * C.s0;
    float d = B.s0 * C.s0;

    temp.s0 = -1*A.s0*a - A.s1*b - A.s1*c + A.s0*d;
    temp.s1 = -1*A.s1*a + A.s0*b + A.s0*c + A.s1*d;

    return temp;
}

__kernel void visi(
    __global float * image,
    __global float2 * dft_x,
    __global float2 * dft_y,
    __global int * image_size,
    __global float2 * output)
{
    float2 visi, A, B;
    visi.s0 = 0;
    visi.s1 = 0;
    B.s0= 0;
    B.s1 = 0;
    
    int image_width = image_size[0];

    int h = get_global_id(0);
    int i = 0;
    int j = 0;
    
    int offset = image_width * h;
 
    for(j=0; j < image_width; j++)
    {
        A = dft_y[offset +  j];
        
        for(i=0; i < image_width; i++)
        {
            B.s0 = image[image_width * j + i];
            visi += MultComplex3(A, dft_x[offset +  i], B);
        }
    }
    
    // Write the result to the output array
    output[h] = visi;
}


