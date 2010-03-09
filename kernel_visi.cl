__kernel void visi(
    __global float * current_image,
    __global float2 * dft_x,
    __global float2 * dft_y,
    __local int model_image_size,
    __global float2 * output)
{
    __local float2 visi;
    visi[0] = 0;
    visi[1] = 0;
    
    float a0, a1, b0, b1, c0, c1;
    float A, B, C, D;
    
    int h = get_global_id(0);
    int i = 0;
    int j = 0;
    
    int offset = model_image_size * h;
    
    for(i=0;  i < model_image_size; i++)
    {
        for(j=0; j < model_image_size; j++)
        {
            a0 = current_image[model_image_size * j + i];
            b0 = dft_x[offset +  i][0];
            b1 = dft_x[offset +  i][1];
            c0 = dft_y[offset +  j][0];
            c1 = dft_y[offset +  j][1];    

            visi[0] += -1*a0*b0*b1*c0*c1;     // Real
            //visi[1] += 0;                 // Imaginary (zero by def of product of two complex numbers)
        }
    }
    
    // TODO: Normalize if (v0 > 0.) visi[uu] /= v0;
    output[h][0] = visi[0];
    output[h][1] = visi[1];
}
