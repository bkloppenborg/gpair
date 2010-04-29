__kernel void visi(
    __global float * image,
    __global float2 * dft_x,
    __global float2 * dft_y,
    __global int * image_size,
    __global float2 * output,
    __global float * invflux)
{
    float2 visi;
    visi.s0 = 0;
    visi.s1 = 0;
    
    int image_width = image_size[0];
    
    float a0, a1, b0, b1, c0, c1;
    
    int h = get_global_id(0);
    int i = 0;
    int j = 0;
    
    int offset = image_width * h;
 
    for(i=0; i < image_width; i++)
    {
        for(j=0; j < image_width; j++)
        {
            a0 = image[ i + image_width * j ];
            a1 = 0;
            b0 = dft_x[offset +  i].s0;
            b1 = dft_x[offset +  i].s1;
            c0 = dft_y[offset +  j].s0;
            c1 = dft_y[offset +  j].s1;
            
            visi.s0 += -1*a0*b1*c1 - a1*b0*c1 - a1*b1*c0 + a0*b0*c0;
            visi.s1 += -1*a1*b1*c1 + a0*b0*c1 + a0*b1*c0 + a1*b0*c0;
        }
    }
    
    // TODO: Normalize if (v0 > 0.) visi[h] /= v0;
    output[h].s0 = visi.s0 * invflux[0];
    output[h].s1 = visi.s1 * invflux[0];
}

