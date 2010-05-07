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
    
    float a0, a1;
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
            a0 = image[image_width * j + i];
            a1 = 0;
            //b0 = dft_x[offset +  i].s0;
            //b1 = dft_x[offset +  i].s1;
            //c0 = dft_y[offset +  j].s0;
            //c1 = dft_y[offset +  j].s1;
            
            visi.s0 += -1*a0*B.s1*C.s1 - a1*B.s0*C.s1 - a1*B.s1*C.s0 + a0*B.s0*C.s0;
            visi.s1 += -1*a1*B.s1*C.s1 + a0*B.s0*C.s1 + a0*B.s1*C.s0 + a1*B.s0*C.s0;
        }
    }
    
    // Write the result to the output array
    output[h] = visi * inv_flux[0];
}


