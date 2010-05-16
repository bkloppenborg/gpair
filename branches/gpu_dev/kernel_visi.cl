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

#define PI      3.14159265358979323
#define RPMAS (3.14159265358979323/180.0)/3600000.0

__kernel void visi(
    __global float * image,
    __global float2 * dft_x,
    __global float2 * dft_y,
    __private int nuv,
    __global int * image_size,
    __global float * inv_flux,
    __global float2 * output,
    __local float2 * sA,
    __global float2 * uv_info,
    __global float * pixellation)
{
    float2 visi;
    visi.s0 = 0;
    visi.s1 = 0;
    
    //float2 A;
    float2 B;
    float2 C;
    
    float arg_x = 0;
    float arg_y = 0;
    
    int image_width = image_size[0];    
    int uv_pnt = get_global_id(0);
    int lsize_x;
    int lid = get_local_id(0);
    int i = 0;
    int j = 0;
    int m = 0;
    
    float2 uv = uv_info[uv_pnt];
  
    float arg = 2.0 * PI * RPMAS * pixellation[0];
 
    for(j=0; j < image_width; j++)
    {
		arg_y = -1 * arg * uv.s1 * (float) j;
        C.s0 = native_cos(arg_y);
        C.s1 = native_sin(arg_y);
        
        //C = dft_y[nuv * j + uv_pnt];
        
        lsize_x = get_local_size(0);
        
        for(i=0; i < image_width; i+= lsize_x)
        {
            if((i + lsize_x) > image_width)
                lsize_x = image_width - i;
                
            if((i + lid) < image_width)
            {
                sA[lid].s0 = image[image_width * j + (i+lid)];
                sA[lid].s1 = 0;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            
            for(m = 0; m < lsize_x; m++)
            {
                arg_x = arg * uv.s0 * (float) (i + m);
                B.s0 = native_cos(arg_x);
                B.s1 = native_sin(arg_x);
                
                //A = image[image_width * j + i];
                //B = dft_x[nuv * (i+m) + uv_pnt];

                visi += MultComplex3Special(sA[m], B, C);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
    // Write the result to the output array
    output[uv_pnt] = visi * inv_flux[0];
}


