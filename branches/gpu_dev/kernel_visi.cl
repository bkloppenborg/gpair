// Compute the visibilities

#define PI      3.14159265358979323
#define RPMAS (3.14159265358979323/180.0)/3600000.0
float2 MultComplex3Special(float A, float B, float C);

// Multiply together three complex numbers.
// NOTE this function is optimized for A.imag = 0, mag(B) = mag(C) = 1
// And is done in polar coordinates
float2 MultComplex3Special(float magA, float argB, float argC)
{
    float2 temp;
    temp.s0 = magA * native_cos(argB + argC);
    temp.s1 = magA * native_sin(argB + argC);
    
    return temp;
}


__kernel void visi(
    __global float * image,
    __private int nuv,
    __global int * image_size,
    __global float * inv_flux,
    __global float2 * output,
    __local float * sA,
    __local float * sB,
    __global float2 * uv_info,
   __global float * pixellation)
{
    float2 visi;
    visi.s0 = 0;
    visi.s1 = 0;
    
    float arg_C;
    
    int image_width = image_size[0];    
    int uv_pnt = get_global_id(0);
    int lsize_x;
    int lid = get_local_id(0);
    int i = 0;
    int j = 0;
    int m = 0;
    
    // Load up the UV information
    float2 uv = uv_info[uv_pnt];
  
    float arg = 2.0 * PI * RPMAS * pixellation[0];
    
    // Iterate over every pixel (i,j) in the image, calculating their contributions to the visibility.
 
    for(j=0; j < image_width; j++)
    {       
        arg_C = -1 * arg * uv.s1 * (float) j;
        
        // Reload the lsize_x, just in case it was modified by a previous loop.
        lsize_x = get_local_size(0);
        
        // Now iterate over the x-direction in the image.
        // We are using shared memory and therefore move in blocks of lsize_x
        for(i=0; i < image_width; i+= lsize_x)
        {
            if((i + lsize_x) > image_width)
                lsize_x = image_width - i;
                
            if((i + lid) < image_width)
            {
                sA[lid] = image[image_width * j + (i + lid)];
                // Save a little computation time by removing one multiplication and addition from each iteration.
                sB[lid] = arg * (float) (i + lid);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            
            for(m = 0; m < lsize_x; m++)
            {
                visi += MultComplex3Special(sA[m], sB[m] * uv.s0, arg_C);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
    // Write the result to the output array
    output[uv_pnt] = visi * inv_flux[0];
}


