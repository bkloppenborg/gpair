// Function prototypes
float2 conj(float2 A);
float creal(float2 A);
float cimag(float2 A);
float2 MultComplex3(float2 A, float2 B, float2 C);
float2 MultComplex2(float2 A, float2 B);

float2 conj(float2 A)
{
    float2 temp;
    temp.s0 = A.s0;
    temp.s1 = -1 * A.s1;
    
    return temp;
}

float creal(float2 A)
{
    return A.s0;
}

float cimag(float2 A)
{
    return A.s1;
}

// Multiply three complex numbers.
float2 MultComplex3(float2 A, float2 B, float2 C)
{
    float2 temp;
    temp = MultComplex2(A, B);
    temp = MultComplex2(temp, C);
    return temp;
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


__kernel void grad_pow(
    __global float * data,
    __global float * data_err,
    __global float * mock,
    __global float2 * dft_x,
    __global float2 * dft_y,
    __global float2 * visi,
    __global float * inv_flux,
    __private int nuv,
    __private int npow,
    __private int image_width,
    __local float * sData,
    __local float * sDataErr,
    __local float * sMock,
    __local float2 * sVisi,
    __global float * data_gradient)
{
    // Load indicies:
    int i = get_global_id(0);
    int j = get_global_id(1);
    int lsize_x = get_local_size(0);
    int lsize = lsize_x * get_local_size(1);
    int lid = lsize_x * get_local_id(1) + get_local_id(0);

    // Setup counters and local variables.
    int k = 0;    
    int l = 0;
    float data_grad = 0;
    float2 temp;
    
    // Pull out variables from global memory:
    float invflux = inv_flux[0];

    // Iterate over the powerspectrum points, adding in their gradients.
    for(k = 0; k < npow; k += lsize)
    {   
        if((k + lsize) > npow)
            lsize = npow - k;
            
        if((i + lid) < npow)
        {
            sData[lid] = data[k + lid];
            sDataErr[lid] = data_err[k + lid];
            sMock[lid] = mock[k + lid];
            sVisi[lid] = visi[k + lid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // The original equation is as thus:
        // data_gradient[ii + jj * image_width] += 
        //    4. * data_err[ kk ] * data_err[ kk ] 
        //       * invflux *  ( mock[ kk ] - data[ kk ] ) 
        //       * creal(conj( visi[ kk ] ) 
        // * ( DFT_tablex[ image_width * kk +  ii ] * DFT_tabley[ image_width * kk +  jj ] - visi[ kk ] ));
        // The complex portion requires the real part of this expansion:
        // (A0 - %i*A1)*(B0 + %i*B1)*(C0 + %i*C1)
        
        for(l = 0; l < lsize; l++)
        {
            temp = MultComplex2(dft_x[(k + l) + i * nuv], dft_y[(k + l) + j * nuv]) - sVisi[l];
            temp = MultComplex2(conj(visi[k]), temp);
            
            data_grad += 4.0 * sDataErr[l] * sDataErr[l] * invflux * (sMock[l] - sData[l]) * creal(temp);   
        } 
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    data_gradient[image_width * j + i] = data_grad;

}
