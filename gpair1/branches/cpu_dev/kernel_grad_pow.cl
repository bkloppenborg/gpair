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
    __private int image_width,
    __private int npow,
    __global float * data_gradient)
{
    // Load indicies:
    int i = get_global_id(0);
    int j = get_global_id(1);

    // Setup counters and local variables.
    int k = 0;    
    float data_grad = 0;
    float2 temp;
    
    // Pull out variables from global memory:
    float invflux = inv_flux[0];

    // Iterate over the powerspectrum points, adding in their gradients.
    for(k = 0; k < npow; k++)
    {   
        // The original equation is as thus:
        // data_gradient[ii + jj * image_width] += 
        //    4. * data_err[ kk ] * data_err[ kk ] 
        //       * invflux *  ( mock[ kk ] - data[ kk ] ) 
        //       * creal(conj( visi[ kk ] ) 
        // * ( DFT_tablex[ image_width * kk +  ii ] * DFT_tabley[ image_width * kk +  jj ] - visi[ kk ] ));
        // The complex portion requires the real part of this expansion:
        // (A0 - %i*A1)*(B0 + %i*B1)*(C0 + %i*C1)
        
        temp = MultComplex2(dft_x[image_width * k + i], dft_y[image_width * k + j]) - visi[k];
        temp = MultComplex2(conj(visi[k]), temp);
        
        data_grad += 4.0 * data_err[k] * data_err[k] * invflux * (mock[k] - data[k]) * creal(temp);    
    }
    
    data_gradient[image_width * j + i] = data_grad;

}
