// Function prototypes
float2 conj(float2 A);
float creal(float2 A);
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
    // real = A.s1*B.s1 + A.s0*B.S0;
    // imag = A.s0*B.s1 + A.s1*B.S0;  
    
    // We can trade off one multiplication for three additional additions
    float k1 = A.s0*(B.s0 + B.s1);
    float k2 = B.s1*(A.s0 + A.s1);
    float k3 = B.s1*(A.s1 - A.s0);
    
    float2 temp;
    temp.s0 = k1 - k2;
    temp.s1 = k1 + k3;
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
    // Load things 
    float invflux = inv_flux[0];

    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = 0;
    float2 t_mock;
    
    float2 temp;

    float data_grad = 0;

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
        
        temp = MultComplex3(conj(visi[k]), dft_x[image_width * k + i], dft_y[image_width * k + j]) - visi[k];
        
        data_grad += 4 * data_err[k] * data_err[k] * invflux * (mock[k] - data[k]) * creal(temp);    
    }
    
    data_gradient[image_width * j + i] = data_grad;

}
