// Multiply three complex numbers.
float2 MultComplex3(float2 A, float2 B, float2 C)
{
    float2 temp;
    temp.s0 = -A.s0*B.s1*C.s1 - A.s1*B.s0*C.s1 - A.s1*B.s1*C.s0 + A.s0*B.s0*C.s0;
    temp.s1 = -A.s1*B.s1*C.s1 + A.s0*B.s0*C.s1 + A.s0*B.s1*C.s0 + A.s1*B.s0*C.s0;

    return temp;
}

// Multiply two complex numbers
float2 MultComplex2(float2 A, float2 B)
{
    float a = A.s0 * B.s0;
    float b = A.s1 * B.s1;
    float c = (A.s0 + A.s1)*(B.s0 + B.s1);
    
    float2 temp;
    temp.s0 = a - b;
    temp.s1 = c - a - b;

    return temp;
}

float2 Square(float2 A)
{
    float2 temp;
    temp.s0 = A.s0 * A.s0 - A.s1 * A.s1;
    temp.s1 = A.s0 * A.s1;
    
    return temp;
}

__kernel void arr_normalize(
    __global float2 * data,
    __global float2 * data_err,
    __global long4 * data_uvpnt,
    __global short4 * data_sign,
    __global float2 * data_bip,
    __global float2 * mock,
    __global float2 * dft_x,
    __global float2 * dft_y,
    __global float2 * visi,
    __global float * flux,
    __private image_width,
    __private nbis,
    __private npow,
    __global float * data_gradient
    
    )
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    float invflux = flux[0];
    
    float2 vab;
    float2 vbc;
    float2 vca;
    float2 vabderr;
    float2 vbcderr;
    float2 vcaderr;
    float2 t3der;
    long4  loc;
    short4 sign;
 
    float data_grad = 0;
    float2 t3der = 0;
    
    int k = 0;
    for(k = 0; k < nbis; k++)
    {
        loc = data_uvpnt[k];
        sign = data_sign[k];

        vab = visi[loc.s0];
        vbc = visi[loc.s1];
        vca = visi[loc.s2];  
    
        // Compute the errors in the visibilities from the DFT matricies.
        // Note, we need to reuse variables as we've used more than 8 registers already.
        vabderr = MultComplex2(dft_x[loc.s0 * image_width + i], dft_y[loc.s0 * image_width + j]);

        vbcderr = MultComplex2(dft_x[loc.s1 * image_width + i], dft_y[loc.s1 * image_width + j]);
        
        vcaderr = MultComplex2(dft_x[loc.s2 * image_width + i], dft_y[loc.s2 * image_width + j]); 
        
        // Take the conjugate
        vab.s1 *= sign.s0;
        vabderr.s1 *= sign.s0;
        vbc.s1 *= sign.s1;
        vbcderr.s1 *= sign.s1;
        vca.s1 *= sign.s2;
        vcaderr.s1 *= sign.s2;
        
        // Now we compute T3, we have to do this in peices because of limited local memory on the GPU.
        // We are going to use D as a temporary variable.
        // t3der = ( (vabder - vab) * vbc * vca + vab * (vbcder - vbc) * vca + vab * vbc * (vcader - vca) ) * data_bisphasor[kk] * invflux ;
        // Step 1: (vabder - vab) * vbc * vca
        t3der = MultComplex3((vabder - vab), vbc, vca);
        // Step 2: + vab * (vbcder - vbc) * vca
        t3der += MultComplex3(vab, (vbcder - vbc), vca);
        // Step 3: + vab * vbc * (vcader - vca)
        t3der += MultComplex3(vab, vbc, (vcader - vca));
        // Step 4: (stuff) * data_bip[k] * fluxinv
        t3der = MultComplex2(A, data_bip[k]) * fluxinv;
        
        // Another nice Math trick after we rearrange 
        // data_grad += 2 * data_err[2 * kk] * data_err[2 * kk]  * ( mock[ npow + 2 * kk] - data[npow + 2 * kk] ) * t3der.s0;
        // data_grad += 2 * data_err[2 * kk + 1] * data_err[2 * kk + 1] * mock[ npow + 2 * kk + 1]  * t3der.s1;        
        
        
        A = 2 * Square(data_err[k]) * t3der;
        data_grad += A.s0 * mock[npow + k].s0 - A.s0 * data[npow + k].s0
        data_grad += A.s1 * mock[npow + k].s1;
         
			
    }

    data_gradient[ii + jj * image_width] = data_grad;
}
