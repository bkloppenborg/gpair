// Multiply three complex numbers.
float2 MultComplex3(float2 A, float2 B, float2 C)
{
    float2 temp = 0;
    temp.s0 = -A.s0*B.s1*C.s1 - A.s1*B.s0*C.s1 - A.s1*B.s1*C.s0 + A.s0*B.s0*C.s0;
    temp.s1 = -A.s1*B.s1*C.s1 + A.s0*B.s0*C.s1 + A.s0*B.s1*C.s0 + A.s1*B.s0*C.s0;

    return temp;
}

// Multiply two complex numbers
float2 MultComplex2(float2 A, float2 B)
{
    float2 temp = 0;
    temp.s0 = A.s1*B.s1 + A.s0*B.S0;
    temp.s1 = A.s0*B.s1 + A.s1*B.S0;  
    return temp;
}



__kernel void arr_normalize(
    __global float * data,
    __global float * data_err,
    __global long * data_uvpnt,
    __global short * data_sign,
    __global float2 * data_bip,
    __global float * mock,
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
    int lid = get_local_id(); // The thread's local ID.
    
    int ab_loc = 0;
    int bc_loc = 0;
    int ca_loc = 0;

    float invflux = flux[0];
    
    float2 vab;
    float2 vbc;
    float2 vca;
    float2 vabderr;
    float2 vbcderr;
    float2 vcaderr;
    float2 t3der;

    
    float data_grad = 0;
    float2 t3der = 0;
    
    int k = 0;
    for(k = 0; k < nbis; k++)
    {
        ab_loc = data_uvpnt[3*k];
        bc_loc = data_uvpnt[3*k + 1];
        ca_loc = data_uvpnt[3*k + 2];

        vab = visi[ab_loc];
        vbc = visi[bc_loc];
        vca = visi[ca_loc];  
    
        // Compute the errors in the visibilities from the DFT matricies.
        // Note, we need to reuse variables as we've used more than 8 registers already.
        vabderr = MultComplex2(dft_x[ab_loc * image_width + i], dft_y[ab_loc * image_width + j]);

        vbcderr = MultComplex2(dft_x[bc_loc * image_width + i], dft_y[bc_loc * image_width + j]);
        
        vcaderr = MultComplex2(dft_x[ca_loc * image_width + i], dft_y[ca_loc * image_width + j]); 
        
        // Take the conjugate
        vab.s1 *= data_sign[3*k];
        vabderr.s1 *= data_sign[3*k];
        vbc.s1 *= data_sign[3*k + 1];
        vbcderr.s1 *= data_sign[3*k + 1];
        vca.s1 *= data_sign[3*k + 2];
        vcaderr.s1 *= data_sign[3*k + 2];
        
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
         
        data_grad += 2 * data_err[2 * kk] * data_err[2 * kk]  * ( mock[ npow + 2 * kk] - data[npow + 2 * kk] ) * t3der.s0;
        data_grad += 2 * data_err[2 * kk + 1] * data_err[2 * kk + 1] * mock[ npow + 2 * kk + 1]  * t3der.s1;			
    }

    data_gradient[ii + jj * image_width] = data_grad;
}
