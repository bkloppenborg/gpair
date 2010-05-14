// Function definitions:
float2 MultComplex2(float2 A, float2 B);
float2 MultComplex3(float2 A, float2 B, float2 C);

// Multiply three complex numbers.
float2 MultComplex3(float2 A, float2 B, float2 C)
{
    float2 temp;
    //temp.s0 = A.s0*B.s0*C.s0 - A.s1*B.s1*C.s0 - A.s0*B.s1*C.s1 - A.s1*B.s0*C.s1;
    //temp.s1 = A.s1*B.s0*C.s0 + A.s0*B.s1*C.s0 + A.s0*B.s0*C.s1 - A.s1*B.s1*C.s1;
    
    temp = MultComplex2(A, B);
    temp = MultComplex2(temp, C);
    return temp;
}

// Multiply three complex numbers.
// NOTE: This function has been modified for when C.s1 = 0.
float2 MultComplex3Special(float2 A, float2 B, float2 C)
{
    float2 temp;
    //temp.s0 = A.s0*B.s0*C.s0 - A.s1*B.s1*C.s0 - A.s0*B.s1*C.s1 - A.s1*B.s0*C.s1;
    //temp.s1 = A.s1*B.s0*C.s0 + A.s0*B.s1*C.s0 + A.s0*B.s0*C.s1 - A.s1*B.s1*C.s1;
    
    temp.s0 = A.s0*B.s0*C.s0 - A.s1*B.s1*C.s0;
    temp.s1 = A.s1*B.s0*C.s0 + A.s0*B.s1*C.s0;    
    return temp;
}

// Multiply two complex numbers
float2 MultComplex2(float2 A, float2 B)
{
    // There is the obvious way to do this:
    float2 temp;
    temp.s0 = A.s0*B.s0 - A.s1*B.s1;
    temp.s1 = A.s0*B.s1 + A.s1*B.s0;  
    
    return temp;
    
/*    // We can trade off one multiplication for three additional additions*/
/*    float k1 = A.s0 * B.s0;*/
/*    float k2 = A.s1 * B.s1;*/
/*    float k3 = (A.s0 + A.s1) * (B.s0 + B.s1);*/
/*    */
/*    float2 temp;*/
/*    temp.s0 = k1 - k2;*/
/*    temp.s1 = k3 - k1 - k2;*/
/*    return temp;*/
}



__kernel void grad_bis(
    __global float * data,
    __global float * data_err,
    __global long4 * data_uvpnt,
    __global short4 * data_sign,
    __global float2 * data_phasor,
    __global float * mock,
    __global float2 * dft_x,
    __global float2 * dft_y,
    __global float2 * visi,
    __global float * flux_inv,
    __private int image_width,
    __private int nbis,
    __private int npow,
    __global float * data_gradient)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    //int lid = get_local_id(); // The thread's local ID.
    
    float2 vab;
    float2 vbc;
    float2 vca;
    float2 vabderr;
    float2 vbcderr;
    float2 vcaderr;
    float2 t3der;
    long4 uvpnt;
    short4 sign;
    
    // Promote the flux to a float2.
    float2 invflux;
    invflux.s0 = flux_inv[0];
    invflux.s1 = 0;
   
    float data_grad = 0;
    
    int k = 0;
    for(k = 0; k < nbis; k++)
    {
        uvpnt = data_uvpnt[k];
        vab = visi[uvpnt.s0];
        vbc = visi[uvpnt.s1];
        vca = visi[uvpnt.s2]; 
    
        // Compute the errors in the visibilities from the DFT matricies.
        vabderr = MultComplex2(dft_x[uvpnt.s0 * image_width + i], dft_y[uvpnt.s0 * image_width + j]);
        vbcderr = MultComplex2(dft_x[uvpnt.s1 * image_width + i], dft_y[uvpnt.s1 * image_width + j]);
        vcaderr = MultComplex2(dft_x[uvpnt.s2 * image_width + i], dft_y[uvpnt.s2 * image_width + j]);
        
        // Take the conjugate when necessary:
        sign = data_sign[k];
        vab.s1 *= sign.s0;
        vabderr.s1 *= sign.s0;
        vbc.s1 *= sign.s1;
        vbcderr.s1  *= sign.s1;       
        vca.s1 *= sign.s2;
        vcaderr.s1 *= sign.s2;
        
        // Now we compute T3, we have to do this in peices because of limited local memory on the GPU.
        // We are going to use D as a temporary variable.
        // t3der = ( (vabder - vab) * vbc * vca + vab * (vbcder - vbc) * vca + vab * vbc * (vcader - vca) ) * data_bisphasor[kk] * invflux ;
        // Step 1: (vabder - vab) * vbc * vca
        t3der = MultComplex3((vabderr - vab), vbc, vca);
        // Step 2: + vab * (vbcder - vbc) * vca
        t3der += MultComplex3(vab, (vbcderr - vbc), vca);
        // Step 3: + vab * vbc * (vcader - vca)
        t3der += MultComplex3(vab, vbc, (vcaderr - vca));
        // Step 4: (stuff) * data_bip[k] * fluxinv
        t3der = MultComplex3Special(t3der, data_phasor[k], invflux);

        // Error from the real part:         
        data_grad += 2 * data_err[npow + 2 * k] * data_err[npow + 2 * k] * ( mock[ npow + 2 * k] - data[npow + 2 * k] ) * t3der.s0;

        // Error from the imaginary part:
        data_grad += 2 * data_err[npow + 2 * k + 1] * data_err[npow + 2 * k + 1] * mock[ npow + 2 * k + 1] * t3der.s1;	   					
    }

    data_gradient[j * image_width + i] += data_grad;
}
