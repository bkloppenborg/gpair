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
    __global float2 * data,     // Notice, this is actually float
    __global float2 * data_err, // Notice, this is actually float
    __global long4 * data_uvpnt,
    __global short4 * data_sign,
    __global float2 * data_phasor,
    __global float2 * mock,     // Notice, this is actually float
    __global float2 * dft_x,
    __global float2 * dft_y,
    __global float2 * visi,
    __global float * flux_inv,
    __private int nuv,
    __private int nbis,
    __private int npow,
    __private int image_width,
    __global float * data_gradient,
    __local long4 * sUVPoint,
    __local short4 * sSigns,
    __local float2 * sData,
    __local float2 * sDataErr,
    __local float2 * sMock,
    __local float2 * sPhasor)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = 0;
    int m = 0;
    int lid = get_local_id(0); // The thread's local ID.
    int lsize = get_local_size(0);
    
    float2 vab;
    float2 vbc;
    float2 vca;
    float2 vabderr;
    float2 vbcderr;
    float2 vcaderr;
    float2 t3der;
   
    float data_grad = 0; 

    for(k = 0; k < nbis; k += lsize)
    {
        if((k + lsize) > nbis)
            lsize = nbis - k;
            
        if((k + lid) < nbis)
        {
            sUVPoint[lid] = data_uvpnt[k + lid];
            sSigns[lid] = data_sign[k + lid];
            sData[lid] = data[npow + 2*(k + lid)];
            sDataErr[lid] = data_err[npow + 2*(k + lid)];
            sMock[lid] = mock[npow + 2*(k + lid)];
            sPhasor[lid] = data_phasor[k + lid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for(m = 0; m < lsize; m++)
        {
            vab = visi[sUVPoint[m].s0];
            vbc = visi[sUVPoint[m].s1];
            vca = visi[sUVPoint[m].s2]; 
            
            vabderr = MultComplex2(dft_x[sUVPoint[m].s0 + nuv * i], dft_y[sUVPoint[m].s0 + nuv * j]);
            vbcderr = MultComplex2(dft_x[sUVPoint[m].s1 + nuv * i], dft_y[sUVPoint[m].s1 + nuv * j]);
            vcaderr = MultComplex2(dft_x[sUVPoint[m].s2 + nuv * i], dft_y[sUVPoint[m].s2 + nuv * j]);
            
            vab.s1 *= sSigns[m].s0;
            vabderr.s1 *= sSigns[m].s0;
            vbc.s1 *= sSigns[m].s1;
            vbcderr.s1  *= sSigns[m].s1;       
            vca.s1 *= sSigns[m].s2;
            vcaderr.s1 *= sSigns[m].s2;      
            
            t3der = MultComplex3((vabderr - vab), vbc, vca);
            // Step 2: + vab * (vbcder - vbc) * vca
            t3der += MultComplex3(vab, (vbcderr - vbc), vca);
            // Step 3: + vab * vbc * (vcader - vca)
            t3der += MultComplex3(vab, vbc, (vcaderr - vca));
            // Step 4: (stuff) * data_bip[k] * fluxinv
            t3der = MultComplex3Special(t3der, sPhasor[m], (float2) (flux_inv[0], 0));
            
            // Now we compute the (straight) product of (err)^2 * t3der
            
            t3der = sDataErr[m] * sDataErr[m] * t3der;
            
            data_grad += t3der.s0 * (sMock[m].s0 - sData[m].s0);
            data_grad += t3der.s1 * sMock[m].s1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    data_gradient[j * image_width + i] += data_grad;
}
