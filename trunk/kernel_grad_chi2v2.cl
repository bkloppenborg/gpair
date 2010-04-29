__kernel void arr_normalize(
    __global float2 * data,
    __global float2 * data_err,
    __global float2 * mock,
    __global float2 * dft_x,
    __global float2 * dft_y,
    __global float2 * visi,
    __global float * flux,
    __private image_width,
    __private npow,
    __global float * data_gradient)
{
    // Load things 
    float invflux = flux[0];

    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = 0;
    
    float2 temp;

    float data_grad = 0;

    for(k = 0; k < npow; k++)
    {
        // Pull stuff from memory.  This should all be fast and coalesced.
        t_mock = mock[k];
        t_data = data[k];
        t_data_err = data_err[k];
        A = vis[k];
        B = dft_x[image_width * k + i];
        C = dft_y[image_width * k + j];
        
        // The original equation is as thus:
        // data_gradient[ii + jj * image_width] += 
        //    4. * data_err[ kk ] * data_err[ kk ] 
        //       * invflux *  ( mock[ kk ] - data[ kk ] ) 
        //       * creal(conj( visi[ kk ] ) 
        // * ( DFT_tablex[ image_width * kk +  ii ] * DFT_tabley[ image_width * kk +  jj ] - visi[ kk ] ));
        // The complex portion requires the real part of this expansion:
        // (A0 - %i*A1)*(B0 + %i*B1)*(C0 + %i*C1)
        
        temp = 
        
        data_grad += 4 * t_data_err * t_data_err * invflux * (t_mock.s0 - t_data.s0) *
            * (-A.s0*B.s1*C.s1 + A.s1*B.s0*C.s1 + A.s1*B.s1*C.s0 + A.s0*B.s0*C.s0 - A.s0)
        
    }
    
    data_gradient[i + j * image_width] = data_grad;

}
