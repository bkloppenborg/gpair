__kernel void update_vis_fluxchange(
    __global float2 * visi_old,
    __global float2 * visi_new,
    __global float2 * dft_x,
    __global float2 * dft_y,
    __private int x_pos,
    __private int y_pos,
    __private int image_width,
    __private float flux_ratio)
{
    // TODO: This kernel is outdated, it needs to be fixed before being put into use again.

    // OpenCL doesn't natively support complex numbers, so we have to do the math out by hand.
    // the original equation is as follows:
    //    visi_new[i] = visi_old[i] * flux_ratio
    //        + (1.0 - flux_ratio) * dft_x[image_width * i +  x_pos] * dft_y[image_width * i +  y_pos];
    // The expansion of the top line is obvious, the bottom line is as follows:
    //  + (a0+a1*i)*(b0+b1*i)*(c0+c1*i)
    // real = -a0*b1*c1 - a1*b0*c1 - a1*b1*c0 + a0*b0*c0
    // imag = -a1*b1*c1 + a0*b0*c1 + a0*b1*c0 + a1*b0*c0
    // Noting that a1 = 0
    // real = a0*b0*c0 - a0*b1*c1 = A*c0 - B*c1
    // imag = a0*b0*c1 + a0*b1*c0 = A*c1 + B*c0
    
    int i = get_global_id(0);
    
    // Load a few items from global memory:
    float2 dft_xi = dft_x[image_width * i + x_pos];
    float2 dft_yi = dft_y[image_width * i + x_pos];
    float2 visi_o = visi_old[i];
    
    float A = (1.0 - flux_ratio) * dft_xi.s0;
    float B = (1.0 - flux_ratio) * dft_xi.s1;
    float c0 = dft_yi.s0;
    float c1 = dft_yi.s1;
    
    visi_new[i].s0 = visi_o.s0 * flux_ratio + (A*c0 - B*c1);
    visi_new[i].s1 = visi_o.s1 * flux_ratio + (A*c1 + B*c0);
    
}
