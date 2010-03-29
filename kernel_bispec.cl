__kernel void compute_bispec(
    __global float2 * vis,
    __global float2 * data_bip,
    __global long * data_uvpnt,
    __global short * data_sign,
    __global float * mock_data_bs,
    __global const int array_offset)
{   
    int i = get_global_id(0);

    float2 vab = vis[data_uvpnt[3*i]];
    float2 vbc = vis[data_uvpnt[3*i + 1]];
    float2 vca = vis[data_uvpnt[3*i + 2]];
    
    vab.s1 *= data_sign[3*i];
    vbc.s1 *= data_sign[3*i + 1];
    vca.s1 *= data_sign[3*i + 2];
    
    // Now compute the triple amplitude and assign the real and imaginary
    // portions to the mock data array. 
    // We compute vab * vbc * vca * data_bip[i]
    // Full expansion (a + bi) == vab...
    // real = b*d*f*h - a*c*f*h - a*d*e*h - b*c*e*h - a*d*f*g - b*c*f*g - b*d*e*g + a*c*e*g
    // img = -1*a*d*f*h - b*c*f*h - b*d*e*h + a*c*e*h - b*d*f*g + a*c*f*g + a*d*e*g + b*c*e*g    
    // As a shortcut, we group and reduce the number of operations:
    // vab[0] = a
    // vab[1] = b
    // vbc[0] = c
    // vbc[1] = d
    // vca[0] = e
    // vca[1] = f
    // data_bip[i][0] = g
    // data_bip[i][1] = h
    
    float A = vab.s1 * vbc.s1;
    float B = vca.s1 * data_bip[i].s1;
    float C = vab.s0 * vbc.s0;
    float D = vab.s0 * vbc.s1;
    float E = vca.s0 * data_bip[i].s1;
    float F = vab.s1 * vbc.s0;
    float G = vca.s1 * data_bip[i].s0;
    float H = vca.s0 * data_bip[i].s0;
  
    mock_data_bs[array_offset + 2*i] = A*B - C*B - D*E - F*E - D*G - F*G - A*H + C*H;
    mock_data_bs[array_offset + 2*i + 1] = F*H + D*H + C*G - A*G + C*E - A*E - F*B - D*B;
}
