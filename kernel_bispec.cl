/*// Multiply four complex numbers.*/
/*float2 MultComplex4(float2 A, float2 B, float2 C, float2 D)*/
/*{*/
/*    float2 temp;*/
/*    float a = A.s1 * B.s1;*/
/*    float b = C.s1 * D.s1;*/
/*    float c = A.s0 * B.s0;*/
/*    float d = A.s0 * B.s1;*/
/*    float e = C.s0 * D.s1;*/
/*    float f = A.s1 * B.s0;*/
/*    float g = C.s1 * D.s0;*/
/*    float h = C.s0 * D.s0;*/
/*  */
/*    temp.s0 = a*b - c*b - d*e - f*e - d*g - f*g - a*h + c*h;*/
/*    temp.s1 = f*h + d*h + c*g - a*g + c*e - a*e - f*b - d*b;*/

/*    return temp;*/
/*}*/

/*// The actual kernel function.*/
/*__kernel void compute_bispec(*/
/*    __global float2 * vis,*/
/*    __global float2 * data_bip,*/
/*    __global long * data_uvpnt,*/
/*    __global short * data_sign,*/
/*    __global float2 * mock_data_bs,*/
/*    __global int * array_offset)*/
/*{   */
/*    int i = get_global_id(0);*/
/*    */
/*    // TODO: Workaround for error introduced in Nvidia 195-series drivers.  Should be able to*/
/*    // pass in an integer to this function without needing it to be an array.*/
/*    int offset = array_offset[0];*/

/*    // Pull some data from global memory:*/
/*    long4 uv_pnt = data_uvpnt[i];*/
/*    float2 vab = vis[uv_pnt.s0];*/
/*    float2 vbc = vis[uv_pnt.s1];*/
/*    float2 vca = vis[uv_pnt.s2];*/
/*    */
/*    short4 sign = data_sign[i];*/
/*    vab.s1 *= sign.s0;*/
/*    vbc.s1 *= sign.s1;*/
/*    vca.s1 *= sign.s2;*/
/*    */
/*    // Get the biphasor:*/
/*    float2 databip = data_bip[i];    */
/*    */
/*    // Store the result.*/
/*    mock_data_bs[offset + i] = MultComplex4(vab, vbc, vca, data_bip[i]);*/
/*}*/

// Multiply four complex numbers.
float2 MultComplex4(float2 A, float2 B, float2 C, float2 D)
{
    float2 temp;
    float a = A.s1 * B.s1;
    float b = C.s1 * D.s1;
    float c = A.s0 * B.s0;
    float d = A.s0 * B.s1;
    float e = C.s0 * D.s1;
    float f = A.s1 * B.s0;
    float g = C.s1 * D.s0;
    float h = C.s0 * D.s0;
  
    temp.s0 = a*b - c*b - d*e - f*e - d*g - f*g - a*h + c*h;
    temp.s1 = f*h + d*h + c*g - a*g + c*e - a*e - f*b - d*b;

    return temp;
}

// The actual kernel function.
__kernel void compute_bispec(
    __global float2 * vis,
    __global float2 * data_bip,
    __global long * data_uvpnt,
    __global short * data_sign,
    __global float2 * mock_data_bs,
    __global int * array_offset)
{   
    int i = get_global_id(0);
    
    // TODO: Workaround for error introduced in Nvidia 195-series drivers.  Should be able to
    // pass in an integer to this function without needing it to be an array.
    int offset = array_offset[0];

    // Pull some data from global memory:
    float2 vab = vis[data_uvpnt[3*i]];
    float2 vbc = vis[data_uvpnt[3*i + 1]];
    float2 vca = vis[data_uvpnt[3*i + 2]];
    
    // Get the biphasor:
    float2 databip = data_bip[i];
    
    vab.s1 *= data_sign[3*i];
    vbc.s1 *= data_sign[3*i + 1];
    vca.s1 *= data_sign[3*i + 2];
    
    // TODO: Convert mock_data_bs over to a float2 array.
    float2 temp = MultComplex4(vab, vbc, vca, data_bip[i]);
    mock_data_bs[offset + i].s0 = temp.s1; //temp.s0;
    mock_data_bs[offset + i].s1 = temp.s1;
}


