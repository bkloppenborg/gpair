// A kernel to compute the temporary gradient
__kernel void criterion_step(
    __global float * current_image,
    __global float * descent_direction,
    __private float steplength,
    __global int * image_width,
    __global float * temp_image)
{
    int i = get_global_id(0);
/*    int j = get_global_id(1);*/
/*    */
/*    int k = image_width[0] * i + j;*/
    
/*    temp_image[k] = current_image[k] + steplength * descent_direction[k];*/
}
