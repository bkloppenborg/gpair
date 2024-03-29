// A kernel to compute the temporary gradient
__kernel void update_image(
    __global float * current_image,
    __global float * descent_direction,
    __private float steplength,
    __private float minvalue,
    __global int * image_width)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    int k = image_width[0] * i + j;
    
    current_image[k] += steplength * descent_direction[k];
    
    if (current_image[k] < minvalue)
        current_image[k] = minvalue;
}

