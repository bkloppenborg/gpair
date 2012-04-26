// A kernel to compute the temporary gradient
__kernel void descent_dir(
    __global float * descent_direction,
    __global float * full_grad,
    __private float beta,
    __global int * image_width)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    int k = image_width[0] * i + j;
    
    descent_direction[k] = beta * descent_direction[k] - full_grad[k];
}
