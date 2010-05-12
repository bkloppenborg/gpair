// A kernel to compute the temporary gradient
__kernel void criterion_grad(
    __global float * data_gradient,
    __global float * entropy_grad,
    __private float h_entropy,
    __global int * image_width,
    __global float * full_grad_new)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    int k = image_width[0] * i + j;
    
    full_grad_new[k] = data_gradient[k] - h_entropy * entropy_grad[k];
}
