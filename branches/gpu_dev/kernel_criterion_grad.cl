// A kernel to compute the temporary gradient
__kernel void criterion_grad(
    __global float * data_gradient,
    __global float * entropy_gradient,
    __local float h_entropy,
    __local int * image_width,
    __global float * output)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    int k = image_width[0] * i + j;
    
    output[k] = data_gradient[k] - h_entropy * entropy_grad[k];
}

