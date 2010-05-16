// A kernel to compute the elements of the Gull Skilling entropy.
// NOTE: Need to run parallel reduction on this after the computation has completed.
__kernel void entropy_gs_grad(
    __global int * image_width,
    __global float * image,
    __global float * default_model,
    __global float * output)
{
    // Get our location in the image
    int i = get_global_id(0);
    int j = get_global_id(1);
		
    // The location in the input data array		
    int k = image_width[0] * i + j;
    
    float t_image = image[k];
    float t_model = default_model[k];
    float dS = output[k];
    
    if ((t_image > 0.) && (t_model > 0.))
        dS = -1 * log(t_image / t_model);
        
    output[k] = dS;
}
