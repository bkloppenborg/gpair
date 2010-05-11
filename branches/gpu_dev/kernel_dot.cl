// A dot product kernel, need to run parallel reduce afterward
__kernel void dot(
    __private int row_width,
    __private int row_height,
    __global float * array1,
    __global float * array2,
    __global float * output)
{
    int i = get_global_id(0);
		
	float sum = 0;			
    for(j = 0; j < row_height; j++)
    {
        sum += array1[j] * array2[row_width * j + i];
    }
    
    output[i] = sum;
}
