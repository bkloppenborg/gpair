#ifndef GPAIR_H
#define GPAIR_H
#endif

// Header file for gpair.c
#include <complex.h>

#ifndef DATA_TYPES_H
#include "data_types.h"
#endif

int read_oifits(char *filename);
void writefits(float * image , char * filename);
void read_fits_image(char* filename, float* array);
void init_data(int do_extrapolation);
float scalprod( float * array1, float * array2);
float linesearch_zoom( float steplength_low, float steplength_high, float criterion_steplength_low, float wolfe_product1,
		float criterion_init, int *criterion_evals, int *grad_evals, float *current_image, float *temp_image,
		float *descent_direction, float *temp_gradient, float *data_gradient,
		float *entropy_gradient, float complex* visi, float* default_model , float hyperparameter_entropy, float *mock, 
		chi2_info * data_info);



