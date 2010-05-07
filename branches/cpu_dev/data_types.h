#ifndef DATA_TYPES_H
#define DATA_TYPES_H
#endif

#ifndef GETOIFITS_H
#include "getoifits.h"
#endif

// TODO: Segment the data information from the DFT and Mock data info.
typedef struct
{
    // Information about the data:
    int npow;
    int nbis;
    int nuv;
    
    float * data;
    float * data_err;
    float complex * data_phasor;
    oi_data * oifits_info;
    
    // Information about the image:
    int image_width;
    
    // DFT Tables
    float complex * dft_x;
    float complex * dft_y;
    
    // Storage locations for computed values:
    float complex * visi;
    float * mock;
} chi2_info;

typedef struct
{
    int * criterion_evals; 
    int * grad_evals; 
    
    float steplength_low; 
    float steplength_high; 
    float criterion_steplength_low; 
    float wolfe_product1; 
    float criterion_init;
    float * current_image; 
    float * temp_image;
    float * descent_direction; 
    float * temp_gradient; 
    float * data_gradient;
    float * entropy_gradient; 
    float complex* visi; 
    float * default_model; 
    float hyperparameter_entropy; 
    float * mock; 
} ls_params;

