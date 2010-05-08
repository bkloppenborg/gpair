#ifndef CPU_H
#define CPU_H
#endif

#include <complex.h>

#ifndef GETOIFITS_H
#include "getoifits.h"
#endif

#ifndef DATA_TYPES_H
#include "data_types.h"
#endif

float compute_flux(int image_width, float* image);

void compute_data_gradient(chi2_info * data_info, float * image, float * data_gradient);

//void conjugate_gradient(chi2_info * data_info, int gradient_method, ls_params * params,
//    float * current_image, float * default_model, 
//    int ndata, int iteration);

float data2chi2(int npow, int nbis,
    float * data, float * data_err,
    float * mock);

float GullSkilling_entropy(int image_width, float * image, float * default_model);

void GullSkilling_entropy_gradient(int image_width, float * image, float * default_model, float * gradient);

float GullSkilling_entropy_diff(int image_width, 
    int x_old, int y_old, int x_new, int y_new , 
    float old_flux, float new_flux, 
    float * default_model);

float image2chi2(chi2_info * info, float * image);

void image2vis(int image_width, int nuv, 
    float * image, float complex * visi, 
    float complex * DFT_tablex, float complex * DFT_tabley);

float L2_entropy(int image_width, float * image, float * default_model);

float L2_entropy_gradient(int image_width, float * image, float * default_model);

float L2_diff(int image_width,
    int x_old, int y_old, int x_new, int y_new, 
    float * image, float * default_model);
    
//float linesearch_zoom(chi2_info * data_info, ls_zoom * linesearch_params);

float scalprod(int array_size, float * array1, float * array2);

void set_model(int image_width, float image_pixellation, 
    int modeltype, float modelwidth, float modelflux, float * default_model);

float square(float number);

void vis2data(int npow, int nbis, 
    oi_data * data_info, float complex * data_phasor, 
    float complex * DFT_tablex, float complex * DFT_tabley,
    float complex * visi, float * mock);

//void update_vis_fluxchange(int x, int y, float flux_old, float flux_new, float complex* visi_old, float complex* visi_new); 

//void update_vis_positionchange(int x_old, int y_old, int x_new, int y_new, float flux, float complex* visi_old, float complex* visi_new) ;

//void update_vis_fluxpositionchange(int x_old, int y_old, int x_new, int y_new, float flux_old, float flux_new, float complex* visi_old, float complex* visi_new);

