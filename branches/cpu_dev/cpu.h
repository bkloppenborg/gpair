#ifndef CPU_H
#define CPU_H
#endif

#include <complex.h>

#ifndef GETOIFITS_H
#include "getoifits.h"
#endif

float compute_flux(int image_width, float* image);

void compute_data_gradient(int image_width, int npow, int nbis, oi_data oifits_info, 
    float * data, float * data_err, float complex * data_phasor,
    float complex * DFT_tablex, float complex * DFT_tabley, 
    float complex * visi, float * mock, float * image, float * data_gradient);

float data2chi2(int npow, int nbis,
    float * data, float * data_err,
    float * mock);

float GullSkilling_entropy(int image_width, float * image, float * default_model);

void GullSkilling_entropy_gradient(int image_width, float * image, float * default_model, float * gradient);

float GullSkilling_entropy_diff(int image_width, 
    int x_old, int y_old, int x_new, int y_new , 
    float old_flux, float new_flux, 
    float * default_model);

void image2vis(int image_width, int nuv, 
    float * image, float complex * visi, 
    float complex * DFT_tablex, float complex * DFT_tabley);

float L2_entropy(int image_width, float * image, float * default_model);

float L2_entropy_gradient(int image_width, float * image, float * default_model);

float L2_diff(int image_width,
    int x_old, int y_old, int x_new, int y_new, 
    float * image, float * default_model);

void set_model(int image_width, float image_pixellation, 
    int modeltype, float modelwidth, float modelflux, float * default_model);

float square(float number);

void vis2data(int npow, int nbis, 
    oi_data oifits_info, float complex * data_phasor, 
    float complex * DFT_tablex, float complex * DFT_tabley,
    float complex * visi, float * mock);

//void update_vis_fluxchange(int x, int y, float flux_old, float flux_new, float complex* visi_old, float complex* visi_new); 

//void update_vis_positionchange(int x_old, int y_old, int x_new, int y_new, float flux, float complex* visi_old, float complex* visi_new) ;

//void update_vis_fluxpositionchange(int x_old, int y_old, int x_new, int y_new, float flux_old, float flux_new, float complex* visi_old, float complex* visi_new);

