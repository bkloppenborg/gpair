// Header file for gpair.c
#include <complex.h>

int read_oifits( );
void write_fits_image( float* image , int* status );
void image2vis(float* image, float complex* visi );
void update_vis_fluxchange(int x, int y, float flux_old, float flux_new, float complex* visi_old, float complex* visi_new); 
void update_vis_positionchange(int x_old, int y_old, int x_new, int y_new, float flux, float complex* visi_old, float complex* visi_new) ;
void update_vis_fluxpositionchange(int x_old, int y_old, int x_new, int y_new, float flux_old, float flux_new, float complex* visi_old, float complex* visi_new);
void vis2data( float complex* visi, float* mock );
float data2chi2( float* mock );
float compute_flux( float* image );
void compute_data_gradient(double complex* visi, double* mock, double* image, double* data_gradient);
float GullSkilling_entropy(float *image, float *default_model);
void GullSkilling_entropy_gradient(float *image, float *default_model, float* gradient);
float GullSkilling_entropy_diff( int x_old, int y_old, int x_new, int y_new , float old_flux, float new_flux, float *default_model);
float L2_entropy(float *image, float *default_model);
float L2_entropy_gradient(float *image, float *default_model);
float L2_diff( int x_old, int y_old, int x_new, int y_new, float *image, float *default_model );
void set_model(int modeltype, float modelwidth, float modelflux, float* default_model );



