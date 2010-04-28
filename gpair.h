// Header file for gpair.c
int read_oifits( );
void vis2data( );
void image2vis( );
float data2chi2( );
void write_fits_image( float* image, int* status );
void update_vis_fluxchange(int x, int y, float flux_old, float flux_new, double * visi_old, double * visi_new);
void update_vis_positionchange(int x_old, int y_old, int x_new, int y_new);
void update_vis_fluxpositionchange(int x_old, int y_old, int x_new, int y_new, float flux_old, float flux_new);
float flux( );
