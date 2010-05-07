#include "cpu.h"


float compute_flux(int image_width, float* image)
{
    register int ii;
    float total=0.;
    for(ii=0; ii < image_width * image_width; ii++)
        total += image[ii];
    return total;
}

void compute_data_gradient(int image_width, int npow, int nbis, oi_data oifits_info, 
    float * data, float * data_err, float complex * data_phasor,
    float complex * DFT_tablex, float complex * DFT_tabley, 
    float complex * visi, float * mock, float * image, float * data_gradient) // need to call vis2data before this
{

    register int ii, jj, kk;
    double complex vab, vbc, vca, vabder, vbcder, vcader, t3der;

    double flux = 0.; // if the flux has already been computed, we could use its value 
    for(ii = 0 ; ii < image_width * image_width ; ii++)
        flux +=  image[ ii ];

    double invflux = 1. / flux;

    for(ii=0; ii < image_width; ii++)
    {
        for(jj=0; jj < image_width; jj++)
        {
            data_gradient[ii + jj * image_width] = 0.;

            // Add gradient of chi2v2
            for(kk = 0 ; kk < npow; kk++)
            {
                data_gradient[ii + jj * image_width] += 4.0 * data_err[ kk ] * data_err[ kk ] * invflux 
                *  ( mock[ kk ] - data[ kk ] ) 
                * creal( conj( visi[ kk ] ) *  ( DFT_tablex[ image_width * kk +  ii ] * DFT_tabley[ image_width * kk +  jj ] - visi[ kk ] ) );
            }

            // Add gradient of chi2bs
            for(kk = 0 ; kk < nbis; kk++)
            {
                vab = visi[oifits_info.bsref[kk].ab.uvpnt];
                vbc = visi[oifits_info.bsref[kk].bc.uvpnt];
                vca = visi[oifits_info.bsref[kk].ca.uvpnt];

                vabder =  DFT_tablex[ oifits_info.bsref[kk].ab.uvpnt * image_width + ii  ] * DFT_tabley[ oifits_info.bsref[kk].ab.uvpnt * image_width + jj  ]  ; 
                vbcder =  DFT_tablex[ oifits_info.bsref[kk].bc.uvpnt * image_width + ii  ] * DFT_tabley[ oifits_info.bsref[kk].bc.uvpnt * image_width + jj  ]  ;
                vcader =  DFT_tablex[ oifits_info.bsref[kk].ca.uvpnt * image_width + ii  ] * DFT_tabley[ oifits_info.bsref[kk].ca.uvpnt * image_width + jj  ]  ;

                if(oifits_info.bsref[kk].ab.sign < 0) { vab = conj(vab);} 
                if(oifits_info.bsref[kk].bc.sign < 0) { vbc = conj(vbc);}
                if(oifits_info.bsref[kk].ca.sign < 0) { vca = conj(vca);}
                if(oifits_info.bsref[kk].ab.sign < 0) { vabder = conj(vabder);} 
                if(oifits_info.bsref[kk].bc.sign < 0) { vbcder = conj(vbcder);}
                if(oifits_info.bsref[kk].ca.sign < 0) { vabder = conj(vcader);}

                t3der = ( (vabder - vab) * vbc * vca + vab * (vbcder - vbc) * vca + vab * vbc * (vcader - vca) ) * data_phasor[kk] * invflux ;

                // gradient from real part
                data_gradient[ii + jj * image_width] += 2. * data_err[npow + 2 * kk] * data_err[npow + 2 * kk]  * ( mock[ npow + 2 * kk] - data[npow + 2 * kk] ); // * creal( t3der );  

                // gradient from imaginary part
                data_gradient[ii + jj * image_width] += 2. * data_err[npow + 2 * kk + 1] * data_err[npow + 2 * kk + 1] * mock[ npow + 2 * kk + 1]; //  * cimag( t3der );			
            }
        }
    }
}	

float data2chi2(int npow, int nbis,
    float * data, float * data_err,
    float * mock)
{
    float chi2 = 0.;
    register int ii = 0;  
    for(ii=0; ii< npow + 2 * nbis; ii++)
    {
        chi2 += square( ( mock[ii] - data[ii] ) * data_err[ii] ) ;
    }

    return chi2;
}

float GullSkilling_entropy(int image_width, float * image, float * default_model)
{
    register int ii;
    float S = 0.;
    
    for(ii=0 ; ii < image_width * image_width; ii++)
    {
        if ((image[ii] > 0.) && (default_model[ii] > 0.))
        {
            S += image[ii] - default_model[ii] - image[ii] * log( image[ii] / default_model[ii] );
        }   
        else 
            S += - default_model[ii];
    }
    return S;
}

void GullSkilling_entropy_gradient(int image_width, float * image, float * default_model, float * gradient)
{
    register int ii;
    for(ii=0 ; ii < image_width * image_width; ii++)
    {
        if((image[ii] > 0.) && (default_model[ii] > 0.))
        {
            gradient[ ii ] = - log( image[ii] / default_model[ii] );
        }
    }
}

float GullSkilling_entropy_diff(int image_width, 
    int x_old, int y_old, int x_new, int y_new, 
    float old_flux, float new_flux, 
    float * default_model)
{
    int position_old = x_old + y_old * image_width ;
    int position_new = x_new + y_new * image_width;
    float S_old, S_new;
    
    if( ( old_flux > 0.) && (default_model[position_old] > 0.) )
        S_old = old_flux - default_model[ position_old ] - old_flux * log( old_flux / default_model[ position_old ] );
    else 
        S_old = - default_model[ position_old ];

    if( ( new_flux > 0.) && (default_model[position_new] > 0.) )
        S_new = new_flux - default_model[ position_new ] - new_flux * log( new_flux / default_model[ position_new ] );
    else 
        S_new = - default_model[ position_new ];

    return S_new - S_old;
}


// A helper function to call necessary functions to compute the chi2.
float image2chi2(int npow, int nbis, int nuv, int image_width, 
    float complex * DFT_tablex, float complex * DFT_tabley,
    float * data, float * data_err, float complex * data_phasor, oi_data oifits_info,
    float * current_image,
    float complex * visi, float * mock)
{   
    float chi2 = 0;
    // Compute the visibilities, data, and chi2.
    image2vis(image_width, nuv, current_image, visi, DFT_tablex, DFT_tabley);
    vis2data(npow, nbis, oifits_info, data_phasor, DFT_tablex, DFT_tabley, visi, mock);
    chi2 = data2chi2(npow, nbis, data, data_err, mock);
    
    return chi2;
}

void image2vis(int image_width, int nuv, 
    float * image, float complex * visi, 
    float complex * DFT_tablex, float complex * DFT_tabley) // DFT implementation
{	 
    register int ii = 0, jj = 0, uu = 0;
    float zeroflux = 0.; // zeroflux 

    for(ii=0 ; ii < image_width * image_width ; ii++) 
    zeroflux += image[ii];

    for(uu=0 ; uu < nuv; uu++)
    {
        visi[uu] = 0.0 + I * 0.0;
        for(ii=0; ii < image_width; ii++)
        {
            for(jj=0; jj < image_width; jj++)
            {
                visi[uu] += image[ ii + image_width * jj ] *  DFT_tablex[ image_width * uu +  ii] * DFT_tabley[ image_width * uu +  jj];
            }
        }
        if (zeroflux > 0.)
            visi[uu] /= zeroflux;
    }
}

/*void update_vis_fluxchange(int x, int y, float flux_old, float flux_new, float complex* visi_old, float complex* visi_new) */
/*// finite difference routine giving visi when changing the flux of an element in (x,y)*/
/*{	*/
/*  // Note : two effects, one due to the flux change in the pixel, the other to the change in total flux*/
/*  // the total flux should be updated outside this loop*/
/*  register int uu;*/
/*  float flux_ratio =  flux_old / flux_new;*/
/*  for(uu=0 ; uu < nuv; uu++)*/
/*      visi_new[uu] = visi_old[uu] *  flux_ratio + ( 1.0 - flux_ratio ) *  DFT_tablex[ image_width * uu +  x] * DFT_tabley[ image_width * uu +  y] ;*/
/*}*/

/*void update_vis_positionchange(int x_old, int y_old, int x_new, int y_new, float flux, float complex* visi_old, float complex* visi_new) */
/*// finite difference routine giving visi when moving one element from (x_old,y_old) to (x_new, y_new)*/
/*{ // Note : no change in total flux in this case*/
/*  register int uu;*/
/*  for(uu=0 ; uu < nuv; uu++)*/
/*      visi_new[uu] = visi_old[uu] */
/*	+ flux  * ( DFT_tablex[ image_width * uu +  x_new] * DFT_tabley[ image_width * uu +  y_new] */
/*		    -  DFT_tablex[ image_width * uu +  x_old] * DFT_tabley[ image_width * uu +  y_old]) ;*/
/*}*/


/*// TBD -- Check the expression used in this function*/
/*void update_vis_fluxpositionchange(int x_old, int y_old, int x_new, int y_new, float flux_old, float flux_new, float complex* visi_old, float complex* visi_new)*/
/*{*/
/*  register int uu;*/
/*  for(uu=0 ; uu < nuv; uu++)*/
/*    visi_new[uu] =  visi_old[uu] * flux_old / flux_new */
/*      +  DFT_tablex[ image_width * uu +  x_new ] * DFT_tabley[ image_width * uu +  y_new ] */
/*      -  flux_old / flux_new * DFT_tablex[ image_width * uu +  x_old ] * DFT_tabley[ image_width * uu +  y_old ] ;*/
/*}*/

float L2_entropy(int image_width, float *image, float *default_model)
{
    register int ii;
    float S = 0.;
    
    for(ii=0 ; ii < image_width * image_width; ii++)
    {
        if( default_model[ii] > 0.)
        {
            S += - image[ii] * image[ii]  / ( 2. * default_model[ii] );
        }   
    }
    return S;
}

float L2_entropy_gradient(int image_width, float * image, float * default_model)
{
    register int ii;
    float S = 0.;
    for(ii=0 ; ii < image_width * image_width; ii++)
    {
        if( default_model[ii] > 0.)
        {
            S += - image[ii] / default_model[ii] ;
        }   
    }
    return S;
}

float L2_diff(int image_width,
    int x_old, int y_old, int x_new, int y_new, 
    float * image, float * default_model )
{
    int position_old = x_old + y_old * image_width ;
    int position_new = x_new + y_new * image_width;
    float S_old = 0., S_new = 0.;
    if( default_model[ position_old ] > 0. )
        S_old = - image[ position_old ] * image[ position_old ] / ( 2. * default_model[ position_old ] );

    if( default_model[ position_new ] > 0. )
        S_new = - image[ position_new ] * image[ position_new ] / ( 2. * default_model[ position_new ] );

    return S_new - S_old;
}

// Prior image
void set_model(int image_width, float image_pixellation, 
    int modeltype, float modelwidth, float modelflux, float * default_model)
{
    float flux = 0.0;
    int i, j;

    switch (modeltype)
    {
        case 0:
        {
            // Flat prior
            printf("Flat prior, total flux: %8.3f\n", modelflux);
            float norm = ((float) image_width * (float) image_width);
            for (i = 0; i < image_width * image_width; i++)
                default_model[ i ] = modelflux / norm;

            break;
        }

        case 1:
        {
            // Centered Dirac (negligible flux around)
            printf("Dirac, flux: %8.3f\n", modelflux);
            for (i = 0; i < image_width * image_width ; i++)
                default_model[ i ] = 1e-8;
                
            default_model[ (image_width * (image_width + 1)) / 2 ] = modelflux;

            break;
        }

        case 2:
        {
            // Centered Uniform disk
            printf("Uniform disk, Radius:%f mas, flux:%f\n", modelwidth, modelflux);
            flux = 0.0;
            float rsq;
            for (i = 0; i < image_width; i++)
            {
                for (j = 0; j < image_width; j++)
                {
                    rsq = square((float) (image_width / 2 - i)) + square((float) (image_width / 2 - j));

                    if (sqrt(rsq) <= (modelwidth / image_pixellation))
                        default_model[ j * image_width + i ] = 1.;
                    else
                        default_model[ j * image_width + i ] = 1e-8;
                        
                    flux += default_model[ j * image_width + i ];
                }
            }

            for (i = 0; i < image_width * image_width; i++)
                default_model[ i ] *= modelflux / flux;

            break;
        }

        case 3:
        {
            // Centered Gaussian
            float sigma = modelwidth / (2. * sqrt(2. * log(2.)));
            printf("Gaussian, FWHM:%f mas, sigma:%f mas, flux:%f\n", modelwidth, sigma, modelflux);
            flux = 0.0;
            
            for (i = 0; i < image_width; i++)
            {
                for (j = 0; j < image_width; j++)
                {
                    default_model[ j * image_width + i ] = exp(-(square((float) image_width / 2 - i) + square((float) image_width / 2 - j)) / (2. * square(sigma
												                      / image_pixellation)));
                    // Fix problem with support at zero
                    //if ( default_model[ j*image_width+i ]<1e-8 ) default_model[ j*image_width+i ]=1e-8;
                    flux += default_model[ j * image_width + i ];
                }
            }
            for (i = 0; i < image_width * image_width; i++)
                default_model[ i ] *= modelflux / flux;
                
            break;
        }

        case 4:
        {
            //Centered Lorentzian
            float sigma = modelwidth;
            printf("Lorentzian, FWHM:%f mas, sigma:%f mas, flux:%f\n", modelwidth, sigma, modelflux);
            flux = 0.0;
            for (i = 0; i < image_width; i++)
            {
                for (j = 0; j < image_width; j++)
                {
                    default_model[ j * image_width + i ] = sigma / (square(sigma) + square((float) image_width / 2 - i) + square((float) image_width / 2 - j));
                    flux += default_model[ j * image_width + i ];
                }
            }
            
            for (i = 0; i < image_width * image_width; i++)
                default_model[ i ] *= modelflux / flux;

            break;
        }
    }
}

// Returns the square of a (floating point) number
float square( float number )
{
    return number*number;
}

// Given the visibilities, compute the resulting data.
void vis2data(int npow, int nbis, 
    oi_data oifits_info, float complex * data_phasor, 
    float complex * DFT_tablex, float complex * DFT_tabley,
    float complex * visi, float * mock)
{
    int ii = 0;
    float complex vab = 0;
    float complex vbc = 0;
    float complex vca = 0;
    float complex t3 = 0;

    for( ii = 0; ii< npow; ii++)
    {
        mock[ ii ] = square ( cabs( visi[ii] ) );
    }

    for( ii = 0; ii< nbis; ii++)
    {
        vab = visi[ oifits_info.bsref[ii].ab.uvpnt ];
        vbc = visi[ oifits_info.bsref[ii].bc.uvpnt ];
        vca = visi[ oifits_info.bsref[ii].ca.uvpnt ];	
        if( oifits_info.bsref[ii].ab.sign < 0) 
            vab = conj(vab);
        if( oifits_info.bsref[ii].bc.sign < 0) 
            vbc = conj(vbc);
        if( oifits_info.bsref[ii].ca.sign < 0) 
            vca = conj(vca);
            
        t3 =  ( vab * vbc * vca ) * data_phasor[ii] ;   
        mock[ npow + 2 * ii ] = creal(t3) ;
        mock[ npow + 2 * ii + 1] = cimag(t3) ;
    } 
    
    // Uncomment to see the mock data array.
/*    int count = npow + 2 * nbis;*/
/*    for(ii = 0; ii < count; ii++)     */
/*        printf("%i %f \n", ii, mock[ii]);*/
/*        */
/*    printf("\n");*/

}


