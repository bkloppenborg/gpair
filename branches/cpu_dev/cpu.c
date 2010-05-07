#include "cpu.h"

void conjugate_gradient(chi2_info * data_info, int gradient_method, float * current_image, float * default_model, 
    int ndata, int iteration)

{
	// Gradient_method: 
	// 0: steepest descent, 1: CG Fletcher-Reeves , 2: CG Polak-Ribiere, 3: CG Hestenes Stiefel

    // TODO: Convert the functions to use this information directly
    int image_width = (*data_info).image_width;
    float complex * visi = (*data_info).visi;
    float * mock = (*data_info).mock;
    int ii = 0;
    
    // Init additional variables:
	double chi2, entropy, hyperparameter_entropy = 0.;
	double criterion;

	// Initialize gradients
	float * data_gradient = malloc(image_width * image_width * sizeof(float));
	float * entropy_gradient = malloc(image_width * image_width * sizeof(float));
	float * full_gradient = malloc(image_width * image_width * sizeof(float));
	float * full_gradient_new = malloc(image_width * image_width * sizeof(float));
	float * temp_gradient = malloc(image_width * image_width * sizeof(float));

	// Init descent direction
	float* descent_direction = malloc(image_width * image_width * sizeof(float));
	memset(descent_direction, 0, image_width * image_width * sizeof(float));

	// Line search
	float* temp_image = malloc(image_width * image_width * sizeof(float));
	float steplength, steplength_old, steplength_max, selected_steplength = 0.;
	int criterion_evals = 0;
	int grad_evals = 0;
	float beta = 0.0;
	float minvalue = 1e-8;
	float criterion_init, criterion_old;
	float wolfe_param1 = 1e-4, wolfe_param2 = 0.1;
	float wolfe_product1 = 0.0, wolfe_product2 = 0.0;
	int linesearch_iteration = 0;

	//
	// Compute the criterion
	//
	chi2 = image2chi2(data_info, current_image);
	entropy = GullSkilling_entropy(image_width, current_image, default_model);
	criterion = chi2 - hyperparameter_entropy * entropy;
	criterion_evals++;

	printf(	"Grad evals: %d J evals: %d Selected coeff %e Beta %e, J = %f, chi2r = %f chi2 = %lf alpha*entropy = %e entropy = %e \n",
			grad_evals, criterion_evals, selected_steplength, beta, criterion, chi2 / (float) ndata, chi2,
			hyperparameter_entropy * entropy, entropy );

    // TODO: Pull FITS writing routines out of gpair.c
	//writefits(current_image, "!reconst.fits");

	//
	// Compute full gradient (data + entropy)
	//

	compute_data_gradient(data_info, current_image, data_gradient);
	GullSkilling_entropy_gradient(image_width, current_image, default_model, entropy_gradient);
	
	for (ii = 0; ii < image_width * image_width; ii++)
		full_gradient_new[ii] = data_gradient[ii] - hyperparameter_entropy * entropy_gradient[ii];
	
	grad_evals++;

	// Compute the modifier of the gradient direction depending on the method
	if ((iteration == 0) || (gradient_method == 0))
	{
		beta = 0.; // steepest descent
		//
		// Compute descent direction
		//
		for (ii = 0; ii < image_width * image_width; ii++)
		  descent_direction[ii] = - full_gradient_new[ii];
	}
	else
	{
		if (gradient_method == 1) // CG
			beta = scalprod(image_width * image_width, full_gradient_new, full_gradient_new) / scalprod(image_width * image_width, full_gradient, full_gradient); // FR
		if (gradient_method == 2)
			beta = (scalprod(image_width * image_width, full_gradient_new, full_gradient_new) - scalprod(image_width * image_width, full_gradient_new, full_gradient)) // PR
					/ scalprod(image_width * image_width, full_gradient, full_gradient);
        if (gradient_method == 3) // HS
			beta = (scalprod(image_width * image_width, full_gradient_new, full_gradient_new) - scalprod(image_width * image_width, full_gradient_new, full_gradient))
					/ (scalprod(image_width * image_width, descent_direction, full_gradient_new) - scalprod(image_width * image_width, descent_direction, full_gradient));
        if(  fabs( scalprod(image_width * image_width, full_gradient, full_gradient_new) ) /  scalprod(image_width * image_width, full_gradient_new , full_gradient_new) > 1.0 )
            beta = 0.;

		//
		// Compute descent direction
		//
		for (ii = 0; ii < image_width * image_width; ii++)
		  descent_direction[ii] = beta * descent_direction[ii] - full_gradient_new[ii];
		
		// Some tests on descent direction
/*            printf("Angle descent direction/gradient %lf \t Descent direction / previous descent direction : %lf \n",*/
/*			       acos (- scalprod(descent_direction, full_gradient_new)*/
/*				/ sqrt( scalprod(full_gradient_new , full_gradient_new) * scalprod(descent_direction, descent_direction) )) / PI * 180.,*/
/*			       fabs( scalprod(full_gradient, full_gradient_new) ) /  scalprod(full_gradient_new , full_gradient_new)   );*/
	  }

	//      writefits(descent_direction, "!gradient.fits");


	//
	// Line search algorithm begins here
	//

	// Compute quantity for Wolfe condition 1
	wolfe_product1 = scalprod(image_width, descent_direction, full_gradient_new);

	// Initialize variables for line search
	selected_steplength = 0.;
	steplength = 1.;
	steplength_old = 0.;
	steplength_max = 100.; // use a clever scheme here
	criterion_init = criterion;
	criterion_old = criterion;
	linesearch_iteration = 1;

	while ( 1 )
	{

		//
		// Evaluate criterion(steplength)
		//

		//  Step 1: compute the temporary image: I1 = I0 - coeff * descent direction
		for (ii = 0; ii < image_width * image_width; ii++)
		{
			temp_image[ii] = current_image[ii] + steplength * descent_direction[ii];
			if (temp_image[ii] < minvalue)
				temp_image[ii] = minvalue;
		}

		// Step 2: Compute criterion(I1)
		chi2 = image2chi2(data_info, temp_image);
		entropy = GullSkilling_entropy(image_width, temp_image, default_model);
		criterion = chi2 - hyperparameter_entropy * entropy;
		criterion_evals++;

		if ((criterion > (criterion_init + wolfe_param1 * steplength * wolfe_product1) ) || ((criterion
				>= criterion_old) && (linesearch_iteration > 1)))
		{
            ls_params params;
            params.criterion_evals = &criterion_evals;
            params.grad_evals = &grad_evals;
            params.steplength_low  =  steplength_old;
            params.steplength_high  =  steplength;
            params.criterion_steplength_low  =  criterion_old;
            params.wolfe_product1  =  wolfe_product1;
            params.criterion_init  =  criterion_init;
            params.current_image  =  current_image;
            params.temp_image  =  temp_image;
            params.descent_direction  = descent_direction;
            params.temp_gradient  = temp_gradient;
            params.data_gradient  = data_gradient;
            params.entropy_gradient  = entropy_gradient;
            params.visi  = visi;
            params.default_model  = default_model;
            params.hyperparameter_entropy  = hyperparameter_entropy;
            params.mock  = mock;           


            //printf("Test 1\t criterion %lf criterion_init %lf criterion_old %lf \n", criterion , criterion_init, criterion_old );
            selected_steplength = linesearch_zoom(data_info, &params);
	
		    break;
		}
		
		//
		// Evaluate wolfe product 2
		//
		
	    compute_data_gradient(data_info, temp_image, data_gradient);
		GullSkilling_entropy_gradient(image_width, current_image, default_model, entropy_gradient);
		for (ii = 0; ii < image_width * image_width; ii++)
		    temp_gradient[ii] = data_gradient[ii] - hyperparameter_entropy * entropy_gradient[ii];
		
		grad_evals++;
		
		wolfe_product2 = scalprod(image_width * image_width, descent_direction, temp_gradient);
		
		if (fabs(wolfe_product2) <= - wolfe_param2 * wolfe_product1 )
        {
            selected_steplength = steplength;
            break;
        }
		
		if (wolfe_product2 >= 0.)
        {
            printf("Test 2\n");
            
            ls_params params;
            params.criterion_evals = &criterion_evals;
            params.grad_evals = &grad_evals;
            params.steplength_low  =  steplength;
            params.steplength_high  =  steplength_old;
            params.criterion_steplength_low  =  criterion_old;
            params.wolfe_product1  =  wolfe_product1;
            params.criterion_init  =  criterion_init;
            params.current_image  =  current_image;
            params.temp_image  =  temp_image;
            params.descent_direction  = descent_direction;
            params.temp_gradient  = temp_gradient;
            params.data_gradient  = data_gradient;
            params.entropy_gradient  = entropy_gradient;
            params.visi  = visi;
            params.default_model  = default_model;
            params.hyperparameter_entropy  = hyperparameter_entropy;
            params.mock  = mock;  
            
            selected_steplength = linesearch_zoom(data_info, &params);
            break;
        }
			
		
		steplength_old = steplength;
		// choose the next steplength
		steplength *= 1.1;
		if (steplength > steplength_max)
		  steplength = steplength_max;
		
		criterion_old = criterion;
		linesearch_iteration++;
		printf("One loop in 3.2 done \n");
	
	}
	// End of line search
	//printf("Double check, selected_steplength = %le \n", selected_steplength); 
	// Update image with the selected step length
	for (ii = 0; ii < image_width * image_width; ii++)
	{
		current_image[ii] += selected_steplength * descent_direction[ii];
		if (current_image[ii] < minvalue)
			current_image[ii] = minvalue;
	}

	// Backup gradient
	if (gradient_method != 0)
		memcpy(full_gradient, full_gradient_new, image_width * image_width * sizeof(double));

    // We're done with the conjugate gradient, free memory:
	free(data_gradient);
	free(entropy_gradient);
	free(full_gradient);
	free(full_gradient_new);
	free(temp_gradient);
	free(descent_direction);
	free(temp_image);

}


float compute_flux(int image_width, float* image)
{
    register int ii;
    float total=0.;
    for(ii=0; ii < image_width * image_width; ii++)
        total += image[ii];
    return total;
}

void compute_data_gradient(chi2_info * data_info, float * image, float * data_gradient)
{
    // TODO: Convert this function over to use data_info directly when possible.
    int image_width = (*data_info).image_width;
    int npow = (*data_info).npow;
    int nbis = (*data_info).nbis;
    oi_data oifits_info = (*(*data_info).oifits_info);    // TODO: Clean this up, this is bad.
    float * data = (*data_info).data;
    float * data_err = (*data_info).data_err;
    float complex * data_phasor = (*data_info).data_phasor;
    float complex * DFT_tablex = (*data_info).dft_x;
    float complex * DFT_tabley = (*data_info).dft_y;    
    float complex * visi = (*data_info).visi;
    float * mock = (*data_info).mock;

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
float image2chi2(chi2_info * info, float * image)
{   
    float chi2 = 0;
    chi2_info i2v_info = *info;
    
    // Compute the visibilities, data, and chi2.
    image2vis(i2v_info.image_width, i2v_info.nuv, image, i2v_info.visi, i2v_info.dft_x, i2v_info.dft_y);
    vis2data(i2v_info.npow, i2v_info.nbis, i2v_info.oifits_info, i2v_info.data_phasor, i2v_info.dft_y, i2v_info.dft_y, i2v_info.visi, i2v_info.mock);
    chi2 = data2chi2(i2v_info.npow, i2v_info.nbis, i2v_info.data, i2v_info.data_err, i2v_info.mock);
    
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

float linesearch_zoom(chi2_info * data_info, ls_params * linesearch_params)
{
    // Pull out the necessary information from the struct:
    ls_params params = *linesearch_params;
    // TODO: Clean up this function, use params directly
    int * criterion_evals = params.criterion_evals; 
    int * grad_evals = params.grad_evals; 
    float steplength_low = params.steplength_low;
    float steplength_high = params.steplength_high;
    float criterion_steplength_low = params.criterion_steplength_low;
    float wolfe_product1 = params.wolfe_product1; 
    float criterion_init = params.criterion_init;
    float * current_image = params.current_image; 
    float * temp_image = params.temp_image;
    float * descent_direction = params.descent_direction;
    float * temp_gradient = params.temp_gradient;
    float * data_gradient = params.data_gradient;
    float * entropy_gradient = params.entropy_gradient;
    //float complex * visi = params.visi;
    float * default_model = params.default_model;
    float hyperparameter_entropy = params.hyperparameter_entropy;
    //float * mock = params.mock;

    int image_width = (*data_info).image_width;

    // Init a few variables
	float chi2, entropy;
	float steplength, selected_steplength = 0., criterion, wolfe_product2;
	int ii;
	int counter = 0;
	float minvalue = 1e-8;
	float wolfe_param1 = 1e-4, wolfe_param2 = 0.1;

	//printf("Entering zoom algorithm \n");

	while( 1 )
	{

		// Interpolation - for the moment by bisection (simple for now)
		steplength = ( steplength_high - steplength_low ) / 2. + steplength_low;
		//printf("Steplength %8.8le Low %8.8le High %8.8le \n", steplength, steplength_low, steplength_high);

		// Evaluate criterion(steplength)
		for(ii=0; ii < image_width * image_width; ii++)
		{
			temp_image[ii] = current_image[ ii ] + steplength * descent_direction[ii];
			if(temp_image[ii] < minvalue)
			temp_image[ii] = minvalue;
		}

		chi2 = image2chi2(data_info, temp_image);
		entropy = GullSkilling_entropy(image_width, temp_image, default_model);
		criterion = chi2 - hyperparameter_entropy * entropy;
		*(criterion_evals) += 1;

	    //printf("Test 1\t criterion %lf criterion_init %lf second member wolfe1 %lf \n", criterion , criterion_init,  criterion_init + wolfe_param1 * steplength * wolfe_product1);
		if ( (criterion > ( criterion_init + wolfe_param1 * steplength * wolfe_product1 ) ) || ( criterion >= criterion_steplength_low ) )
		{
			steplength_high = steplength;
		}
		else
		{
		  
            // Evaluate wolfe product 2
            compute_data_gradient(data_info, temp_image, data_gradient);
            GullSkilling_entropy_gradient(image_width, current_image, default_model, entropy_gradient);
            for (ii = 0; ii < image_width * image_width; ii++)
                temp_gradient[ii] = data_gradient[ii] - hyperparameter_entropy * entropy_gradient[ii];
            
            (*grad_evals) += 1;
            wolfe_product2 = scalprod(image_width * image_width, descent_direction, temp_gradient );
		  
            //printf("Wolfe products: %le %le Second member wolfe2 %le \n", wolfe_product1, wolfe_product2, - wolfe_param2 * wolfe_product1);
		 
            if( ( wolfe_product2 >= wolfe_param2 * wolfe_product1 ) || (  counter > 10 ))
            {
                selected_steplength = steplength;
                break;
            }

            if( wolfe_product2 * ( steplength_high - steplength_low ) >= 0. )
                steplength_high = steplength_low;

            steplength_low = steplength;
		  
		}	
		
		counter++;	
	}

	return selected_steplength;
}

float scalprod(int array_size, float * array1, float * array2)
{
    float total = 0.0;
    register int ii;
    
    for (ii = 0; ii < array_size; ii++)
        total += array1[ii] * array2[ii];
    return total;
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
    oi_data * data_info, float complex * data_phasor, 
    float complex * DFT_tablex, float complex * DFT_tabley,
    float complex * visi, float * mock)
{
    int ii = 0;
    float complex vab = 0;
    float complex vbc = 0;
    float complex vca = 0;
    float complex t3 = 0;
    
    oi_data oifits_info = *data_info;

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


