#include "gpair.h"
#include <time.h>
#include "cpu.h"

#ifndef GETOIFITS_H
#include "getoifits.h"
#endif

#ifndef DATA_TYPES_H
#include "data_types.h"
#endif

// Preprocessor directive for the GPU:
#define USE_GPU

#ifdef USE_GPU
#include "gpu.h"
#endif

#define SEP "-----------------------------------------------------------\n"

// Global variables
int image_width = 0;
float image_pixellation = 0;
int status = 0;
oi_usersel usersel;
oi_data oifits_info; // stores all the info from oifits files


float * data = NULL; // stores the quantities derived from the data
float * data_err = NULL; // stores the error bars on the data
float complex *data_phasor = NULL; // bispectrum rotation precomputed value
float * data_gradient = NULL;

// DFT precomputed coefficient tables
float complex* DFT_tablex = NULL;
float complex* DFT_tabley = NULL;

int npow = 0;
int nbis = 0;
int nuv = 0;
int ndof = 0;

int main(int argc, char *argv[])
{
	register int ii = 0;
	register int uu = 0;
	image_width = 128;
	char filename[200] = "2004contest1.oifits";
	char modelfile[200] = "";
	float image_pixellation = 0.15;
	float modelwidth = 5., modelflux = 10.;
	int modeltype = 3;
	
	// Parse command line arguments:
	for (ii = 1; ii < argc; ii += 2)
	{
		if (strcmp(argv[ii], "-d") == 0)
		{
			sscanf(argv[ii + 1], "%s", filename);
			printf("Filename = %s\n", filename);
		}
		else if (strcmp(argv[ii], "-s") == 0)
		{
			sscanf(argv[ii + 1], "%f", &image_pixellation);
			printf("Pixellation = %f\n", image_pixellation);
		}
		else if (strcmp(argv[ii], "-w") == 0)
		{
			sscanf(argv[ii + 1], "%d", &image_width);
			printf("Pixellation = %d\n", image_width);
		}
		else if (strcmp(argv[ii], "-mw") == 0)
		{
			sscanf(argv[ii + 1], "%f", &modelwidth);
			printf("Modelwidth = %f\n", modelwidth);
		}

		else if (strcmp(argv[ii], "-mf") == 0)
		{
			sscanf(argv[ii + 1], "%f", &modelflux);
			printf("Model Flux = %f\n", modelflux);
		}

		else if (strcmp(argv[ii], "-mt") == 0)
		{
			sscanf(argv[ii + 1], "%d", &modeltype);
			printf("Model Type = %d\n", modeltype);
		}
		if (strcmp(argv[ii], "-i") == 0)
		{
			sscanf(argv[ii + 1], "%s", modelfile);
			printf("Model image = %s\n", modelfile);
		}		
		
	}
	
	float chi2 = 0;

	// read OIFITS and visibilities
	read_oifits(filename);

	// setup visibility/current data matrices

	npow = oifits_info.npow;
	nbis = oifits_info.nbis;
	nuv = oifits_info.nuv;
	ndof = npow + 2 * nbis;

	// TODO: GPU Memory Allocations simply round to the nearest power of two, make this more intelligent
	// by rounding to the nearest sum of powers of 2 (if it makes sense).

	// First thing we do is make all data occupy memory elements that are powers of two.  This makes
	// the GPU code much easier to speed up.
	int data_size = npow + 2 * nbis;
	int data_alloc = pow(2, ceil(log(data_size) / log(2))); // Arrays are allocated to be powers of 2
	int data_alloc_uv = pow(2, ceil(log(nuv) / log(2)));
	int data_alloc_phasor = pow(2, ceil(log(nbis) / log(2)));

	// Allocate memory for the data, error, and mock arrays:
	data = malloc(data_alloc * sizeof(float));
	data_err = malloc(data_alloc * sizeof(float));
	data_phasor = malloc(data_alloc_phasor * sizeof( float complex ));
	init_data(1);

	// Pad the arrays with zeros and ones after the data
	for (ii = data_size; ii < data_alloc; ii++)
	{
		data[ii] = 0;
		data_err[ii] = 0;
	}
	
	for (ii = nbis; ii < data_alloc_phasor ; ii++)
		data_phasor[ii] = 0;


	printf("%d data read : %d powerspectrum, %d bispectrum and Ndof = %d\n", npow + 2 * nbis, npow, nbis, ndof);

	
	float complex * visi = malloc(data_alloc_uv * sizeof( float complex)); // current visibilities 
	//float complex * new_visi= malloc(data_alloc_uv * sizeof( float complex)); // tentative visibilities  
	float * mock = malloc(data_alloc * sizeof(float)); // stores the mock current pseudo-data derived from the image

	for (ii = 0; ii < data_alloc; ii++)
		mock[ii] = 0.;

	// Init the default model:
	float * default_model = malloc(image_width * image_width * sizeof(float));
	if(strcmp(modelfile, "") == 0)
		set_model(image_width, image_pixellation, modeltype, modelwidth, modelflux, default_model);
	else 
		read_fits_image(modelfile, default_model);
	
	writefits(default_model, "!model.fits");
	
	// setup initial image as 128x128 pixel, centered Dirac of flux = 1.0, pixellation of 1.0 mas/pixel
	int image_size = image_width * image_width;
	printf("Image Buffer Size %i \n", image_size);
	float * current_image = malloc(image_size * sizeof(float));
	memset(current_image, 0, image_size);
	for (ii = 0; ii < image_size; ii++)
		current_image[ii] = default_model[ii];

	// setup precomputed DFT table
	int dft_size = nuv * image_width;
	int dft_alloc = pow(2, ceil(log(dft_size) / log(2))); // Amount of space to allocate on the GPU for each axis of the DFT table. 
	DFT_tablex = malloc( dft_size * sizeof(float complex));
	memset(DFT_tablex, 0, dft_size);
	DFT_tabley = malloc( dft_size * sizeof(float complex));
	memset(DFT_tabley, 0, dft_size);
	for (uu = 0; uu < nuv; uu++)
	{
		for (ii = 0; ii < image_width; ii++)
		{
			DFT_tablex[image_width * uu + ii] = cexp(2.0 * I * PI * RPMAS * image_pixellation * oifits_info.uv[uu].u * (float) ii);
			DFT_tabley[image_width * uu + ii] = cexp(-2.0 * I * PI * RPMAS * image_pixellation * oifits_info.uv[uu].v * (float) ii);
		}
	}

	// TODO: Only output if we are in a verbose mode.
	printf("Data Size: %i, Data Allocation: %i \n", data_size, data_alloc);
	printf("UV Size: %i, Allocation: %i \n", nuv, data_alloc_uv);
	printf("POW Size: %i \n", npow);
	printf("BIS Size: %i, Allocation: %i \n", nbis, data_alloc_phasor);
	printf("DFT Size: %i , DFT Allocation: %i \n", dft_size, dft_alloc);

	// TODO: Remove after testing
	int iterations = 10000;

	// Init variables for the line search:
	int criterion_evals = 0;
	int grad_evals = 0;
	int linesearch_iteration = 0;
	float steplength = 0.0;
	float steplength_old = 0.0;
	float steplength_max = 0.0;
	float steplength_temp = 0.0;
	float selected_steplength = 0.0;
	float beta = 0.0;
	float minvalue = 1e-8;
	float criterion_init = 0.0;
	float criterion_old = 0.0;
	float wolfe_param1 = 1e-4;
	float wolfe_param2 = 0.1;
	float wolfe_product1 = 0.0;
	float wolfe_product2 = 0.0;

	float entropy, hyperparameter_entropy = 1000.;
	float criterion;
	int gradient_method = 1;

// Only perform the CPU calculations if we are not using the GPU
#ifndef USE_GPU
	// #########
	// CPU Code:
	// #########

	chi2_info i2v_info;
	i2v_info.npow = npow;
	i2v_info.nbis = nbis;
	i2v_info.nuv = nuv;
	i2v_info.data = data;
	i2v_info.data_err = data_err;
	i2v_info.data_phasor = data_phasor;
	i2v_info.oifits_info = &oifits_info;
	i2v_info.dft_x = DFT_tablex;
	i2v_info.dft_y = DFT_tabley;
	i2v_info.visi = visi;
	i2v_info.mock = mock;
	i2v_info.image_width = image_width;

	float * data_gradient = malloc(image_size * sizeof(float));
	float * entropy_gradient = malloc(image_size * sizeof(float));
	float * full_gradient = malloc(image_size * sizeof(float));
	float * full_gradient_new = malloc(image_size * sizeof(float));
	float * temp_gradient = malloc(image_size * sizeof(float));
	float * temp_image = malloc(image_size * sizeof(float));

	// Init descent direction
	float * descent_direction = malloc(image_size * sizeof(float));
	memset(descent_direction, 0, image_size * sizeof(float));

	// Test 1 : compute mock data, powerspectra + bispectra from scratch
	//clock_t tick = clock();
	//clock_t tock = 0;
	for (uu = 0; uu < iterations; uu++)
	{

		//
		// Compute the criterion
		//
		chi2 = image2chi2(&i2v_info, current_image);
		entropy = GullSkilling_entropy(image_width, current_image, default_model);
		criterion = chi2 - hyperparameter_entropy * entropy;
		criterion_evals++;

		printf(
				"Grad evals: %d J evals: %d Selected coeff %e Beta %e, J = %f, chi2r = %f chi2 = %lf alpha*entropy = %e entropy = %e \n",
				grad_evals, criterion_evals, selected_steplength, beta, criterion, chi2 / (float) ndof, chi2,
				hyperparameter_entropy * entropy, entropy);

		// TODO: Re-enable this:
		if(uu%2 == 0)
			writefits(current_image, "!reconst.fits");

		//
		// Compute full gradient (data + entropy)
		//

		compute_data_gradient(&i2v_info, current_image, data_gradient);
		GullSkilling_entropy_gradient(image_width, current_image, default_model, entropy_gradient);
		for (ii = 0; ii < image_size; ii++)
			full_gradient_new[ii] = data_gradient[ii] - hyperparameter_entropy * entropy_gradient[ii];

		grad_evals++;

		// Compute the modifier of the gradient direction depending on the method
		if ((uu == 0) || (gradient_method == 0))
		{
			beta = 0.; // steepest descent
			//
			// Compute descent direction
			//
		}
		else
		{

			if (gradient_method == 1) // CG
				beta = scalprod(full_gradient_new, full_gradient_new) / scalprod(full_gradient, full_gradient); // FR
			if (gradient_method == 2)
			{
				beta = (scalprod(full_gradient_new, full_gradient_new) - scalprod(full_gradient_new, full_gradient)) // PR
						/ scalprod(full_gradient, full_gradient);
				if (beta < 0.)
					beta = 0.;
			}

			if (gradient_method == 3) // HS
			{
				beta = (scalprod(full_gradient_new, full_gradient_new) - scalprod(full_gradient_new, full_gradient))
						/ (scalprod(descent_direction, full_gradient_new) - scalprod(descent_direction, full_gradient));
				if (beta < 0.)
					beta = 0.;
			}

			if (fabs(scalprod(full_gradient_new, full_gradient) / scalprod(full_gradient_new, full_gradient_new)) > .5)
				beta = 0.;
		}

		//
		// Compute descent direction
		//
		for (ii = 0; ii < image_size; ii++)
			descent_direction[ii] = beta * descent_direction[ii] - full_gradient_new[ii];

		// Some tests on descent direction
	/*	printf("Angle descent direction/gradient %f \t Descent direction / previous descent direction : %f \n", acos(
				-scalprod(descent_direction, full_gradient_new) / sqrt(scalprod(full_gradient_new, full_gradient_new)
						* scalprod(descent_direction, descent_direction))) / PI * 180., fabs(scalprod(full_gradient,
				full_gradient_new)) / scalprod(full_gradient_new, full_gradient_new));
*/
		//      writefits(descent_direction, "!gradient.fits");


		//
		// Line search algorithm begins here
		//

		// Compute quantity for Wolfe condition 1
		wolfe_product1 = scalprod(descent_direction, full_gradient_new);

		// Initialize variables for line search
		selected_steplength = 0.;
		//if(uu > 0)
	//		steplength = 2. * (criterion - criterion_old) / wolfe_product1 ;
	//	else steplength = 1.;
		steplength = 1.;
		steplength_old = 0.;
		steplength_max = 100.; // use a clever scheme here
		criterion_init = criterion;
		criterion_old = criterion;
		linesearch_iteration = 1;

		while (1)
		{

			//
			// Evaluate criterion(steplength)
			//

			//  Step 1: compute the temporary image: I1 = I0 - coeff * descent direction
			for (ii = 0; ii < image_size; ii++)
			{
				temp_image[ii] = current_image[ii] + steplength * descent_direction[ii];
				if (temp_image[ii] < minvalue)
					temp_image[ii] = minvalue;
			}

			// Step 2: Compute criterion(I1)
			chi2 = image2chi2(&i2v_info, temp_image);
			entropy = GullSkilling_entropy(image_width, temp_image, default_model);
			criterion = chi2 - hyperparameter_entropy * entropy;
			criterion_evals++;

			if ((criterion > (criterion_init + wolfe_param1 * steplength * wolfe_product1)) || ((criterion
					>= criterion_old) && (linesearch_iteration > 1)))
			{
				//printf("Test 1\t criterion %lf criterion_init %lf criterion_old %lf \n", criterion , criterion_init, criterion_old );
				selected_steplength = linesearch_zoom(steplength_old, steplength, criterion_old, wolfe_product1,
						criterion_init, &criterion_evals, &grad_evals, current_image, temp_image, descent_direction,
						temp_gradient, data_gradient, entropy_gradient, visi, default_model, hyperparameter_entropy,
						mock, &i2v_info);

				break;
			}

			//
			// Evaluate wolfe product 2
			//

			compute_data_gradient(&i2v_info, temp_image, data_gradient);
			GullSkilling_entropy_gradient(image_width, current_image, default_model, entropy_gradient);
			for (ii = 0; ii < image_size; ii++)
				temp_gradient[ii] = data_gradient[ii] - hyperparameter_entropy * entropy_gradient[ii];
			grad_evals++;

			wolfe_product2 = scalprod(descent_direction, temp_gradient);

			if (fabs(wolfe_product2) <= -wolfe_param2 * wolfe_product1)
			{
				selected_steplength = steplength;
				break;
			}

			if (wolfe_product2 >= 0.)
			{
				printf("Test 2\n");

				selected_steplength = linesearch_zoom(steplength, steplength_old, criterion, wolfe_product1,
						criterion_init, &criterion_evals, &grad_evals, current_image, temp_image, descent_direction,
						temp_gradient, data_gradient, entropy_gradient, visi, default_model, hyperparameter_entropy,
						mock, &i2v_info);

				break;
			}

			steplength_old = steplength;
			criterion_old = criterion;

			// choose the next steplength
			//if((linesearch_iteration > 0) && ( criterion_old - criterion_init - wolfe_product1 * steplength_old ) != 0.)
			//					steplength_temp = wolfe_product1 * steplength_old * steplength_old / (2. * ( criterion_old - criterion_init - wolfe_product1 * steplength_old ) );
			//else 
			steplength_temp = 10.0 * steplength;
			
			steplength = steplength_temp;
			
			if (steplength > steplength_max)
				steplength = steplength_max;
			
			linesearch_iteration++;
			printf("Steplength %f Steplength old %f\n", steplength, steplength_old);

		}
		// End of line search
		//printf("Double check, selected_steplength = %le \n", selected_steplength); 
		// Update image with the selected step length
		for (ii = 0; ii < image_size; ii++)
		{
			current_image[ii] += selected_steplength * descent_direction[ii];
			if (current_image[ii] < minvalue)
				current_image[ii] = minvalue;
		}

		// Backup gradient
		if (gradient_method != 0)
			memcpy(full_gradient, full_gradient_new, image_size * sizeof(float));

	} // End Conjugated Gradient.

	//	tock = clock();
	//	float time_chi2 = (float) (tock - tick) / (float) CLOCKS_PER_SEC;
	//	printf(SEP);
	//	printf("Full DFT Calculation (CPU)\n");
	//	printf(SEP);
	//	printf("CPU time (s): = %f\n", time_chi2);
	//	printf("CPU Chi2: %f (CPU only)\n", chi2);

	// Test 2 : recompute mock data, powerspectra + bispectra when changing only the flux of one pixel
	//	tick = clock();
	//	float total_flux = compute_flux(image_width, current_image);
	//	float inc = 1.1;
	//	int x_changed = 64;
	//	int y_changed = 4;

	// Disabled for now, disagreement between CPU and GPU values.
	/*    for(ii=0; ii < iterations; ii++)*/
	/*    {*/
	/*        //compute complex visibilities and the chi2*/
	/*	    update_vis_fluxchange(x_changed, y_changed, current_image[ x_changed + y_changed * image_width ],  current_image[ x_changed + y_changed * image_width ] + inc, visi, new_visi ) ;*/
	/*	    current_image[ x_changed + y_changed * image_width ] += inc;*/
	/*	    total_flux += inc;*/
	/*	    vis2data(new_visi, mock);*/
	/*	    chi2 = data2chi2( mock );*/
	/*    }       */
	/*    tock=clock();*/
	/*    time_chi2 = (float)(tock - tick) / (float)CLOCKS_PER_SEC;*/
	/*    printf(SEP);*/
	/*    printf("Atomic change (CPU)\n");*/
	/*    printf(SEP);*/
	/*    printf("CPU time (s): = %f\n", time_chi2);*/
	/*    printf("CPU Chi2: %f (CPU only)\n", chi2);*/

	// Compute the gradient of the mock data.  Note, vis2data should have been caled before this call.
	//compute_data_gradient(&i2v_info, current_image, data_gradient);
	
	// Free CPU-based Memory
	free(entropy_gradient);
	free(full_gradient);
	free(full_gradient_new);
	free(temp_gradient);
	free(descent_direction);
#endif

// Use the GPU if specified.
#ifdef USE_GPU
	// #########
	// GPU Code:  
	// #########

	// Convert visi over to a cl_float2 in format <real, imaginary>
	cl_float2 * gpu_visi = NULL;
	gpu_visi = malloc(data_alloc_uv * sizeof(cl_float2));
	int i;
	for(i = 0; i < nuv; i++)
	{
		gpu_visi[i].s0 = __real__ visi[i];
		gpu_visi[i].s1 = __imag__ visi[i];
	}
	// Pad the remainder
	for(i = nuv; i < data_alloc_uv; i++)
	{
		gpu_visi[i].s0 = 0;
		gpu_visi[i].s1 = 0;
	}

	// Convert the biphasor over to a cl_float2 in format <real, imaginary>    
	cl_float2 * gpu_phasor = NULL;
	gpu_phasor = malloc(data_alloc_phasor * sizeof(cl_float2));
	for(i = 0; i < nbis; i++)
	{
		gpu_phasor[i].s0 = creal(data_phasor[i]);
		gpu_phasor[i].s1 = cimag(data_phasor[i]);
	}
	// Pad the remainder
	for(i = nbis; i < data_alloc_phasor; i++)
	{
		gpu_phasor[i].s0 = 0;
		gpu_phasor[i].s1 = 0;
	}

	// We will also need the uvpnt and sign information for bisepctrum computations.
	// Although we waste a little space, we use cl_long4 and cl_float4 so that we may have
	// coalesced loads on the GPU.
	cl_long4 * gpu_bsref_uvpnt = NULL;
	cl_short4 * gpu_bsref_sign = NULL;
	int data_alloc_bsref = data_alloc_phasor; // TODO: Update code below for this.
	gpu_bsref_uvpnt = malloc(data_alloc_bsref * sizeof(cl_long4));
	gpu_bsref_sign = malloc(data_alloc_bsref * sizeof(cl_short4));
	for(i = 0; i < nbis; i++)
	{
		gpu_bsref_uvpnt[i].s0 = oifits_info.bsref[i].ab.uvpnt;
		gpu_bsref_uvpnt[i].s1 = oifits_info.bsref[i].bc.uvpnt;
		gpu_bsref_uvpnt[i].s2 = oifits_info.bsref[i].ca.uvpnt;
		gpu_bsref_uvpnt[i].s3 = 0;

		gpu_bsref_sign[i].s0 = oifits_info.bsref[i].ab.sign;
		gpu_bsref_sign[i].s1 = oifits_info.bsref[i].bc.sign;
		gpu_bsref_sign[i].s2 = oifits_info.bsref[i].ca.sign;
		gpu_bsref_sign[i].s3 = 0;
	}

	// Copy the DFT table over to a GPU-friendly format:
	cl_float2 * gpu_dft_x = NULL;
	cl_float2 * gpu_dft_y = NULL;
	gpu_dft_x = malloc(dft_alloc * sizeof(cl_float2));
	gpu_dft_y = malloc(dft_alloc * sizeof(cl_float2));
	for(uu=0; uu < nuv; uu++)
	{
		for(ii=0; ii < image_width; ii++)
		{
			i = image_width * uu + ii;
			gpu_dft_x[i].s0 = __real__ DFT_tablex[i];
			gpu_dft_x[i].s1 = __imag__ DFT_tablex[i];
			gpu_dft_y[i].s0 = __real__ DFT_tabley[i];
			gpu_dft_y[i].s1 = __imag__ DFT_tabley[i];
		}
	}

	// Pad out the remainder of the array with zeros:
	for(i = nuv * image_width; i < dft_alloc; i++)
	{
		gpu_dft_x[i].s0 = 0;
		gpu_dft_x[i].s1 = 0;
		gpu_dft_y[i].s0 = 0;
		gpu_dft_y[i].s1 = 0;
	}

	// Initalize the GPU, copy data, and build the kernels.
	gpu_init();

	gpu_copy_data(data, data_err, data_alloc, data_alloc_uv, gpu_phasor, data_alloc_phasor,
			npow, gpu_bsref_uvpnt, gpu_bsref_sign, data_alloc_bsref,
			default_model,
			image_size,	image_width);
			
	gpu_copy_image(current_image, image_width, image_width);

	gpu_build_kernels(data_alloc, image_size, image_width);
	gpu_copy_dft(gpu_dft_x, gpu_dft_y, dft_alloc);

	// Free variables used to store values pepared for the GPU
	free(gpu_visi);
	free(gpu_phasor);
	free(gpu_bsref_uvpnt);
	free(gpu_bsref_sign);
	free(gpu_dft_x);
	free(gpu_dft_y);
	
	cl_mem * pFull_gradient_new = gpu_getp_fgn();
	cl_mem * pFull_gradient = gpu_getp_fg();
	cl_mem * pDescent_direction = gpu_getp_dd();
	cl_mem * pTemp_gradient= gpu_getp_tg();
	
    printf("Entering Main CG Loop.\n");

	for (uu = 0; uu < iterations; uu++)
	{

		//
		// Compute the criterion
		//
		chi2 = gpu_get_chi2_curr(nuv, npow, nbis, data_alloc, data_alloc_uv);
		entropy = gpu_get_entropy_curr(image_width);
		criterion = chi2 - hyperparameter_entropy * entropy;
		criterion_evals++;

		printf("Grad evals: %d J evals: %d Selected coeff %e Beta %e, J = %f, chi2r = %f chi2 = %lf alpha*entropy = %e entropy = %e \n",
				grad_evals, criterion_evals, selected_steplength, beta, criterion, chi2 / (float) ndof, chi2,
				hyperparameter_entropy * entropy, entropy);

		if(uu%2 == 0)
			writefits(current_image, "!reconst.fits");

		//
		// Compute full gradient (data + entropy)
		//
		
		gpu_compute_data_gradient_curr(npow, nbis, image_width);
		gpu_compute_entropy_gradient_curr(image_width);
        
        // Now compute the criterion gradient:
        gpu_compute_criterion_gradient(image_width, hyperparameter_entropy);
		grad_evals++;

		// Compute the modifier of the gradient direction depending on the method
		if ((uu == 0) || (gradient_method == 0))
		{
			beta = 0.; // steepest descent
		}
		else
		{
			if (gradient_method == 1) // CG
				beta = gpu_get_scalprod(image_width, image_width, pFull_gradient_new, pFull_gradient_new) 
				        / gpu_get_scalprod(image_width, image_width, pFull_gradient, pFull_gradient); // FR
			if (gradient_method == 2)
			{
				beta = (gpu_get_scalprod(image_width, image_width, pFull_gradient_new, pFull_gradient_new) 
				        - gpu_get_scalprod(image_width, image_width, pFull_gradient_new, pFull_gradient)) // PR
						/ gpu_get_scalprod(image_width, image_width, pFull_gradient, pFull_gradient);
				if (beta < 0.)
					beta = 0.;
			}

			if (gradient_method == 3) // HS
			{
				beta = (gpu_get_scalprod(image_width, image_width, pFull_gradient_new, pFull_gradient_new) 
				        - gpu_get_scalprod(image_width, image_width, pFull_gradient_new, pFull_gradient))
					    / (gpu_get_scalprod(image_width, image_width, pDescent_direction, pFull_gradient_new) 
					        - gpu_get_scalprod(image_width, image_width, pDescent_direction, pFull_gradient));
				if (beta < 0.)
					beta = 0.;
			}

			if (fabs(gpu_get_scalprod(image_width, image_width, pFull_gradient_new, pFull_gradient) 
			    / gpu_get_scalprod(image_width, image_width, pFull_gradient_new, pFull_gradient_new)) > .5)
				beta = 0.;
		}

		//
		// Compute descent direction
		//
		gpu_compute_descent_dir(image_width, beta);

		// Some tests on descent direction
		// TODO: Note this hasn't been rewritten for the GPU side yet:
	/*	printf("Angle descent direction/gradient %f \t Descent direction / previous descent direction : %f \n", acos(
				-scalprod(descent_direction, full_gradient_new) / sqrt(scalprod(full_gradient_new, full_gradient_new)
						* scalprod(descent_direction, descent_direction))) / PI * 180., fabs(scalprod(full_gradient,
				full_gradient_new)) / scalprod(full_gradient_new, full_gradient_new));
*/
		//      writefits(descent_direction, "!gradient.fits");


		//
		// Line search algorithm begins here
		//

		// Compute quantity for Wolfe condition 1
		wolfe_product1 = gpu_get_scalprod(image_width, image_width, pDescent_direction, pFull_gradient_new);

		// Initialize variables for line search
		selected_steplength = 0.;
		//if(uu > 0)
	//		steplength = 2. * (criterion - criterion_old) / wolfe_product1 ;
	//	else steplength = 1.;
		steplength = 1.;
		steplength_old = 0.;
		steplength_max = 100.; // use a clever scheme here
		criterion_init = criterion;
		criterion_old = criterion;
		linesearch_iteration = 1;

		while (1)
		{

			//
			// Evaluate criterion(steplength)
			//

			//  Step 1: compute the temporary image: I1 = I0 - coeff * descent direction
			gpu_update_tempimage(image_width, steplength, minvalue, pDescent_direction);

			// Step 2: Compute criterion(I1)
		    chi2 = gpu_get_chi2_temp(nuv, npow, nbis, data_alloc, data_alloc_uv);
		    entropy = gpu_get_entropy_temp(image_width);
			criterion = chi2 - hyperparameter_entropy * entropy;
			criterion_evals++;

			if ((criterion > (criterion_init + wolfe_param1 * steplength * wolfe_product1)) || ((criterion
					>= criterion_old) && (linesearch_iteration > 1)))
			{
			    selected_steplength = gpu_linesearch_zoom(nuv, npow, nbis, data_alloc, data_alloc_uv, image_width,
			        steplength_old, steplength, criterion_old, wolfe_product1, criterion_init,
			        &criterion_evals, &grad_evals,
			        pDescent_direction, pTemp_gradient,
			        hyperparameter_entropy);
			
				//printf("Test 1\t criterion %lf criterion_init %lf criterion_old %lf \n", criterion , criterion_init, criterion_old );
/*				selected_steplength = linesearch_zoom(steplength_old, steplength, criterion_old, wolfe_product1,*/
/*						criterion_init, &criterion_evals, &grad_evals, current_image, temp_image, descent_direction,*/
/*						temp_gradient, data_gradient, entropy_gradient, visi, default_model, hyperparameter_entropy,*/
/*						mock, &i2v_info);*/

				break;
			}

			//
			// Evaluate wolfe product 2
			//

            gpu_compute_data_gradient_temp(npow, nbis, image_width);
			gpu_compute_entropy_gradient_temp(image_width);
			gpu_compute_criterion_gradient(image_width, hyperparameter_entropy);
			
			grad_evals++;

			wolfe_product2 = gpu_get_scalprod(image_width, image_width, pDescent_direction, pTemp_gradient);

			if (fabs(wolfe_product2) <= -wolfe_param2 * wolfe_product1)
			{
				selected_steplength = steplength;
				break;
			}

			if (wolfe_product2 >= 0.)
			{
				printf("Test 2\n");

			    selected_steplength = gpu_linesearch_zoom(nuv, npow, nbis, data_alloc, data_alloc_uv, image_width,
			        steplength, steplength_old, criterion, wolfe_product1, criterion_init,
			        &criterion_evals, &grad_evals,
			        pDescent_direction, pTemp_gradient,
			        hyperparameter_entropy);

/*				selected_steplength = linesearch_zoom(steplength, steplength_old, criterion, wolfe_product1,*/
/*						criterion_init, &criterion_evals, &grad_evals, current_image, temp_image, descent_direction,*/
/*						temp_gradient, data_gradient, entropy_gradient, visi, default_model, hyperparameter_entropy,*/
/*						mock, &i2v_info);*/

				break;
			}

			steplength_old = steplength;
			criterion_old = criterion;

			// choose the next steplength
			//if((linesearch_iteration > 0) && ( criterion_old - criterion_init - wolfe_product1 * steplength_old ) != 0.)
			//					steplength_temp = wolfe_product1 * steplength_old * steplength_old / (2. * ( criterion_old - criterion_init - wolfe_product1 * steplength_old ) );
			//else 
			steplength_temp = 10.0 * steplength;
			
			steplength = steplength_temp;
			
			if (steplength > steplength_max)
				steplength = steplength_max;
			
			linesearch_iteration++;
			printf("Steplength %f Steplength old %f\n", steplength, steplength_old);

		}
		// End of line search
		//printf("Double check, selected_steplength = %le \n", selected_steplength); 
		// Update image with the selected step length
		gpu_update_image(image_width, selected_steplength, minvalue, pDescent_direction);

		// Backup gradient
		gpu_backup_gradient(image_width * image_width, pFull_gradient, pFull_gradient_new);

	} // End Conjugated Gradient.

	// Cleanup, shutdown, were're done.
	gpu_cleanup();

#endif  // End of ifdef USE_GPU

	free(mock);
	free(data);
	free(data_err);
	free( data_phasor);
	free( visi);
	free(current_image);
	free( DFT_tablex);
	free( DFT_tabley);

	return 0;

}

int read_oifits(char * filename)
{
	// Read the image
	strcpy(usersel.file, filename);
	get_oi_fits_selection(&usersel, &status);
	get_oi_fits_data(usersel, &oifits_info, &status);
	printf("OIFITS File read\n");
	return 1;
}

void writefits(float* image, char* fitsimage)
{
	fitsfile *fptr;
	int error = 0;
	int* status = &error;
	long fpixel = 1, naxis = 2, nelements;
	long naxes[2];

	/*Initialise storage*/
	naxes[0] = (long) image_width;
	naxes[1] = (long) image_width;
	nelements = naxes[0] * naxes[1];

	/*Create new file*/
	if (*status == 0)
		fits_create_file(&fptr, fitsimage, status);

	/*Create primary array image*/
	if (*status == 0)
		fits_create_img(fptr, DOUBLE_IMG, naxis, naxes, status);
	/*Write a keywords (datafile, target, image pixelation) */
	if (*status == 0)
		fits_update_key(fptr, TSTRING, "DATAFILE", "dummy", "Data File Name", status);
	if (*status == 0)
		fits_update_key(fptr, TSTRING, "TARGET", "dummy", "Target Name", status);
	if (*status == 0)
		fits_update_key(fptr, TFLOAT, "PIXSIZE", &image_pixellation, "Pixelation (mas)", status);
	if (*status == 0)
		fits_update_key(fptr, TINT, "WIDTH", &image_width, "Size (pixels)", status);

	/*Write image*/
	if (*status == 0)
		fits_write_img(fptr, TFLOAT, fpixel, nelements, &image[0], status);

	/*Close file*/
	if (*status == 0)
		fits_close_file(fptr, status);

	/*Report any errors*/
	fits_report_error(stderr, *status);

}

void init_data(int do_extrapolation)
{
	register int ii;
	float infinity = 1e8;
	int warning_extrapolation = 0;
	float pow1, powerr1, pow2, powerr2, pow3, powerr3, sqamp1, sqamp2, sqamp3, sqamperr1, sqamperr2, sqamperr3;
	// Set elements [0, npow - 1] equal to the power spectrum
	for (ii = 0; ii < npow; ii++)
	{
		data[ii] = oifits_info.pow[ii];
		data_err[ii] = 1 / oifits_info.powerr[ii];
	}

	// Let j = npow, set elements [j, j + nbis - 1] to the powerspectrum data.
	for (ii = 0; ii < nbis; ii++)
	{
		
		if ((do_extrapolation == 1) && ((oifits_info.bisamperr[ii] <= 0.) || (oifits_info.bisamperr[ii] > infinity))) // Missing triple amplitudes
		{
			if ((oifits_info.bsref[ii].ab.uvpnt < npow) && (oifits_info.bsref[ii].bc.uvpnt < npow)
					&& (oifits_info.bsref[ii].ca.uvpnt < npow))
			{
				// if corresponding powerspectrum points are available
				// Derive pseudo-triple amplitudes from powerspectrum data
				// First select the relevant powerspectra
				pow1 = oifits_info.pow[oifits_info.bsref[ii].ab.uvpnt];
				powerr1 = oifits_info.powerr[oifits_info.bsref[ii].ab.uvpnt];
				pow2 = oifits_info.pow[oifits_info.bsref[ii].bc.uvpnt];
				powerr2 = oifits_info.powerr[oifits_info.bsref[ii].bc.uvpnt];
				pow3 = oifits_info.pow[oifits_info.bsref[ii].ca.uvpnt];
				powerr3 = oifits_info.powerr[oifits_info.bsref[ii].ca.uvpnt];
				// Derive optimal visibility amplitudes + noise variance
				sqamp1 = (pow1 + sqrt(square(pow1) + 2.0 * square(powerr1))) / 2.;
				sqamperr1 = 1. / (1. / sqamp1 + 2. * (3. * sqamp1 - pow1) / square(powerr1));
				sqamp2 = (pow2 + sqrt(square(pow2) + 2.0 * square(powerr2))) / 2.;
				sqamperr2 = 1. / (1. / sqamp2 + 2. * (3. * sqamp2 - pow2) / square(powerr2));
				sqamp3 = (pow3 + sqrt(square(pow3) + 2.0 * square(powerr3))) / 2.;
				sqamperr3 = 1. / (1. / sqamp3 + 2. * (3. * sqamp3 - pow3) / square(powerr3));
				// And form the triple amplitude statistics
				oifits_info.bisamp[ii] = sqrt(sqamp1 * sqamp2 * sqamp3);
				oifits_info.bisamperr[ii] = oifits_info.bisamp[ii] * sqrt(sqamperr1 / sqamp1 + sqamperr2 / sqamp2
						+ sqamperr3 / sqamp3);
				if(warning_extrapolation == 0)
				{
					printf("*************************  Warning - Recalculating T3amp from Powerspectra  ********************\n");
					warning_extrapolation = 1;
				}
				
			}

			else // missing powerspectrum points -> cannot extrapolate bispectrum
			{
				printf(
						"WARNING: triple amplitude extrapolation from powerspectrum failed because of missing powerspectrum\n");
				oifits_info.bisamp[ii] = 1.0;
				oifits_info.bisamperr[ii] = infinity;
				// TDB - decrease the number of degrees of freedom
				ndof--;
			}

		}

		data[npow + 2 * ii] = fabs(oifits_info.bisamp[ii]);
		data[npow + 2 * ii + 1] = 0.;
		if (oifits_info.bisamperr[ii] < infinity / 2.)
			data_err[npow + 2 * ii] = 1 / oifits_info.bisamperr[ii];
		else
			data_err[npow + 2 * ii] = 0.;

		data_err[npow + 2 * ii + 1] = 1 / (fabs(oifits_info.bisamp[ii] * oifits_info.bisphserr[ii] ));
		
		data_phasor[ii] = cexp(-I * oifits_info.bisphs[ii]);
		// printf("Debug phs: %f phs_err: %f\n", oifits_info.bisphs[ii], oifits_info.bisphserr[ii]);
	}

}

float linesearch_zoom( float steplength_low, float steplength_high, float criterion_steplength_low, float wolfe_product1,
		float criterion_init, int *criterion_evals, int *grad_evals, float *current_image, float *temp_image,
		float *descent_direction, float *temp_gradient, float *data_gradient,
		float *entropy_gradient, float complex* visi, float* default_model , float hyperparameter_entropy, float *mock,
		chi2_info * data_info)
{
	float chi2, entropy;
	float steplength =0., selected_steplength = 0., criterion = 0., criterion_old = 0., steplength_old = 0., wolfe_product2;
	int ii;
	int counter = 0;
	float minvalue = 1e-8;
	float wolfe_param1 = 1e-4, wolfe_param2 = 0.1;

	//printf("Entering zoom algorithm \n");

	while( 1 )
	{

		// Interpolation - for the moment by bisection (simple for now)
		//steplength = ( steplength_high - steplength_low ) / 2. + steplength_low;
		printf("Steplength %8.8le Low %8.8le High %8.8le \n", steplength, steplength_low, steplength_high);

		if((counter > 0) && ( criterion_old - criterion_init - wolfe_product1 * steplength_old ) != 0.)
		steplength = fabs(wolfe_product1 * steplength_old * steplength_old / (2. * ( criterion_old - criterion_init - wolfe_product1 * steplength_old ) ));

		if((counter == 0) || (steplength < steplength_low ) || ( steplength > steplength_high))
		steplength = ( steplength_high - steplength_low ) / 2. + steplength_low;

		if( fabs( steplength_high - steplength_low ) < 1e-14)
		{
			selected_steplength=steplength_low;
			break;
		}

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
		*criterion_evals++;

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

			*grad_evals++;
			wolfe_product2 = scalprod( descent_direction, temp_gradient );

			//printf("Wolfe products: %le %le Second member wolfe2 %le \n", wolfe_product1, wolfe_product2, - wolfe_param2 * wolfe_product1);
			if( ( wolfe_product2 >= wolfe_param2 * wolfe_product1 ) || ( counter > 10 ))
			{
				selected_steplength = steplength;
				break;
			}

			if( wolfe_product2 * ( steplength_high - steplength_low ) >= 0. )
			steplength_high = steplength_low;

			steplength_low = steplength;

		}

		steplength_old = steplength;
		criterion_old = criterion;

		counter++;
	}

	return selected_steplength;
}

float scalprod(float * array1, float * array2)
{
	float total = 0.0;
	register int ii;
	for (ii = 0; ii < image_width * image_width; ii++)
		total += array1[ii] * array2[ii];
	return total;
}

void read_fits_image(char* filename, float* array)
{
	fitsfile *fptr;       /* pointer to the FITS file, defined in fitsio.h */
	int error = 0;
	int* status = &error;
	int  nfound, anynull;
	long naxes[2], fpixel, npixels;
	float nullval;

	if (*status==0)
		fits_open_file(&fptr, filename, READONLY, status);
	
	if (*status==0)
			fits_read_keys_lng(fptr, "NAXIS", 1, 2, naxes, &nfound, status);
		
	npixels  = naxes[0] * naxes[1];         /* number of pixels in the image */
	fpixel   = 1;
	nullval  = 0;                /* don't check for null values in the image */

	if(*status==0)
			fits_read_img(fptr, TFLOAT, fpixel, npixels, &nullval, array, &anynull, status);

	if(*status==0)
			fits_close_file(fptr, status);

	if(*status != 0)
	{
		printf("Error reading image %s\n", filename);
		getchar();
	}
}




