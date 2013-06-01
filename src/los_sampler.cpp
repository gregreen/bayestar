/*
 * los_sampler.cpp
 * 
 * Samples from posterior distribution of line-of-sight extinction
 * model, given a set of stellar posterior densities in DM, E(B-V).
 * 
 * This file is part of bayestar.
 * Copyright 2012 Gregory Green
 * 
 * Bayestar is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 * 
 */

#include "los_sampler.h"


/*
 *  Discrete cloud model
 */

void sample_los_extinction_clouds(std::string out_fname, TMCMCOptions &options, TLOSMCMCParams &params,
                                  unsigned int N_clouds, uint64_t healpix_index) {
	timespec t_start, t_write, t_end;
	clock_gettime(CLOCK_MONOTONIC, &t_start);
	
	/*double x[] = {8., 4., -0.693, -1.61};
	gsl_rng *r;
	seed_gsl_rng(&r);
	//gen_rand_los_extinction_clouds(&(x[0]), 4, r, params);
	double lnp_tmp = lnp_los_extinction_clouds(&(x[0]), 4, params);
	std::cout << lnp_tmp << std::endl;
	gsl_rng_free(r);*/
	
	std::cout << "subpixel: " << std::endl;
	for(size_t i=0; i<params.subpixel.size(); i++) {
		std::cout << " " << params.subpixel[i];
	}
	std::cout << std::endl;
	
	TNullLogger logger;
	
	unsigned int max_attempts = 2;
	unsigned int N_steps = options.steps;
	unsigned int N_samplers = options.samplers;
	unsigned int N_threads = options.N_threads;
	unsigned int ndim = 2 * N_clouds;
	
	std::vector<double> GR_transf;
	TLOSCloudTransform transf(ndim);
	double GR_threshold = 1.25;
	
	TAffineSampler<TLOSMCMCParams, TNullLogger>::pdf_t f_pdf = &lnp_los_extinction_clouds;
	TAffineSampler<TLOSMCMCParams, TNullLogger>::rand_state_t f_rand_state = &gen_rand_los_extinction_clouds;
	
	std::cerr << std::endl;
	std::cout << "Line-of-Sight Extinction Profile" << std::endl;
	std::cout << "====================================" << std::endl;
	
	//std::cerr << "# Setting up sampler" << std::endl;
	TParallelAffineSampler<TLOSMCMCParams, TNullLogger> sampler(f_pdf, f_rand_state, ndim, N_samplers*ndim, params, logger, N_threads);
	sampler.set_scale(2.);
	sampler.set_replacement_bandwidth(0.25);
	
	// Burn-in
	std::cerr << "# Burn-in ..." << std::endl;
	sampler.step(int(N_steps*25./100.), false, 0., 0., 0.);
	sampler.step(int(N_steps*20./100.), false, 0., options.p_replacement, 0.);
	sampler.step(int(N_steps*20./100.), false, 0., 0.85, 0.);
	sampler.step(int(N_steps*20./100.), false, 0., options.p_replacement, 0.);
	sampler.step(int(N_steps*20./100.), false, 0., 0.85, 0.);
	sampler.print_stats();
	sampler.clear();
	
	std::cerr << "# Main run ..." << std::endl;
	bool converged = false;
	size_t attempt;
	for(attempt = 0; (attempt < max_attempts) && (!converged); attempt++) {
		sampler.step((1<<attempt)*N_steps, true, 0., options.p_replacement, 0.);
		
		std::cout << std::endl << "Transformed G-R Diagnostic:";
		sampler.calc_GR_transformed(GR_transf, &transf);
		for(unsigned int k=0; k<ndim; k++) {
			std::cout << "  " << std::setprecision(3) << GR_transf[k];
		}
		std::cout << std::endl << std::endl;
		
		converged = true;
		for(size_t i=0; i<ndim; i++) {
			if(GR_transf[i] > GR_threshold) {
				converged = false;
				if(attempt != max_attempts-1) {
					sampler.print_stats();
					std::cerr << "# Extending run ..." << std::endl;
					sampler.step(int(N_steps*1./5.), false, 0., 1., 0.);
					sampler.clear();
					//logger.clear();
				}
				break;
			}
		}
	}
	
	clock_gettime(CLOCK_MONOTONIC, &t_write);
	
	//std::stringstream group_name;
	//group_name << "/pixel " << healpix_index;
	//group_name << "/los clouds";
	//chain.save(out_fname, group_name.str(), 0, "Delta mu, Delta E(B-V)", 3, 100, converged);
	
	std::stringstream group_name;
	group_name << "/pixel " << healpix_index;
	TChain chain = sampler.get_chain();
	
	TChainWriteBuffer writeBuffer(ndim, 100, 1);
	writeBuffer.add(chain, converged);
	writeBuffer.write(out_fname, group_name.str(), "clouds");
	
	clock_gettime(CLOCK_MONOTONIC, &t_end);
	
	sampler.print_stats();
	std::cout << std::endl;
	
	
	
	if(!converged) {
		std::cerr << "# Failed to converge." << std::endl;
	}
	std::cerr << "# Number of steps: " << (1<<(attempt-1))*N_steps << std::endl;
	std::cerr << "# Time elapsed: " << std::setprecision(2) << (t_end.tv_sec - t_start.tv_sec) + 1.e-9*(t_end.tv_nsec - t_start.tv_nsec) << " s" << std::endl;
	std::cerr << "# Sample time: " << std::setprecision(2) << (t_write.tv_sec - t_start.tv_sec) + 1.e-9*(t_write.tv_nsec - t_start.tv_nsec) << " s" << std::endl;
	std::cerr << "# Write time: " << std::setprecision(2) << (t_end.tv_sec - t_write.tv_sec) + 1.e-9*(t_end.tv_nsec - t_write.tv_nsec) << " s" << std::endl << std::endl;
}

void los_integral_clouds(TImgStack &img_stack, const double *const subpixel, double *const ret, const double *const Delta_mu,
                         const double *const logDelta_EBV, unsigned int N_clouds) {
	int x = 0;
	int x_next = ceil((Delta_mu[0] - img_stack.rect->min[0]) / img_stack.rect->dx[0]);
	
	double y_0 = -img_stack.rect->min[1] / img_stack.rect->dx[1];
	double y = 0.;
	int y_max = img_stack.rect->N_bins[1];
	double y_ceil, y_floor, dy, y_scaled;
	
	for(size_t i=0; i<img_stack.N_images; i++) { ret[i] = 0.; }
	
	for(int i=0; i<N_clouds+1; i++) {
		if(i == N_clouds) {
			x_next = img_stack.rect->N_bins[0];
		} else if(i != 0) {
			x_next += ceil(Delta_mu[i] / img_stack.rect->dx[0]);
		}
		
		if(x_next > img_stack.rect->N_bins[0]) {
			x_next = img_stack.rect->N_bins[0];
		} else if(x_next < 0) {
			x_next = 0;
		}
		
		if(i != 0) {
			y += exp(logDelta_EBV[i-1]) / img_stack.rect->dx[1];
		}
		
		int x_start = x;
		for(int k=0; k<img_stack.N_images; k++) {
			y_scaled = y_0 + y*subpixel[k];
			y_floor = floor(y_scaled);
			y_ceil = y_floor + 1.;
			if((int)y_ceil >= y_max) { break; }
			if((int)y_floor < 0) { break; }
			
			for(x = x_start; x<x_next; x++) {
				ret[k] += (y_ceil - y_scaled) * img_stack.img[k]->at<double>(x, (int)y_floor)
				          + (y_scaled - y_floor) * img_stack.img[k]->at<double>(x, (int)y_ceil);
			}
		}
	}
}

double lnp_los_extinction_clouds(const double* x, unsigned int N, TLOSMCMCParams& params) {
	size_t N_clouds = N / 2;
	const double *Delta_mu = x;
	const double *logDelta_EBV = x + N_clouds;
	
	double lnp = 0.;
	
	// Delta_mu must be positive
	double mu_tot = 0.;
	for(size_t i=0; i<N_clouds; i++) {
		if(Delta_mu[i] <= 0.) { return neg_inf_replacement; }
		mu_tot += Delta_mu[i];
	}
	
	// Don't consider clouds outside of the domain under consideration
	if(Delta_mu[0] < params.img_stack->rect->min[0]) { return neg_inf_replacement; }
	if(mu_tot > params.img_stack->rect->max[0]) { return neg_inf_replacement; }
	
	double EBV_tot = 0.;
	double tmp;
	for(size_t i=0; i<N_clouds; i++) {
		tmp = exp(logDelta_EBV[i]);
		EBV_tot += tmp;
		
		// Prior to prevent EBV from straying high
		lnp -= 0.5 * tmp * tmp / (10. * 10.);
	}
	
	// Extinction must not exceed maximum value
	if(EBV_tot * params.subpixel_max >= params.img_stack->rect->max[1]) { return neg_inf_replacement; }
	
	// Prior on total extinction
	if((params.EBV_max > 0.) && (EBV_tot > params.EBV_max)) {
		lnp -= (EBV_tot - params.EBV_max) * (EBV_tot - params.EBV_max) / (2. * 0.25 * 0.25 * params.EBV_max * params.EBV_max);
	}
	
	// Wide Gaussian prior on Delta_EBV to prevent fit from straying drastically
	const double bias = -5.;
	const double sigma = 25.;
	for(size_t i=0; i<N_clouds; i++) {
		lnp -= (logDelta_EBV[i] - bias) * (logDelta_EBV[i] - bias) / (2. * sigma * sigma);
		
		//lnp -= Delta_mu[i] * Delta_mu[i] / (0.5 * 5. * 5.);
	}
	
	// Repulsive force to keep clouds from collapsing into one
	for(size_t i=1; i<N_clouds; i++) {
		lnp -= 1. / Delta_mu[i];
	}
	
	// Compute line integrals through probability surfaces
	double *line_int = params.get_line_int(omp_get_thread_num());
	los_integral_clouds(*(params.img_stack), params.subpixel.data(), line_int, Delta_mu, logDelta_EBV, N_clouds);
	
	// Soften and multiply line integrals
	for(size_t i=0; i<params.img_stack->N_images; i++) {
		if(line_int[i] < 1.e5*params.p0) {
			line_int[i] += params.p0 * exp(-line_int[i]/params.p0);
		}
		lnp += log(line_int[i]);
		//std::cerr << line_int[i] << std::endl;
	}
	
	return lnp;
}

void gen_rand_los_extinction_clouds(double *const x, unsigned int N, gsl_rng *r, TLOSMCMCParams &params) {
	double mu_floor = params.img_stack->rect->min[0];
	double mu_ceil = params.img_stack->rect->max[0];
	double EBV_ceil = params.img_stack->rect->max[1] / params.subpixel_max;
	unsigned int N_clouds = N / 2;
	
	double logEBV_mean = log(1.5 * params.EBV_guess_max / params.subpixel_max / (double)N_clouds);
	double mu_mean = (mu_ceil - mu_floor) / N_clouds;
	double EBV_sum = 0.;
	double mu_sum = mu_floor;
	
	double *Delta_mu = x;
	double *logDelta_EBV = x + N_clouds;
	
	double log_mu_mean = log(0.5 * mu_mean);
	for(size_t i=0; i<N_clouds; i++) {
		logDelta_EBV[i] = logEBV_mean + gsl_ran_gaussian_ziggurat(r, 1.5);
		EBV_sum += exp(logDelta_EBV[i]);
		
		Delta_mu[i] = exp(log_mu_mean + gsl_ran_gaussian_ziggurat(r, 1.5));
		mu_sum += Delta_mu[i];
	}
	Delta_mu[0] += mu_floor;
	
	// Ensure that reddening is not more than allowed
	if(EBV_sum >= 0.95 * EBV_ceil) {
		double factor = log(0.95 * EBV_ceil / EBV_sum);
		for(size_t i=0; i<N_clouds; i++) {
			logDelta_EBV[i] += factor;
		}
	}
	
	// Ensure that distance to farthest cloud is not more than allowed
	if(mu_sum >= 0.95 * mu_ceil) {
		double factor = 0.95 * mu_ceil / mu_sum;
		for(size_t i=0; i<N_clouds; i++) {
			Delta_mu[i] *= factor;
		}
	}
}



/*
 *  Piecewise-linear line-of-sight model
 */

void sample_los_extinction(std::string out_fname, TMCMCOptions &options, TLOSMCMCParams &params,
                           unsigned int N_regions, uint64_t healpix_index) {
	timespec t_start, t_write, t_end;
	clock_gettime(CLOCK_MONOTONIC, &t_start);
	
	std::cout << "guess of EBV max = " << params.EBV_guess_max << std::endl;
	
	guess_EBV_profile(options, params, N_regions);
	//monotonic_guess(img_stack, N_regions, params.EBV_prof_guess, options);
	
	TNullLogger logger;
	
	unsigned int max_attempts = 2;
	unsigned int N_steps = options.steps;
	unsigned int N_samplers = options.samplers;
	unsigned int N_threads = options.N_threads;
	unsigned int ndim = N_regions + 1;
	
	std::vector<double> GR_transf;
	TLOSTransform transf(ndim);
	double GR_threshold = 1.25;
	
	TAffineSampler<TLOSMCMCParams, TNullLogger>::pdf_t f_pdf = &lnp_los_extinction;
	TAffineSampler<TLOSMCMCParams, TNullLogger>::rand_state_t f_rand_state = &gen_rand_los_extinction_from_guess;
	
	std::cerr << std::endl;
	std::cout << "Line-of-Sight Extinction Profile" << std::endl;
	std::cout << "====================================" << std::endl;
	
	TParallelAffineSampler<TLOSMCMCParams, TNullLogger> sampler(f_pdf, f_rand_state, ndim, N_samplers*ndim, params, logger, N_threads);
	sampler.set_scale(1.1);
	sampler.set_replacement_bandwidth(0.50);	// TODO: Scale with number of regions
	
	// Burn-in
	std::cerr << "# Burn-in ..." << std::endl;
	sampler.step(int(N_steps*40./100.), false, 0., 0.4, 0.);
	sampler.step(int(N_steps*10./100), false, 0., 1.0, 0., false);
	sampler.step(int(N_steps*40./100.), false, 0., 0.4, 0.);
	sampler.step(int(N_steps*10./100), false, 0., 0.8, 0.);
	//sampler.step(N_steps, false, 0., options.p_replacement, 0.);
	//sampler.step(N_steps/2., false, 0., 1., 0.);
	sampler.print_stats();
	sampler.clear();
	
	std::cerr << "# Main run ..." << std::endl;
	bool converged = false;
	size_t attempt;
	for(attempt = 0; (attempt < max_attempts) && (!converged); attempt++) {
		sampler.step((1<<attempt)*N_steps, true, 0., options.p_replacement, 0.);
		
		std::cout << std::endl << "Transformed G-R Diagnostic:";
		sampler.calc_GR_transformed(GR_transf, &transf);
		for(unsigned int k=0; k<ndim; k++) {
			std::cout << "  " << std::setprecision(3) << GR_transf[k];
		}
		std::cout << std::endl << std::endl;
		
		converged = true;
		for(size_t i=0; i<ndim; i++) {
			if(GR_transf[i] > GR_threshold) {
				converged = false;
				if(attempt != max_attempts-1) {
					sampler.print_stats();
					std::cerr << "# Extending run ..." << std::endl;
					sampler.step(int(N_steps*1./5.), false, 0., 1., 0.);
					sampler.clear();
					//logger.clear();
				}
				break;
			}
		}
	}
	
	clock_gettime(CLOCK_MONOTONIC, &t_write);
	
	std::stringstream group_name;
	group_name << "/pixel " << healpix_index;
	TChain chain = sampler.get_chain();
	
	TChainWriteBuffer writeBuffer(ndim, 500, 1);
	writeBuffer.add(chain, converged);
	writeBuffer.write(out_fname, group_name.str(), "los");
	
	std::stringstream los_group_name;
	los_group_name << group_name.str() << "/los";
	H5Utils::add_watermark<double>(out_fname, los_group_name.str(), "DM_min", params.img_stack->rect->min[0]);
	H5Utils::add_watermark<double>(out_fname, los_group_name.str(), "DM_max", params.img_stack->rect->max[0]);
	
	clock_gettime(CLOCK_MONOTONIC, &t_end);
	
	sampler.print_stats();
	std::cout << std::endl;
	
	if(!converged) {
		std::cerr << "# Failed to converge." << std::endl;
	}
	std::cerr << "# Number of steps: " << (1<<(attempt-1))*N_steps << std::endl;
	std::cerr << "# Time elapsed: " << std::setprecision(2) << (t_end.tv_sec - t_start.tv_sec) + 1.e-9*(t_end.tv_nsec - t_start.tv_nsec) << " s" << std::endl;
	std::cerr << "# Sample time: " << std::setprecision(2) << (t_write.tv_sec - t_start.tv_sec) + 1.e-9*(t_write.tv_nsec - t_start.tv_nsec) << " s" << std::endl;
	std::cerr << "# Write time: " << std::setprecision(2) << (t_end.tv_sec - t_write.tv_sec) + 1.e-9*(t_end.tv_nsec - t_write.tv_nsec) << " s" << std::endl << std::endl;
}


void los_integral(TImgStack &img_stack, const double *const subpixel, double *const ret,
                                        const double *const logEBV, unsigned int N_regions) {
	assert(img_stack.rect->N_bins[0] % N_regions == 0);
	
	unsigned int N_samples = img_stack.rect->N_bins[0] / N_regions;
	int y_max = img_stack.rect->N_bins[1];
	
	int x = 0;
	double y = exp(logEBV[0]) / img_stack.rect->dx[1];
	double y_0 = -img_stack.rect->min[1] / img_stack.rect->dx[1];
	double y_ceil, y_floor, dy, y_scaled;
	
	for(size_t i=0; i<img_stack.N_images; i++) { ret[i] = 0.; }
	
	for(int i=1; i<N_regions+1; i++) {
		dy = (double)(exp(logEBV[i])) / (double)(N_samples) / img_stack.rect->dx[1];
		//std::cout << "(" << x << ", " << y << ", " << tmp << ") ";
		for(int j=0; j<N_samples; j++, x++, y+=dy) {
			for(int k=0; k<img_stack.N_images; k++) {
				y_scaled = y_0 + y * subpixel[k];
				y_floor = floor(y_scaled);
				y_ceil = y_floor + 1.;
				if( ((int)y_floor >= 0) && ((int)y_ceil < y_max) ) {
					ret[k] += (y_ceil - y_scaled) * img_stack.img[k]->at<double>(x, (int)y_floor)
					           + (y_scaled - y_floor) * img_stack.img[k]->at<double>(x, (int)y_ceil);
				}
			}
		}
		
		//if((int)y_ceil >= y_max) { break; }
		//if((int)y_floor < 0) { break; }
	}
}

double lnp_los_extinction(const double *const logEBV, unsigned int N, TLOSMCMCParams& params) {
	double lnp = 0.;
	
	// Extinction must not exceed maximum value
	double EBV_tot = 0.;
	double EBV_tmp;
	for(size_t i=0; i<N; i++) {
		EBV_tmp = exp(logEBV[i]);
		EBV_tot += EBV_tmp;
		
		// Prior to prevent EBV from straying high
		//lnp -= 0.5 * (EBV_tmp * EBV_tmp) / (10. * 10.);
	}
	if(EBV_tot * params.subpixel_max >= params.img_stack->rect->max[1]) { return neg_inf_replacement; }
	
	// Prior on total extinction
	if((params.EBV_max > 0.) && (EBV_tot > params.EBV_max)) {
		lnp -= (EBV_tot - params.EBV_max) * (EBV_tot - params.EBV_max) / (2. * 0.05 * 0.05 * params.EBV_max * params.EBV_max);
	}
	
	// Wide Gaussian prior on logEBV to prevent fit from straying drastically
	const double bias = -5.;
	const double sigma = 25.;
	for(size_t i=0; i<N; i++) {
		lnp -= (logEBV[i] - bias) * (logEBV[i] - bias) / (2. * sigma * sigma);
	}
	
	// Compute line integrals through probability surfaces
	double *line_int = params.get_line_int(omp_get_thread_num());
	los_integral(*(params.img_stack), params.subpixel.data(), line_int, logEBV, N-1);
	
	// Soften and multiply line integrals
	for(size_t i=0; i<params.img_stack->N_images; i++) {
		if(line_int[i] < 1.e5*params.p0) {
			line_int[i] += params.p0 * exp(-line_int[i]/params.p0);
		}
		lnp += log(line_int[i]);
	}
	
	return lnp;
}

void gen_rand_los_extinction(double *const logEBV, unsigned int N, gsl_rng *r, TLOSMCMCParams &params) {
	double EBV_ceil = params.img_stack->rect->max[1] / params.subpixel_max;
	double mu = log(1.5 * params.EBV_guess_max / params.subpixel_max / (double)N);
	double EBV_sum = 0.;
	for(size_t i=0; i<N; i++) {
		logEBV[i] = mu + gsl_ran_gaussian_ziggurat(r, 2.5);
		EBV_sum += exp(logEBV[i]);
	}
	
	// Ensure that reddening is not more than allowed
	if(EBV_sum >= 0.95 * EBV_ceil) {
		double factor = log(0.95 * EBV_ceil / EBV_sum);
		for(size_t i=0; i<N; i++) {
			logEBV[i] += factor;
		}
	}
}

// Guess upper limit for E(B-V) based on stacked probability surfaces
double guess_EBV_max(TImgStack &img_stack) {
	cv::Mat stack, row_avg, col_avg;
	
	// Stack images
	img_stack.stack(stack);
	
	// Sum across each EBV
	cv::reduce(stack, row_avg, 0, CV_REDUCE_AVG);
	double max_sum = *std::max_element(row_avg.begin<double>(), row_avg.end<double>());
	int max = 1;
	for(int i = row_avg.cols - 1; i > 0; i--) {
		if(row_avg.at<double>(0, i) > 0.001 * max_sum) {
			max = i;
			break;
		}
	}
	
	// Convert bin index to E(B-V)
	return max * img_stack.rect->dx[1] + img_stack.rect->min[1];
}

void guess_EBV_profile(TMCMCOptions &options, TLOSMCMCParams &params, unsigned int N_regions) {
	TNullLogger logger;
	
	unsigned int N_steps = options.steps / 8;
	if(N_steps < 40) { N_steps = 40; }
	unsigned int N_samplers = options.samplers;
	unsigned int N_threads = options.N_threads;
	unsigned int ndim = N_regions + 1;
	
	TAffineSampler<TLOSMCMCParams, TNullLogger>::pdf_t f_pdf = &lnp_los_extinction;
	TAffineSampler<TLOSMCMCParams, TNullLogger>::rand_state_t f_rand_state = &gen_rand_los_extinction;
	
	std::cout << "Generating Guess ..." << std::endl;
	
	TParallelAffineSampler<TLOSMCMCParams, TNullLogger> sampler(f_pdf, f_rand_state, ndim, N_samplers*ndim, params, logger, N_threads);
	sampler.set_scale(1.05);
	sampler.set_replacement_bandwidth(0.75);
	
	sampler.step(int(N_steps*30./100.), true, 0., 0.5, 0.);
	sampler.step(int(N_steps*20./100), true, 0., 1., 0., true);
	sampler.step(int(N_steps*30./100.), true, 0., 0.5, 0.);
	sampler.step(int(N_steps*20./100), true, 0., 1., 0., true);
	
	sampler.print_stats();
	std::cout << std::endl << std::endl;
	
	sampler.get_chain().get_best(params.EBV_prof_guess);
	for(size_t i=0; i<ndim; i++) {
		//params.EBV_prof_guess[i] = log(params.EBV_prof_guess[i]);
		std::cout << "\t" << params.EBV_prof_guess[i] << std::endl;
	}
	std::cout << std::endl;
}


struct TEBVGuessParams {
	std::vector<double> EBV;
	std::vector<double> sigma_EBV;
	std::vector<double> sum_weight;
	double EBV_max, EBV_ceil;
	
	TEBVGuessParams(std::vector<double>& _EBV, std::vector<double>& _sigma_EBV, std::vector<double>& _sum_weight, double _EBV_ceil)
		: EBV(_EBV.size()), sigma_EBV(_sigma_EBV.size()), sum_weight(_sum_weight.size())
	{
		assert(_EBV.size() == _sigma_EBV.size());
		assert(_sum_weight.size() == _sigma_EBV.size());
		std::copy(_EBV.begin(), _EBV.end(), EBV.begin());
		std::copy(_sigma_EBV.begin(), _sigma_EBV.end(), sigma_EBV.begin());
		std::copy(_sum_weight.begin(), _sum_weight.end(), sum_weight.begin());
		EBV_max = -1.;
		for(unsigned int i=0; i<EBV.size(); i++) {
			if(EBV[i] > EBV_max) { EBV_max = EBV[i]; }
		}
		EBV_ceil = _EBV_ceil;
	}
};

double lnp_monotonic_guess(const double* Delta_EBV, unsigned int N, TEBVGuessParams& params) {
	double lnp = 0;
	
	double EBV = 0.;
	double tmp;
	for(unsigned int i=0; i<N; i++) {
		if(Delta_EBV[i] < 0.) { return neg_inf_replacement; }
		EBV += Delta_EBV[i];
		if(params.sum_weight[i] > 1.e-10) {
			tmp = (EBV - params.EBV[i]) / params.sigma_EBV[i];
			lnp -= 0.5 * tmp * tmp; //params.sum_weight[i] * tmp * tmp;
		}
	}
	
	return lnp;
}

void gen_rand_monotonic(double *const Delta_EBV, unsigned int N, gsl_rng *r, TEBVGuessParams &params) {
	double EBV_sum = 0.;
	double mu = 2. * params.EBV_max / (double)N;
	for(size_t i=0; i<N; i++) {
		Delta_EBV[i] = mu * gsl_rng_uniform(r);
		EBV_sum += Delta_EBV[i];
	}
	
	// Ensure that reddening is not more than allowed
	if(EBV_sum >= 0.95 * params.EBV_ceil) {
		double factor = EBV_sum / (0.95 * params.EBV_ceil);
		for(size_t i=0; i<N; i++) { Delta_EBV[i] *= factor; }
	}
}

void monotonic_guess(TImgStack &img_stack, unsigned int N_regions, std::vector<double>& Delta_EBV, TMCMCOptions& options) {
	std::cout << "stacking images" << std::endl;
	// Stack images
	cv::Mat stack;
	img_stack.stack(stack);
	
	std::cout << "calculating weighted mean at each distance" << std::endl;
	// Weighted mean of each distance
	double * dist_y_sum = new double[stack.rows];
	double * dist_y2_sum = new double[stack.rows];
	double * dist_sum = new double[stack.rows];
	for(int k = 0; k < stack.rows; k++) {
		dist_y_sum[k] = 0.;
		dist_y2_sum[k] = 0.;
		dist_sum[k] = 0.;
	}
	double y = 0.5;
	for(int j = 0; j < stack.cols; j++, y += 1.) {
		for(int k = 0; k < stack.rows; k++) {
			dist_y_sum[k] += y * stack.at<double>(k,j);
			dist_y2_sum[k] += y*y * stack.at<double>(k,j);
			dist_sum[k] += stack.at<double>(k,j);
		}
	}
	
	for(int k = 0; k < stack.rows; k++) {
		std::cout << k << "\t" << dist_y_sum[k]/dist_sum[k] << "\t" << sqrt(dist_y2_sum[k]/dist_sum[k]) << "\t" << dist_sum[k] << std::endl;
	}
	
	std::cout << "calculating weighted mean about each anchor" << std::endl;
	// Weighted mean in region of each anchor point
	std::vector<double> y_sum(N_regions+1, 0.);
	std::vector<double> y2_sum(N_regions+1, 0.);
	std::vector<double> w_sum(N_regions+1, 0.);
	int kStart = 0;
	int kEnd;
	double width = (double)(stack.rows) / (double)(N_regions);
	for(int n = 0; n < N_regions+1; n++) {
		std::cout << "n = " << n << std::endl;
		if(n == N_regions) {
			kEnd = stack.rows;
		} else {
			kEnd = ceil(((double)n + 0.5) * width);
		}
		for(int k = kStart; k < kEnd; k++) {
			y_sum[n] += dist_y_sum[k];
			y2_sum[n] += dist_y2_sum[k];
			w_sum[n] += dist_sum[k];
		}
		kStart = kEnd + 1;
	}
	
	delete[] dist_sum;
	delete[] dist_y_sum;
	delete[] dist_y2_sum;
	
	std::cout << "Covert to EBV and sigma_EBV" << std::endl;
	// Create non-monotonic guess
	Delta_EBV.resize(N_regions+1);
	std::vector<double> sigma_EBV(N_regions+1, 0.);
	for(int i=0; i<N_regions+1; i++) { Delta_EBV[i] = 0; }
	for(int n = 0; n < N_regions+1; n++) {
		Delta_EBV[n] = img_stack.rect->min[1] + img_stack.rect->dx[1] * y_sum[n] / w_sum[n];
		sigma_EBV[n] = img_stack.rect->dx[1] * sqrt( (y2_sum[n] - (y_sum[n] * y_sum[n] / w_sum[n])) / w_sum[n] );
		std::cout << n << "\t" << Delta_EBV[n] << "\t+-" << sigma_EBV[n] << std::endl;
	}
	
	// Fit monotonic guess
	unsigned int N_steps = 100;
	unsigned int N_samplers = 2 * N_regions;
	unsigned int N_threads = options.N_threads;
	unsigned int ndim = N_regions + 1;
	
	std::cout << "Setting up params" << std::endl;
	TEBVGuessParams params(Delta_EBV, sigma_EBV, w_sum, img_stack.rect->max[1]);
	TNullLogger logger;
	
	TAffineSampler<TEBVGuessParams, TNullLogger>::pdf_t f_pdf = &lnp_monotonic_guess;
	TAffineSampler<TEBVGuessParams, TNullLogger>::rand_state_t f_rand_state = &gen_rand_monotonic;
	
	std::cout << "Setting up sampler" << std::endl;
	TParallelAffineSampler<TEBVGuessParams, TNullLogger> sampler(f_pdf, f_rand_state, ndim, N_samplers*ndim, params, logger, N_threads);
	sampler.set_scale(1.1);
	sampler.set_replacement_bandwidth(0.75);
	
	std::cout << "Stepping" << std::endl;
	sampler.step(int(N_steps*40./100.), true, 0., 0.5, 0.);
	sampler.step(int(N_steps*10./100), true, 0., 1., 0.);
	sampler.step(int(N_steps*40./100.), true, 0., 0.5, 0.);
	sampler.step(int(N_steps*10./100), true, 0., 1., 0., true);
	
	sampler.print_stats();
	
	std::cout << "Getting best value" << std::endl;
	Delta_EBV.clear();
	sampler.get_chain().get_best(Delta_EBV);
	
	std::cout << "Monotonic guess" << std::endl;
	double EBV_sum = 0.;
	for(size_t i=0; i<Delta_EBV.size(); i++) {
		EBV_sum += Delta_EBV[i];
		std::cout << EBV_sum << std::endl;
		Delta_EBV[i] = log(Delta_EBV[i]);
	}
	std::cout << std::endl;
}



void gen_rand_los_extinction_from_guess(double *const logEBV, unsigned int N, gsl_rng *r, TLOSMCMCParams &params) {
	assert(params.EBV_prof_guess.size() == N);
	double EBV_ceil = params.img_stack->rect->max[1];
	double EBV_sum = 0.;
	for(size_t i=0; i<N; i++) {
		logEBV[i] = params.EBV_prof_guess[i] + gsl_ran_gaussian_ziggurat(r, 2.5);
		EBV_sum += logEBV[i];
	}
	
	// Switch adjacent reddenings
	int n_switches = gsl_rng_uniform_int(r, 3);
	size_t k;
	double tmp_log_EBV;
	int max_dist = std::min((int)(N-1)/2, 5);
	for(int i=0; i<n_switches; i++) {
		int dist = gsl_rng_uniform_int(r, max_dist+1);
		k = gsl_rng_uniform_int(r, N-dist);
		tmp_log_EBV = logEBV[k];
		logEBV[k] = logEBV[k+dist];
		logEBV[k+dist] = tmp_log_EBV;
	}
	
	// Ensure that reddening is not more than allowed
	if(EBV_sum >= 0.95 * EBV_ceil) {
		double factor = log(0.95 * EBV_ceil / EBV_sum);
		for(size_t i=0; i<N; i++) {
			logEBV[i] += factor;
		}
	}
}


/****************************************************************************************************************************
 * 
 * TLOSMCMCParams
 * 
 ****************************************************************************************************************************/

TLOSMCMCParams::TLOSMCMCParams(TImgStack* _img_stack, double _p0,
                               unsigned int _N_threads, double _EBV_max)
	: img_stack(_img_stack), subpixel(_img_stack->N_images, 1.), N_threads(_N_threads), line_int(NULL)
{
	line_int = new double[_img_stack->N_images * N_threads];
	std::cout << "Allocated line_int[" << _img_stack->N_images * N_threads << "] (" << _img_stack->N_images << " images, " << N_threads << " threads)" << std::endl;
	p0 = _p0;
	lnp0 = log(p0);
	EBV_max = _EBV_max;
	EBV_guess_max = guess_EBV_max(*img_stack);
	subpixel_max = 1.;
	subpixel_min = 1.;
}

TLOSMCMCParams::~TLOSMCMCParams() {
	if(line_int != NULL) { delete[] line_int; }
}

void TLOSMCMCParams::set_p0(double _p0) {
	p0 = _p0;
	lnp0 = log(p0);
}

void TLOSMCMCParams::set_subpixel_mask(TStellarData& data) {
	assert(data.star.size() == img_stack->N_images);
	subpixel.clear();
	subpixel_max = 0.;
	subpixel_min = inf_replacement;
	double EBV;
	for(size_t i=0; i<data.star.size(); i++) {
		EBV = data.star[i].EBV;
		if(EBV > subpixel_max) { subpixel_max = EBV; }
		if(EBV < subpixel_min) { subpixel_min = EBV; }
		subpixel.push_back(EBV);
	}
}

void TLOSMCMCParams::set_subpixel_mask(std::vector<double>& new_mask) {
	assert(new_mask.size() == img_stack->N_images);
	subpixel.clear();
	subpixel_max = 0.;
	subpixel_min = inf_replacement;
	for(size_t i=0; i<new_mask.size(); i++) {
		if(new_mask[i] > subpixel_max) { subpixel_max = new_mask[i]; }
		if(new_mask[i] < subpixel_min) { subpixel_min = new_mask[i]; }
		subpixel.push_back(new_mask[i]);
	}
}

double* TLOSMCMCParams::get_line_int(unsigned int thread_num) {
	assert(thread_num < N_threads);
	return line_int + img_stack->N_images * thread_num;
}




/****************************************************************************************************************************
 * 
 * TImgStack
 * 
 ****************************************************************************************************************************/

TImgStack::TImgStack(size_t _N_images) {
	N_images = _N_images;
	img = new cv::Mat*[N_images];
	for(size_t i=0; i<N_images; i++) {
		img[i] = new cv::Mat;
	}
	rect = NULL;
}

TImgStack::TImgStack(size_t _N_images, TRect& _rect) {
	N_images = _N_images;
	img = new cv::Mat*[N_images];
	for(size_t i=0; i<N_images; i++) { img[i] = NULL; }
	rect = new TRect(_rect);
}

TImgStack::~TImgStack() {
	if(img != NULL) {
		for(size_t i=0; i<N_images; i++) {
			if(img[i] != NULL) {
				delete img[i];
			}
		}
		delete[] img;
	}
	if(rect != NULL) { delete rect; }
}

void TImgStack::resize(size_t _N_images) {
	if(img != NULL) {
		for(size_t i=0; i<N_images; i++) {
			if(img[i] != NULL) {
				delete img[i];
			}
		}
		delete[] img;
	}
	if(rect != NULL) { delete rect; }
	
	N_images = _N_images;
	img = new cv::Mat*[N_images];
	for(size_t i=0; i<N_images; i++) {
		img[i] = new cv::Mat;
	}
}

void TImgStack::cull(const std::vector<bool> &keep) {
	assert(keep.size() == N_images);
	
	size_t N_tmp = 0;
	for(std::vector<bool>::const_iterator it = keep.begin(); it != keep.end(); ++it) {
		if(*it) { N_tmp++; }
	}
	
	cv::Mat **img_tmp = new cv::Mat*[N_tmp];
	size_t i = 0;
	size_t k = 0;
	for(std::vector<bool>::const_iterator it = keep.begin(); it != keep.end(); ++it, ++i) {
		if(*it) {
			img_tmp[k] = img[i];
			k++;
		} else {
			delete img[i];
		}
	}
	
	delete[] img;
	img = img_tmp;
	N_images = N_tmp;
}

void TImgStack::set_rect(TRect& _rect) {
	if(rect == NULL) {
		rect = new TRect(_rect);
	} else {
		*rect = _rect;
	}
}

void TImgStack::stack(cv::Mat& dest) {
	if(N_images > 0) {
		dest = *(img[0]);
		for(size_t i=1; i<N_images; i++) {
			dest += *(img[i]);
		}
	} else {
		dest.setTo(0);
	}
}

TLOSTransform::TLOSTransform(unsigned int ndim)
	: TTransformParamSpace(ndim), _ndim(ndim)
{}

TLOSTransform::~TLOSTransform()
{}

void TLOSTransform::transform(const double *const x, double *const y) {
	y[0] = exp(x[0]);
	for(unsigned int i=1; i<_ndim; i++) {
		y[i] = y[i-1] + exp(x[i]);
	}
}

TLOSCloudTransform::TLOSCloudTransform(unsigned int ndim)
	: TTransformParamSpace(ndim), _ndim(ndim)
{
	assert(!(ndim & 1));
	n_clouds = ndim / 2;
}

TLOSCloudTransform::~TLOSCloudTransform()
{}

void TLOSCloudTransform::transform(const double *const x, double *const y) {
	y[0] = x[0];
	y[n_clouds] = exp(x[n_clouds]);
	for(unsigned int i=1; i<n_clouds; i++) {
		y[i] = x[i];
		y[n_clouds+i] = exp(x[n_clouds+i]);
	}
}
