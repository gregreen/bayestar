/*
 * sampler.cpp
 * 
 * Samples from posterior distribution of line-of-sight model.
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


#include "sampler.h"


/****************************************************************************************************************************
 * 
 * TMCMCParams
 * 
 ****************************************************************************************************************************/

TMCMCParams::TMCMCParams(TGalacticLOSModel *_gal_model, TSyntheticStellarModel *_synth_stellar_model, TStellarModel *_emp_stellar_model,
			  TExtinctionModel *_ext_model, TStellarData *_data, double _EBV_SFD, unsigned int _N_DM, double _DM_min, double _DM_max)
	: gal_model(_gal_model), synth_stellar_model(_synth_stellar_model), emp_stellar_model(_emp_stellar_model),
          ext_model(_ext_model), data(_data), EBV_SFD(_EBV_SFD), N_DM(_N_DM), DM_min(_DM_min), DM_max(_DM_max)
{
	N_stars = data->star.size();
	EBV_interp = new TLinearInterp(DM_min, DM_max, N_DM);
	
	// Defaults
	lnp0 = -10.;
	idx_star = 0;
	
	EBV_floor = 0.;
	
	vary_RV = false;
	RV_mean = 3.1;
	RV_variance = 0.2*0.2;
}

TMCMCParams::~TMCMCParams() {
	delete EBV_interp;
}


void TMCMCParams::update_EBV_interp(const double *x) {
	double EBV = 0.;
	for(unsigned int i=0; i<N_DM; i++) {
		EBV += exp(x[1+i]);
		(*EBV_interp)[i] = EBV;
		if(i == 0) { EBV_min = EBV; }
	}
	EBV_max = EBV;
}

double TMCMCParams::get_EBV(double DM) {
	if(DM <= DM_min) { return EBV_min; } else if(DM >= DM_max) { return EBV_max; }
	return (*EBV_interp)(DM);
}



/****************************************************************************************************************************
 * 
 * Probability density functions
 * 
 ****************************************************************************************************************************/


// Natural logarithm of posterior probability density for one star, given parameters x, where
//
//     x = {DM, Log_10(Mass_init), Log_10(Age), [Fe/H]}
double logP_single_star_synth(const double *x, double EBV, double RV,
                              const TGalacticLOSModel &gal_model, const TSyntheticStellarModel &stellar_model,
                              TExtinctionModel &ext_model, const TStellarData::TMagnitudes &d, TSED *tmp_sed) {
	#define neginf -std::numeric_limits<double>::infinity()
	double logP = 0.;
	
	/*
	 *  Likelihood
	 */
	bool del_sed = false;
	if(tmp_sed == NULL) {
		del_sed = true;
		tmp_sed = new TSED(true);
	}
	if(!stellar_model.get_sed(x+1, *tmp_sed)) {
		if(del_sed) { delete tmp_sed; }
		return neginf;
	}
	
	double logL = 0.;
	double tmp;
	for(unsigned int i=0; i<NBANDS; i++) {
		tmp = d.m[i] - x[_DM] - EBV * ext_model.get_A(RV, i);	// De-reddened absolute magnitude
		tmp = (tmp_sed->absmag[i] - tmp) / d.err[i];
		logL -= 0.5*tmp*tmp;
	}
	logP += logL - d.lnL_norm;
	
	if(del_sed) { delete tmp_sed; }
	
	/*
	 *  Priors
	 */
	logP += gal_model.log_prior_synth(x);
	
	//double lnp0 = -100.;
	//tmp = exp(logP - lnp0);
	//logP = lnp0 + log(tmp + exp(-tmp));	// p --> p + p0 exp(-p/p0)  (Smooth floor on outliers)
	
	#undef neginf
	return logP;
}

// Natural logarithm of posterior probability density for one star, given parameters x, where
//
//     x = {DM, M_r, [Fe/H]}
double logP_single_star_emp(const double *x, double EBV, double RV,
                            const TGalacticLOSModel &gal_model, const TStellarModel &stellar_model,
                            TExtinctionModel &ext_model, const TStellarData::TMagnitudes &d, TSED *tmp_sed) {
	#define neginf -std::numeric_limits<double>::infinity()
	double logP = 0.;
	
	/*
	 *  Likelihood
	 */
	bool del_sed = false;
	if(tmp_sed == NULL) {
		del_sed = true;
		tmp_sed = new TSED(true);
	}
	if(!stellar_model.get_sed(x+1, *tmp_sed)) {
		if(del_sed) { delete tmp_sed; }
		return neginf;
	}
	
	double logL = 0.;
	double tmp;
	for(unsigned int i=0; i<NBANDS; i++) {
		tmp = d.m[i] - x[_DM] - EBV * ext_model.get_A(RV, i);	// De-reddened absolute magnitude
		tmp = (tmp_sed->absmag[i] - tmp) / d.err[i];
		logL -= 0.5*tmp*tmp;
	}
	logP += logL - d.lnL_norm;
	
	if(del_sed) { delete tmp_sed; }
	
	/*
	 *  Priors
	 */
	logP += gal_model.log_prior_emp(x) + stellar_model.get_log_lf(x[1]);
	
	//double lnp0 = -100.;
	//tmp = exp(logP - lnp0);
	//logP = lnp0 + log(tmp + exp(-tmp));	// p --> p + p0 exp(-p/p0)  (Smooth floor on outliers)
	
	#undef neginf
	return logP;
}


double logP_EBV(TMCMCParams &p) {
	double logP = 0.;
	
	// SFD prior
	logP -= (p.EBV_max * p.EBV_max) / (2. * p.EBV_SFD * p.EBV_SFD);
	
	// Lognormal
	double tmp;
	for(double DM = p.DM_min; DM <= p.DM_max; DM += (p.DM_max-p.DM_min)/10.) {
		tmp = p.get_EBV(DM);
		logP -= tmp*tmp / 2.;
	}
	
	return logP;
}

double logP_los_synth(const double *x, unsigned int N, TMCMCParams &p, double *lnP_star) {
	double logP = 0.;
	
	// Prior on RV
	double RV = x[0];
	if(p.ext_model->in_model(RV)) {
		logP -= (RV - 3.1)*(RV - 3.1) / (2. * 0.1 * 0.1);
	} else {
		return -std::numeric_limits<double>::infinity();
	}
	
	// Prior on extinction
	logP += logP_EBV(p);
	
	// Probabilities of stars
	double tmp;
	const double *x_star;
	for(unsigned int i=0; i<p.N_stars; i++) {
		x_star = &(x[1 + p.N_DM + 4*i]);
		tmp = logP_single_star_synth(x_star, p.get_EBV(x_star[_DM]), RV, *p.gal_model, *p.synth_stellar_model, *p.ext_model, p.data->star[i]);
		logP += tmp;
		if(lnP_star != NULL) { lnP_star[i] = tmp; }
	}
	
	return logP;
}



/****************************************************************************************************************************
 * 
 * Sampling functions
 * 
 ****************************************************************************************************************************/

void sample_model_synth(TGalacticLOSModel &galactic_model, TSyntheticStellarModel &stellar_model, TExtinctionModel &extinction_model, TStellarData &stellar_data, double EBV_SFD) {
	unsigned int N_DM = 20;
	double DM_min = 5.;
	double DM_max = 20.;
	TMCMCParams params(&galactic_model, &stellar_model, NULL, &extinction_model, &stellar_data, EBV_SFD, N_DM, DM_min, DM_max);
	TMCMCParams params_tmp(&galactic_model, &stellar_model, NULL, &extinction_model, &stellar_data, EBV_SFD, N_DM, DM_min, DM_max);
	
	// Random number generator
	gsl_rng *r;
	seed_gsl_rng(&r);
	
	// Vector describing position in probability space
	size_t length = 1 + params.N_DM + 4*params.N_stars;
	// x = {RV, Delta_EBV_1, ..., Delta_EBV_M, Theta_1, ..., Theta_N}, where Theta = {DM, logMass, logtau, FeH}.
	double *x = new double[length];
	
	// Random starting point for reddening profile
	x[0] = 3.1;// + gsl_ran_gaussian_ziggurat(r, 0.2);	// RV
	for(size_t i=0; i<params.N_DM; i++) { x[i+1] = EBV_SFD/(double)N_DM * gsl_ran_chisq(r, 1.); }		// Delta_EBV
	
	// Random starting point for each star
	TSED sed_tmp(true);
	for(size_t i = 1 + params.N_DM; i < 1 + params.N_DM + 4*params.N_stars; i += 4) {
		x[i] = 5. + 13.*gsl_rng_uniform(r);
		double logMass, logtau, FeH, tau;
		bool in_lib = false;
		while(!in_lib) {
			logMass = gsl_ran_gaussian_ziggurat(r, 0.5);
			tau = -1.;
			while(tau <= 0.) {
				tau = 1.e9 * (5. + gsl_ran_gaussian_ziggurat(r, 2.));
			}
			logtau = log10(tau);
			FeH = -1.0 + gsl_ran_gaussian_ziggurat(r, 1.);
			
			in_lib = stellar_model.get_sed(logMass, logtau, FeH, sed_tmp);
		}
		x[i+1] = logMass;
		x[i+2] = logtau;
		x[i+3] = FeH;
	}
	
	params.update_EBV_interp(x);
	double *lnp_star = new double[params.N_stars];
	double lnp_los = logP_los_synth(x, length, params, lnp_star);
	std::cerr << "# ln p(x_0) = " << lnp_los << std::endl;
	
	double *x_tmp = new double[length];
	double Theta_tmp[4];
	double sigma_Theta[4] = {0.1, 0.1, 0.1, 0.1};
	double sigma_RV = 0.05;
	double sigma_lnEBV = 0.1;
	double lnp_tmp;
	double *lnp_star_tmp = new double[params.N_stars];
	double p;
	
	unsigned int N_steps = 1000000;
	
	TChain chain(length, N_steps);
	TStats EBV_stats(N_DM);
	
	// In each step
	unsigned int N_star = 0;
	unsigned int N_accept_star = 0;
	unsigned int N_los = 0;
	unsigned int N_accept_los = 0;
	bool accept;
	bool burn_in = true;
	for(unsigned int i=0; i<N_steps; i++) {
		if(i == N_steps/2) {
			sigma_Theta[0] = 0.05;
			sigma_Theta[1] = 0.05;
			sigma_Theta[2] = 0.05;
			sigma_Theta[3] = 0.05;
			sigma_RV = 0.005;
			sigma_lnEBV = 0.05;
			burn_in = false;
		}
		
		// Step each star
		for(unsigned int n=0; n<params.N_stars; n++) {
			if(!burn_in) { N_star++; }
			
			rand_gaussian_vector(&Theta_tmp[0], &x[1+N_DM+4*n], &sigma_Theta[0], 4, r);
			lnp_tmp = logP_single_star_synth(&Theta_tmp[0], params.get_EBV(Theta_tmp[_DM]), x[0], galactic_model, stellar_model, extinction_model, stellar_data.star[n]);
			
			accept = false;
			if(lnp_tmp > lnp_star[n]) {
				accept = true;
			} else {
				p = gsl_rng_uniform(r);
				if((p > 0.) && (log(p) < lnp_tmp - lnp_star[n])) {
					accept = true;
				}
			}
			
			if(accept) {
				if(!burn_in) { N_accept_star++; }
				for(size_t k=0; k<4; k++) { x[1+N_DM+4*n+k] = Theta_tmp[k]; }
				lnp_los += lnp_tmp - lnp_star[n];
				lnp_star[n] = lnp_tmp;
			}
		}
		
		// Step reddening profile
		if(!burn_in) { N_los++; }
		for(size_t k=0; k<length; k++) { x_tmp[k] = x[k]; }
		//if(!burn_in) { x_tmp[0] += gsl_ran_gaussian_ziggurat(r, sigma_RV); }
		for(unsigned int m=0; m<params.N_DM; m++) { x_tmp[1+m] += gsl_ran_gaussian_ziggurat(r, sigma_lnEBV); }
		
		params_tmp.update_EBV_interp(x_tmp);
		lnp_tmp = logP_los_synth(x_tmp, length, params_tmp, lnp_star_tmp);
		//if(isinf(lnp_tmp)) {
		//	lnp_tmp = logP_los(x, length, params_tmp, lnp_star_tmp);
		//}
		//std::cerr << "#     ln p(y) = " << lnp_tmp << std::endl;
		
		accept = false;
		if(lnp_tmp > lnp_los) {
			accept = true;
		} else if(log(gsl_rng_uniform(r)) < lnp_tmp - lnp_los) {
			accept = true;
		}
		
		if(accept) {
			if(!burn_in) { N_accept_los++; }
			for(size_t k=0; k<1+N_DM; k++) { x[k] = x_tmp[k]; }
			for(size_t k=0; k<params.N_stars; k++) { lnp_star[k] = lnp_star_tmp[k]; }
			lnp_los = lnp_tmp;
			params.update_EBV_interp(x);
			//std::cerr << "# ln p(x) = " << lnp_los << std::endl;
		}
		
		if(!burn_in) {
			chain.add_point(x, lnp_los, 1.);
			
			x_tmp[0] = exp(x[1]);
			for(size_t k=1; k<N_DM; k++) {
				x_tmp[k] = x_tmp[k-1] + exp(x[k]);
			}
			EBV_stats(x_tmp, 1);
		}
	}
	
	std::cerr << "# ln p(x) = " << lnp_los << std::endl;
	std::cout.precision(4);
	std::cerr << std::endl;
	std::cerr << "# % acceptance: " << 100. * (double)N_accept_star / (double)N_star << " (stars)" << std::endl;
	std::cerr << "                " << 100. * (double)N_accept_los / (double)N_los << " (extinction)" << std::endl;
	std::cerr << "# R_V = " << x[0] << std::endl << std::endl;
	std::cerr << "#  DM   E(B-V)" << std::endl;
	std::cerr << "# =============" << std::endl;
	for(double DM=5.; DM<20.; DM+=1.) {
		std::cerr << "#  " << DM << " " << params.get_EBV(DM) << std::endl;
	}
	std::cerr << std::endl;
	EBV_stats.print();
	std::cerr << std::endl;
	
	delete[] x;
	delete[] x_tmp;
	delete[] lnp_star;
	delete[] lnp_star_tmp;
}

void gen_rand_state_synth(double *const x, unsigned int N, gsl_rng *r, TMCMCParams &params) {
	assert(N == 1 + params.N_DM + 4*params.N_stars);
	
	// R_V
	x[0] = 3.1 + gsl_ran_gaussian_ziggurat(r, 0.2);
	
	// Delta_EBV
	for(size_t i=0; i<params.N_DM; i++) { x[i+1] = params.EBV_SFD/(double)params.N_DM * gsl_ran_chisq(r, 1.); }
	
	// Stars
	TSED sed_tmp(true);
	for(size_t i = 1 + params.N_DM; i < 1 + params.N_DM + 4*params.N_stars; i += 4) {
		// DM
		x[i] = 5. + 13. * gsl_rng_uniform(r);
		
		// Stellar type
		double logMass, logtau, FeH, tau;
		bool in_lib = false;
		while(!in_lib) {
			logMass = gsl_ran_gaussian_ziggurat(r, 0.5);
			tau = -1.;
			while(tau <= 0.) {
				tau = 1.e9 * (5. + gsl_ran_gaussian_ziggurat(r, 2.));
			}
			logtau = log10(tau);
			FeH = -1.0 + gsl_ran_gaussian_ziggurat(r, 1.);
			
			in_lib = params.synth_stellar_model->get_sed(logMass, logtau, FeH, sed_tmp);
		}
		x[i+1] = logMass;
		x[i+2] = logtau;
		x[i+3] = FeH;
	}
}

double logP_los_simple_synth(const double *x, unsigned int N, TMCMCParams &params) {
	params.update_EBV_interp(x);
	return logP_los_synth(x, N, params, NULL);
}

void sample_model_affine_synth(TGalacticLOSModel &galactic_model, TSyntheticStellarModel &stellar_model, TExtinctionModel &extinction_model, TStellarData &stellar_data, double EBV_SFD) {
	unsigned int N_DM = 20;
	double DM_min = 5.;
	double DM_max = 20.;
	TMCMCParams params(&galactic_model, &stellar_model, NULL, &extinction_model, &stellar_data, EBV_SFD, N_DM, DM_min, DM_max);
	TStats EBV_stats(N_DM);
	
	unsigned int N_steps = 100;
	unsigned int N_samplers = 4;
	
	typename TAffineSampler<TMCMCParams, TStats>::pdf_t f_pdf = &logP_los_simple_synth;
	typename TAffineSampler<TMCMCParams, TStats>::rand_state_t f_rand_state = &gen_rand_state_synth;
	
	std::cerr << "# Setting up sampler" << std::endl;
	unsigned int ndim = 1 + params.N_DM + 4*params.N_stars;
	TParallelAffineSampler<TMCMCParams, TStats> sampler(f_pdf, f_rand_state, ndim, 10*ndim, params, EBV_stats, N_samplers);
	
	std::cerr << "# Burn-in" << std::endl;
	sampler.set_scale(1.1);
	sampler.set_replacement_bandwidth(0.5);
	sampler.step(N_steps, false, 0, 0.01, 0.);
	sampler.clear();
	std::cerr << "# Main run" << std::endl;
	sampler.step(N_steps, true, 0, 0.01, 0.);
	
	std::cout << "Sampler stats:" << std::endl;
	sampler.print_stats();
	std::cout << std::endl;
	
	std::cout << "E(B-V) statistics:" << std::endl;
	EBV_stats.print();
	std::cout << std::endl;
	
	/*
	std::cerr << "# ln p(x) = " << lnp_los << std::endl;
	std::cout.precision(4);
	std::cerr << std::endl;
	std::cerr << "# % acceptance: " << 100. * (double)N_accept_star / (double)N_star << " (stars)" << std::endl;
	std::cerr << "                " << 100. * (double)N_accept_los / (double)N_los << " (extinction)" << std::endl;
	std::cerr << "# R_V = " << x[0] << std::endl << std::endl;
	std::cerr << "#  DM   E(B-V)" << std::endl;
	std::cerr << "# =============" << std::endl;
	for(double DM=5.; DM<20.; DM+=1.) {
		std::cerr << "#  " << DM << " " << params.get_EBV(DM) << std::endl;
	}
	std::cerr << std::endl;
	EBV_stats.print();
	std::cerr << std::endl;
	*/
}

void gen_rand_state_indiv_synth(double *const x, unsigned int N, gsl_rng *r, TMCMCParams &params) {
	assert(N == 5);
	
	// Stars
	TSED sed_tmp(true);
	
	// E(B-V)
	x[0] = 1.5 * params.EBV_SFD * gsl_rng_uniform(r);
	
	// DM
	x[1] = 5. + 13. * gsl_rng_uniform(r);
	
	// Stellar type
	double logMass, logtau, FeH, tau;
	bool in_lib = false;
	while(!in_lib) {
		logMass = gsl_ran_gaussian_ziggurat(r, 0.5);
		//tau = -1.;
		//while(tau <= 0.) {
		//	tau = 1.e9 * (5. + gsl_ran_gaussian_ziggurat(r, 2.));
		//}
		//logtau = log10(tau);
		logtau = 8. + gsl_ran_gaussian_ziggurat(r, 1.);
		FeH = -1.0 + gsl_ran_gaussian_ziggurat(r, 1.);
		
		in_lib = params.synth_stellar_model->get_sed(logMass, logtau, FeH, sed_tmp);
	}
	x[2] = logMass;
	x[3] = logtau;
	x[4] = FeH;
	
	if(params.vary_RV) {
		double RV = -1.;
		while((RV <= 2.1) || (RV >= 5.)) {
			RV = params.RV_mean + gsl_ran_gaussian_ziggurat(r, 1.5*params.RV_variance*params.RV_variance);
		}
		x[5] = RV;
	}
}

void gen_rand_state_indiv_emp(double *const x, unsigned int N, gsl_rng *r, TMCMCParams &params) {
	if(params.vary_RV) { assert(N == 5); } else { assert(N == 4); }
	
	// Stars
	TSED sed_tmp(true);
	
	// E(B-V)
	x[0] = 1.5 * params.EBV_SFD * gsl_rng_uniform(r);
	
	// DM
	x[1] = 6. + 12. * gsl_rng_uniform(r);
	
	// Stellar type
	double Mr, FeH;
	Mr = -0.5 + 15.5 * gsl_rng_uniform(r);
	FeH = -2.45 + 2.4 * gsl_rng_uniform(r);
	
	x[2] = Mr;
	x[3] = FeH;
	
	if(params.vary_RV) {
		double RV = -1.;
		while((RV <= 2.1) || (RV >= 5.)) {
			RV = params.RV_mean + gsl_ran_gaussian_ziggurat(r, 1.5*params.RV_variance*params.RV_variance);
		}
		x[4] = RV;
	}
}

double logP_indiv_simple_synth(const double *x, unsigned int N, TMCMCParams &params) {
	if(x[0] < params.EBV_floor) { return -std::numeric_limits<double>::infinity(); }
	double RV;
	double logp = 0;
	if(params.vary_RV) {
		RV = x[5];
		if((RV <= 2.1) || (RV >= 5.)) {
			return -std::numeric_limits<double>::infinity();
		}
		logp = -0.5*(RV-params.RV_mean)*(RV-params.RV_mean)/params.RV_variance;
	} else {
		RV = params.RV_mean;
	}
	logp += logP_single_star_synth(x+1, x[0], RV, *params.gal_model, *params.synth_stellar_model, *params.ext_model, params.data->star[params.idx_star], NULL);
	return logp;
}

double logP_indiv_simple_emp(const double *x, unsigned int N, TMCMCParams &params) {
	if(x[0] < params.EBV_floor) { return -std::numeric_limits<double>::infinity(); }
	double RV;
	double logp = 0;
	if(params.vary_RV) {
		RV = x[4];
		if((RV <= 2.1) || (RV >= 5.)) {
			return -std::numeric_limits<double>::infinity();
		}
		logp = -0.5*(RV-params.RV_mean)*(RV-params.RV_mean)/params.RV_variance;
	} else {
		RV = params.RV_mean;
	}
	logp += logP_single_star_emp(x+1, x[0], RV, *params.gal_model, *params.emp_stellar_model, *params.ext_model, params.data->star[params.idx_star], NULL);
	return logp;
}

void sample_indiv_synth(TGalacticLOSModel &galactic_model, TSyntheticStellarModel &stellar_model, TExtinctionModel &extinction_model, TStellarData &stellar_data, double EBV_SFD, double RV_sigma) {
	unsigned int N_DM = 20;
	double DM_min = 5.;
	double DM_max = 20.;
	TMCMCParams params(&galactic_model, &stellar_model, NULL, &extinction_model, &stellar_data, EBV_SFD, N_DM, DM_min, DM_max);
	
	if(RV_sigma > 0.) {
		params.vary_RV = true;
		params.RV_variance = RV_sigma*RV_sigma;
	}
	
	std::string fname = "synth_out.hdf5";
	std::string dim_name[6] = {"E(B-V)", "DM", "LogMass", "Logtau", "FeH", "R_V"};
	
	//double min[5] = {0.0, 0.0, -1.0, 6.0, -2.5};
	//double max[5] = {10., 25.,  1.2, 11.,  0.5};
	//unsigned int N_bins[5] = {1000, 500, 100, 100, 100};
	//TSparseBinner logger(&min[0], &max[0], &N_bins[0], 5);
	TNullLogger logger;
	
	unsigned int max_attempts = 3;
	unsigned int N_steps = 1000;
	unsigned int N_samplers = 15;
	unsigned int N_threads = 4;
	unsigned int ndim;
	
	if(params.vary_RV) { ndim = 6; } else { ndim = 5; }
	
	double *GR = new double[ndim];
	double GR_threshold = 1.1;
	
	typename TAffineSampler<TMCMCParams, TNullLogger>::pdf_t f_pdf = &logP_indiv_simple_synth;
	typename TAffineSampler<TMCMCParams, TNullLogger>::rand_state_t f_rand_state = &gen_rand_state_indiv_synth;
	
	timespec t_start, t_write, t_end;
	//bool write_success;
	
	std::cerr << std::endl;
	std::remove(fname.c_str());
	
	for(size_t n=0; n<params.N_stars; n++) {
		params.idx_star = n;
		
		clock_gettime(CLOCK_MONOTONIC, &t_start);
		
		std::cout << "Star #" << n+1 << " of " << params.N_stars << std::endl;
		std::cout << "====================================" << std::endl;
		
		//std::cerr << "# Setting up sampler" << std::endl;
		TParallelAffineSampler<TMCMCParams, TNullLogger> sampler(f_pdf, f_rand_state, ndim, N_samplers*ndim, params, logger, N_threads);
		sampler.set_scale(1.2);
		sampler.set_replacement_bandwidth(0.2);
		
		//std::cerr << "# Burn-in" << std::endl;
		sampler.step(N_steps, false, 0., 0.2, 0.);
		sampler.clear();
		
		//std::cerr << "# Main run" << std::endl;
		bool converged = false;
		size_t attempt;
		for(attempt = 0; (attempt < max_attempts) && (!converged); attempt++) {
			sampler.step((1<<attempt)*N_steps, true, 0., 0.2, 0.);
			
			converged = true;
			sampler.get_GR_diagnostic(GR);
			for(size_t i=0; i<ndim; i++) {
				if(GR[i] > GR_threshold) {
					converged = false;
					if(attempt != max_attempts-1) {
						sampler.clear();
						//logger.clear();
					}
					break;
				}
			}
		}
		
		clock_gettime(CLOCK_MONOTONIC, &t_write);
		
		std::stringstream group_name;
		group_name << "/star " << n;
		//logger.write(fname, group_name.str(), dset_name, &dim_name[0], 1, 25000);
		std::stringstream dim_name_all;
		for(size_t i=0; i<ndim; i++) { dim_name_all << (i == 0 ? "" : " ") << dim_name[i]; }
		sampler.get_chain().save(fname, group_name.str(), dim_name_all.str(), 1, 1000, 5000);
		
		clock_gettime(CLOCK_MONOTONIC, &t_end);
		
		//std::cout << "Sampler stats:" << std::endl;
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
	
	delete[] GR;
}

void sample_indiv_emp(TGalacticLOSModel& galactic_model, TStellarModel& stellar_model,
                      TExtinctionModel& extinction_model, TStellarData& stellar_data,
                      double EBV_SFD, TImgStack& img_stack, double RV_sigma) {
	unsigned int N_DM = 20;
	double DM_min = 5.;
	double DM_max = 20.;
	TMCMCParams params(&galactic_model, NULL, &stellar_model, &extinction_model, &stellar_data, EBV_SFD, N_DM, DM_min, DM_max);
	
	if(RV_sigma > 0.) {
		params.vary_RV = true;
		params.RV_variance = RV_sigma*RV_sigma;
	}
	
	std::string fname = "emp_out.hdf5";
	std::string dim_name[5] = {"E(B-V)", "DM", "Mr", "FeH", "R_V"};
	
	double min[2] = {DM_min, 0.};
	double max[2] = {DM_max, 5.};
	unsigned int N_bins[2] = {120, 500};
	TRect rect(min, max, N_bins);
	
	img_stack.resize(params.N_stars);
	img_stack.set_rect(rect);
	
	TNullLogger logger;
	
	unsigned int max_attempts = 3;
	unsigned int N_steps = 500;
	unsigned int N_samplers = 15;
	unsigned int N_threads = 4;
	unsigned int ndim;
	
	if(params.vary_RV) { ndim = 5; } else { ndim = 4; }
	
	double *GR = new double[ndim];
	double GR_threshold = 1.1;
	
	typename TAffineSampler<TMCMCParams, TNullLogger>::pdf_t f_pdf = &logP_indiv_simple_emp;
	typename TAffineSampler<TMCMCParams, TNullLogger>::rand_state_t f_rand_state = &gen_rand_state_indiv_emp;
	
	timespec t_start, t_write, t_end;
	//bool write_success;
	
	std::cerr << std::endl;
	std::remove(fname.c_str());
	
	for(size_t n=0; n<params.N_stars; n++) {
		params.idx_star = n;
		
		clock_gettime(CLOCK_MONOTONIC, &t_start);
		
		std::cout << "Star #" << n+1 << " of " << params.N_stars << std::endl;
		std::cout << "====================================" << std::endl;
		
		//std::cerr << "# Setting up sampler" << std::endl;
		TParallelAffineSampler<TMCMCParams, TNullLogger> sampler(f_pdf, f_rand_state, ndim, N_samplers*ndim, params, logger, N_threads);
		sampler.set_scale(1.5);
		sampler.set_replacement_bandwidth(0.2);
		
		//std::cerr << "# Burn-in" << std::endl;
		sampler.step(N_steps, false, 0., 0.2, 0.);
		sampler.clear();
		
		//std::cerr << "# Main run" << std::endl;
		bool converged = false;
		size_t attempt;
		for(attempt = 0; (attempt < max_attempts) && (!converged); attempt++) {
			sampler.step((1<<attempt)*N_steps, true, 0., 0.2, 0.);
			
			converged = true;
			sampler.get_GR_diagnostic(GR);
			for(size_t i=0; i<ndim; i++) {
				if(GR[i] > GR_threshold) {
					converged = false;
					if(attempt != max_attempts-1) {
						sampler.clear();
						//logger.clear();
					}
					break;
				}
			}
		}
		
		clock_gettime(CLOCK_MONOTONIC, &t_write);
		
		std::stringstream group_name;
		group_name << "/star " << n;
		//logger.write(fname, group_name.str(), dset_name, &dim_name[0], 1, 25000);
		std::stringstream dim_name_all;
		for(size_t i=0; i<ndim; i++) { dim_name_all << (i == 0 ? "" : " ") << dim_name[i]; }
		
		TChain chain = sampler.get_chain();
		chain.save(fname, group_name.str(), dim_name_all.str(), 3, 500, 500);
		chain.get_image(*(img_stack.img[n]), rect, 1, 0, true, 0.02, 0.02, 50.);
		
		std::stringstream img_name;
		img_name << group_name.str() << "/DM_EBV";
		save_mat_image(*(img_stack.img[n]), rect, fname, img_name.str(), "DM", "E(B-V)", 3);
		
		clock_gettime(CLOCK_MONOTONIC, &t_end);
		
		//std::cout << "Sampler stats:" << std::endl;
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
	
	delete[] GR;
}


/*************************************************************************
 * 
 *   Auxiliary Functions
 * 
 *************************************************************************/

#ifndef __SEED_GSL_RNG_
#define __SEED_GSL_RNG_
// Seed a gsl_rng with the Unix time in nanoseconds
inline void seed_gsl_rng(gsl_rng **r) {
	timespec t_seed;
	clock_gettime(CLOCK_REALTIME, &t_seed);
	long unsigned int seed = 1e9*(long unsigned int)t_seed.tv_sec;
	seed += t_seed.tv_nsec;
	*r = gsl_rng_alloc(gsl_rng_taus);
	gsl_rng_set(*r, seed);
}
#endif


void rand_vector(double*const x, double* min, double* max, size_t N, gsl_rng* r) {
	for(size_t i=0; i<N; i++) {
		x[i] = min[i] + gsl_rng_uniform(r) * (max[i] - min[i]);
	}
}

void rand_vector(double*const x, size_t N, gsl_rng* r, double A) {
	for(size_t i=0; i<N; i++) { x[i] = A*gsl_rng_uniform(r); }
}

void rand_gaussian_vector(double*const x, double mu, double sigma, size_t N, gsl_rng* r) {
	for(size_t i=0; i<N; i++) { x[i] = mu + gsl_ran_gaussian_ziggurat(r, sigma); }
}

void rand_gaussian_vector(double*const x, double* mu, double* sigma, size_t N, gsl_rng* r) {
	for(size_t i=0; i<N; i++) { x[i] = mu[i] + gsl_ran_gaussian_ziggurat(r, sigma[i]); }
}









/*
double calc_logP(const double *const x, unsigned int N, MCMCParams &p) {
	#define neginf -std::numeric_limits<double>::infinity()
	double logP = 0.;
	
	//double x_tmp[4] = {x[0],x[1],x[2],x[3]};
	
	// P(Ar|G): Flat prior for Ar > 0. Don't allow DM < 0
	if((x[_Ar] < 0.) || (x[_DM] < 0.)) { return neginf; }
	
	// Make sure star is in range of template spectra
	if((x[_Mr] < p.model.Mr_min) || (x[_Mr] > p.model.Mr_max) || (x[_FeH] < p.model.FeH_min) || (x[_FeH] > p.model.FeH_max)) { return neginf; }
	
	// If the giant or dwarf flag is set, make sure star is appropriate type
	if(p.giant_flag == 1) {		// Dwarfs only
		if(x[_Mr] < 4.) { return neginf; }
	} else if(p.giant_flag == 2) {	// Giants only
		if(x[_Mr] > 4.) { return neginf; }
	}
	
	// P(Mr|G) from luminosity function
	double loglf_tmp = p.model.lf(x[_Mr]);
	logP += loglf_tmp;
	
	// P(DM|G) from model of galaxy
	double logdn_tmp = p.log_dn_interp(x[_DM]);
	logP += logdn_tmp;
	
	// P(FeH|DM,G) from Ivezich et al (2008)
	double logpFeH_tmp = p.log_p_FeH_fast(x[_DM], x[_FeH]);
	logP += logpFeH_tmp;
	
	// P(g,r,i,z,y|Ar,Mr,DM) from model spectra
	double M[NBANDS];
	FOR(0, NBANDS) { M[i] = p.m[i] - x[_DM] - x[_Ar]*p.model.Acoef[i]; }	// Calculate absolute magnitudes from observed magnitudes, distance and extinction
	
	TSED sed_bilin_interp = (*p.model.sed_interp)(x[_Mr], x[_FeH]);
	double logL = logL_SED(M, p.err, sed_bilin_interp);
	logP += logL;
	
	#undef neginf
	return logP;
}

// Generates a random state, with a flat distribution in each parameter
void ran_state(double *const x_0, unsigned int N, gsl_rng *r, MCMCParams &p) {
	x_0[_DM] = gsl_ran_flat(r, 5.1, 19.9);
	x_0[_Ar] = gsl_ran_flat(r, 0.1, 3.0);
	if(p.giant_flag == 0) {
		x_0[_Mr] = gsl_ran_flat(r, -0.5, 27.5);	// Both giants and dwarfs
	} else if(p.giant_flag == 1) {
		x_0[_Mr] = gsl_ran_flat(r, 4.5, 27.5);	// Dwarfs only
	} else {
		x_0[_Mr] = gsl_ran_flat(r, -0.5, 3.5);	// Giants only
	}
	x_0[_FeH] = gsl_ran_flat(r, -2.4, -0.1);
}

*/