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
 * TSparseBinner
 * 
 ****************************************************************************************************************************/

TSparseBinner::TSparseBinner(double *_min, double *_max, unsigned int *_N_bins, unsigned int _N) {
	N = _N;
	min = new double[N];
	max = new double[N];
	N_bins = new unsigned int[N];
	dx = new double[N];
	multiplier = new uint64_t[N];
	multiplier[0] = 1;
	max_index = 1;
	for(unsigned int i=0; i<N; i++) {
		min[i] = _min[i];
		max[i] = _max[i];
		N_bins[i] = _N_bins[i];
		dx[i] = (max[i] - min[i]) / (double)(N_bins[i]);
		if(i != 0) { multiplier[i] = multiplier[i-1] * N_bins[i]; }
		max_index *= N_bins[i];
	}
}

TSparseBinner::~TSparseBinner() {
	delete min;
	delete max;
	delete N_bins;
	delete multiplier;
}

uint64_t TSparseBinner::coord_to_index(double* coord) {
	uint64_t index = 0;
	uint64_t k;
	for(unsigned int i=0; i<N; i++) {
		if((coord[i] >= max[i]) || (coord[i] < min[i])) { return UINT64_MAX; }
		k = (coord[i] - min[i]) / dx[i];
		index += multiplier[i] * k;
	}
	return index;
}

bool TSparseBinner::index_to_coord(uint64_t index, double* coord) {
	if(index >= max_index) { return false; }
	uint64_t k = index % N_bins[0];
	coord[0] = min[0] + ((double)k + 0.5) * dx[0];
	for(unsigned int i=1; i<N; i++) {
		index = (index - k) / N_bins[i-1];
		k = index % N_bins[i];
		coord[i] = min[i] + ((double)k + 0.5) * dx[i];
	}
	return true;
}

void TSparseBinner::add_point(double* x, double weight) {
	uint64_t index = coord_to_index(x);
	if(index != UINT64_MAX) { bins[index] += weight; }
}

void TSparseBinner::operator()(double* x, double weight) {
	add_point(x, weight);
}

double TSparseBinner::get_bin(double* x) {
	uint64_t index;
	if((index = coord_to_index(x)) != UINT64_MAX) {
		return bins[index];
	} else {
		return -1.;
	}
}


/****************************************************************************************************************************
 * 
 * TMCMCParams
 * 
 ****************************************************************************************************************************/

TMCMCParams::TMCMCParams(TGalacticLOSModel *_gal_model, TStellarModel *_stellar_model, TExtinctionModel *_ext_model, TStellarData *_data, double _EBV_SFD, unsigned int _N_DM, double _DM_min, double _DM_max)
	: gal_model(_gal_model), stellar_model(_stellar_model), ext_model(_ext_model), data(_data), EBV_SFD(_EBV_SFD), N_DM(_N_DM), DM_min(_DM_min), DM_max(_DM_max)
{
	N_stars = data->star.size();
	EBV_interp = new TLinearInterp(DM_min, DM_max, N_DM);
	
	// Defaults
	lnp0 = -10.;
}

TMCMCParams::~TMCMCParams() {
	delete EBV_interp;
}


void TMCMCParams::update_EBV_interp(double *x) {
	double EBV = 0.;
	for(unsigned int i=0; i<N_DM; i++) {
		EBV += x[1+i];
		(*EBV_interp)[i] = EBV;
	}
	EBV_min = x[0];
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

double logP_single_star(const double *x, double EBV, double RV, TGalacticLOSModel *gal_model, TStellarModel *stellar_model, TExtinctionModel *ext_model, TStellarData::TMagnitudes d) {
	#define neginf -std::numeric_limits<double>::infinity()
	double logP = 0.;
	
	// P(EBV|G): Flat prior for E(B-V) > 0. Don't allow DM < 0
	if((EBV < 0.) || (x[_DM] < 0.)) { return neginf; }
	
	// Make sure star is in range of template spectra
	if(!(stellar_model->in_model(x[_Mr], x[_FeH]))) { return neginf; }
	
	// P(Mr|G) from luminosity function
	double loglf_tmp = stellar_model->get_log_lf(x[_Mr]);
	logP += loglf_tmp;
	
	// P(DM|G) from model of galaxy
	double logdn_tmp = gal_model->log_dNdmu(x[_DM]);
	logP += logdn_tmp;
	
	// P(FeH|DM,G) from Ivezich et al (2008)
	double logpFeH_tmp = gal_model->log_p_FeH(x[_DM], x[_FeH]);
	logP += logpFeH_tmp;
	
	// P(g,r,i,z,y|DM,Ar,Mr,FeH) from model magnitudes
	TSED sed_bilin_interp = stellar_model->get_sed(x[_Mr], x[_FeH]);
	double logL = 0.;
	double tmp;
	for(unsigned int i=0; i<NBANDS; i++) {
		tmp = d.m[i] - x[_DM] - EBV * ext_model->get_A(RV, i);	// Re-reddened absolute magnitude
		tmp = (sed_bilin_interp.absmag[i] - tmp) / d.err[i];
		logL -= 0.5*tmp*tmp;
	}
	logP += logL - d.lnL_norm;
	
	#undef neginf
	return logP;
}


double logP_EBV(TMCMCParams &p) {
	double logP = 0.;
	
	// SFD prior
	logP -= (p.EBV_max * p.EBV_max) / (2. * p.EBV_SFD * p.EBV_SFD);
	
	return logP;
}

double logP_los(const double *x, unsigned int N, TMCMCParams &p) {
	double logP = 0.;
	
	// Prior on extinction
	logP += logP_EBV(p);
	
	// Probabilities of stars
	double tmp;
	double lnp0 = -10.;
	const double *x_star;
	for(unsigned int i=0; i<p.N_stars; i++) {
		x_star = &(x[1 + p.N_DM + 3*i]);
		tmp = logP_single_star(x_star, p.get_EBV(x_star[_DM]), x[0], p.gal_model, p.stellar_model, p.ext_model, p.data->star[i]);
		tmp = exp(tmp - lnp0);
		logP += lnp0 + log(tmp + exp(-tmp));	// p --> p + p0 exp(-p/p0)  (Smooth floor on outliers)
	}
	
	return logP;
}



/****************************************************************************************************************************
 * 
 * Sampling function
 * 
 ****************************************************************************************************************************/

void sample_model(TGalacticLOSModel &galactic_model, TStellarModel &stellar_model, TExtinctionModel &extinction_model, TStellarData &stellar_data, double EBV_SFD) {
	unsigned int N_DM = 20;
	double DM_min = 5.;
	double DM_max = 20.;
	TMCMCParams params(&galactic_model, &stellar_model, &extinction_model, &stellar_data, EBV_SFD, N_DM, DM_min, DM_max);
	
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