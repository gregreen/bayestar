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

#ifndef _SAMPLER_H__
#define _SAMPLER_H__

#include <iostream>
#include <iomanip>
#include <map>
#include <string>
#include <cstring>
#include <sstream>
#include <math.h>
#include <time.h>

//#define __STDC_LIMIT_MACROS
#include <stdint.h>

#include <unistd.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include <H5Cpp.h>
#include <H5Exception.h>

#ifdef _USE_PARALLEL_TERMPERING
#include <ptsampler.h>
#endif // _USE_PARALLEL_TERMPERING

#include "h5utils.h"

#include "model.h"
#include "data.h"

#include "affine_sampler.h"
#include "chain.h"
#include "binner.h"
#include "los_sampler.h"

//#ifndef GSL_RANGE_CHECK_OFF
//#define GSL_RANGE_CHECK_OFF
//#endif // GSL_RANGE_CHECK_OFF


// Wrapper for parameters needed by the sampler
struct TMCMCParams {
	TMCMCParams(TGalacticLOSModel* _gal_model,
				TSyntheticStellarModel* _synth_stellar_model,
				TStellarModel* _emp_stellar_model,
				TExtinctionModel* _ext_model,
                TStellarData* _data,
				unsigned int _N_DM, double _DM_min, double _DM_max);
	~TMCMCParams();

	// Model
	TSyntheticStellarModel *synth_stellar_model;
	TStellarModel *emp_stellar_model;
	TGalacticLOSModel *gal_model;
	TExtinctionModel *ext_model;
	double EBV_SFD, EBV_floor;
	double DM_min, DM_max;
	unsigned int N_DM, N_stars;

	// Data
	TStellarData *data;

	// Single-star probability density floor
	double lnp0;

	// Auxiliary info for E(B-V) curve
	TLinearInterp *EBV_interp;
	double EBV_min, EBV_max;
	void update_EBV_interp(const double* x);
	double get_EBV(double DM);

	// Index of star to fit, when sampling from individual stellar posteriors
	unsigned int idx_star;

	bool vary_RV;
	double RV_mean, RV_variance;

	bool use_priors;
};


// Probability densities
double logP_EBV(TMCMCParams &p);
double logP_los_synth(const double* x, unsigned int N, TMCMCParams& p, double* lnP_star = 0);

double logP_single_star_synth(const double *x, double EBV, double RV,
                              const TGalacticLOSModel &gal_model, const TSyntheticStellarModel &stellar_model,
                              TExtinctionModel &ext_model, const TStellarData::TMagnitudes &d, TSED *tmp_sed=NULL);
double logP_single_star_emp(const double *x, double EBV, double RV,
                            const TGalacticLOSModel &gal_model, const TStellarModel &stellar_model,
                            TExtinctionModel &ext_model, const TStellarData::TMagnitudes &d, TSED *tmp_sed=NULL);
double logP_single_star_emp_noprior(const double *x, double EBV, double RV,
                                    const TGalacticLOSModel &gal_model, const TStellarModel &stellar_model,
                                    TExtinctionModel &ext_model, const TStellarData::TMagnitudes &d, TSED *tmp_sed=NULL);

// Sampling routines
void sample_model_synth(TGalacticLOSModel& galactic_model, TSyntheticStellarModel& stellar_model, TExtinctionModel& extinction_model, TStellarData& stellar_data);
void sample_model_affine_synth(TGalacticLOSModel& galactic_model, TSyntheticStellarModel& stellar_model, TExtinctionModel& extinction_model, TStellarData& stellar_data);

void sample_indiv_synth(std::string &out_fname, TMCMCOptions &options, TGalacticLOSModel& galactic_model,
                        TSyntheticStellarModel& stellar_model,TExtinctionModel& extinction_model, TStellarData& stellar_data,
                        TImgStack& img_stack, std::vector<bool> &conv, std::vector<double> &lnZ,
                        double RV_sigma=-1., double minEBV=0., const bool saveSurfs=false, const bool gatherSurfs=true,
                        int verbosity=1);

void sample_indiv_emp(std::string &out_fname, TMCMCOptions &options, TGalacticLOSModel& galactic_model,
                      TStellarModel& stellar_model, TExtinctionModel& extinction_model, TEBVSmoothing& EBV_smoothing,
                      TStellarData& stellar_data, TImgStack& img_stack, std::vector<bool> &conv, std::vector<double> &lnZ,
                      double RV_mean=3.1, double RV_sigma=-1., double minEBV=0., const bool saveSurfs=false,
                      const bool gatherSurfs=true, const bool use_priors=true, int verbosity=1);

#ifdef _USE_PARALLEL_TEMPERING__
void sample_indiv_emp_pt(
	std::string &out_fname,
	TMCMCOptions &options,
	TGalacticLOSModel& galactic_model,
	TStellarModel& stellar_model,
	TExtinctionModel& extinction_model,
	TEBVSmoothing& EBV_smoothing,
	TStellarData& stellar_data,
	TImgStack& img_stack,
	std::vector<bool> &conv,
	std::vector<double> &lnZ,
	double RV_mean=3.1, double RV_sigma=-1., double minEBV=0.,
	const bool saveSurfs=false, const bool gatherSurfs=true,
	const bool use_priors=true, int verbosity=1
);
#endif // _USE_PARALLEL_TEMPERING__


// Auxiliary functions
void seed_gsl_rng(gsl_rng **r);

void rand_vector(double *const x, double *min, double *max, size_t N, gsl_rng *r);
void rand_vector(double *const x, size_t N, gsl_rng* r, double A=1.);
void rand_gaussian_vector(double *const x, double mu, double sigma, size_t N, gsl_rng* r);
void rand_gaussian_vector(double *const x, double *mu, double *sigma, size_t N, gsl_rng *r);


#endif // _SAMPLER_H__
