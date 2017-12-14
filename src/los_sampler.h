/*
 * los_sampler.h
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

#ifndef _LOS_SAMPLER_H__
#define _LOS_SAMPLER_H__

#include <iostream>
#include <iomanip>
#include <map>
#include <string>
#include <cstring>
#include <sstream>
#include <math.h>
#include <numeric>
#include <time.h>
#include <memory>

#include <stdint.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "definitions.h"

#include "model.h"
#include "data.h"

#include "affine_sampler.h"
#include "chain.h"
#include "binner.h"


// Parameters commonly passed to sampling routines
struct TMCMCOptions {
	unsigned int steps;
	unsigned int samplers;
	double p_replacement;
	unsigned int N_runs;

	TMCMCOptions(unsigned int _steps, unsigned int _samplers,
	             double _p_replacement, unsigned int _N_runs)
		: steps(_steps), samplers(_samplers),
		  p_replacement(_p_replacement), N_runs(_N_runs)
	{}
};

struct TImgStack {
	cv::Mat **img;
	TRect *rect;

	size_t N_images;

	TImgStack(size_t _N_images);
	TImgStack(size_t _N_images, TRect &_rect);
	~TImgStack();

	void cull(const std::vector<bool>& keep);
	void crop(double x_min, double x_max, double y_min, double y_max);

	void resize(size_t _N_images);
	void set_rect(TRect& _rect);
	void stack(cv::Mat& dest);

	bool initialize_to_zero(unsigned int img_idx);

	void smooth(std::vector<double> sigma, double n_sigma=5);
};

struct TLOSMCMCParams {
	TImgStack *img_stack;
	std::vector<double> p0_over_Z, ln_p0_over_Z, inv_p0_over_Z;
	double p0, lnp0;

	double *line_int;
	float *Delta_EBV;
	unsigned int N_runs;
	unsigned int N_threads;
	unsigned int N_regions;

	double EBV_max;
	double EBV_guess_max;
	std::vector<double> EBV_prof_guess;
	gsl_matrix *guess_cov, *guess_sqrt_cov;

	std::vector<double> subpixel;
	double subpixel_min, subpixel_max;

	double *Delta_EBV_prior;
	double *log_Delta_EBV_prior;
	double *sigma_log_Delta_EBV;
	double alpha_skew;

	TLOSMCMCParams(TImgStack* _img_stack, const std::vector<double>& _lnZ,
				   double _p0, unsigned int _N_runs, unsigned int _N_threads,
				   unsigned int _N_regions, double _EBV_max=-1.);
	~TLOSMCMCParams();

	void set_p0(double _p0);
	void set_subpixel_mask(TStellarData& data);
	void set_subpixel_mask(std::vector<double>& new_mask);

	void calc_Delta_EBV_prior(TGalacticLOSModel& gal_los_model,
							  double log_Delta_EBV_floor,
							  double log_Delta_EBV_ceil,
							  double sigma, double EBV_tot,
							  int verbosity=1);

	void gen_guess_covariance(double scale_length);

	double* get_line_int(unsigned int thread_num);
	float* get_Delta_EBV(unsigned int thread_num);

};


struct TDiscreteLosMcmcParams {
    TImgStack* img_stack;   // Stack of (distance, reddening) posteriors for stars
    double y_zero_idx;		// y-index corresponding to zero reddening

    double* line_int;       // Line integral through line of sight for each thread
    int16_t* E_pix_idx;     // LOS reddening profile, in the form of the pixel y-index at each distance (for each thread)

    unsigned int n_dists, n_E;  // # of distance and reddening pixels, respectively
    unsigned int N_runs;    // # of times to repeat inference (to check convergence)
	unsigned int N_threads; // # of threads (can be less than # of runs)

	// Priors on Delta E in each distance bin
	double mu_log_dE, sigma_log_dE;
	double mu_log_dy, inv_sigma_log_dy;
	double inv_sigma_dy_neg;

	// Distance-dependent priors on Delta E
	std::shared_ptr<cv::Mat> log_P_dy;

	TDiscreteLosMcmcParams(TImgStack *_img_stack,
						   unsigned int _N_runs,
						   unsigned int _N_threads);
	~TDiscreteLosMcmcParams();

	// Access fit information for one thread
	double* get_line_int(unsigned int thread_num);
	int16_t* get_E_pix_idx(unsigned int thread_num);

	// Line-of-sight integrals
	void los_integral_discrete(const int16_t *const y_idx,
							   double *const line_int_ret);

	// Prior on line-of-sight dust distribution
	floating_t log_dy_prior(const int16_t x_idx, const int16_t dy);	// Indiv. dist.
	floating_t log_prior(const int16_t *const y_idx);	// Entire profile

	// Step proposal
	void los_integral_diff_step(const int16_t x_idx,
								const int16_t y_idx_old,
                                const int16_t y_idx_new,
								double *const delta_line_int_ret);

	floating_t log_prior_diff_step(const int16_t x_idx,
							  const int16_t *const y_idx_los_old,
                              const int16_t y_idx_new);

	// Swap proposal
    void los_integral_diff_swap(const int16_t x0_idx,
								const int16_t *const y_idx,
                                double *const delta_line_int_ret);

	floating_t log_prior_diff_swap(const int16_t x0_idx,
		                      const int16_t *const y_idx_los_old);

	// Shift-right proposal
	bool shift_r_step_valid(
		const int16_t x_idx,
		const int16_t dy,
		const int16_t *const y_idx_old);

	void los_integral_diff_shift_r(
		const int16_t x_idx,
		const int16_t dy,
		const int16_t *const y_idx_old,
		double *const delta_line_int_ret);

	floating_t log_prior_diff_shift_r(
		const int16_t x_idx,
		const int16_t dy,
	    const int16_t *const y_idx_los_old);

	// Shift-left proposal
	bool shift_l_step_valid(
		const int16_t x_idx,
		const int16_t dy,
		const int16_t *const y_idx_old);

	void los_integral_diff_shift_l(
		const int16_t x_idx,
		const int16_t dy,
		const int16_t *const y_idx_old,
		double *const delta_line_int_ret);

	floating_t log_prior_diff_shift_l(
		const int16_t x_idx,
		const int16_t dy,
	    const int16_t *const y_idx_los_old);

	// Miscellaneous functions
	void los_integral_diff_shift_compare_operations(
		const int16_t x_idx,
		const int16_t dy,
		const int16_t *const y_idx_old,
		unsigned int& n_eval_diff,
		unsigned int& n_eval_cumulative);

    void guess_EBV_profile_discrete(int16_t *const y_idx_ret, gsl_rng *r);

	void initialize_priors(
		TGalacticLOSModel& gal_los_model,
		double log_Delta_EBV_floor,
		double log_Delta_EBV_ceil,
		int verbosity=0);
};


// Transform from log(DeltaEBV) to cumulative EBV for piecewise-linear l.o.s. fit
class TLOSTransform : public TTransformParamSpace {
private:
	size_t _ndim;
public:
	TLOSTransform(unsigned int ndim);
	virtual ~TLOSTransform();

	virtual void transform(const double *const x, double *const y);
};

// Transform to cumulative EBV for cloud l.o.s. fit
class TLOSCloudTransform : public TTransformParamSpace {
private:
	size_t _ndim;
	size_t n_clouds;

public:
	TLOSCloudTransform(unsigned int ndim);
	virtual ~TLOSCloudTransform();

	virtual void transform(const double *const x, double *const y);
};

// Testing functions
void test_extinction_profiles(TLOSMCMCParams &params);

// Sample piecewise-linear model

void sample_los_extinction(const std::string& out_fname, const std::string& group_name,
                           TMCMCOptions &options, TLOSMCMCParams &params,
                           int verbosity=1);

double lnp_los_extinction(const double *const Delta_EBV, unsigned int N_regions, TLOSMCMCParams &params);

void gen_rand_los_extinction_from_guess(double *const logEBV, unsigned int N, gsl_rng *r, TLOSMCMCParams &params);

void gen_rand_los_extinction(double *const Delta_EBV, unsigned int N, gsl_rng *r, TLOSMCMCParams &params);

void los_integral(TImgStack& img_stack, const double *const subpixel, double *const ret,
                  const float *const Delta_EBV, unsigned int N_regions);

double guess_EBV_max(TImgStack &img_stack);

void guess_EBV_profile(TMCMCOptions &options, TLOSMCMCParams &params, int verbosity=1);

void monotonic_guess(TImgStack &img_stack, unsigned int N_regions, std::vector<double>& Delta_EBV, TMCMCOptions& options);

double switch_log_Delta_EBVs(double *const _X, double *const _Y, unsigned int _N, gsl_rng* r, TLOSMCMCParams& _params);

double switch_adjacent_log_Delta_EBVs(double *const _X, double *const _Y, unsigned int _N, gsl_rng* r, TLOSMCMCParams& _params);

double mix_log_Delta_EBVs(double *const _X, double *const _Y, unsigned int _N, gsl_rng* r, TLOSMCMCParams& _params);

double step_one_Delta_EBV(double *const _X, double *const _Y, unsigned int _N, gsl_rng* r, TLOSMCMCParams& _params);


// Sample cloud model
void sample_los_extinction_clouds(const std::string& out_fname, const std::string& group_name,
                                  TMCMCOptions &options, TLOSMCMCParams &params,
                                  unsigned int N_clouds, int verbosity=1);

double lnp_los_extinction_clouds(const double* x, unsigned int N, TLOSMCMCParams& params);

void gen_rand_los_extinction_clouds(double *const x, unsigned int N, gsl_rng *r, TLOSMCMCParams &params);

void los_integral_clouds(TImgStack &img_stack, const double *const subpixel, double *const ret, const double *const Delta_mu,
                         const double *const logDelta_EBV, unsigned int N_clouds);


// Sample discrete line-of-sight model
void sample_los_extinction_discrete(const std::string& out_fname, const std::string& group_name,
                           TMCMCOptions &options, TDiscreteLosMcmcParams &params,
                           int verbosity);


#endif // _LOS_SAMPLER_H__
