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
#include <time.h>

#include <stdint.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

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
	unsigned int N_threads;
	
	TMCMCOptions(unsigned int _steps, unsigned int _samplers,
	             double _p_replacement, unsigned int _N_threads)
		: steps(_steps), samplers(_samplers),
		  p_replacement(_p_replacement), N_threads(_N_threads)
	{}
};

struct TImgStack {
	cv::Mat **img;
	TRect *rect;
	
	size_t N_images;
	
	TImgStack(size_t _N_images);
	TImgStack(size_t _N_images, TRect &_rect);
	~TImgStack();
	
	void cull(const std::vector<bool> &keep);
	void resize(size_t _N_images);
	void set_rect(TRect &_rect);
	void stack(cv::Mat &dest);
};

struct TLOSMCMCParams {
	TImgStack *img_stack;
	double p0, lnp0;
	double EBV_max;
	double EBV_guess_max;
	
	TLOSMCMCParams(TImgStack* _img_stack, double _p0, double _EBV_max = -1.);
	~TLOSMCMCParams();
	
	void set_p0(double _p0);
};


void sample_los_extinction(std::string out_fname, TMCMCOptions &options, TImgStack& img_stack,
                           unsigned int N_regions, double p0, double EBV_max, uint64_t healpix_index);

void los_integral(TImgStack& img_stack, double* ret,
                  const double* Delta_EBV, unsigned int N_regions);

double lnp_los_extinction(const double *Delta_EBV, unsigned int N_regions, TLOSMCMCParams &params);
void gen_rand_los_extinction(double *const Delta_EBV, unsigned int N, gsl_rng *r, TLOSMCMCParams &params);

double guess_EBV_max(TImgStack &img_stack);

#endif // _LOS_SAMPLER_H__