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

#include "model.h"
#include "data.h"

#include <iostream>
#include <map>
#include <string>
#include <math.h>

//#define __STDC_LIMIT_MACROS
#include <stdint.h>

#define _DM 0
#define _Mr 1
#define _FeH 2


// Class for binning sparse data with minimal memory usage. This is especially useful
// when the domain of the function being binned is of high dimension.
class TSparseBinner {
public:
	TSparseBinner(double *_min, double *_max, unsigned int *_N_bins, unsigned int _N);
	~TSparseBinner();
	
	bool write(std::string fname);	// Save the binned data to a binary file
	
	// Add weight to a point in space
	void add_point(double *x, double weight);
	void operator()(double *x, double weight);
	
	// Determine the weight of a particular bin;
	double get_bin(double *x);
	
private:
	// Variables describing bounds of region to be binned
	double *min, *max, *dx;
	unsigned int *N_bins;	// # of bins along each axis
	uint64_t *multiplier;	// used in calculating index of element in array
	uint64_t max_index;	// Total # of bins in volume
	unsigned int N;		// Dimensionality of posterior
	
	std::map<uint64_t, double> bins;	// Bins are stored as index/weight pairs in a stdlib map
	
	uint64_t coord_to_index(double *x);			// Translate a coordinate into a bin number
	bool index_to_coord(uint64_t index, double* coord);	// Translate a bin number to a coordinate
};


// Wrapper for parameters needed by the sampler
struct TMCMCParams {
	TMCMCParams(TGalacticLOSModel* _gal_model, TStellarModel* _stellar_model, TExtinctionModel* _ext_model, TStellarData* _data, double _EBV_SFD, unsigned int _N_DM, double _DM_min, double _DM_max);
	~TMCMCParams();
	
	// Model
	TStellarModel *stellar_model;
	TGalacticLOSModel *gal_model;
	TExtinctionModel *ext_model;
	double EBV_SFD;
	double DM_min, DM_max;
	unsigned int N_DM, N_stars;
	
	// Data
	TStellarData *data;
	
	// Single-star probability density floor
	double lnp0;
	
	// Auxiliary info for E(B-V) curve
	TLinearInterp *EBV_interp;
	double EBV_min, EBV_max;
	void update_EBV_interp(double *x);
	double get_EBV(double DM);
};


// Probability densities
double logP_single_star(const double *x, double EBV, double RV, TGalacticLOSModel *gal_model, TStellarModel *stellar_model, TExtinctionModel *ext_model, TStellarData::TMagnitudes d);
double logP_EBV(TMCMCParams &p);
double logP_los(const double *x, unsigned int N, TMCMCParams &p);

// Sampling routines
void sample_model(TGalacticLOSModel& galactic_model, TStellarModel& stellar_model, TExtinctionModel& ext_model, TStellarData& stellar_data);


#endif // _SAMPLER_H__