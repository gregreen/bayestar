/*
 * binner.h
 * 
 * Defines classes to bin posterior densities.
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

#ifndef _BINNER_H__
#define _BINNER_H__

#include <iostream>
#include <iomanip>
#include <map>
#include <string>
#include <cstring>
#include <sstream>
#include <math.h>
#include <time.h>

#include <stdint.h>

#include <H5Cpp.h>
#include <H5Exception.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// Class for binning sparse data with minimal memory usage. This is especially useful
// when the domain of the function being binned is of high dimension. In order to achieve
// this, only the non-empty bins are stored.
class TSparseBinner {
public:
	TSparseBinner(double *_min, double *_max, unsigned int *_N_bins, unsigned int _N);
	TSparseBinner(std::string fname, std::string dset_name, std::string group_name);
	~TSparseBinner();
	
	// Save the binned data to an HDF5 file
	bool write(std::string fname, std::string dset_name,
	           std::string group_name, std::string *dim_name,
	           int compression=1, hsize_t chunk=1000);
	
	// Load in binned data from an HDF5 file
	bool load(std::string fname, std::string dset_name,
	          std::string group_name);
	
	// Add weight to a point in space
	void add_point(double *x, double weight);
	void operator()(double *x, double weight);
	
	// Determine the weight of a particular bin;
	double get_bin(double *x);
	
	// Return an image, optionally with smoothing
	void get_image(cv::Mat &mat, unsigned int dim1, unsigned int dim2,
	                             double sigma1=-1., double sigma2=-1.);
	
	// Clear the binner
	void clear();
	
private:
	// Variables describing bounds of region to be binned
	double *min, *max, *dx;
	unsigned int *N_bins;	// # of bins along each axis
	uint64_t *multiplier;	// used in calculating index of element in array
	uint64_t max_index;	// Total # of bins in volume
	unsigned int N;		// Dimensionality of posterior
	
	// Bins are stored as index/weight pairs in a stdlib map
	std::map<uint64_t, double> bins;
	
	// Translate a coordinate into a bin number
	uint64_t coord_to_index(double *x);
	
	// Translate a bin number to a coordinate
	bool index_to_coord(uint64_t index, double* coord);		// Physical coordinate
	bool index_to_array_coord(uint64_t index, uint64_t* coord);	// Array coordinate
	
	// Needed in order to write to HDF5
	struct TIndexValue {
		uint64_t index;
		float value;
	};
	
	// Needed in order to write attributes to HDF5
	struct TDimDesc {
		char* name;
		float min, max;
		unsigned int N_bins;
	};
};

#endif // _BINNER_H__