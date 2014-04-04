/*
 *  cpp_utils.h
 *  
 *  Contains various useful functions.
 *  
 *  This file is part of cpp-utils, a library of useful functions
 *  written in c++.
 *  
 *  Copyright (C) 2013 Gregory Green
 *  
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *  
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *  
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef CPP_UTILS_H__
#define CPP_UTILS_H__


#include <algorithm>
#include <math.h>
#include <vector>
#include <assert.h>


// Return a percentile within a vector.
// Rearranges the vector. Use percentile_const if
// this behavior is unacceptable.
template <typename T>
T percentile(std::vector<T>& x, double pctile) {
	assert(pctile < 100.);
	assert(x.size() != 0);
	if(x.size() == 1) { return x.at(0); }
	
	double idx = pctile / 100. * (x.size()-1);
	size_t low_idx = floor(idx);
	double a = idx - low_idx;
	
	std::partial_sort(x.begin(), x.end(), x.begin() + low_idx + 1);
	
	return (1. - a) * x.at(low_idx) + a * x.at(low_idx + 1);
};

// Return a percentile within a vector, without rearranging
// the vector.
template <typename T>
T percentile_const(const std::vector<T>& x, double pctile) {
	std::vector<T> y = x;
	
	return percentile(y, pctile);
};


#endif // CPP_UTILS_H__