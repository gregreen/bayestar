/*
 * definitions.h
 *
 * Definitions needed by a number of header/source files.
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

#ifndef _DEFINITIONS_H__
#define _DEFINITIONS_H__

#include <limits>

/*************************************************************************
 *   Replacements for infinity
 *************************************************************************/

#ifndef INF_REPLACEMENT
#define INF_REPLACEMENT

const double inf_replacement = std::numeric_limits<double>::max();
const double neg_inf_replacement = -std::numeric_limits<double>::max();

const double large_double_replacement = inf_replacement / 10.;
const double neg_large_double_replacement = neg_inf_replacement / 10.;

const double min_replacement = 10. * std::numeric_limits<double>::min();
const double neg_min_replacement = -10. * std::numeric_limits<double>::min();

inline bool is_pos_inf_replacement(double x) {
	return (x >= large_double_replacement);
}

inline bool is_neg_inf_replacement(double x) {
	return (x <= neg_large_double_replacement);
}

inline bool is_inf_replacement(double x) {
	return (x >= large_double_replacement) || (x <= neg_large_double_replacement);
}

#endif // INF_REPLACEMENT


/*************************************************************************
 *   Mathematical constants
 *************************************************************************/

#ifndef PI
#define PI 3.14159265358979323
#endif // PI

#ifndef SQRTPI
#define SQRTPI 1.7724538509055159
#endif // SQRTPI

#ifndef SQRT2
#define SQRT2 1.4142135623730951
#endif // SQRT2

#ifndef INV_SQRT2
#define INV_SQRT2 0.70710678118654746
#endif // INV_SQRT2

#ifndef SQRT3
#define SQRT3 1.7320508075688772
#endif // SQRT3

#ifndef SQRT2PI
#define SQRT2PI 2.5066282746310002
#endif // SQRT2PI

#ifndef LN10
#define LN10 2.3025850929940459
#endif // LN10


/*************************************************************************
 *   Custom types
 *************************************************************************/

// Easily switch between 32- and 64-bit floating-point types
typedef float floating_t;
#ifndef CV_FLOATING_TYPE
#define CV_FLOATING_TYPE CV_32F
#endif // CV_FLOATING_TYPE


#endif // _DEFINITIONS_H__
