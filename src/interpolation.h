/*
 * interpolation.h
 * 
 * Classes which implement linear and bilinear interpolation on
 * arbitrary objects. The objects must have addition and scalar
 * multiplication defined on them.
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

#ifndef __INTERPOLATION_H_
#define __INTERPOLATION_H_

#include <math.h>
#include <limits>
#include <vector>
#include <assert.h>
#include <cstddef>


// 1-D (linear) interpolation //////////////////////////////////////////////////////////////////////////////////////////////////////////////

class TLinearInterp {
	double *f_x;
	double x_min, x_max, dx, inv_dx;
	unsigned int N;
	
public:
	typedef double (*func1d_t)(double x);
	
	TLinearInterp(double _x_min, double _x_max, unsigned int _N);
	TLinearInterp(func1d_t func, double _x_min, double _x_max, unsigned int _N);
	~TLinearInterp();
	
	double operator()(double x) const;
	double& operator[](unsigned int index);
	
	double get_x(unsigned int index) const;
	double dfdx(double x) const;
	
	void fill(func1d_t func);
};




// 2-D (bilinear) interpolation ////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<class T>
class TBilinearInterp {
	T *f;				// y is more significant than x, i.e. idx = x + Nx*y
	double x_min, x_max, dx, inv_dx;
	double y_min, y_max, dy, inv_dy;
	double dxdy, inv_dxdy;
	unsigned int Nx, Ny;
	
public:
	typedef T (*func2d_t)(double x, double y);
	typedef T* (*func2d_ptr_t)(double x, double y);
	
	TBilinearInterp(double _x_min, double _x_max, unsigned int Nx, double _y_min, double _y_max, unsigned int Ny);
	TBilinearInterp(func2d_t func, double _x_min, double _x_max, unsigned int Nx, double _y_min, double _y_max, unsigned int Ny);
	TBilinearInterp(func2d_ptr_t func, double _x_min, double _x_max, unsigned int Nx, double _y_min, double _y_max, unsigned int Ny);
	~TBilinearInterp();
	
	T operator()(double x, double y) const;
	T& operator[](unsigned int index);
	unsigned int get_index(double x, double y) const;
	void get_xy(unsigned int i, unsigned int j, double &x, double &y) const;
	
	void fill(func2d_t func);
	void fill(func2d_ptr_t func);
};


template<class T>
TBilinearInterp<T>::TBilinearInterp(double _x_min, double _x_max, unsigned int _Nx, double _y_min, double _y_max, unsigned int _Ny)
	: x_min(_x_min), x_max(_x_max), Nx(_Nx), y_min(_y_min), y_max(_y_max), Ny(_Ny), f(NULL)
{
	f = new T[Nx*Ny];
	dx = (x_max - x_min) / (double)(Nx - 1);
	dy = (y_max - y_min) / (double)(Ny - 1);
	dxdy = dx*dy;
	inv_dx = 1./dx;
	inv_dy = 1./dy;
	inv_dxdy = 1./dxdy;
}

template<class T>
TBilinearInterp<T>::TBilinearInterp(func2d_t func, double _x_min, double _x_max, unsigned int _Nx, double _y_min, double _y_max, unsigned int _Ny)
	: x_min(_x_min), x_max(_x_max), Nx(_Nx), y_min(_y_min), y_max(_y_max), Ny(_Ny), f(NULL)
{
	f = new T[Nx*Ny];
	dx = (x_max - x_min) / (double)(Nx - 1);
	dy = (y_max - y_min) / (double)(Ny - 1);
	dxdy = dx*dy;
	inv_dx = 1./dx;
	inv_dy = 1./dy;
	inv_dxdy = 1./dxdy;
	fill(func);
}

template<class T>
TBilinearInterp<T>::TBilinearInterp(func2d_ptr_t func, double _x_min, double _x_max, unsigned int _Nx, double _y_min, double _y_max, unsigned int _Ny)
	: x_min(_x_min), x_max(_x_max), Nx(_Nx), y_min(_y_min), y_max(_y_max), Ny(_Ny), f(NULL)
{
	f = new T[Nx*Ny];
	dx = (x_max - x_min) / (double)(Nx - 1);
	dy = (y_max - y_min) / (double)(Ny - 1);
	dxdy = dx*dy;
	inv_dx = 1./dx;
	inv_dy = 1./dy;
	inv_dxdy = 1./dxdy;
	fill(func);
}

template<class T>
TBilinearInterp<T>::~TBilinearInterp() {
	delete[] f;
}

template<class T>
unsigned int TBilinearInterp<T>::get_index(double x, double y) const {
	assert((x >= x_min) && (x <= x_max) && (y >= y_min) && (y <= y_max));
	return (unsigned int)((x-x_min)*inv_dx + 0.5) + Nx*(unsigned int)((y-y_min)*inv_dy + 0.5);
}

template<class T>
T TBilinearInterp<T>::operator()(double x, double y) const {
	double idx = floor((x-x_min)*inv_dx);
	assert((idx >= 0) && (idx < Nx));
	double idy = floor((y-y_min)*inv_dy);
	assert((idy >= 0) && (idy < Ny));
	double Delta_x = x - x_min - dx*idx;
	double Delta_y = y - y_min - dy*idy;
	unsigned int N00 = (unsigned int)idx + Nx*(unsigned int)idy;
	unsigned int N10 = N00 + 1;
	unsigned int N01 = N00 + Nx;
	unsigned int N11 = N00 + 1 + Nx;
	T tmp = inv_dxdy*(f[N00]*(dx-Delta_x)*(dy-Delta_y) + f[N10]*Delta_x*(dy-Delta_y) + f[N01]*(dx-Delta_x)*Delta_y + f[N11]*Delta_x*Delta_y);
	return tmp;
}

template<class T>
T& TBilinearInterp<T>::operator[](unsigned int index) {
	assert(index < Nx*Ny);
	return f[index];
}

template<class T>
void TBilinearInterp<T>::get_xy(unsigned int i, unsigned int j, double &x, double &y) const {
	assert((i < Nx) && (j < Ny));
	x = dx*(double)i - x_min;
	y = dy*(double)j - y_min;
}

template<class T>
void TBilinearInterp<T>::fill(func2d_t func) {
	double x, y;
	for(unsigned int i=0; i<Nx; i++) {
		for(unsigned int j=0; j<Ny; j++) {
			get_xy(i, j, x, y);
			f[i + Nx*j] = func(x, y);
		}
	}
}

template<class T>
void TBilinearInterp<T>::fill(func2d_ptr_t func) {
	double x, y;
	for(unsigned int i=0; i<Nx; i++) {
		for(unsigned int j=0; j<Ny; j++) {
			get_xy(i, j, x, y);
			f[i + Nx*j] = *func(x, y);
		}
	}
}



// N-Dimensional (multilinear) interpolation ////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<class T>
class TMultiLinearInterp {
	T *f;					// Function values
	double *min, *max, *inv_dx;
	unsigned int *N;
	
	unsigned int ndim;
	unsigned int *coeff;
	
	std::vector<bool> filled;
	T empty;
	
	double *lower;
	unsigned int *Delta_idx;
	unsigned int N_Delta;
	
public:
	typedef T (*func_t)(double *x);
	typedef T* (*func_ptr_t)(double *x);
	
	TMultiLinearInterp(double *_min, double *_max, unsigned int *_N, unsigned int _ndim, T &_empty);
	~TMultiLinearInterp();
	
	T operator()(double *x);
	
	void set(double *x, double fx);
	
	bool fill(double *x, double *fx);
	bool fill(typename std::vector<T>::iterator begin, typename std::vector<T>::iterator end, std::vector<double> &fx);
	void fill(func_t func);
	void fill(func_ptr_t func);
	
private:
	int get_index(double *x) const;
	int get_lower(double *x) const;
	void set_index_arr(double *x);
};


template<class T>
TMultiLinearInterp<T>::TMultiLinearInterp(double *_min, double *_max, unsigned int *_N, unsigned int _ndim, T &_empty)
	: ndim(_ndim), empty(_empty)
{
	unsigned int length = 1;
	coeff = new double[ndim];
	min = new double[ndim];
	max = new double[ndim];
	inv_dx = new double[ndim];
	N = new unsigned int[ndim];
	for(int i=0; i<ndim; i++) {
		coeff[i] = length;
		length *= N[i];
		min[i] = _min[i];
		max[i] = _max[i];
		N[i] = _N[i];
		inv_dx[i] = (double)(N[i] - 1) / (max[i] - min[i]);
	}
	
	f = new T[length];
	filled.resize(length);
	std::fill(filled.begin(), filled.end(), false);
	
	lower = new double[ndim];
	
	// Compute Deltas (difference in index from lower corner) to corners of box
	N_Delta = (1 << ndim);
	Delta_idx = new unsigned int[N_Delta];
	for(unsigned int i=0; i<N_Delta; i++) {
		Delta_idx[i] = 0;
		for(unsigned int k=0; k<ndim; k++) {
			Delta_idx[i] += coeff[k] * ((i >> k) & 1);
		}
	}
}

template<class T>
TMultiLinearInterp<T>::~TMultiLinearInterp() {
	delete[] f;
	delete[] min;
	delete[] max;
	delete[] inv_dx;
	delete[] coeff;
	delete[] lower;
	delete[] Delta_idx;
}


template<class T>
int TMultiLinearInterp<T>::get_index(double* x) const {
	int index = 0;
	int k;
	for(int i=0; i<ndim; i++) {
		k = (x[i] - min[i]) * inv_dx[i] + 0.5;
#ifndef INTERP_NO_BOUNDS_CHECK
		assert((k >= 0) && (k < N[i]));
#endif // INTERP_NO_BOUNDS_CHECK
		index += coeff[i] * k;
	}
	return index;
}

template<class T>
int TMultiLinearInterp<T>::get_lower(double* x) const {
	int index = 0;
	int k;
	for(int i=0; i<ndim; i++) {
		k = (x[i] - min[i]) * inv_dx[i];
#ifndef INTERP_NO_BOUNDS_CHECK
		assert((k >= 0) && (k < N[i]));
#endif // INTERP_NO_BOUNDS_CHECK
		index += coeff[i] * k;
	}
	return index;
}

template<class T>
void TMultiLinearInterp<T>::set_index_arr(double* x) {
	int k = 0;
	for(int i=0; i<ndim; i++) {
		k = (x[i] - min[i]) * inv_dx[i];
#ifndef INTERP_NO_BOUNDS_CHECK
		assert((k >= 0) && (k < N[i]));
#endif // INTERP_NO_BOUNDS_CHECK
		lower[i] = (x[i] - min[i]) * inv_dx[i] - (double)k;
	}
}


template<class T>
void TMultiLinearInterp<T>::set(double* x, double fx) {
	int idx = get_index(x);
	f[idx] = fx;
	filled[idx] = true;
}

template<class T>
T TMultiLinearInterp<T>::operator()(double* x) {
	int idx = get_lower(x);
	set_index_arr(x);
	
	double sum = 0.;
	double term;
	
	unsigned int i_max = (1 << ndim);
	for(unsigned int i=0; i<i_max; i++) {
		term = f[idx + Delta_idx[i]];
		for(unsigned int k=0; k<ndim; k++) {
			if((i >> k) & 1) {
				term *= (1. - lower[k]);
			} else {
				term *= lower[k];
			}
		}
		sum += term;
	}
	
	return sum;
}





#endif	// __INTERPOLATION_H_