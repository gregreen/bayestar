#include "interpolation.h"

// 1-D (linear) interpolation //////////////////////////////////////////////////////////////////////////////////////////////////////////////

TLinearInterp::TLinearInterp(double _x_min, double _x_max, unsigned int _N)
	: x_min(_x_min), x_max(_x_max), N(_N), f_x(NULL)
{
	f_x = new double[N];
	dx = (x_max - x_min) / (double)(N - 1);
	inv_dx = 1./dx;
}

TLinearInterp::TLinearInterp(func1d_t func, double _x_min, double _x_max, unsigned int _N)
	: x_min(_x_min), x_max(_x_max), N(_N), f_x(NULL)
{
	f_x = new double[N];
	dx = (x_max - x_min) / (double)(N - 1);
	inv_dx = 1./dx;
	fill(func);
}

TLinearInterp::~TLinearInterp() {
	delete[] f_x;
}


double TLinearInterp::operator()(double x) const {
	if((x < x_min) || (x > x_max)) {
		return std::numeric_limits<double>::quiet_NaN();
	} else if(x == x_max) {
		return f_x[N-1];
	}
	
	double index_dbl = (x - x_min) * inv_dx;
	unsigned int index_lower = (unsigned int)index_dbl;
	double Delta_lower = index_dbl - (double)index_lower;
	
	return Delta_lower * f_x[index_lower + 1] + (1. - Delta_lower) * f_x[index_lower];
	
	//unsigned int index_nearest = (unsigned int)(index_dbl + 0.5);
	//return f_x[index_nearest];
	/*double dist = (index_dbl - (double)(index_nearest)) * dx;
	if(index_nearest == N) { return f_x[N-1] - dist*(f_x[N-1]-f_x[N-2])*inv_dx; }
	if(dist == 0) {
		return f_x[index_nearest];
	} else if(dist > 0) {
		return f_x[index_nearest] + dist * (f_x[index_nearest+1]-f_x[index_nearest])*inv_dx;
	} else {
		return f_x[index_nearest] + dist * (f_x[index_nearest]-f_x[index_nearest-1])*inv_dx;
	}*/
}

double TLinearInterp::dfdx(double x) const {
	if((x < x_min) || (x > x_max)) { return std::numeric_limits<double>::quiet_NaN(); }
	double index_dbl = (x - x_min) * inv_dx;
	unsigned int index_nearest = (unsigned int)(index_dbl + 0.5);
	if(index_nearest == 0) {
		return (f_x[1]-f_x[0])*inv_dx;
	} else if((index_nearest == N-1) || (index_nearest == N)) {
		return (f_x[N-1]-f_x[N-2])*inv_dx;
	}
	double diff = index_dbl - (double)(index_nearest);
	if(diff >= 0) {
		return (f_x[index_nearest+1]-f_x[index_nearest])*inv_dx;
	} else {
		return (f_x[index_nearest]-f_x[index_nearest-1])*inv_dx;
	}
}

double TLinearInterp::get_x(unsigned int index) const {
	assert(index < N);
	return x_min + (double)index * dx;
}

double& TLinearInterp::operator[](unsigned int index) {
	assert(index < N);
	return f_x[index];
}

void TLinearInterp::fill(func1d_t func) {
	double x;
	for(unsigned int i=0; i<N; i++) {
		x = x_min + dx*(double)i;
		f_x[i] = func(x);
	}
}
