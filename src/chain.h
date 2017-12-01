/*
 * chain.h
 *
 * Defines class representing a Markov Chain, for use in MCMC routines.
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

#ifndef _CHAIN_H__
#define _CHAIN_H__

#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sstream>
#include <cstring>
#include <stdint.h>
#include <vector>
#include <map>
#include <algorithm>
#include <limits>
#include <assert.h>

#include <unistd.h>

#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_sf_gamma.h>

#include <gsl/gsl_errno.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "definitions.h"
#include "h5utils.h"
#include "stats.h"

#ifndef PI
#define PI 3.14159265358979323
#endif

#ifndef SQRTPI
#define SQRTPI 1.77245385
#endif


// Gaussian mixture structure
//     Stores data necessary for representing a mixture of Gaussians, along
//     with workspaces for computing inverses.
struct TGaussianMixture {
	// Data
	unsigned int ndim, nclusters;
	double *w;
	double *mu;
	gsl_matrix **cov;
	gsl_matrix **inv_cov;
	gsl_matrix **sqrt_cov;
	double *det_cov;

	// Workspaces
	gsl_permutation *p;
	gsl_matrix *LU;
	gsl_eigen_symmv_workspace* esv;
	gsl_vector *eival;
	gsl_matrix *eivec;
	gsl_matrix *sqrt_eival;
	gsl_rng *r;

	// Constructor / Destructor
	TGaussianMixture(unsigned int _ndim, unsigned int _nclusters);
	~TGaussianMixture();

	// Accessors
	gsl_matrix* get_cov(unsigned int k);
	double get_w(unsigned int k);
	double* get_mu(unsigned int k);
	void draw(double *x);
	void print();

	// Mutators
	void invert_covariance();

	void density(const double *x, unsigned int N, double *res);
	double density(const double *x);

	void expectation_maximization(const double *x, const double *w, unsigned int N, unsigned int iterations=10);
};


class TRect {
public:
	double dx[2];
	uint32_t N_bins[2];
	double min[2];
	double max[2];

	TRect(double _min[2], double _max[2], uint32_t _N_bins[2]);
	TRect(const TRect& rect);
	~TRect();

	bool get_index(double x1, double x2,
				   unsigned int &i1, unsigned int &i2) const;
	bool get_index(double x1, double x2,
				   double& i1, double& i2) const;

	bool get_interpolant(double x1, double x2,
					     unsigned int& i1, unsigned int& i2,
						 double& a1, double& a2) const;

	TRect& operator =(const TRect& rhs);
};


/*************************************************************************
 *   Chain Class Prototype
 *************************************************************************/

class TChain {
private:
	std::vector<double> x;			// Elements in chain. Each point takes up N contiguous slots in array
	std::vector<double> L;			// Likelihood of each point in chain
	std::vector<double> w;			// Weight of each point in chain
	double total_weight;			// Sum of the weights
	unsigned int N, length, capacity;	// # of dimensions, length and capacity of chain

	std::vector<double> x_min;
	std::vector<double> x_max;

	struct TChainAttribute {
		char *dim_name;
		float total_weight;
		uint64_t ndim, length;
	};

public:
	TStats stats;				// Keeps track of statistics of chain

	TChain(unsigned int _N, unsigned int _capacity);
	TChain(const TChain& c);
	TChain(std::string filename, bool reserve_extra=false);	// Construct the chain from a file
	~TChain();

	// Mutators
	void add_point(const double *const element, double L_i, double w_i);		// Add a point to the end of the chain
	void clear();								// Remove all the points from the chain
	void set_capacity(unsigned int _capacity);				// Set the capacity of the vectors used in the chain
	double append(const TChain& chain, bool reweight=false, bool use_peak=true, double nsigma_max=1.,
	              double nsigma_peak=0.1, double chain_frac=0.05, double threshold=1.e-5);	// Append a second chain to this one

	// Accessors
	unsigned int get_capacity() const;			// Return the capacity of the vectors used in the chain
	unsigned int get_length() const;			// Return the number of unique points in the chain
	double get_total_weight() const;			// Return the sum of the weights in the chain
	const double* get_element(unsigned int i) const;	// Return the i-th point in the chain
	void get_best(std::vector<double> &x) const;		// Return best point in chain
	unsigned int get_index_of_best() const;
	double get_L(unsigned int i) const;			// Return the likelihood of the i-th point
	double get_w(unsigned int i) const;			// Return the weight of the i-th point
	unsigned int get_ndim() const;

	// Computations on chain

	// Estimate the Bayesian Evidence of the posterior using the bounded Harmonic Mean Approximation
	double get_ln_Z_harmonic(bool use_peak=true, double nsigma_max=1.,
	                         double nsigma_peak=0.1, double chain_frac=0.1) const;

	// Estimate coordinates with peak density by binning
	void density_peak(double* const peak, double nsigma) const;

	// Find a point in space with high density by picking a random point, drawing an ellipsoid,
	// taking the mean coordinate within the ellipsoid, and then iterating
	void find_center(double* const center, gsl_matrix *const cov, gsl_matrix *const inv_cov,
	                 double* det_cov, double dmax=1., unsigned int iterations=5) const;

	void fit_gaussian_mixture(TGaussianMixture *gm, unsigned int iterations=10);

	// Return an image, optionally with smoothing
	void get_image(cv::Mat &mat, const TRect &grid,
	               unsigned int dim1, unsigned int dim2, bool norm=true,
	               double sigma1=-1., double sigma2=-1., double nsigma=5.,
				   bool sigma_pix_units=false) const;

	// File IO
	// Save the chain to an HDF5 file
	bool save(std::string fname, std::string group_name, size_t index,
	          std::string dim_name, int compression=1, int subsample=-1,
	          bool converged=true, float lnZ=std::numeric_limits<float>::quiet_NaN()) const;
	bool load(std::string filename, bool reserve_extra=false);	// Load the chain from file

	// Operators
	const double* operator [](unsigned int i);	// Calls get_element
	void operator +=(const TChain& rhs);		// Calls append
	TChain& operator =(const TChain& rhs);		// Assignment operator
};


/*************************************************************************
 *   Class to write multiple chains to HDF5
 *************************************************************************/

class TChainWriteBuffer {
public:
	TChainWriteBuffer(unsigned int nDim, unsigned int nSamples, unsigned int nReserved = 10);
	~TChainWriteBuffer();

	void add(const TChain &chain,
	         bool converged = true,
	         double lnZ = std::numeric_limits<double>::quiet_NaN(),
		     double * GR = NULL,
			 bool subsample = true);

	void reserve(unsigned int nReserved);

	void write(const std::string& fname, const std::string& group,
	           const std::string& chain, const std::string& meta="");

private:
	float *buf;
	unsigned int nDim_, nSamples_, nReserved_, length_;
	gsl_rng *r;
	std::vector<double> samplePos;

	struct TChainMetadata {
		bool converged;
		float lnZ;
	};

	std::vector<TChainMetadata> metadata;

};

/*************************************************************************
 *   Class to write stack of images to HDF5
 *************************************************************************/

class TImgWriteBuffer {
public:
	TImgWriteBuffer(const TRect& rect, unsigned int nReserved = 10);
	~TImgWriteBuffer();

	void add(const cv::Mat& img);

	void reserve(unsigned int nReserved);

	void write(const std::string& fname, const std::string& group, const std::string& img);

private:
	float *buf;
	unsigned int nReserved_, length_;
	TRect rect_;
};



/*************************************************************************
 *   Convergence diagnostics in transformed parameter space
 *       (e.g. observable space)
 *************************************************************************/

class TTransformParamSpace {
public:
	TTransformParamSpace(unsigned int ndim);
	virtual ~TTransformParamSpace();

	virtual void transform(const double *const x, double *const y);
	void operator()(const double *const x, double *const y);

private:
	unsigned int _ndim;
};


void Gelman_Rubin_diagnostic(const std::vector<TChain*>& chains, std::vector<double>& R, TTransformParamSpace *const transf);



/*************************************************************************
 *   Auxiliary functions
 *************************************************************************/

// Save an image stored in an OpenCV matrix, with dimensions corresponding to
// those encoded in the TRect class, to an HDF5 file
bool save_mat_image(cv::Mat& img, TRect& rect, std::string fname, std::string group_name,
                    std::string dset_name, std::string dim1, std::string dim2,
                    int compression=1);//, hsize_t chunk=0);

// Load an image stored in an HDF5 file to an OpenCV matrix, and read
// dimensions of image into TRect class
bool load_mat_image(cv::Mat &img, TRect &rect, std::string fname,
                    std::string group_name, std::string dim_name);


#ifndef __SEED_GSL_RNG_
#define __SEED_GSL_RNG_
// Seed a gsl_rng with the Unix time in nanoseconds
inline void seed_gsl_rng(gsl_rng **r) {
	timespec t_seed;
	clock_gettime(CLOCK_REALTIME, &t_seed);
	long unsigned int seed = 1e9*(long unsigned int)t_seed.tv_sec;
	seed += t_seed.tv_nsec;
	seed ^= (long unsigned int)getpid();
	*r = gsl_rng_alloc(gsl_rng_taus);
	gsl_rng_set(*r, seed);
}
#endif

// Sets inv_A to the inverse of A, and returns the determinant of A. If inv_A is NULL, then
// A is inverted in place. If worspaces p and LU are provided, the function does not have to
// allocate its own workspaces.
double invert_matrix(gsl_matrix* A, gsl_matrix* inv_A=NULL,
                     gsl_permutation* p=NULL, gsl_matrix* LU=NULL);

// Find B s.t. B B^T = A. This is useful for generating vectors from a multivariate normal distribution.
// Operates on A in-place if sqrt_A == NULL.
void sqrt_matrix(gsl_matrix* A, gsl_matrix* sqrt_A, gsl_eigen_symmv_workspace* esv,
                 gsl_vector *eival, gsl_matrix *eivec, gsl_matrix* sqrt_eival);
void sqrt_matrix(gsl_matrix* A, gsl_matrix* sqrt_A=NULL);

// Draw a normal varariate from a covariance matrix. The square-root of the covariance (as defined in sqrt_matrix) must be provided.
void draw_from_cov(double* x, const gsl_matrix* sqrt_cov, unsigned int N, gsl_rng* r);

#endif // _CHAIN_H__
