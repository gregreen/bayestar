/*
 * affine_sampler.h
 * 
 * Implementation of affine sampler from Goodman & Weare (2010),
 * incorporating both stretch and replacement steps.
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

#ifndef _AFFINE_SAMPLER_H__
#define _AFFINE_SAMPLER_H__

#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <time.h>
#include <limits>
#include <assert.h>
#include <omp.h>

#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>

#include <boost/cstdint.hpp>

//#include <opencv2/flann/flann.hpp>

#include "chain.h"
#include "stats.h"

#ifndef GSL_RANGE_CHECK_OFF
#define GSL_RANGE_CHECK_OFF
#endif // GSL_RANGE_CHECK_OFF

/*************************************************************************
 *   Function Prototypes
 *************************************************************************/

class tm;
void seed_gsl_rng(gsl_rng **r);

static void Gelman_Rubin_diagnostic(TStats **stats_arr, unsigned int N_chains, double *R);


/*************************************************************************
 *   Affine Sampler class protoype
 *************************************************************************/

/* An affine-invariant ensemble sampler, introduced by Goodman & Weare (2010). */
template<class TParams, class TLogger>
class TAffineSampler {
	
	// Sampler settings
	unsigned int N;		// Dimensionality of parameter space
	unsigned int L;		// Number of component states in ensemble
	double sqrta;		// Square-root of dimensionless step scale a (a = 2 by default). Can be tuned to achieve desired acceptance rate.
	double h, log_h, twopiN;
	bool use_log;		// If true, <pdf> returns log(pi(X)). Else, <pdf> returns pi(X). Default value is <true>.
	
	// Current state
	struct TState;
	TState* X;		// Ensemble of states
	
	// Proposal states
	TState* Y;		// One proposal per state in ensemble
	bool* accept;		// Whether to accept this state
	
	// Working space for replacement moves
	double* W;
	gsl_vector* wv;
	gsl_eigen_symmv_workspace* ws;
	gsl_matrix* wm1;
	gsl_matrix* wm2;
	gsl_permutation* wp;
	
	// Statistics on ensemble
	double* ensemble_mean;
	gsl_matrix* ensemble_cov;
	gsl_matrix* sqrt_ensemble_cov;
	gsl_matrix* inv_ensemble_cov;
	double det_ensemble_cov;
	double log_norm_ensemble_cov;
	
	// Model for Gaussian mixture proposals
	TGaussianMixture *gm_target;
	
	TParams& params;	// Constant model parameters
	
	// Information about chain
	//TStats stats;		// Stores expectation values, covariance, etc.
	TChain chain;		// Contains the entire chain
	TLogger& logger;	// Object which logs states in the chain
	TState X_ML;		// Maximum likelihood point encountered
	boost::uint64_t N_accepted, N_rejected;		// # of steps which have been accepted and rejected. Used to tune and track acceptance rate.
	boost::uint64_t N_replacements_accepted, N_replacements_rejected;		// # of replacement steps which have been accepted and rejected. Used to track effectiveness of long-range steps.
	
	// Random number generator
	gsl_rng* r;
	
	// Private member functions
	void get_proposal(unsigned int j, double scale);		// Generate a proposal state for sampler j, with the given step scale, using the stretch algorithm (default)
	void replacement_proposal(unsigned int j);			// Generate a proposal state for sampler j, with the given step scale, using the replacement algorithm (long-range steps)
	void mixture_proposal(unsigned int j);				// Generate a proposal state for sampler j from a Gaussian mixture model designed to resemble the target distribution
	void update_ensemble_cov();					// Calculate the covariance of the ensemble, as well as its inverse, determinant and square-root (A A^T = Cov)
	double log_gaussian_density(const TState *const x, const TState *const y);	// Log gaussian density at (x-y) given covariance matrix of ensemble
	
public:
	typedef double (*pdf_t)(const double *const _X, unsigned int _N, TParams& _params);
	typedef void (*rand_state_t)(double *const _X, unsigned int _N, gsl_rng* r, TParams& _params);
	
	// Constructor & destructor
	TAffineSampler(pdf_t _pdf, rand_state_t _rand_state, unsigned int _N, unsigned int _L, TParams& _params, TLogger& _logger, bool _use_log=true);
	~TAffineSampler();
	
	// Mutators
	void step(bool record_step=true, double p_replacement=0.1, double p_mixture=0.1, bool unbalanced=false);	// Advance each sampler in ensemble by one step
	void set_scale(double a);			// Set dimensionless step scale
	void set_replacement_bandwidth(double _h);	// Set smoothing scale to be used for replacement steps, in units of the covariance
	void flush(bool record_steps=true);		// Clear the weights in the ensemble and record the outstanding component states
	void clear();					// Clear the stats, acceptance information and weights
	
	void init_gaussian_mixture_target(unsigned int nclusters, unsigned int iterations=100);
	
	// Accessors
	TLogger& get_logger() { return logger; }
	TParams& get_params() { return params; }
	TStats& get_stats() { return chain.stats; }
	TChain& get_chain() { return chain; }
	double get_scale() { return sqrta*sqrta; }
	double get_replacement_bandwidth() { return h; }
	double get_acceptance_rate() { return (double)N_accepted/(double)(N_accepted+N_rejected); }
	boost::uint64_t get_N_replacements_accepted() { return N_replacements_accepted; }
	boost::uint64_t get_N_replacements_rejected() { return N_replacements_rejected; }
	double get_ln_Z_harmonic(bool use_peak=true, double nsigma_max=1., double nsigma_peak=0.1, double chain_frac=0.1) { return chain.get_ln_Z_harmonic(use_peak, nsigma_max, nsigma_peak, chain_frac); }
	void print_state();
	void print_clusters() { gm_target->print(); };
	
private:
	rand_state_t rand_state;	// Function which generates a random state
	pdf_t pdf;			// pi(X), a function proportional to the target distribution
};


/*************************************************************************
 *   Parallel Affine Sampler Prototype
 *************************************************************************/

template<class TParams, class TLogger>
class TParallelAffineSampler {
	TAffineSampler<TParams, TLogger>** sampler;
	unsigned int N;
	unsigned int N_samplers;
	TStats stats;
	TStats** component_stats;
	TLogger& logger;
	TParams& params;
	double *R;
	
public:
	// Constructor & Destructor
	TParallelAffineSampler(typename TAffineSampler<TParams, TLogger>::pdf_t _pdf, typename TAffineSampler<TParams, TLogger>::rand_state_t _rand_state, unsigned int _N, unsigned int _L, TParams& _params, TLogger& _logger, unsigned int _N_samplers, bool _use_log=true);
	~TParallelAffineSampler();
	
	// Mutators
	void step(unsigned int N_steps, bool record_steps, double cycle=0, double p_replacement=0.1, double p_mixture=0.1, bool unbalanced=false);		// Take the given number of steps in each affine sampler
	void set_scale(double a) { for(unsigned int i=0; i<N_samplers; i++) { sampler[i]->set_scale(a); } };				// Set the dimensionless step size a
	void set_replacement_bandwidth(double h) { for(unsigned int i=0; i<N_samplers; i++) { sampler[i]->set_replacement_bandwidth(h); } };	// Set the dimensionless step size a
	void init_gaussian_mixture_target(unsigned int nclusters, unsigned int iterations=100) { for(unsigned int i=0; i<N_samplers; i++) { sampler[i]->init_gaussian_mixture_target(nclusters, iterations); } };
	void clear() { for(unsigned int i=0; i<N_samplers; i++) { sampler[i]->clear(); }; stats.clear(); };
	
	// Accessors
	TLogger& get_logger() { return logger; }
	TParams& get_params() { return params; }
	TStats& get_stats() { return stats; }
	TStats& get_stats(unsigned int index) { assert(index < N_samplers); return sampler[index]->get_stats(); }
	TChain get_chain();
	void get_GR_diagnostic(double *const GR) { for(unsigned int i=0; i<N; i++) { GR[i] = R[i]; } }
	double get_GR_diagnostic(unsigned int index) { return R[index]; }
	double get_scale(unsigned int index) { assert(index < N_samplers); return sampler[index]->get_scale(); }
	void print_stats();
	void print_state() { for(unsigned int i=0; i<N_samplers; i++) { sampler[i]->print_state(); } }
	void print_clusters() { for(unsigned int i=0; i<N_samplers; i++) { std::cout << std::endl; sampler[i]->print_clusters(); } } 
	TAffineSampler<TParams, TLogger>* const get_sampler(unsigned int index) { assert(index < N_samplers); return sampler[index]; }
	
	// Calculate the GR diagnostic on a transformed space
	void calc_GR_transformed(std::vector<double>& GR, TTransformParamSpace* transf);
};


/*************************************************************************
 *   Structs
 *************************************************************************/

// Component state type
template<class TParams, class TLogger>
struct TAffineSampler<TParams, TLogger>::TState {
	double *element;
	unsigned int N;
	double pi;		// pdf(X) = likelihood of state (up to normalization)
	unsigned int weight;	// # of times the chain has remained on this state
	double replacement_factor;	// Factor of pi(X|Y) / pi(Y|X) used when evaluating acceptance probability of replacement step
	
	TState() : N(0), element(NULL) {}
	TState(unsigned int _N) : N(_N) { element = new double[N]; }
	~TState() { if(element != NULL) { delete[] element; } }
	
	void initialize(unsigned int _N) {
		N = _N;
		if(element == NULL) { element = new double[N]; }
	}
	
	double& operator[](unsigned int index) { return element[index]; }
	
	// Assignment operator
	TState& operator=(const TState& rhs) {
		for(unsigned int i=0; i<N; i++) { element[i] = rhs.element[i]; }
		pi = rhs.pi;
		weight = rhs.weight;
		return *this;
	}
	
	// Compares everything but weight
	bool operator==(const TState& rhs) {
		assert(rhs.N == N);
		if(pi != rhs.pi){ return false; }
		for(unsigned int i=0; i<N; i++) { if(element[i] != rhs.element[i]) { return false; } }
		return true;
	}
	bool operator!=(const TState& rhs) {
		assert(rhs.N == N);
		if(pi != rhs.pi){ return true; }
		for(unsigned int i=0; i<N; i++) { if(element[i] != rhs.element[i]) { return true; } }
		return false;
	}
	
	// The operators > and < compare the likelihood of two states
	bool operator>(const TState& rhs) { return pi > rhs.pi; }
	bool operator<(const TState& rhs) { return pi < rhs.pi; }
	bool operator>(const double& rhs) { return pi > rhs; }
	bool operator<(const double& rhs) { return pi < rhs; }
};



/*************************************************************************
 *   Affine Sampler Class Member Functions
 *************************************************************************/

/*************************************************************************
 *   Constructor and destructor
 *************************************************************************/

// Constructor
// Inputs:
// 	_pdf		Target distribution, up to a normalization constant
// 	_rand_state	Function which generates a random state, used for initialization of the chain
// 	_L		# of concurrent states in the ensemble
// 	_params		Misc. constant model parameters needed by _pdf
// 	_logger		Object which logs the chain in some way. It must have an operator()(double state[N], unsigned int weight).
// 			The logger could, for example, bin the chain, or just push back each state into a vector.
template<class TParams, class TLogger>
TAffineSampler<TParams, TLogger>::TAffineSampler(pdf_t _pdf, rand_state_t _rand_state, unsigned int _N, unsigned int _L, TParams& _params, TLogger& _logger, bool _use_log)
	: pdf(_pdf), rand_state(_rand_state), params(_params), logger(_logger), N(_N), L(_L), X(NULL), Y(NULL), accept(NULL),
	  r(NULL), use_log(_use_log), chain(_N, 1000*_L), W(NULL), ensemble_mean(NULL), ensemble_cov(NULL), sqrt_ensemble_cov(NULL),
	  inv_ensemble_cov(NULL), wv(NULL), ws(NULL), wm1(NULL), wm2(NULL), wp(NULL), gm_target(NULL)
{
	// Seed the random number generator
	seed_gsl_rng(&r);
	
	// Generate the initial state and record the most likely point
	X = new TState[L];
	Y = new TState[L];
	accept = new bool[L];
	for(unsigned int i=0; i<L; i++) {
		X[i].initialize(N);
		Y[i].initialize(N);
	}
	unsigned int index_of_best = 0;
	unsigned int max_tries = 100;
	unsigned int tries;
	for(unsigned int i=0; i<L; i++) {
		rand_state(X[i].element, N, r, params);
		X[i].pi = pdf(X[i].element, N, params);
		
		// Re-seed points that land at zero probability
		tries = 0;
		while((   (_use_log && isinf(X[i].pi))
		       || (!_use_log && X[i].pi <=  2. * std::numeric_limits<double>::min()) )
		       && (tries < max_tries)) {
			rand_state(X[i].element, N, r, params);
			X[i].pi = pdf(X[i].element, N, params);
			tries++;
			//#pragma omp critical
			//{
			//std::cerr << "! Re-seeding: " << tries << std::endl;
			//}
		}
		
		X[i].weight = 1;
		if(X[i] > X[index_of_best]) { index_of_best = i; }
	}
	X_ML = X[index_of_best];
	
	// Create working space for replacement move
	W = new double[N];
	ensemble_mean = new double[N];
	ensemble_cov = gsl_matrix_alloc(N, N);
	sqrt_ensemble_cov = gsl_matrix_alloc(N, N);
	inv_ensemble_cov = gsl_matrix_alloc(N, N);
	wv = gsl_vector_alloc(N);
	ws = gsl_eigen_symmv_alloc(N);
	wm1 = gsl_matrix_alloc(N, N);
	wm2 = gsl_matrix_alloc(N, N);
	wp = gsl_permutation_alloc(N);
	twopiN = pow(2.*3.14159265358979, (double)N);
	
	// Replacement move smoothing scale, in units of the ensemble covariance
	set_replacement_bandwidth(0.20);
	
	// Set the initial step scale. 2 is good for most situations.
	set_scale(2);
	
	// Initialize number of accepted and rejected steps to zero
	N_accepted = 0;
	N_rejected = 0;
	N_replacements_accepted = 0;
	N_replacements_rejected = 0;
}

// Destructor
template<class TParams, class TLogger>
TAffineSampler<TParams, TLogger>::~TAffineSampler() {
	gsl_rng_free(r);
	if(X != NULL) { delete[] X; X = NULL; }
	if(Y != NULL) { delete[] Y; Y = NULL; }
	if(accept != NULL) { delete[] accept; accept = NULL; }
	if(W != NULL) { delete[] W; W = NULL; }
	if(ensemble_mean != NULL) { delete[] ensemble_mean; ensemble_mean = NULL; }
	gsl_matrix_free(ensemble_cov);
	gsl_matrix_free(sqrt_ensemble_cov);
	gsl_matrix_free(inv_ensemble_cov);
	gsl_vector_free(wv);
	gsl_eigen_symmv_free(ws);
	gsl_matrix_free(wm1);
	gsl_matrix_free(wm2);
	gsl_permutation_free(wp);
	if(gm_target != NULL) { delete gm_target; }
}


/*************************************************************************
 *   Private functions
 *************************************************************************/

// Generate a proposal state
template<class TParams, class TLogger>
inline void TAffineSampler<TParams, TLogger>::get_proposal(unsigned int j, double scale) {
	// Choose a sampler to stretch from
	unsigned int k = gsl_rng_uniform_int(r, (long unsigned int)L - 1);
	if(k >= j) { k += 1; }
	// Determine the coordinates of the proposal
	for(unsigned int i=0; i<N; i++) {
		Y[j].element[i] = (1. - scale) * X[k].element[i] + scale * X[j].element[i];
	}
	// Get pdf(Y) and initialize weight of proposal point to unity
	Y[j].pi = pdf(Y[j].element, N, params);
	Y[j].weight = 1;
	Y[j].replacement_factor = 1.;
}


// Calculate the transformation matrix A s.t. AA^T = S, where S is the covariance matrix.
// wv, wm1, wm2 and wm3 are workspaces required by the algorithm. The dimensions of wv and ws must be N, while wm1 and wm2 must have dimensions NxN.
static void calc_sqrt_A(gsl_matrix *A, const gsl_matrix * const S, gsl_vector* wv, gsl_eigen_symmv_workspace* ws, gsl_matrix* wm1, gsl_matrix* wm2) {
	assert(A->size1 == A->size2);
	assert(S->size1 == S->size2);
	assert(A->size1 == S->size1);
	size_t N = S->size1;
	// Calculate the eigendecomposition of the covariance matrix
	gsl_matrix_memcpy(A, S);
	gsl_eigen_symmv(A, wv, wm1, ws);	// wv = eigenvalues, wm1 = eigenvectors
	gsl_matrix_set_zero(wm2);	// wm2 will have sqrt of eigenvalues along diagonal
	double tmp;
	for(size_t i=0; i<N; i++) {
		tmp = gsl_vector_get(wv, i);
		gsl_matrix_set(wm2, i, i, sqrt(fabs(tmp)));
		if(tmp < 0.) {
			for(size_t j=0; j<N; j++) { gsl_matrix_set(wm1, j, i, -gsl_matrix_get(wm1, j, i)); }
		}
	}
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1., wm1, wm2, 0., A);
}

template<class TParams, class TLogger>
void TAffineSampler<TParams, TLogger>::update_ensemble_cov() {
	// Mean
	for(unsigned int i=0; i<N; i++) {
		ensemble_mean[i] = 0.;
		for(unsigned int n=0; n<L; n++) { ensemble_mean[i] += X[n].element[i]; }
		ensemble_mean[i] /= (double)L;
	}
	
	// Covariance
	double tmp;
	for(unsigned int j=0; j<N; j++) {
		for(unsigned int k=j; k<N; k++) {
			tmp = 0.;
			for(unsigned int n=0; n<L; n++) { tmp += (X[n].element[j] - ensemble_mean[j]) * (X[n].element[k] - ensemble_mean[k]); }
			tmp /= (double)(L - 1);
			if(k == j) {
				gsl_matrix_set(ensemble_cov, j, k, tmp);//*1.005 + 0.005);		// Small factor added in to avoid singular matrices
			} else {
				gsl_matrix_set(ensemble_cov, j, k, tmp);
				gsl_matrix_set(ensemble_cov, k, j, tmp);
			}
		}
	}
	
	// Inverse and Sqrt of Covariance
	det_ensemble_cov = invert_matrix(ensemble_cov, inv_ensemble_cov, wp, wm1);
	sqrt_matrix(ensemble_cov, sqrt_ensemble_cov, ws, wv, wm1, wm2);
	log_norm_ensemble_cov = -0.5 * log(fabs(det_ensemble_cov) * twopiN);
}

// Get the density Gaussian proposal distribution
template<class TParams, class TLogger>
double TAffineSampler<TParams, TLogger>::log_gaussian_density(const TState *const x, const TState *const y) {
	double sum = 0.;
	double tmp;
	for(unsigned int i=0; i<N; i++) {
		tmp = (x->element[i] - y->element[i]);
		sum += tmp * gsl_matrix_get(inv_ensemble_cov, i, i) * tmp;
		for(unsigned int j=i+1; j<N; j++) {
			sum += 2. * tmp * gsl_matrix_get(inv_ensemble_cov, i, j) * (x->element[j] - y->element[j]);
		}
	}
	//double w;
	//for(unsigned int i=0; i<N; i++) {
	//	w = 0.;
	//	for(unsigned int j=0; j<N; j++) { w += (x->element[j] - y->element[j]) * gsl_matrix_get(inv_ensemble_cov, i, j); }
	//	sum += w * (x->element[i] - y->element[i]);
	//}
	return -(double)N * log_h + log_norm_ensemble_cov - sum/(2.*h*h);
}

template<class TParams, class TLogger>
void TAffineSampler<TParams, TLogger>::replacement_proposal(unsigned int j) {
//#pragma omp critical (replacement)
//{
	// Choose a sampler to step from
	unsigned int k = gsl_rng_uniform_int(r, (long unsigned int)L);
	
	// Determine step vector
	draw_from_cov(W, sqrt_ensemble_cov, N, r);
	
	// Determine the coordinates of the proposal
	//#pragma omp critical
	//{
	//std::cout << "W = ";
	for(unsigned int i=0; i<N; i++) {
		Y[j].element[i] = X[k].element[i] + h * W[i];
	//	std::cout << W[i] << " ";
	}
	//std::cout << std::endl;
	//}
	
//#pragma omp critical
//{
	// Determine pi_S(X_j | Y_j , X_{-j})
	double tmp;
	double max = -std::numeric_limits<double>::infinity();
	double cutoff = 4.;
	double pi_XY = 0.;
	for(unsigned int i=0; i<L; i++) {
		if(i != j) {
			tmp = log_gaussian_density(&(X[i]), &(X[j]));
//			std::cout << "log(p) = " << tmp << std::endl;
			if(tmp > max) { max = tmp; }
			if(tmp >= max - cutoff) { pi_XY += exp(tmp); }
		}
	}
	tmp = log_gaussian_density(&(Y[j]), &(X[j]));
	if(tmp >= max - cutoff) { pi_XY += exp(tmp); }
//	std::cout << "pi(X|Y) = " << pi_XY << "\t" << "(max = " << max << ")" << std::endl;
	
	// Determine pi_S(Y_j | X)
	max = -std::numeric_limits<double>::infinity();
	double pi_YX = 0.;
	for(unsigned int i=0; i<L; i++) {
		tmp = log_gaussian_density(&(X[i]), &(Y[j]));
		if(tmp > max) { max = tmp; }
		if(tmp >= max - cutoff) { pi_YX += exp(tmp); }
	}
//	std::cout << "pi(Y|X) = " << pi_YX << "\t" << "(max = " << max << ")" << std::endl << std::endl;

	// Get pdf(Y) and initialize weight of proposal point to unity
	Y[j].pi = pdf(Y[j].element, N, params);
	Y[j].weight = 1.;
	Y[j].replacement_factor = pi_XY / pi_YX;
//}
	
//	std::cout << Y[j].pi << "\t" << X[j].pi << "\t" << Y[j].replacement_factor << std::endl << std::endl;
}

template<class TParams, class TLogger>
void TAffineSampler<TParams, TLogger>::mixture_proposal(unsigned int j) {
	// Draw from Gaussian mixture
	gm_target->draw(Y[j].element);
	
	Y[j].pi = pdf(Y[j].element, N, params);
	Y[j].weight = 1.;
	
	// Determine Q(Y) / Q(X)
	Y[j].replacement_factor = gm_target->density(X[j].element) / gm_target->density(Y[j].element);
}

template<class TParams, class TLogger>
void TAffineSampler<TParams, TLogger>::init_gaussian_mixture_target(unsigned int nclusters, unsigned int iterations) {
	if(gm_target != NULL) { delete gm_target; }
	gm_target = new TGaussianMixture(N, nclusters);
	get_chain().fit_gaussian_mixture(gm_target, iterations);
}


/*************************************************************************
 *   Mutators
 *************************************************************************/

template<class TParams, class TLogger>
void TAffineSampler<TParams, TLogger>::step(bool record_step, double p_replacement, double p_mixture, bool unbalanced) {
	double scale, alpha, p;
	unsigned int step_type;
	bool ensemble_cov_updated = false;
	for(unsigned int j=0; j<L; j++) {
		// Make either a stretch or a replacement step
		p = gsl_rng_uniform(r);
		if(p < p_replacement) {
			if(!ensemble_cov_updated) {
				update_ensemble_cov();
				ensemble_cov_updated = true;
			}
			replacement_proposal(j);
			step_type = 1;
		} else if((p < p_replacement + p_mixture) && (gm_target != NULL)) {
			mixture_proposal(j);
			step_type = 2;
		} else {
			// Determine the step scale and draw a proposal
			scale = (sqrta - 1./sqrta) * gsl_rng_uniform(r) + 1./sqrta;
			scale *= scale;
			get_proposal(j, scale);
			step_type = 0;
		}
		
		// Determine if the proposal is the maximum-likelihood point
		if(Y[j].pi > X_ML.pi) { X_ML = Y[j]; }
		
		// Determine whether to accept or reject
		accept[j] = false;
		if(use_log) {	// If <pdf> returns log probability
			// Determine the acceptance probability
			if(isinff(X[j].pi) && !(isinff(Y[j].pi))) {
				alpha = 1;	// Accept the proposal if the current state has zero probability and the proposed state doesn't
			} else {
				if(step_type == 0) {
					alpha = (double)(N - 1) * log(scale) + Y[j].pi - X[j].pi;
				} else if(unbalanced) {
					alpha = Y[j].pi - X[j].pi;	// Ignore detailed balance. Use carefully - does not sample from target!
				} else {
					alpha = Y[j].pi - X[j].pi + log(Y[j].replacement_factor);
					//std::cout << "alpha = " << alpha << std::endl;
				}
			}
			// Decide whether to accept or reject
			if(alpha > 0.) {	// Accept if probability of acceptance is greater than unity
				accept[j] = true;
			} else {
				p = gsl_rng_uniform(r);
				if((p == 0.) && (Y[j] > -std::numeric_limits<double>::infinity())) {	// Accept if zero is rolled but proposal has nonzero probability
					accept[j] = true;
				} else if(log(p) < alpha) {
					accept[j] = true;
				}
			}
		} else {	// If <pdf> returns bare probability
			// Determine the acceptance probability
			if((X[j].pi == 0) && (Y[j].pi != 0)) {
				alpha = 2;	// Accept the proposal if the current state has zero probability and the proposed state doesn't
			} else {
				if(step_type == 0) {
					alpha = pow(scale, (double)(N - 1)) * Y[j].pi / X[j].pi;
				} else {
					alpha = Y[j].pi / X[j].pi * Y[j].replacement_factor;
				}
			}
			// Decide whether to accept or reject
			if(alpha > 0.) {	// Accept if probability of acceptance is greater than unity
				accept[j] = true;
			} else {
				p = gsl_rng_uniform(r);
				if((p == 0.) && (Y[j] != 0.)) {	// Accept if zero is rolled but proposal has nonzero probability
					accept[j] = true;
				} else if(p < alpha) {
					accept[j] = true;
				}
			}
		}
	}
	// Update ensemble
	for(unsigned int j=0; j<L; j++) {
		// Update sampler j
		if(accept[j]) {
			if(record_step) {
				//stats(X[j].element, X[j].weight);
				chain.add_point(X[j].element, X[j].pi, (double)(X[j].weight));
				#pragma omp critical (logger)
				logger(X[j].element, X[j].weight);
			}
			//if(step_type == 1) {
			//	#pragma omp critical
			//	std::cout << Y[j].element[0] - X[j].element[0] << std::endl;
			//}
			X[j] = Y[j];
			N_accepted++;
			if(step_type) { N_replacements_accepted++; }
		} else {
			X[j].weight++;
			N_rejected++;
			if(step_type) { N_replacements_rejected++; }
		}
	}
}

// Set the dimensionless step scale
template<class TParams, class TLogger>
void TAffineSampler<TParams, TLogger>::set_scale(double a) {
	sqrta = sqrt(a);
}

template<class TParams, class TLogger>
void TAffineSampler<TParams, TLogger>::set_replacement_bandwidth(double _h) {
	h = _h;
	log_h = log(h);
}

template<class TParams, class TLogger>
void TAffineSampler<TParams, TLogger>::flush(bool record_steps) {
	for(unsigned int i=0; i<L; i++) {
		if(record_steps) {
			//stats(X[i].element, X[i].weight);
			chain.add_point(X[i].element, X[i].pi, (double)(X[i].weight));
			#pragma omp critical (logger)
			logger(X[i].element, X[i].weight);
		}
		X[i].weight = 0;
	}
}

// Clear the stats, acceptance information and weights
template<class TParams, class TLogger>
void TAffineSampler<TParams, TLogger>::clear() {
	for(unsigned int i=0; i<L; i++) {
		X[i].weight = 0;
	}
	//stats.clear();
	chain.clear();
	N_accepted = 0;
	N_rejected = 0;
	N_replacements_accepted = 0;
	N_replacements_rejected = 0;
}



/*************************************************************************
 *   Accessors
 *************************************************************************/

template<class TParams, class TLogger>
void TAffineSampler<TParams, TLogger>::print_state() {
	for(unsigned int i=0; i<L; i++) {
		std::cout << "p(X) = " << X[i].pi << std::endl;
		std::cout << "Weight = " << X[i].weight << std::endl << "X [" << i << "] = { ";
		for(unsigned int j=0; j<N; j++) { std::cout << (j == 0 ? "" : " ") << std::setprecision(3) << X[i].element[j]; }
		std::cout << " }" << std::endl << std::endl;
	}
}



/*************************************************************************
 *   Parallel Affine Sampler Class Member Functions
 *************************************************************************/

template<class TParams, class TLogger>
TParallelAffineSampler<TParams, TLogger>::TParallelAffineSampler(typename TAffineSampler<TParams, TLogger>::pdf_t _pdf, typename TAffineSampler<TParams, TLogger>::rand_state_t _rand_state, unsigned int _N, unsigned int _L, TParams& _params, TLogger& _logger, unsigned int _N_samplers, bool _use_log)
	: logger(_logger), params(_params), N(_N), sampler(NULL), component_stats(NULL), R(NULL), stats(_N)
{
	assert(_N_samplers > 1);
	N_samplers = _N_samplers;
	sampler = new TAffineSampler<TParams, TLogger>*[N_samplers];
	component_stats = new TStats*[N_samplers];
	for(unsigned int i=0; i<N_samplers; i++) { sampler[i] = NULL; component_stats[i] = NULL; }
	#pragma omp parallel for
	for(unsigned int i=0; i<N_samplers; i++) {
		sampler[i] = new TAffineSampler<TParams, TLogger>(_pdf, _rand_state, N, _L, _params, _logger, _use_log);
		component_stats[i] = &(sampler[i]->get_stats());
	}
	
	R = new double[N];
}

template<class TParams, class TLogger>
TParallelAffineSampler<TParams, TLogger>::~TParallelAffineSampler() {
	if(sampler != NULL) {
		for(unsigned int i=0; i<N_samplers; i++) { if(sampler[i] != NULL) { delete sampler[i]; } }
		delete[] sampler;
	}
	if(component_stats != NULL) { delete[] component_stats; }
	if(R != NULL) { delete[] R; }
}

template<class TParams, class TLogger>
void TParallelAffineSampler<TParams, TLogger>::step(unsigned int N_steps, bool record_steps, double cycle, double p_replacement, double p_mixture, bool unbalanced) {
	//omp_set_num_threads(N_samplers);
	#pragma omp parallel firstprivate(record_steps, N_steps, cycle, p_replacement, p_mixture, unbalanced) num_threads(N_samplers)
	{
		unsigned int thread_ID = omp_get_thread_num();
		double base_a = sampler[thread_ID]->get_scale();
		for(unsigned int i=0; i<N_steps; i++) {
			if(cycle > 1) {
				if((i % 10) == 0) {
					sampler[thread_ID]->set_scale(base_a*cycle);
				} else if((i % 10) == 1) {
					sampler[thread_ID]->set_scale(base_a);
				}
			}
			sampler[thread_ID]->step(record_steps, p_replacement, p_mixture, unbalanced);
		}
		sampler[thread_ID]->flush(record_steps);
		#pragma omp critical (append_stats)
		stats += sampler[thread_ID]->get_stats();
		
		#pragma omp barrier
	}
	Gelman_Rubin_diagnostic(component_stats, N_samplers, R, N);
}

template<class TParams, class TLogger>
void TParallelAffineSampler<TParams, TLogger>::print_stats() {
	stats.print();
	std::cout << std::endl << "Gelman-Rubin diagnostic:" << std::endl;
	for(unsigned int i=0; i<N; i++) { std::cout << (i==0 ? "" : "\t") << std::setprecision(5) << R[i]; }
	std::cout << std::endl;
	std::cout << "Acceptance rate: ";
	for(unsigned int i=0; i<N_samplers; i++) { std::cout << std::setprecision(3) << 100.*get_sampler(i)->get_acceptance_rate() << "%" << (i != N_samplers - 1 ? " " : ""); }
	std::cout << std::endl;
	std::cout << "Replacements accepted/rejected: ";
	for(unsigned int i=0; i<N_samplers; i++) { std::cout << get_sampler(i)->get_N_replacements_accepted() << "/" << get_sampler(i)->get_N_replacements_rejected() << (i != N_samplers - 1 ? " " : ""); }
	std::cout << std::endl;
}

template<class TParams, class TLogger>
TChain TParallelAffineSampler<TParams, TLogger>::get_chain() {
	unsigned int capacity = 0;
	for(unsigned int i=0; i<N_samplers; i++) {
		capacity += sampler[i]->get_chain().get_length();
	}
	TChain tmp(N, capacity);
	for(unsigned int i=0; i<N_samplers; i++) {
		tmp += sampler[i]->get_chain();
	}
	return tmp;
}

template<class TParams, class TLogger>
void TParallelAffineSampler<TParams, TLogger>::calc_GR_transformed(std::vector<double>& GR, TTransformParamSpace* transf) {
	TStats **transf_stats = new TStats*[N_samplers];
	for(size_t n=0; n<N_samplers; n++) {
		transf_stats[n] = new TStats(N);
	}
	
	#pragma omp parallel num_threads(N_samplers)
	{
		size_t n = omp_get_thread_num();
		TStats& transf_comp_stat = *(transf_stats[n]);
		TChain& chain = sampler[n]->get_chain();
		size_t n_points = chain.get_length();
		
		double* y = new double[N];
		
		for(size_t i=0; i<n_points; i++) {
			(*transf)(chain.get_element(i), y);
			transf_comp_stat(y, (unsigned int)(chain.get_w(i)));
		}
		
		delete y;
		
		#pragma omp barrier
	}
	
	GR.resize(N);
	Gelman_Rubin_diagnostic(transf_stats, N_samplers, GR.data(), N);
	
	for(size_t n=0; n<N_samplers; n++) {
		delete transf_stats[n];
	}
	delete[] transf_stats;
}



/*************************************************************************
 *   Auxiliary Functions
 *************************************************************************/

#ifndef __SEED_GSL_RNG_
#define __SEED_GSL_RNG_
// Seed a gsl_rng with the Unix time in nanoseconds
inline void seed_gsl_rng(gsl_rng **r) {
	timespec t_seed;
	clock_gettime(CLOCK_REALTIME, &t_seed);
	long unsigned int seed = 1e9*(long unsigned int)t_seed.tv_sec;
	seed += t_seed.tv_nsec;
	*r = gsl_rng_alloc(gsl_rng_taus);
	gsl_rng_set(*r, seed);
}
#endif


/*************************************************************************
 *   Null logger:
 * 	Fulfills the role of a logger for the affine sampler,
 * 	but doesn't actually log anything.
 *************************************************************************/

struct TNullLogger {
	void operator()(double* element, double weight) {}
};




#endif // _AFFINE_SAMPLER_H__