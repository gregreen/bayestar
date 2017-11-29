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

#include <unistd.h>

#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>

#include <boost/cstdint.hpp>

//#include <opencv2/flann/flann.hpp>

#include "definitions.h"
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
	double logL;		// Log of ensemble size
	double sqrta;		// Square-root of dimensionless step scale a (a = 2 by default). Can be tuned to achieve desired acceptance rate.
	double h, log_h, h_MH, log_h_MH;
	double replacement_accept_bias;
	double twopiN;
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
	double sigma_min;
	
	// Diagonal approximation of ensemble covariance
	double* diag_cov;
	double* sqrt_diag_cov;
	double* inv_diag_cov;
	double det_diag_cov;
	double log_norm_diag_cov;
	
	// Model for Gaussian mixture proposals
	TGaussianMixture *gm_target;
	
	TParams& params;	// Constant model parameters
	
	// Information about chain
	//TStats stats;		// Stores expectation values, covariance, etc.
	TChain chain;		// Contains the entire chain
	TLogger& logger;	// Object which logs states in the chain
	TState X_ML;		// Maximum likelihood point encountered
	boost::uint64_t N_accepted, N_rejected;		// # of steps which have been accepted and rejected. Used to tune and track acceptance rate.
	boost::uint64_t N_stretch_accepted, N_stretch_rejected;	// # of stretch steps accepted/rejected
	boost::uint64_t N_replacements_accepted, N_replacements_rejected;	// # of replacement steps which have been accepted and rejected. Used to track effectiveness of long-range steps.
	boost::uint64_t N_MH_accepted, N_MH_rejected;	// # of Metroplis-Hastings steps accepted/rejected
	boost::uint64_t N_custom_accepted, N_custom_rejected;	// # of custom reversible steps accepted/rejected
	
	// Random number generator
	gsl_rng* r;
	
	// Private member functions
	void affine_proposal(unsigned int j, double& scale);		// Generate a proposal state for sampler j, with the given step scale, using the stretch algorithm (default)
	void replacement_proposal(unsigned int j, bool unbalanced);	// Generate a proposal state for sampler j using the replacement algorithm (long-range steps)
	void replacement_proposal_diag(unsigned int j, bool unbalanced);	// Geenrate proposal state using replacement algorithm (with diagonal covariance)
	void mixture_proposal(unsigned int j);				// Generate a proposal state for sampler j from a Gaussian mixture model designed to resemble the target distribution
	void MH_proposal(unsigned int j);				// Generate a Metropolis-Hastings proposal for sampler j
	void update_ensemble_cov();					// Calculate the covariance of the ensemble, as well as its inverse, determinant and square-root (A A^T = Cov)
	double log_gaussian_density(const TState *const x, const TState *const y);	// Log gaussian density at (x-y) given covariance matrix of ensemble
	double log_gaussian_density_diag(const TState *const x, const TState *const y);	// Log gaussian density at (x-y) given diagonal approximation of covariance matrix of ensemble
	
public:
	typedef double (*pdf_t)(const double *const _X, unsigned int _N, TParams& _params);
	typedef void (*rand_state_t)(double *const _X, unsigned int _N, gsl_rng* r, TParams& _params);
	typedef double (*reversible_step_t)(double *const _X, double *const _Y, unsigned int _N, gsl_rng* r, TParams& _params);
	
	// Constructor & destructor
	TAffineSampler(pdf_t _pdf, rand_state_t _rand_state, unsigned int _N, unsigned int _L, TParams& _params, TLogger& _logger, bool _use_log=true);
	~TAffineSampler();
	
	// Mutators
	void step(bool record_step=true, double p_replacement=0.1,
	          bool unbalanced=false, bool diag_approx=false);	// Advance each sampler in ensemble by one step
	void step_affine(bool record_step=true);					
	void step_replacement(bool record_step=true, bool unbalanced=false, bool diag_approx=false);	// Replacement step using full covariance (affine invariant)
	void step_MH(bool record_step=true);		// Advance each sampler using Metropolis-Hastings step
	void step_custom_reversible(reversible_step_t f_reversible_step, bool record_step=true);
	void set_scale(double a);			// Set dimensionless step scale
	void set_replacement_bandwidth(double _h);	// Set smoothing scale to be used for replacement steps, in units of the covariance
	void set_MH_bandwidth(double _h);
	void set_replacement_accept_bias(double epsilon);
	void set_sigma_min(double _sigma_min);
	void flush(bool record_steps=true);		// Clear the weights in the ensemble and record the outstanding component states
	void clear();					// Clear the stats, acceptance information and weights
	
	void init_gaussian_mixture_target(unsigned int nclusters, unsigned int iterations=100);
	
	// Accessors
	TLogger& get_logger() { return logger; }
	TParams& get_params() { return params; }
	TStats& get_stats() { return chain.stats; }
	TChain& get_chain() { return chain; }
	unsigned int get_N_walkers() { return L; }
	double get_scale() { return sqrta*sqrta; }
	double get_replacement_bandwidth() { return h; }
	double get_MH_bandwidth() { return h_MH; }
	double get_acceptance_rate() { return (double)N_accepted/(double)(N_accepted+N_rejected); }
	double get_stretch_acceptance_rate() { return (double)(N_stretch_accepted) / (double)(N_stretch_accepted + N_stretch_rejected); }
	double get_replacement_acceptance_rate() { return (double)N_replacements_accepted / (double)(N_replacements_accepted + N_replacements_rejected); }
	double get_MH_acceptance_rate() { return (double)N_MH_accepted / (double)(N_MH_accepted + N_MH_rejected); }
	double get_custom_acceptance_rate() { return (double)(N_custom_accepted) / (double)(N_custom_accepted + N_custom_rejected); }
	boost::uint64_t get_N_stretch_accepted() { return N_stretch_accepted; }
	boost::uint64_t get_N_stretch_rejected() { return N_stretch_rejected; }
	boost::uint64_t get_N_replacements_accepted() { return N_replacements_accepted; }
	boost::uint64_t get_N_replacements_rejected() { return N_replacements_rejected; }
	boost::uint64_t get_N_MH_accepted() { return N_MH_accepted; }
	boost::uint64_t get_N_MH_rejected() { return N_MH_rejected; }
	boost::uint64_t get_N_custom_accepted() { return N_custom_accepted; }
	boost::uint64_t get_N_custom_rejected() { return N_custom_rejected; }
	double get_ln_Z_harmonic(bool use_peak=true, double nsigma_max=1., double nsigma_peak=0.1, double chain_frac=0.1) { return chain.get_ln_Z_harmonic(use_peak, nsigma_max, nsigma_peak, chain_frac); }
	void print_state();
	void print_stats();
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
	void step(unsigned int N_steps, bool record_steps, double cycle=0,
	          double p_replacement=0.1, bool unbalanced=false, bool diag_approx=false);	// Take the given number of steps in each affine sampler
	void step_MH(unsigned int N_steps, bool record_steps);		// Take the given number of Metropolis-Hastings steps in each affine sampler
	void step_custom_reversible(unsigned int N_steps,
	                            typename TAffineSampler<TParams, TLogger>::reversible_step_t f_reversible_step,
	                            bool record_steps);	// Take given number of steps using custom user-provided reversible step
	void tune_stretch(unsigned int N_rounds, double target_acceptance);	// Adjust stretch scale to achieve desired acceptance rate
	void tune_MH(unsigned int N_rounds, double target_acceptance);		// Adjust step size to achieve desired acceptance rate
	void set_scale(double a) { for(unsigned int i=0; i<N_samplers; i++) { sampler[i]->set_scale(a); } };				// Set the dimensionless step size a
	void set_replacement_bandwidth(double h) { for(unsigned int i=0; i<N_samplers; i++) { sampler[i]->set_replacement_bandwidth(h); } };	// Set size of replacement steps (in units of covariance) 
	void set_MH_bandwidth(double h) { for(unsigned int i=0; i<N_samplers; i++) { sampler[i]->set_MH_bandwidth(h); } };	// Set size of M-H steps (in units of covariance) 
	void set_replacement_accept_bias(double epsilon) { for(unsigned int i=0; i<N_samplers; i++) { sampler[i]->set_replacement_accept_bias(epsilon); } };
	void set_sigma_min(double _sigma_min) { for(unsigned int i=0; i<N_samplers; i++) { sampler[i]->set_sigma_min(_sigma_min); } };
	void init_gaussian_mixture_target(unsigned int nclusters, unsigned int iterations=100) { for(unsigned int i=0; i<N_samplers; i++) { sampler[i]->init_gaussian_mixture_target(nclusters, iterations); } };
	void clear() { for(unsigned int i=0; i<N_samplers; i++) { sampler[i]->clear(); }; stats.clear(); };
	
	// Accessors
	TLogger& get_logger() { return logger; }
	TParams& get_params() { return params; }
	void calc_stats();
	TStats& get_stats() { calc_stats(); return stats; }
	TStats& get_stats(unsigned int index) { assert(index < N_samplers); return sampler[index]->get_stats(); }
	TChain get_chain();
	void get_GR_diagnostic(double *const GR) { for(unsigned int i=0; i<N; i++) { GR[i] = R[i]; } }
	double get_GR_diagnostic(unsigned int index) { return R[index]; }
	double get_scale(unsigned int index) { assert(index < N_samplers); return sampler[index]->get_scale(); }
	double get_replacement_bandwidth(unsigned int index) { assert(index < N_samplers); return sampler[index]->get_replacement_bandwidth(); }
	double get_MH_bandwidth(unsigned int index) { assert(index < N_samplers); return sampler[index]->get_MH_bandwidth(); }
	unsigned int get_N_samplers() { return N_samplers; }
	void print_stats();
	void print_diagnostics();
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
	double replacement_factor;	// Factor of Q(Y->X) / Q(X->Y) used when evaluating acceptance probability of replacement step
	
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
	  inv_ensemble_cov(NULL), wv(NULL), ws(NULL), wm1(NULL), wm2(NULL), wp(NULL), gm_target(NULL),
	  diag_cov(NULL), sqrt_diag_cov(NULL), inv_diag_cov(NULL)
{
	// Seed the random number generator
	seed_gsl_rng(&r);
	
	logL = log(L);
	
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
		while((   (_use_log && is_neg_inf_replacement(X[i].pi))
		       || (!_use_log && X[i].pi <=  min_replacement) )
		       && (tries < max_tries)) {
			rand_state(X[i].element, N, r, params);
			X[i].pi = pdf(X[i].element, N, params);
			tries++;
		}
		if(tries >= max_tries) {
			#pragma omp critical
			{
			std::cerr << "! Re-seeding failed !" << std::endl;
			std::cerr << "p(X) = " << X[i].pi << std::endl;
			std::cerr << "X =";
			for(int k=0; k<N; k++) {
				std::cerr << " " << X[i].element[k];
			}
			std::cerr << std::endl;
			}
			
			//X[i].pi = pdf(X[i].element, N, params);
			
			abort();
		}
		
		//#pragma omp critical
		//{
		//std::cout << tries << std::endl;
		//}
		
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
	
	diag_cov = new double[N];
	sqrt_diag_cov = new double[N];
	inv_diag_cov = new double[N];
	
	// Replacement move smoothing scale, in units of the ensemble covariance
	set_replacement_bandwidth(0.50);
	
	// Set Metropolis-Hastings step size, in units of ensemble covariance
	set_MH_bandwidth(0.25);
	
	// Set the initial step scale. 2 is good for most situations.
	set_scale(2.);
	
	// Set the replacement sampler to be ergodic
	set_replacement_accept_bias(0.);
	
	// Set minimum proposal kernel size for M-H and replacement steps
	set_sigma_min(0.);
	
	// Initialize number of accepted and rejected steps to zero
	N_accepted = 0;
	N_rejected = 0;
	N_stretch_accepted = 0;
	N_stretch_rejected = 0;
	N_replacements_accepted = 0;
	N_replacements_rejected = 0;
	N_MH_accepted = 0;
	N_MH_rejected = 0;
	N_custom_accepted = 0;
	N_custom_rejected = 0;
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
	if(diag_cov != NULL) { delete[] diag_cov; diag_cov = NULL; }
	if(sqrt_diag_cov != NULL) { delete[] sqrt_diag_cov; sqrt_diag_cov = NULL; }
	if(inv_diag_cov != NULL) { delete[] inv_diag_cov; inv_diag_cov = NULL; }
}


/*************************************************************************
 *   Private functions
 *************************************************************************/

// Generate a proposal state
template<class TParams, class TLogger>
inline void TAffineSampler<TParams, TLogger>::affine_proposal(unsigned int j, double& scale) {
	// Determine stretch scale
	scale = (sqrta - 1./sqrta) * gsl_rng_uniform(r) + 1./sqrta;
	scale *= scale;
	
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
	double sum_weight = 0.;
	double weight;
	
	// Find probability density of best point in ensemble
	double pi_0 = neg_inf_replacement;
	for(unsigned int n=0; n<L; n++) {
		if(X[n].pi > pi_0) { pi_0 = X[n].pi; }
	}
	
	// Mean
	for(unsigned int i=0; i<N; i++) { ensemble_mean[i] = 0.; }
	
	if(use_log) {
		for(unsigned int n=0; n<L; n++) {
			weight = exp(X[n].pi - pi_0);
			sum_weight += weight;
			for(unsigned int i=0; i<N; i++) { ensemble_mean[i] += weight * X[n].element[i]; }
		}
	} else {
		for(unsigned int n=0; n<L; n++) {
			weight = X[n].pi / pi_0;
			sum_weight += weight;
			for(unsigned int i=0; i<N; i++) { ensemble_mean[i] += weight * X[n].element[i]; }
		}
	}
	
	for(unsigned int i=0; i<N; i++) { ensemble_mean[i] /= sum_weight; }
	
	// Covariance
	double tmp;
	
	if(use_log) {
		for(unsigned int j=0; j<N; j++) {
			for(unsigned int k=j; k<N; k++) {
				gsl_matrix_set(ensemble_cov, j, k, 0.);
			}
		}
		
		for(unsigned int n=0; n<L; n++) {
			weight = exp(X[n].pi - pi_0);
			
			for(unsigned int j=0; j<N; j++) {
				for(unsigned int k=j; k<N; k++) {
					tmp = gsl_matrix_get(ensemble_cov, j, k);
					tmp += weight * (X[n].element[j] - ensemble_mean[j]) * (X[n].element[k] - ensemble_mean[k]);
					gsl_matrix_set(ensemble_cov, j, k, tmp);
				}
			}
		}
		
		for(unsigned int j=0; j<N; j++) {
			for(unsigned int k=j; k<N; k++) {
				tmp = gsl_matrix_get(ensemble_cov, j, k) / sum_weight;
				gsl_matrix_set(ensemble_cov, j, k, tmp);
				gsl_matrix_set(ensemble_cov, k, j, tmp);
			}
		}
		
		/*for(unsigned int j=0; j<N; j++) {
			for(unsigned int k=j; k<N; k++) {
				tmp = 0.;
				sum_weight = 0;
				for(unsigned int n=0; n<L; n++) {
					weight = exp(X[n].pi);
					tmp += weight * (X[n].element[j] - ensemble_mean[j]) * (X[n].element[k] - ensemble_mean[k]);
					sum_weight += weight;
				}
				tmp /= sum_weight;
				if(k == j) {
					gsl_matrix_set(ensemble_cov, j, k, tmp);//*1.005 + 0.005);	// Small factor added in to avoid singular matrices
				} else {
					gsl_matrix_set(ensemble_cov, j, k, tmp);
					gsl_matrix_set(ensemble_cov, k, j, tmp);
				}
			}
		}*/
	} else {
		for(unsigned int j=0; j<N; j++) {
			for(unsigned int k=j; k<N; k++) {
				tmp = 0.;
				sum_weight = 0.;
				for(unsigned int n=0; n<L; n++) {
					weight = X[n].pi / pi_0;
					tmp += weight * (X[n].element[j] - ensemble_mean[j]) * (X[n].element[k] - ensemble_mean[k]);
				}
				tmp /= (double)(L - 1) * sum_weight;
				if(k == j) {
					gsl_matrix_set(ensemble_cov, j, k, tmp);//*1.005 + 0.005);	// Small factor added in to avoid singular matrices
				} else {
					gsl_matrix_set(ensemble_cov, j, k, tmp);
					gsl_matrix_set(ensemble_cov, k, j, tmp);
				}
			}
		}
	}
	
	// Add in small constant along diagonals
	if(sigma_min > 0.) {
		for(unsigned int j=0; j<N; j++) {
			tmp = gsl_matrix_get(ensemble_cov, j, j);
			tmp = sqrt(tmp*tmp + sigma_min*sigma_min);
			gsl_matrix_set(ensemble_cov, j, j, tmp);
		}
	}
	
	/*#pragma omp critical (cout)
	{
	for(int k=0; k<N; k++) {
		std::cerr << sqrt(gsl_matrix_get(ensemble_cov, k, k)) << "  ";
	}
	std::cerr << std::endl;
	}*/
	
	// Inverse and Sqrt of Covariance
	det_ensemble_cov = invert_matrix(ensemble_cov, inv_ensemble_cov, wp, wm1);
	sqrt_matrix(ensemble_cov, sqrt_ensemble_cov, ws, wv, wm1, wm2);
	log_norm_ensemble_cov = -0.5 * log(fabs(det_ensemble_cov) * twopiN);
	
	// Diagonal covariance information
	det_diag_cov = 1.;
	//#pragma omp critical
	//{
	for(unsigned int j=0; j<N; j++) {
		tmp = 0.;
		for(unsigned int n=0; n<L; n++) { tmp += (X[n].element[j] - ensemble_mean[j]) * (X[n].element[j] - ensemble_mean[j]); }
		tmp /= (double)(L - 1);
		diag_cov[j] = tmp;
		sqrt_diag_cov[j] = sqrt(tmp);
		inv_diag_cov[j] = 1. / tmp;
		det_diag_cov *= tmp;
	//	std::cout << "diag_cov[" << j << "] = " << tmp << std::endl;
	}
	log_norm_diag_cov = -0.5 * log(fabs(det_diag_cov) * twopiN);
	
	//std::cout << "Det = " << det_ensemble_cov << " = " << det_diag_cov << std::endl;
	//}
}

// Get the density Gaussian proposal distribution
template<class TParams, class TLogger>
double TAffineSampler<TParams, TLogger>::log_gaussian_density(const TState *const x, const TState *const y) {
	double sum = 0.;
	double tmp;
	//double *inv = inv_ensemble_cov->data;
	for(unsigned int i=0; i<N; i++) {
		tmp = (x->element[i] - y->element[i]);
		//sum += tmp * tmp * inv[i + N*i];
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

// Get the density Gaussian proposal distribution, using only the diagonal terms in the covariance matrix
template<class TParams, class TLogger>
double TAffineSampler<TParams, TLogger>::log_gaussian_density_diag(const TState *const x, const TState *const y) {
	double sum = 0.;
	double tmp;
	for(unsigned int i=0; i<N; i++) {
		tmp = (x->element[i] - y->element[i]);
		sum += tmp * tmp * inv_diag_cov[i];
	}
	return -(double)N * log_h + log_norm_diag_cov - sum/(2.*h*h);
}

template<class TParams, class TLogger>
void TAffineSampler<TParams, TLogger>::replacement_proposal(unsigned int j, bool unbalanced) {
	// Choose a sampler to step from
	unsigned int k = gsl_rng_uniform_int(r, (long unsigned int)L);
	
	// Determine step vector
	draw_from_cov(W, sqrt_ensemble_cov, N, r);
	
	// Determine the coordinates of the proposal
	for(unsigned int i=0; i<N; i++) {
		Y[j].element[i] = X[k].element[i] + h * W[i];
	}
	
	if(unbalanced) {
		Y[j].replacement_factor = 1.;
	} else {
		// Determine pi_S(X_j | Y_j , X_{-j}) and pi_S(Y_j | X)
		double tmp;
		double XY_max = neg_inf_replacement;
		double YX_max = neg_inf_replacement;
		double YX_cutoff = neg_inf_replacement;
		double XY_cutoff = neg_inf_replacement;
		const double cutoff = 3. + logL;
		double pi_XY = 0.;
		double pi_YX = 0.;
		for(unsigned int i=0; i<j; i++) {
			tmp = log_gaussian_density(&(X[i]), &(X[j]));
			if(tmp > XY_cutoff) {
				pi_XY += exp(tmp);
				if(tmp > XY_max) {
					XY_max = tmp;
					XY_cutoff = XY_max - cutoff;
				}
			}
			
			tmp = log_gaussian_density(&(X[i]), &(Y[j]));
			if(tmp > YX_cutoff) {
				pi_YX += exp(tmp);
				if(tmp > YX_max) {
					YX_max = tmp;
					YX_cutoff = YX_max - cutoff;
				}
			}
			
		}
		for(unsigned int i=j+1; i<L; i++) {
			tmp = log_gaussian_density(&(X[i]), &(X[j]));
			if(tmp > XY_cutoff) {
				pi_XY += exp(tmp);
				if(tmp > XY_max) {
					XY_max = tmp;
					XY_cutoff = XY_max - cutoff;
				}
			}
			
			tmp = log_gaussian_density(&(X[i]), &(Y[j]));
			if(tmp > YX_cutoff) {
				pi_YX += exp(tmp);
				if(tmp > YX_max) {
					YX_max = tmp;
					YX_cutoff = YX_max - cutoff;
				}
			}
		}
		tmp = log_gaussian_density(&(Y[j]), &(X[j]));
		if(tmp > XY_cutoff) { pi_XY += exp(tmp); }
		if(tmp > YX_cutoff) { pi_YX += exp(tmp); }
		
		// Factor to ensure reversibility
		Y[j].replacement_factor = pi_XY / pi_YX + replacement_accept_bias;
	}
	
	// Get pdf(Y) and initialize weight of proposal point to unity
	Y[j].pi = pdf(Y[j].element, N, params);
	Y[j].weight = 1.;
}

template<class TParams, class TLogger>
void TAffineSampler<TParams, TLogger>::replacement_proposal_diag(unsigned int j, bool unbalanced) {
	// Choose a sampler to step from
	unsigned int k = gsl_rng_uniform_int(r, (long unsigned int)L);
	
	// Determine step vector
	//draw_from_cov(W, sqrt_ensemble_cov, N, r);
	
	// Determine the coordinates of the proposal
	for(unsigned int i=0; i<N; i++) {
		//Y[j].element[i] = X[k].element[i] + h * W[i];
		Y[j].element[i] = X[k].element[i] + h * sqrt_diag_cov[i] * gsl_ran_gaussian_ziggurat(r, 1.);
	}
	
	if(unbalanced) {
		Y[j].replacement_factor = 1.;
	} else {
		// Determine pi_S(X_j | Y_j , X_{-j}) and pi_S(Y_j | X)
		double tmp;
		double XY_max = neg_inf_replacement;
		double YX_max = neg_inf_replacement;
		double YX_cutoff = neg_inf_replacement;
		double XY_cutoff = neg_inf_replacement;
		const double cutoff = 3. + logL;
		double pi_XY = 0.;
		double pi_YX = 0.;
		for(unsigned int i=0; i<j; i++) {
			tmp = log_gaussian_density_diag(&(X[i]), &(X[j]));
			if(tmp > XY_cutoff) {
				pi_XY += exp(tmp);
				if(tmp > XY_max) {
					XY_max = tmp;
					XY_cutoff = XY_max - cutoff;
				}
			}
			
			tmp = log_gaussian_density_diag(&(X[i]), &(Y[j]));
			if(tmp > YX_cutoff) {
				pi_YX += exp(tmp);
				if(tmp > YX_max) {
					YX_max = tmp;
					YX_cutoff = YX_max - cutoff;
				}
			}
			
		}
		for(unsigned int i=j+1; i<L; i++) {
			tmp = log_gaussian_density_diag(&(X[i]), &(X[j]));
			if(tmp > XY_cutoff) {
				pi_XY += exp(tmp);
				if(tmp > XY_max) {
					XY_max = tmp;
					XY_cutoff = XY_max - cutoff;
				}
			}
			
			tmp = log_gaussian_density_diag(&(X[i]), &(Y[j]));
			if(tmp > YX_cutoff) {
				pi_YX += exp(tmp);
				if(tmp > YX_max) {
					YX_max = tmp;
					YX_cutoff = YX_max - cutoff;
				}
			}
		}
		tmp = log_gaussian_density_diag(&(Y[j]), &(X[j]));
		if(tmp > XY_cutoff) { pi_XY += exp(tmp); }
		if(tmp > YX_cutoff) { pi_YX += exp(tmp); }
		
		Y[j].replacement_factor = pi_XY / pi_YX + replacement_accept_bias;
	}
	
	// Get pdf(Y) and initialize weight of proposal point to unity
	Y[j].pi = pdf(Y[j].element, N, params);
	Y[j].weight = 1.;
}

template<class TParams, class TLogger>
void TAffineSampler<TParams, TLogger>::MH_proposal(unsigned int j) {
	// Determine step vector
	draw_from_cov(W, sqrt_ensemble_cov, N, r);
	
	// Determine the coordinates of the proposal
	for(unsigned int i=0; i<N; i++) {
		Y[j].element[i] = X[j].element[i] + h_MH * W[i];
	}
	
	// Get pdf(Y) and initialize weight of proposal point to unity
	Y[j].pi = pdf(Y[j].element, N, params);
	Y[j].weight = 1.;
	Y[j].replacement_factor = 1.;
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
void TAffineSampler<TParams, TLogger>::step(bool record_step, double p_replacement,
                                            bool unbalanced, bool diag_approx) {
	// Make either a stretch or a replacement step
	double p = gsl_rng_uniform(r);
	//#pragma omp critical
	//{
	if(p < p_replacement) {
		//std::cerr << "replacement" << std::endl;
		step_replacement(record_step, unbalanced, diag_approx);
	} else {
		//std::cerr << "affine" << std::endl;
		step_affine(record_step);
	}
	//}
}

template<class TParams, class TLogger>
void TAffineSampler<TParams, TLogger>::step_affine(bool record_step) {
	double scale, alpha, p;
	for(unsigned int j=0; j<L; j++) {
		// Draw a proposal
		affine_proposal(j, scale);
		
		// Determine if the proposal is the maximum-likelihood point
		if(Y[j].pi > X_ML.pi) { X_ML = Y[j]; }
		
		// Determine whether to accept or reject
		accept[j] = false;
		if(use_log) {	// If <pdf> returns log probability
			// Determine the acceptance probability
			if(is_neg_inf_replacement(X[j].pi) && !(is_neg_inf_replacement(Y[j].pi))) {
				alpha = 1;	// Accept the proposal if the current state has zero probability and the proposed state doesn't
			} else {
				alpha = (double)(N - 1) * log(scale) + Y[j].pi - X[j].pi;
			}
			
			// Decide whether to accept or reject
			if(alpha > 0.) {	// Accept if probability of acceptance is greater than unity
				accept[j] = true;
			} else {
				p = gsl_rng_uniform(r);
				if((p == 0.) && (Y[j] > neg_inf_replacement)) {	// Accept if zero is rolled but proposal has nonzero probability
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
				alpha = pow(scale, (double)(N - 1)) * Y[j].pi / X[j].pi;
			}
			
			// Decide whether to accept or reject
			if(alpha > 1.) {	// Accept if probability of acceptance is greater than unity
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
		
		// Update sampler j
		if(accept[j]) {
		    if(is_neg_inf_replacement(Y[j].pi)) {
		        #pragma omp critical (cout)
		        {
		        std::cerr << "!!! Accepted -infinity point! (affine step)" << std::endl;
		        }
		    }
			if(record_step) {
				chain.add_point(X[j].element, X[j].pi, (double)(X[j].weight));
				
				#pragma omp critical (logger)
				logger(X[j].element, X[j].weight);
			}
			
			X[j] = Y[j];
			
			N_accepted++;
			N_stretch_accepted++;
		} else {
			X[j].weight++;
			
			N_rejected++;
			N_stretch_rejected++;
		}
	}
}

template<class TParams, class TLogger>
void TAffineSampler<TParams, TLogger>::step_replacement(bool record_step, bool unbalanced, bool diag_approx) {
	update_ensemble_cov();
	
	double alpha, p;
	for(unsigned int j=0; j<L; j++) {
		if(diag_approx) {
			replacement_proposal_diag(j, unbalanced);
		} else {
			replacement_proposal(j, unbalanced);
		}
		
		// Determine if the proposal is the maximum-likelihood point
		if(Y[j].pi > X_ML.pi) { X_ML = Y[j]; }
		
		// Determine whether to accept or reject
		accept[j] = false;
		if(use_log) {	// If <pdf> returns log probability
			// Determine the acceptance probability
			if(is_neg_inf_replacement(X[j].pi) && !(is_neg_inf_replacement(Y[j].pi))) {
				alpha = 1;	// Accept the proposal if the current state has zero probability and the proposed state doesn't
			} else {
				if(unbalanced) {
					alpha = Y[j].pi - X[j].pi;	// Ignore detailed balance. Use carefully - does not sample from target!
				} else {
					alpha = Y[j].pi - X[j].pi + log(Y[j].replacement_factor);
					
					/*
					#pragma omp critical (cout)
					{
						//std::cout << Y[j].pi << std::endl;
						//std::cout << X[j].pi << std::endl;
						std::cout << Y[j].pi - X[j].pi << std::endl;
						std::cout << log(Y[j].replacement_factor) << std::endl;
						std::cout << std::endl;
					}
					*/
				}
			}
			
			// Decide whether to accept or reject
			if(is_neg_inf_replacement(Y[j].pi)) {
			    accept[j] = false;
			} else if(alpha > 0.) {	// Accept if probability of acceptance is greater than unity
				accept[j] = true;
			} else {
				p = gsl_rng_uniform(r);
				if((p == 0.) && (Y[j] > neg_inf_replacement)) {	// Accept if zero is rolled but proposal has nonzero probability
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
				alpha = Y[j].pi / X[j].pi * Y[j].replacement_factor;
			}
			
			// Decide whether to accept or reject
			if(alpha > 1.) {	// Accept if probability of acceptance is greater than unity
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
		
		// Update sampler j
		if(accept[j]) {
		    if(is_neg_inf_replacement(Y[j].pi)) {
		        #pragma omp critical (cout)
		        {
		        std::cerr << "!!! Accepted -infinity point! (replacement step)" << std::endl;
		        }
		    }
		    
			if(record_step) {
				chain.add_point(X[j].element, X[j].pi, (double)(X[j].weight));
				
				#pragma omp critical (logger)
				logger(X[j].element, X[j].weight);
			}
			
			X[j] = Y[j];
			
			N_accepted++;
			N_replacements_accepted++;
		} else {
			X[j].weight++;
			
			N_rejected++;
			N_replacements_rejected++;
		}
	}
}

template<class TParams, class TLogger>
void TAffineSampler<TParams, TLogger>::step_MH(bool record_step) {
	double alpha, p;
	
	// Update statistics on ensemble
	update_ensemble_cov();
	
	for(unsigned int j=0; j<L; j++) {
		// Generate proposal
		MH_proposal(j);
		
		// Determine if the proposal is the maximum-likelihood point
		if(Y[j].pi > X_ML.pi) { X_ML = Y[j]; }
		
		// Determine whether to accept or reject
		accept[j] = false;
		if(use_log) {	// If <pdf> returns log probability
			// Determine the acceptance probability
			if(is_neg_inf_replacement(X[j].pi) && !(is_neg_inf_replacement(Y[j].pi))) {
				alpha = 1;	// Accept the proposal if the current state has zero probability and the proposed state doesn't
			} else {
				alpha = Y[j].pi - X[j].pi;
			}
			// Decide whether to accept or reject
			if(alpha > 0.) {	// Accept if probability of acceptance is greater than unity
				accept[j] = true;
			} else {
				p = gsl_rng_uniform(r);
				if((p == 0.) && (Y[j] > neg_inf_replacement)) {	// Accept if zero is rolled but proposal has nonzero probability
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
				alpha = Y[j].pi / X[j].pi;
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
		
		// Update sampler j
		if(accept[j]) {
		    if(is_neg_inf_replacement(Y[j].pi)) {
		        #pragma omp critical (cout)
		        {
		        std::cerr << "!!! Accepted -infinity point! (MH step)" << std::endl;
		        }
		    }
		    
			if(record_step) {
				chain.add_point(X[j].element, X[j].pi, (double)(X[j].weight));
				
				#pragma omp critical (logger)
				logger(X[j].element, X[j].weight);
			}
			
			X[j] = Y[j];
			
			N_accepted++;
			N_MH_accepted++;
		} else {
			X[j].weight++;
			
			N_rejected++;
			N_MH_rejected++;
		}
	}
}

template<class TParams, class TLogger>
void TAffineSampler<TParams, TLogger>::step_custom_reversible(reversible_step_t f_reversible_step, bool record_step) {
	double alpha, p, Q_factor;
	
	for(unsigned int j=0; j<L; j++) {
		// Generate proposal from custom user function. Assume step probability is symmetric in X and Y.
		Q_factor = f_reversible_step(X[j].element, Y[j].element, N, r, params);
		
		// Get pdf(Y) and initialize weight of proposal point to unity
		Y[j].pi = pdf(Y[j].element, N, params);
		Y[j].weight = 1;
		Y[j].replacement_factor = 1.;
		
		// Determine if the proposal is the maximum-likelihood point
		if(Y[j].pi > X_ML.pi) { X_ML = Y[j]; }
		
		// Determine whether to accept or reject
		accept[j] = false;
		if(use_log) {	// If <pdf> returns log probability
			// Determine the acceptance probability
			if(is_neg_inf_replacement(X[j].pi) && !(is_neg_inf_replacement(Y[j].pi))) {
				alpha = 1;	// Accept the proposal if the current state has zero probability and the proposed state doesn't
			} else {
				alpha = Y[j].pi - X[j].pi + Q_factor;
			}
			// Decide whether to accept or reject
			if(alpha > 0.) {	// Accept if probability of acceptance is greater than unity
				accept[j] = true;
			} else {
				p = gsl_rng_uniform(r);
				if((p == 0.) && (Y[j] > neg_inf_replacement)) {	// Accept if zero is rolled but proposal has nonzero probability
					accept[j] = true;
				} else if(log(p) < alpha) {
					accept[j] = true;
				}
			}
		} else {	// If <pdf> returns bare probability
			// Determine the acceptance probability
			if((X[j].pi == 0) && (Y[j].pi != 0)) {
				alpha = 2.;	// Accept the proposal if the current state has zero probability and the proposed state doesn't
			} else {
				alpha = Y[j].pi / X[j].pi * Q_factor;
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
		
		// Update sampler j
		if(accept[j]) {
			if(record_step) {
				chain.add_point(X[j].element, X[j].pi, (double)(X[j].weight));
				
				#pragma omp critical (logger)
				logger(X[j].element, X[j].weight);
			}
			
			X[j] = Y[j];
			
			N_accepted++;
			N_custom_accepted++;
		} else {
			X[j].weight++;
			
			N_rejected++;
			N_custom_rejected++;
		}
	}
}

// Set the dimensionless step scale
template<class TParams, class TLogger>
void TAffineSampler<TParams, TLogger>::set_scale(double a) {
	assert(a > 0);
	sqrta = sqrt(a);
}

template<class TParams, class TLogger>
void TAffineSampler<TParams, TLogger>::set_replacement_bandwidth(double _h) {
	assert(_h > 0.);
	h = _h;
	log_h = log(h);
}

template<class TParams, class TLogger>
void TAffineSampler<TParams, TLogger>::set_MH_bandwidth(double _h) {
	assert(_h > 0);
	h_MH = _h;
	log_h_MH = log(h_MH);
}

template<class TParams, class TLogger>
void TAffineSampler<TParams, TLogger>::set_sigma_min(double _sigma_min) {
	assert(_sigma_min >= 0.);
	sigma_min = _sigma_min;
}

template<class TParams, class TLogger>
void TAffineSampler<TParams, TLogger>::set_replacement_accept_bias(double epsilon) {
	assert(epsilon >= 0.);
	replacement_accept_bias = epsilon;
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
	N_stretch_accepted = 0;
	N_stretch_rejected = 0;
	N_replacements_accepted = 0;
	N_replacements_rejected = 0;
	N_MH_accepted = 0;
	N_MH_rejected = 0;
	N_custom_accepted = 0;
	N_custom_rejected = 0;
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

template<class TParams, class TLogger>
void TAffineSampler<TParams, TLogger>::print_stats() {
	TStats &stats = get_stats();
	stats.print();
	
	std::cout << std::endl;
	std::cout << "Acceptance rate: ";
	std::cout << std::setprecision(3) << 100.*get_acceptance_rate() << "%" << std::endl;
	
	uint64_t acc_tmp, rej_tmp;
	
	unsigned int N_steps_tmp = 0;
	
	N_steps_tmp = get_N_stretch_accepted();
	N_steps_tmp += get_N_stretch_rejected();
	
	if(N_steps_tmp != 0) {
		std::cout << "Stretch steps accepted:rejected: ";
		acc_tmp = get_N_stretch_accepted();
		rej_tmp = get_N_stretch_rejected();
		std::cout << std::fixed << acc_tmp << ":" << rej_tmp
		          << " (" << std::setprecision(1) << 100. * (double)acc_tmp / (double)(acc_tmp + rej_tmp) << "%)";
		std::cout << std::endl;
	}
	
	N_steps_tmp = get_N_replacements_accepted();
	N_steps_tmp += get_N_replacements_rejected();
	
	if(N_steps_tmp != 0) {
		std::cout << "Replacements accepted:rejected: ";
		acc_tmp = get_N_replacements_accepted();
		rej_tmp = get_N_replacements_rejected();
		std::cout << std::fixed << acc_tmp << ":" << rej_tmp
		          << " (" << std::setprecision(1) << 100. * (double)acc_tmp / (double)(acc_tmp + rej_tmp) << "%)";
		std::cout << std::endl;
	}
	
	N_steps_tmp = get_N_MH_accepted();
	N_steps_tmp += get_N_MH_rejected();
	
	if(N_steps_tmp != 0) {
		std::cout << "M-H steps accepted:rejected: ";
		acc_tmp = get_N_MH_accepted();
		rej_tmp = get_N_MH_rejected();
		std::cout << std::fixed << acc_tmp << ":" << rej_tmp
		          << " (" << std::setprecision(1) << 100. * (double)acc_tmp / (double)(acc_tmp + rej_tmp) << "%)";
		std::cout << std::endl;
	}
	
	N_steps_tmp = get_N_custom_accepted();
	N_steps_tmp += get_N_custom_rejected();
	
	if(N_steps_tmp != 0) {
		std::cout << "Custom reversible steps accepted:rejected: ";
		acc_tmp = get_N_custom_accepted();
		rej_tmp = get_N_custom_rejected();
		std::cout << std::fixed << acc_tmp << ":" << rej_tmp
		          << " (" << std::setprecision(1) << 100. * (double)acc_tmp / (double)(acc_tmp + rej_tmp) << "%)";
		std::cout << std::endl;
	}
	
	std::cout << std::setprecision(6);
}


/*************************************************************************
 *   Parallel Affine Sampler Class Member Functions
 *************************************************************************/

template<class TParams, class TLogger>
TParallelAffineSampler<TParams, TLogger>::TParallelAffineSampler(typename TAffineSampler<TParams, TLogger>::pdf_t _pdf, typename TAffineSampler<TParams, TLogger>::rand_state_t _rand_state,
                                                                 unsigned int _N, unsigned int _L, TParams& _params, TLogger& _logger, unsigned int _N_samplers, bool _use_log)
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
void TParallelAffineSampler<TParams, TLogger>::step(unsigned int N_steps, bool record_steps, double cycle,
                                                    double p_replacement, bool unbalanced, bool diag_approx) {
	#pragma omp parallel for schedule(dynamic) firstprivate(record_steps, N_steps, cycle, p_replacement, unbalanced, diag_approx)
	for(int sampler_num = 0; sampler_num < N_samplers; sampler_num++) {
		for(unsigned int i=0; i<N_steps; i++) {
			sampler[sampler_num]->step(record_steps, p_replacement, unbalanced, diag_approx);
		}
		sampler[sampler_num]->flush(record_steps);
	}
	#pragma omp barrier
	Gelman_Rubin_diagnostic(component_stats, N_samplers, R, N);
}

template<class TParams, class TLogger>
void TParallelAffineSampler<TParams, TLogger>::step_MH(unsigned int N_steps, bool record_steps) {
	#pragma omp parallel for schedule(dynamic) firstprivate(record_steps, N_steps)
	for(int sampler_num = 0; sampler_num < N_samplers; sampler_num++) {
		for(unsigned int i=0; i<N_steps; i++) {
			sampler[sampler_num]->step_MH(record_steps);
		}
		sampler[sampler_num]->flush(record_steps);
	}
	#pragma omp barrier
	Gelman_Rubin_diagnostic(component_stats, N_samplers, R, N);
}

template<class TParams, class TLogger>
void TParallelAffineSampler<TParams, TLogger>::step_custom_reversible(unsigned int N_steps,
	                                                              typename TAffineSampler<TParams, TLogger>::reversible_step_t f_reversible_step,
	                                                              bool record_steps) {
	#pragma omp parallel for schedule(dynamic) firstprivate(record_steps, N_steps)
	for(int sampler_num = 0; sampler_num < N_samplers; sampler_num++) {
		for(unsigned int i=0; i<N_steps; i++) {
			sampler[sampler_num]->step_custom_reversible(f_reversible_step, record_steps);
		}
		sampler[sampler_num]->flush(record_steps);
	}
	#pragma omp barrier
	Gelman_Rubin_diagnostic(component_stats, N_samplers, R, N);
}

template<class TParams, class TLogger>
void TParallelAffineSampler<TParams, TLogger>::tune_MH(unsigned int N_rounds, double target_acceptance) {
	#pragma omp parallel for
	for(int sampler_num = 0; sampler_num < N_samplers; sampler_num++) {
		unsigned int N_steps = 100. / ((double)(sampler[sampler_num]->get_N_walkers()) * target_acceptance);
		if(N_steps < 3) { N_steps = 3; }
		
		//#pragma omp critical
		//std::cout << "Tuning steps: " << N_steps << std::endl;
		
		double acceptance_tmp, bandwidth_tmp;
		
		for(int k=0; k<N_rounds; k++) {
			sampler[sampler_num]->clear();
			for(unsigned int i=0; i<N_steps; i++) {
				sampler[sampler_num]->step_MH(false);
			}
			sampler[sampler_num]->flush(false);
			
			acceptance_tmp = sampler[sampler_num]->get_MH_acceptance_rate();
			if(acceptance_tmp < 0.9 * target_acceptance) {
				bandwidth_tmp = sampler[sampler_num]->get_MH_bandwidth();
				sampler[sampler_num]->set_MH_bandwidth(0.9 * bandwidth_tmp);
				
				//#pragma omp critical
				//std::cout << "Thread " << thread_ID << ": " << bandwidth_tmp << " -> " << 0.8 * bandwidth_tmp << " (" << 100. * acceptance_tmp << "%)" << std::endl;
			} else if(acceptance_tmp > 1.1 * target_acceptance) {
				bandwidth_tmp = sampler[sampler_num]->get_MH_bandwidth();
				sampler[sampler_num]->set_MH_bandwidth(1.1 * bandwidth_tmp);
				
				//#pragma omp critical
				//std::cout << "Thread " << thread_ID << ": " << bandwidth_tmp << " -> " << 1.2 * bandwidth_tmp << " (" << 100. * acceptance_tmp << "%)" << std::endl;
			}
		}
	}
}

template<class TParams, class TLogger>
void TParallelAffineSampler<TParams, TLogger>::tune_stretch(unsigned int N_rounds, double target_acceptance) {
	#pragma omp parallel for
	for(int sampler_num = 0; sampler_num < N_samplers; sampler_num++) {
		unsigned int N_steps = 100. / ((double)(sampler[sampler_num]->get_N_walkers()) * target_acceptance);
		if(N_steps < 3) { N_steps = 3; }
		
		//#pragma omp critical
		//std::cout << "Tuning steps: " << N_steps << std::endl;
		
		double acceptance_tmp, scale_Delta;
		
		for(int k=0; k<N_rounds; k++) {
			sampler[sampler_num]->clear();
			for(unsigned int i=0; i<N_steps; i++) {
				sampler[sampler_num]->step(false, 0., false);
			}
			sampler[sampler_num]->flush(false);
			
			acceptance_tmp = sampler[sampler_num]->get_stretch_acceptance_rate();
			if(acceptance_tmp < 0.9 * target_acceptance) {
				scale_Delta = sampler[sampler_num]->get_scale() - 1.;
				sampler[sampler_num]->set_scale(1. + 0.9 * scale_Delta);
				
				//#pragma omp critical
				//std::cout << "Thread " << thread_ID << ": " << 1. + scale_Delta << " -> " << 1. + 0.8 * scale_Delta << " (" << 100. * acceptance_tmp << "%)" << std::endl;
			} else if(acceptance_tmp > 1.1 * target_acceptance) {
				scale_Delta = sampler[sampler_num]->get_scale() - 1.;
				sampler[sampler_num]->set_scale(1. + 1.1 * scale_Delta);
				
				//#pragma omp critical
				//std::cout << "Thread " << thread_ID << ": " << 1. + scale_Delta << " -> " << 1. + 1.2 * scale_Delta << " (" << 100. * acceptance_tmp << "%)" << std::endl;
			}
		}
	}
}


template<class TParams, class TLogger>
void TParallelAffineSampler<TParams, TLogger>::calc_stats() {
	stats.clear();
	for(int i=0; i<N_samplers; i++) {
		stats += sampler[i]->get_stats();
	}
}


template<class TParams, class TLogger>
void TParallelAffineSampler<TParams, TLogger>::print_stats() {
	calc_stats();
	stats.print();
	
	std::cout << std::endl << "Gelman-Rubin diagnostic:" << std::endl;
	for(unsigned int i=0; i<N; i++) { std::cout << (i==0 ? "" : "\t") << std::setprecision(5) << R[i]; }
	std::cout << std::endl;
	std::cout << "Acceptance rate: ";
	for(unsigned int i=0; i<N_samplers; i++) { std::cout << std::setprecision(3) << 100.*get_sampler(i)->get_acceptance_rate() << "%" << (i != N_samplers - 1 ? " " : ""); }
	std::cout << std::endl;
	
	uint64_t acc_tmp, rej_tmp;
	
	unsigned int N_steps_tmp = 0;
	for(int i=0; i<N_samplers; i++) {
		N_steps_tmp += get_sampler(i)->get_N_stretch_accepted();
		N_steps_tmp += get_sampler(i)->get_N_stretch_rejected();
		
	}
	
	if(N_steps_tmp != 0) {
		std::cout << "Stretch steps accepted:rejected: ";
		for(unsigned int i=0; i<N_samplers; i++) {
			acc_tmp = get_sampler(i)->get_N_stretch_accepted();
			rej_tmp = get_sampler(i)->get_N_stretch_rejected();
			std::cout << std::fixed << acc_tmp << ":" << rej_tmp
				<< " (" << std::setprecision(1) << 100. * (double)acc_tmp / (double)(acc_tmp + rej_tmp) << "%)"
				<< (i != N_samplers - 1 ? " " : "");
		}
		std::cout << std::endl;
	}
	
	N_steps_tmp = 0;
	for(int i=0; i<N_samplers; i++) {
		N_steps_tmp += get_sampler(i)->get_N_replacements_accepted();
		N_steps_tmp += get_sampler(i)->get_N_replacements_rejected();
		
	}
	
	if(N_steps_tmp != 0) {
		std::cout << "Replacements accepted:rejected: ";
		for(unsigned int i=0; i<N_samplers; i++) {
			acc_tmp = get_sampler(i)->get_N_replacements_accepted();
			rej_tmp = get_sampler(i)->get_N_replacements_rejected();
			std::cout << std::fixed << acc_tmp << ":" << rej_tmp
				<< " (" << std::setprecision(1) << 100. * (double)acc_tmp / (double)(acc_tmp + rej_tmp) << "%)"
				<< (i != N_samplers - 1 ? " " : "");
		}
		std::cout << std::endl;
	}
	
	N_steps_tmp = 0;
	for(int i=0; i<N_samplers; i++) {
		N_steps_tmp += get_sampler(i)->get_N_MH_accepted();
		N_steps_tmp += get_sampler(i)->get_N_MH_rejected();
		
	}
	
	if(N_steps_tmp != 0) {
		std::cout << "M-H steps accepted:rejected: ";
		for(unsigned int i=0; i<N_samplers; i++) {
			acc_tmp = get_sampler(i)->get_N_MH_accepted();
			rej_tmp = get_sampler(i)->get_N_MH_rejected();
			std::cout << std::fixed << acc_tmp << ":" << rej_tmp
				<< " (" << std::setprecision(1) << 100. * (double)acc_tmp / (double)(acc_tmp + rej_tmp) << "%)"
				<< (i != N_samplers - 1 ? " " : "");
		}
		std::cout << std::endl;
	}
	
	N_steps_tmp = 0;
	for(int i=0; i<N_samplers; i++) {
		N_steps_tmp += get_sampler(i)->get_N_custom_accepted();
		N_steps_tmp += get_sampler(i)->get_N_custom_rejected();
		
	}
	
	if(N_steps_tmp != 0) {
		std::cout << "Custom reversible steps accepted:rejected: ";
		for(unsigned int i=0; i<N_samplers; i++) {
			acc_tmp = get_sampler(i)->get_N_custom_accepted();
			rej_tmp = get_sampler(i)->get_N_custom_rejected();
			std::cout << std::fixed << acc_tmp << ":" << rej_tmp
				<< " (" << std::setprecision(1) << 100. * (double)acc_tmp / (double)(acc_tmp + rej_tmp) << "%)"
				<< (i != N_samplers - 1 ? " " : "");
		}
		std::cout << std::endl;
	}
	
	std::cout << std::setprecision(6);
}

template<class TParams, class TLogger>
void TParallelAffineSampler<TParams, TLogger>::print_diagnostics() {
	std::cout << "Gelman-Rubin diagnostic:" << std::endl;
	for(unsigned int i=0; i<N; i++) { std::cout << (i==0 ? "" : "\t") << std::setprecision(5) << R[i]; }
	std::cout << std::endl;
	std::cout << "Acceptance rate: ";
	for(unsigned int i=0; i<N_samplers; i++) { std::cout << std::setprecision(3) << 100.*get_sampler(i)->get_acceptance_rate() << "%" << (i != N_samplers - 1 ? " " : ""); }
	std::cout << std::endl;
	
	uint64_t acc_tmp, rej_tmp;
	
	unsigned int N_steps_tmp = 0;
	for(int i=0; i<N_samplers; i++) {
		N_steps_tmp += get_sampler(i)->get_N_stretch_accepted();
		N_steps_tmp += get_sampler(i)->get_N_stretch_rejected();
		
	}
	
	if(N_steps_tmp != 0) {
		std::cout << "Stretch steps accepted:rejected: ";
		for(unsigned int i=0; i<N_samplers; i++) {
			acc_tmp = get_sampler(i)->get_N_stretch_accepted();
			rej_tmp = get_sampler(i)->get_N_stretch_rejected();
			std::cout << std::fixed << acc_tmp << ":" << rej_tmp
				<< " (" << std::setprecision(1) << 100. * (double)acc_tmp / (double)(acc_tmp + rej_tmp) << "%)"
				<< (i != N_samplers - 1 ? " " : "");
		}
		std::cout << std::endl;
	}
	
	N_steps_tmp = 0;
	for(int i=0; i<N_samplers; i++) {
		N_steps_tmp += get_sampler(i)->get_N_replacements_accepted();
		N_steps_tmp += get_sampler(i)->get_N_replacements_rejected();
		
	}
	
	if(N_steps_tmp != 0) {
		std::cout << "Replacements accepted:rejected: ";
		for(unsigned int i=0; i<N_samplers; i++) {
			acc_tmp = get_sampler(i)->get_N_replacements_accepted();
			rej_tmp = get_sampler(i)->get_N_replacements_rejected();
			std::cout << std::fixed << acc_tmp << ":" << rej_tmp
				<< " (" << std::setprecision(1) << 100. * (double)acc_tmp / (double)(acc_tmp + rej_tmp) << "%)"
				<< (i != N_samplers - 1 ? " " : "");
		}
		std::cout << std::endl;
	}
	
	N_steps_tmp = 0;
	for(int i=0; i<N_samplers; i++) {
		N_steps_tmp += get_sampler(i)->get_N_MH_accepted();
		N_steps_tmp += get_sampler(i)->get_N_MH_rejected();
		
	}
	
	if(N_steps_tmp != 0) {
		std::cout << "M-H steps accepted:rejected: ";
		for(unsigned int i=0; i<N_samplers; i++) {
			acc_tmp = get_sampler(i)->get_N_MH_accepted();
			rej_tmp = get_sampler(i)->get_N_MH_rejected();
			std::cout << std::fixed << acc_tmp << ":" << rej_tmp
				<< " (" << std::setprecision(1) << 100. * (double)acc_tmp / (double)(acc_tmp + rej_tmp) << "%)"
				<< (i != N_samplers - 1 ? " " : "");
		}
		std::cout << std::endl;
	}
	
	N_steps_tmp = 0;
	for(int i=0; i<N_samplers; i++) {
		N_steps_tmp += get_sampler(i)->get_N_custom_accepted();
		N_steps_tmp += get_sampler(i)->get_N_custom_rejected();
		
	}
	
	if(N_steps_tmp != 0) {
		std::cout << "Custom reversible steps accepted:rejected: ";
		for(unsigned int i=0; i<N_samplers; i++) {
			acc_tmp = get_sampler(i)->get_N_custom_accepted();
			rej_tmp = get_sampler(i)->get_N_custom_rejected();
			std::cout << std::fixed << acc_tmp << ":" << rej_tmp
				<< " (" << std::setprecision(1) << 100. * (double)acc_tmp / (double)(acc_tmp + rej_tmp) << "%)"
				<< (i != N_samplers - 1 ? " " : "");
		}
		std::cout << std::endl;
	}
	
	std::cout << std::setprecision(6);
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
	
	#pragma omp parallel for
	for(int sampler_num = 0; sampler_num < N_samplers; sampler_num++) {
		TStats& transf_comp_stat = *(transf_stats[sampler_num]);
		TChain& chain = sampler[sampler_num]->get_chain();
		size_t n_points = chain.get_length();
		
		double* y = new double[N];
		
		for(size_t i=0; i<n_points; i++) {
			(*transf)(chain.get_element(i), y);
			transf_comp_stat(y, (unsigned int)(chain.get_w(i)));
		}
		
		delete[] y;
	}
	#pragma omp barrier
	
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
	seed ^= (long unsigned int)getpid();
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
