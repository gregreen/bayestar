#ifndef _STATS_H__
#define _STATS_H__


#include <iostream>
#include <fstream>
#include <iomanip>
#include <assert.h>
#include <omp.h>
#include <stdint.h>
#include <vector>
#include <math.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>


class TStats {
	double *E_k;
	double *E_ij;
	unsigned int N;
	uint64_t N_items_tot;
	
public:
	// Constructor & Destructor
	TStats(unsigned int _N);
	TStats(const TStats& s);	// Copy
	~TStats();
	
	// Mutators
	void clear();							// Clear the contents of the statistics object
	void update(const double *const x, unsigned int weight);	// Update the chain from a an array of doubles with a weight
	void update(const TStats *const stats);
	
	void operator()(const double *const x, unsigned int weight);	// proxy for update()
	void operator()(const TStats *const stats);			// proxy for update()
	
	TStats& operator+=(const TStats &rhs);				// Add the data in another stats object to this one
	TStats& operator*=(double a);					// Multiply by scalar
	TStats& operator=(const TStats &rhs);				// Copy data from another stats object to this one, replacing existing data
	
	friend TStats operator*(double a, const TStats& stats);
	friend TStats operator*(const TStats& stats, double a);
	
	// Accessors
	double mean(unsigned int i) const;				// Return < x_i >
	double cov(unsigned int i, unsigned int j) const;		// Return covariance element Cov(i,j)
	void get_cov_matrix(gsl_matrix* Sigma, gsl_matrix* invSigma, double* detSigma) const;	// Calculates the covariance matrix Sigma, alongside Sigma^{-1} and det(Sigma)
	uint64_t get_N_items() const;
	unsigned int get_dim() const;
	
	// I/O
	void print() const;											// Print out statistics
	bool write_binary(std::string fname, bool converged, double evidence, bool append_to_file=false) const;	// Write statistics to binary file, possibly appending to existing file
	bool write_binary_old(std::string fname, std::ios::openmode writemode = std::ios::out) const;		// Write statistics to binary file. Pass std::ios::app as writemode to append to end of existing file.
	bool read_binary(std::string fname, std::streampos read_offset=std::streampos(0));			// Read statistics from file. Pass read_offset if the stats object is offset in the binary file.
};

// Overloaded arithmetic operations with TStats

TStats operator*(double a, const TStats& stats);
TStats operator*(const TStats& stats, double a);

void Gelman_Rubin_diagnostic(TStats **stats_arr, unsigned int N_chains, double *R, unsigned int N);

double metric_dist2(const gsl_matrix* g, const double* x_1, const double* x_2, unsigned int N);

#endif // _STATS_H__