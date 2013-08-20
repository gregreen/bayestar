
#include "stats.h"

// Standard constructor
TStats::TStats(unsigned int _N)
	: E_k(NULL), E_ij(NULL), N(_N)
{
	E_k = new double[N];
	E_ij = new double[N*N];
	clear();
}

// Copy constructor
TStats::TStats(const TStats& s)
	: E_k(NULL), E_ij(NULL)
{
	N = s.N;
	E_k = new double[N];
	E_ij = new double[N*N];
	N_items_tot = s.N_items_tot;
	for(unsigned int i=0; i<N; i++) {
		E_k[i] = s.E_k[i];
		for(unsigned int j=0; j<N; j++) { E_ij[i+N*j] = s.E_ij[i+N*j]; }
	}
}

// Destructor
TStats::~TStats() {
	delete[] E_k;
	delete[] E_ij;
}


// Clear all the contents of the statistics object
void TStats::clear() {
	for(unsigned int i=0; i<N; i++) {
		E_k[i] = 0.;
		for(unsigned int j=0; j<N; j++) { E_ij[i+N*j] = 0.; }
	}
	N_items_tot = 0;
}

// Update the chain from a an array of doubles with a weight
void TStats::update(const double *const x, unsigned int weight) {
	if(weight != 0) {
		for(unsigned int i=0; i<N; i++) {
			E_k[i] += x[i] * (double)weight;
			for(unsigned int j=i; j<N; j++) {
				E_ij[i+N*j] += x[i] * x[j] * (double)weight;
				E_ij[N*i+j] = E_ij[i+N*j];
			}
		}
		N_items_tot += (uint64_t)weight;
	}
}

// Update the chain from the statistics in another TStats object
void TStats::update(const TStats *const stats) {
	assert(stats->N == N);
	for(unsigned int i=0; i<N; i++) {
		E_k[i] += stats->E_k[i];
		for(unsigned int j=i; j<N; j++) {
			E_ij[i+N*j] += stats->E_ij[i+N*j];
			E_ij[N*i+j] = E_ij[i+N*j];
		}
	}
	N_items_tot += stats->N_items_tot;
}

// Update the chain from the statistics in another TStats object
void TStats::operator()(const TStats *const stats) { update(stats); }

// Update the chain from a an array of doubles with a weight
void TStats::operator()(const double *const x, unsigned int weight) { update(x, weight); }

// Add the data in another stats object to this one
TStats& TStats::operator+=(const TStats &rhs) {
	assert(rhs.N == N);
	N_items_tot += rhs.N_items_tot;
	for(unsigned int i=0; i<N; i++) {
		E_k[i] += rhs.E_k[i];
		for(unsigned int j=0; j<N; j++) { E_ij[i+N*j] += rhs.E_ij[i+N*j]; }
	}
	return *this;
}

// Multiply stats by a scalar (Changes total weight of stats object, but doesn't change means or covariance)
TStats& TStats::operator*=(double a) {
	N_items_tot = ceil(a * (double)N_items_tot);
	for(unsigned int i=0; i<N; i++) {
		E_k[i] *= a;
		for(unsigned int j=0; j<N; j++) { E_ij[i+N*j] *= a; }
	}
	return *this;
}

// Copy data from another stats object to this one, replacing existing data
TStats& TStats::operator=(const TStats &rhs) {
	if(&rhs != this) {
		// Resize the expectation-value arrays if necessary
		if(rhs.N != N) {
			delete[] E_k;
			delete[] E_ij;
			N = rhs.N;
			E_k = new double[N];
			E_ij = new double[N*N];
		}
		// Copy in the data from the rhs object
		N_items_tot = rhs.N_items_tot;
		for(unsigned int i=0; i<N; i++) {
			E_k[i] = rhs.E_k[i];
			for(unsigned int j=0; j<N; j++) { E_ij[i+N*j] = rhs.E_ij[i+N*j]; }
		}
	}
	return *this;
}

// Multiply a statistics operator by a scalar
TStats operator*(double a, const TStats& stats) {
	TStats tmp(stats.N);
	tmp.N_items_tot = ceil(a * stats.N_items_tot);
	for(unsigned int i=0; i<stats.N; i++) {
		tmp.E_k[i] = a * stats.E_k[i];
		for(unsigned int j=0; j<stats.N; j++) { tmp.E_ij[i+stats.N*j] = a * stats.E_ij[i+stats.N*j]; }
	}
	return tmp;
}

TStats operator*(const TStats &stats, double a) {
	TStats tmp(stats.N);
	tmp.N_items_tot = ceil(a * stats.N_items_tot);
	for(unsigned int i=0; i<stats.N; i++) {
		tmp.E_k[i] = a * stats.E_k[i];
		for(unsigned int j=0; j<stats.N; j++) { tmp.E_ij[i+stats.N*j] = a * stats.E_ij[i+stats.N*j]; }
	}
	return tmp;
}

// Return covariance element Cov(i,j)
double TStats::cov(unsigned int i, unsigned int j) const { return (E_ij[i+N*j] - E_k[i]*E_k[j]/(double)N_items_tot)/(double)N_items_tot; }

// Return < x_i >
double TStats::mean(unsigned int i) const { return E_k[i] / (double)N_items_tot; }

uint64_t TStats::get_N_items() const { return N_items_tot; }

unsigned int TStats::get_dim() const { return N; }

// Calculates the covariance matrix Sigma, alongside Sigma^{-1} and det(Sigma)
void TStats::get_cov_matrix(gsl_matrix* Sigma, gsl_matrix* invSigma, double* detSigma) const {
	// Check that the matrices are the correct size
	assert(Sigma->size1 == N);
	assert(Sigma->size2 == N);
	assert(invSigma->size1 == N);
	assert(invSigma->size2 == N);
	
	// Calculate the covariance matrix Sigma
	double tmp;
	for(unsigned int i=0; i<N; i++) {
		for(unsigned int j=i; j<N; j++) {
			tmp = cov(i,j);
			gsl_matrix_set(Sigma, i, j, tmp);
			if(i != j) { gsl_matrix_set(Sigma, j, i, tmp); }
		}
	}
	
	// Get the inverse of Sigma
	int s;
	gsl_permutation* p = gsl_permutation_alloc(N);
	gsl_matrix* LU = gsl_matrix_alloc(N, N);
	gsl_matrix_memcpy(LU, Sigma);
	gsl_linalg_LU_decomp(LU, p, &s);
	gsl_linalg_LU_invert(LU, p, invSigma);
	
	// Get the determinant of sigma
	*detSigma = gsl_linalg_LU_det(LU, s);
	
	// Cleanup
	gsl_matrix_free(LU);
	gsl_permutation_free(p);
}

// Print out statistics
void TStats::print() const {
	//std::cout << "N_items_tot: " << N_items_tot << std::endl;
	std::cout << "Mean:" << std::endl;
	for(unsigned int i=0; i<N; i++) { std::cout << "\t" << std::setprecision(3) << mean(i) << "\t+-\t" << sqrt(cov(i, i)) << std::endl; }
	std::cout << std::endl;
	std::cout << "Covariance:" << std::endl;
	for(unsigned int i=0; i<N; i++) {
		for(unsigned int j=0; j<N; j++) {
			if(i != j) {
				std::cout << "\t" << cov(i, j) / sqrt(cov(i, i) * cov(j, j));
			} else {
				std::cout << "\t" << sqrt(cov(i, j));
			}
		}
		std::cout << std::endl;
	}
}

// Write statistics to binary file. Pass std::ios::app as writemode to append to end of existing file.
bool TStats::write_binary_old(std::string fname, std::ios::openmode writemode) const {
	std::fstream f;
	f.open(fname.c_str(), writemode | std::ios::out | std::ios::binary);
	if(!f) { f.close(); return false; }	// Return false if the file could not be opened
	
	// Write number of dimensions
	f.write(reinterpret_cast<const char*>(&N), sizeof(N));
	
	// Write mean values
	double tmp;
	for(unsigned int i=0; i<N; i++) {
		tmp = mean(i);
		f.write(reinterpret_cast<char*>(&tmp), sizeof(tmp));
	}
	
	// Write upper triangle (including diagonal) of covariance matrix
	for(unsigned int i=0; i<N; i++) {
		for(unsigned int j=i; j<N; j++) {
			tmp = cov(i, j);
			f.write(reinterpret_cast<char*>(&tmp), sizeof(tmp));
		}
	}
	
	// Write raw data
	f.write(reinterpret_cast<char*>(E_k), N * sizeof(double));
	f.write(reinterpret_cast<char*>(E_ij), N*N * sizeof(double));
	f.write(reinterpret_cast<const char*>(&N_items_tot), sizeof(N_items_tot));
	
	// Return false if there was a write error, else true
	if(!f) { f.close(); return false; }
	f.close();
	return true;
}

// Write statistics to binary file, possibly appending to existing file.
bool TStats::write_binary(std::string fname, bool converged, double evidence, bool append_to_file) const {
	// If writing to new file, delete file, if it already exists
	if(!append_to_file) { std::remove(fname.c_str()); }
	
	std::ios_base::openmode mode = std::ios::binary | std::ios::out;
	if(append_to_file) { mode |= std::ios::in; }
	std::fstream outfile(fname.c_str(), mode);
	if(outfile.fail()) {
		std::cerr << "Failed to open " << fname << "." << std::endl;
		return false;
	}
	
	// Header:
	unsigned int tmp_N_files;
	if(append_to_file) {
		// If appending to existing file, increment the number of objects in file
		outfile.read(reinterpret_cast<char *>(&tmp_N_files), sizeof(unsigned int));
		tmp_N_files++;
		outfile.seekp(std::ios_base::beg);
		outfile.write(reinterpret_cast<char *>(&tmp_N_files), sizeof(unsigned int));
		outfile.seekp(0, std::ios::end);
	} else {
		// Write header if not appending to existing file
		tmp_N_files = 1;
		outfile.write(reinterpret_cast<char *>(&tmp_N_files), sizeof(unsigned int));
		outfile.write(reinterpret_cast<const char *>(&N), sizeof(unsigned int));
	}
	
	// Data:
	
	// Write whether converged
	outfile.write(reinterpret_cast<char *>(&converged), sizeof(bool));
	outfile.write(reinterpret_cast<char *>(&evidence), sizeof(double));
	
	// Write mean values
	double tmp;
	for(unsigned int i=0; i<N; i++) {
		tmp = mean(i);
		outfile.write(reinterpret_cast<char*>(&tmp), sizeof(double));
	}
	
	// Write covariance matrix
	for(unsigned int i=0; i<N; i++) {
		for(unsigned int j=0; j<N; j++) {
			tmp = cov(i, j);
			outfile.write(reinterpret_cast<char*>(&tmp), sizeof(double));
		}
	}
	
	// Write raw data
	outfile.write(reinterpret_cast<char*>(E_k), N * sizeof(double));
	outfile.write(reinterpret_cast<char*>(E_ij), N*N * sizeof(double));
	outfile.write(reinterpret_cast<const char*>(&N_items_tot), sizeof(N_items_tot));
	
	// Return false if something has gone wrong in the writing
	if(outfile.bad()) {
		std::cout << "Something has gone wrong in writing stats to " << fname << "." << std::endl;
		outfile.close();
		return false;
	}
	
	outfile.close();
	
	return true;
}

// Read statistics from a binary file
bool TStats::read_binary(std::string fname, std::streampos read_offset) {
	std::fstream f;
	f.open(fname.c_str(), std::ios::in | std::ios::binary);
	// Skip to the point in the file designated by read_offset
	f.seekg(read_offset);
	
	if(!f.good()) { f.close(); return false; }	// Return false if the file could not be opened or read_offset was past the end of the file
	
	// Read the number of dimensions
	unsigned int N_tmp;
	f.read(reinterpret_cast<char*>(&N_tmp), sizeof(N_tmp));
	
	// If necessary, resize arrays in stats object
	if(N_tmp != N) {
		N = N_tmp;
		delete[] E_k;
		delete[] E_ij;
		E_k = new double[N];
		E_ij = new double[N*N];
	}
	
	// Skip past summary information
	f.ignore((N + N*(N+1)/2) * sizeof(double));
	
	// Read in raw data
	f.read(reinterpret_cast<char*>(E_k), N * sizeof(double));
	f.read(reinterpret_cast<char*>(E_ij), N*N * sizeof(double));
	f.read(reinterpret_cast<char*>(&N_items_tot), sizeof(N_items_tot));
	
	// Return false if there was a write error, else true
	if(!f.good()) { f.close(); return false; }
	
	f.close();
	return true;
}

void Gelman_Rubin_diagnostic(TStats **stats_arr, unsigned int N_chains, double *R, unsigned int N) {
	// Run some basic checks on the input to ensure that G-R statistics can be calculated
	assert(N_chains > 1);	// More than one chain
	unsigned int N_items_tot = stats_arr[0]->get_N_items();
	for(unsigned int i=1; i<N_chains; i++) { assert(stats_arr[i]->get_N_items() == N_items_tot); }	// Each chain is of the same length
	
	std::vector<double> W(N, 0.);		// Mean within-chain variance
	std::vector<double> B(N, 0.);		// Between-chain variance
	std::vector<double> Theta(N, 0.);	// Mean of means (overall mean)
	
	// Calculate mean within chain variance and overall mean
	for(unsigned int i=0; i<N_chains; i++) {
		for(unsigned int k=0; k<N; k++) {
			W[k] += stats_arr[i]->cov(k,k);
			Theta[k] += stats_arr[i]->mean(k);
		}
	}
	for(unsigned int k=0; k<N; k++) {
		W[k] /= (double)N_chains;
		Theta[k] /= (double)N_chains;
	}
	
	// Calculate variance between chains
	double tmp;
	for(unsigned int i=0; i<N_chains; i++) {
		for(unsigned int k=0; k<N; k++) {
			tmp = stats_arr[i]->mean(k) - Theta[k];
			B[k] += tmp*tmp;
		}
	}
	for(unsigned int k=0; k<N; k++) { B[k] /= (double)N_chains - 1.; }
	
	// Calculate estimated variance
	for(unsigned int k=0; k<N; k++) { R[k] = 1. - 1./(double)N_items_tot + B[k]/W[k]; }
}

double metric_dist2(const gsl_matrix* g, const double* x_1, const double* x_2, unsigned int N) {
	double dist2 = 0.;
	for(unsigned int i=0; i<N; i++) {
		for(unsigned int j=0; j<N; j++) {
			dist2 += (x_2[i] - x_1[i]) * (x_2[j] - x_1[j]) * gsl_matrix_get(g, i, j);
		}
	}
	return dist2;
}
