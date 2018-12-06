/*
 * model.h
 *
 * Defines the stellar and galactic models.
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

#ifndef _MODEL_H__
#define _MODEL_H__

#include <string>
#include <vector>
#include <limits.h>
#include <stddef.h>

#include <H5Cpp.h>

#include <gsl/gsl_spline.h>

#include "definitions.h"
#include "interpolation.h"

#define _DM 0
#define _LOGMASS 1
#define _LOGTAU 2
#define _FEH 3

#define NBANDS 8



// Spectral energy distribution object, with operators necessary for interpolation
struct TSED {
	//double Mr, FeH;
	double absmag[NBANDS];

	TSED();				// Data initialized to zero
	TSED(bool uninitialized);	// Overloaded version leaves data uninitialized
	~TSED();

	// Comparison based on absolute r-magnitude
	//bool operator<(const TSED &b) const { return Mr < b.Mr || (Mr == b.Mr && FeH < b.FeH); }

	// Operators required for bilinear interpolation of SEDs
	TSED& operator=(const TSED &rhs);
	friend TSED operator+(const TSED &sed1, const TSED &sed2);
	friend TSED operator-(const TSED &sed1, const TSED &sed2);
	friend TSED operator*(const TSED &sed, const double &a);
	friend TSED operator*(const double &a, const TSED &sed);
	friend TSED operator/(const TSED &sed, const double &a);
	TSED& operator*=(double a);
	TSED& operator/=(double a);
	TSED& operator+=(const TSED &rhs);
	TSED& operator+=(double a);
};


// A stellar template library and luminosity function
class TStellarModel {
public:
	TStellarModel(std::string lf_fname, std::string seds_fname);
	~TStellarModel();

	// Access by parameter value
	bool get_sed(const double* x, TSED& sed) const;
	bool get_sed(double Mr, double FeH, TSED& sed) const;
	TSED get_sed(double Mr, double FeH);
	bool in_model(double Mr, double FeH);

	// Access by grid index
	bool get_sed(unsigned int Mr_idx, unsigned int FeH_idx,
				 TSED& sed, double& Mr, double& FeH) const;
	unsigned int get_N_FeH() const;
	unsigned int get_N_Mr() const;
    
    // Look up (Mr, FeH) corresponding to (i, j) index in grid
    bool get_Mr_FeH(unsigned int Mr_idx, unsigned int FeH_idx,
                    double& Mr, double& FeH) const;
    
	// Luminosity function
	double get_log_lf(double Mr) const;

private:
	// Template library data
	double dMr_seds, dFeH_seds, Mr_min_seds, FeH_min_seds, Mr_max_seds, FeH_max_seds;	// Sample spacing for stellar SEDs
	unsigned int N_FeH_seds, N_Mr_seds;
	TBilinearInterp<TSED> *sed_interp;	// Bilinear interpolation of stellar SEDs in Mr and FeH

	// Luminosity function library data
	TLinearInterp *log_lf_interp;
	double log_lf_norm;

	bool load_lf(std::string lf_fname);
	bool load_seds(std::string seds_fname);
};

// Returns a normalized creation function C(logM, tau),
// where logM is the log (base 10) of stellar mass, and
// tau (positive) is the time in the past. The creation
// function is defined as
//     C(logM, tau) = \frac{d N(logM, tau)}{d logM d t} .
class TStellarAbundance {
public:
	TStellarAbundance(int component);
	~TStellarAbundance();

	double IMF(double logM) const;	// Initial mass function
	double SFR(double tau) const;	// Star formation rate

	void set_IMF(double _logM_norm, double _logM_c, double _sigma_logM, double _x);
	void set_SFR(double _A_burst, double _tau_burst, double _sigma_tau, double _tau_max);

private:
	double IMF_norm, SFR_norm;	// Normalization constants for IMF and SFR

	// Chabrier (2003) IMF parameters
	double A_21, logM_norm, logM_c, sigma_logM_2, x;

	// Star formation rate parameters
	double A_burst, tau_burst, sigma_tau_2, tau_max;
};

// Synthetic stellar library, with accompanying initial mass function
// and star formation rate.
class TSyntheticStellarModel {
public:
	TSyntheticStellarModel(std::string seds_fname);
	~TSyntheticStellarModel();

	bool get_sed(const double *MtZ, TSED &sed) const;
	bool get_sed(double logMass, double logtau, double FeH, TSED &sed);

private:
	struct TSynthSED {
		float Z;
		float logtau;
		float logMass_init;
		float logTeff;
		float logg;
		float M_g;
		float M_r;
		float M_i;
		float M_z;
		float M_y;
	};

	struct TGridDim {
		uint32_t N_Z;
		uint32_t N_logtau;
		uint32_t N_logMass_init;
		float Z_min;
		float Z_max;
		float logtau_min;
		float logtau_max;
		float logMass_init_min;
		float logMass_init_max;
	};

	TMultiLinearInterp<TSED> *sed_interp;
	TGridDim grid_dim;
	double Theta[3];
};

// Dust extinction model
class TExtinctionModel {
public:
	TExtinctionModel(std::string A_RV_fname);
	~TExtinctionModel();

	double get_A(double RV, unsigned int i);	// Get A_i(EBV=1), where i is a bandpass
	bool in_model(double RV);

private:
	double RV_min, RV_max;
	gsl_spline **A_spl;
	gsl_interp_accel **acc;
};

// Luminosity function
/*struct TLuminosityFunc {
	double Mr0, dMr;
	std::vector<double> lf;
	TLinearInterp *lf_interp;
	double log_lf_norm;

	TLuminosityFunc(const std::string &fn) : lf_interp(NULL) { load(fn); }
	~TLuminosityFunc() { delete lf_interp; }

	// return the LF at position Mr (linear interpolation)
	double operator()(double Mr) const {
		return (*lf_interp)(Mr) - log_lf_norm;
	}

	void load(const std::string &fn);
};*/

struct TGalStructParams {
	double R0, Z0;					// Solar position
	double H1, L1;					// Thin disk
	double f_thick, H2, L2;				// Thick disk
	double L_epsilon;				// Disk smoothing scale
	double fh, qh, nh, R_br, nh_outer, R_epsilon;	// Halo
	double mu_FeH_inf, delta_mu_FeH, H_mu_FeH;	// Metallicity
	double H_ISM, L_ISM, dH_dR_ISM, R_flair_ISM;	// Smooth ISM disk

	TGalStructParams();
};


// A model of the galaxy, for producing priors on number density and metallicity of stars
class TGalacticModel {
public:
	// Set default model parameters
	TGalacticModel();
	TGalacticModel(const TGalStructParams& gal_struct_params);

	~TGalacticModel();

	// Set custom model parameters
	void set_struct_params(const TGalStructParams& gal_struct_params);

	// Stellar density
	double rho_halo(double R, double Z) const;
	double rho_disk(double R, double Z) const;

	// ISM density
	double rho_ISM(double R, double Z) const;

	// Stellar metallicity
	double mu_FeH_disk(double Z) const;
	double log_p_FeH(double FeH, double R, double Z) const;

	// Priors (component = {0 for disk, 1 for halo})
	double p_FeH(double FeH, double R, double Z, int component) const;
	double IMF(double logM, int component) const;	// Initial mass function
	double SFR(double tau, int component) const;	// Star formation rate
	//double lnp_Mr(double Mr) const;		// Luminosity function

	void set_IMF(int component, double _logM_norm, double _logM_c, double _sigma_logM, double _x);
	void set_SFR(int component, double _A_burst, double _tau_burst, double _sigma_tau, double _tau_max);

	//void load_lf(std::string lf_fname);

protected:
	// Density parameters
	double R0, Z0;					// Solar position
	double H1, L1;					// Thin disk (exponential)
	double f_thick, H2, L2;				// Thick disk (exponential)
	double L_epsilon;				// Smoothing of disk near Galactic center
	double fh, qh, nh, R_br, nh_outer, R_epsilon2;	// Halo (broken power law)
	double fh_outer;
	double H_ISM, L_ISM, dH_dR_ISM, R_flair_ISM;

	// Metallicity parameters
	double mu_FeH_inf;
	double delta_mu_FeH;
	double H_mu_FeH;

	// IMF and SFR of Galaxy
	TStellarAbundance *halo_abundance;
	TStellarAbundance *disk_abundance;

	// Luminosity Function
	//TLuminosityFunc *lf;
};


// A model of the galaxy, interpolated along one line of sight
class TGalacticLOSModel : public TGalacticModel {
public:
	// Set default model parameters
	TGalacticLOSModel(double _l, double _b);

	// Set custom model parameters
	TGalacticLOSModel(double _l, double _b, const TGalStructParams& gal_struct_params);

	~TGalacticLOSModel();

	// Volume element
	double dV(double DM) const;

	// Stars per unit solid angle per unit distance modulus (up to a normalizing factor)
	double log_dNdmu(double DM) const;
	double log_dNdmu_full(double DM) const;

	// Fraction of stars in the halo at a given distance modulus (rho_halo / rho_disk)
	double f_halo(double DM) const;
	double f_halo_full(double DM) const;

	// Probability density of star being at given distance modulus with given metallicity
	double log_p_FeH_fast(double DM, double FeH, double f_H=-1.) const;

	double p_FeH_fast(double DM, double FeH, int component) const;

	double log_prior_synth(double DM, double logM, double logtau, double FeH) const;
	double log_prior_synth(const double* x) const;

	// Full priors on (DM, Mr, [Fe/H]) for empirical stellar model
	double log_prior_emp(double DM, double Mr, double FeH) const;
	double log_prior_emp(const double* x) const;
    
	// Expected dust reddening, up to normalizing constant
	double dA_dmu(double DM) const;

	// Convert from distance modulus to R and Z
	void DM_to_RZ(double DM, double &R, double &Z) const;

	double get_log_dNdmu_norm() const;

	// Densities
	double rho_disk_los(double DM) const;
	double rho_halo_los(double DM) const;
	double rho_ISM_los(double DM) const;

	// Direction of l.o.s.
	void get_lb(double &l, double &b) const;

private:
	double cos_l, sin_l, cos_b, sin_b, l, b;
	double DM_min, DM_max, DM_samples, log_dNdmu_norm;
	TLinearInterp *log_dNdmu_arr, *f_halo_arr, *mu_FeH_disk_arr;

	void init(double _l, double _b);

	// Stellar density
	double rho_halo_interp(double DM) const;
	double rho_disk_interp(double DM) const;

	// Stellar metallicity
	double mu_FeH_disk_interp(double DM) const;
};

// A class for calculating the desired percent smoothing in the E(B-V) direction.
class TEBVSmoothing {
public:
	TEBVSmoothing(double alpha_coeff[2], double beta_coeff[2],
	              double pct_smoothing_min, double pct_smoothing_max);
	~TEBVSmoothing();

	void calc_pct_smoothing(unsigned int nside,
                            double EBV_min, double EBV_max, int n_samples,
                            std::vector<double>& sigma_pct) const;

	double get_pct_smoothing_min() const;
	double get_pct_smoothing_max() const;

private:
	double _alpha_coeff[2];	// Coefficients for the E^2 coefficient
	double _beta_coeff[2];	// Coefficients for the E coefficient
	double _pct_smoothing_min;	// Minimum smoothing, in percent
	double _pct_smoothing_max;	// Maximum smoothing, in percent
	double _healpix_scale;	// Angular scale of a HEALPix nside=1 pixel

	double nside_2_arcmin(unsigned int nside) const;
};


double chi2_parallax(double DM, double parallax, double parallax_err);


#endif // _MODEL_H__
