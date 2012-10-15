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

#include "interpolation.h"

#ifndef PI
#define PI 3.14159265358979323
#endif // PI

#ifndef SQRTPI
#define SQRTPI 1.7724538509055159
#endif // SQRTPI

#ifndef SQRT2
#define SQRT2 1.4142135623730951
#endif // SQRT2

#ifndef LN10
#define LN10 2.3025850929940459
#endif // LN10

#define NBANDS 5


// A model of the galaxy, for producing priors on number density and metallicity of stars
class TGalacticModel {
public:
	// Set default model parameters
	TGalacticModel();
	
	// Set custom model parameters
	TGalacticModel(double _R0, double _Z0, double _H1, double _L1, double _f_thick, double _H2, double _L2, double _fh, double _qh, double _nh, double _R_br, double _nh_outer, double _mu_FeH_inf, double _delta_mu_FeH, double _H_mu_FeH);
	
	~TGalacticModel();
	
	// Stellar density
	double rho_halo(double R, double Z) const;
	double rho_disk(double R, double Z) const;
	
	// Stellar metallicity
	double mu_FeH_disk(double Z) const;
	double log_p_FeH(double FeH, double R, double Z) const;
	
protected:
	// Density parameters
	double R0, Z0;				// Solar position
	double H1, L1;				// Thin disk
	double f_thick, H2, L2;			// Galactic structure (thin and thick disk)
	double fh, qh, nh, R_br, nh_outer;	// Galactic structure (power-law halo)
	double fh_outer;
	
	// Metallicity parameters
	double mu_FeH_inf;
	double delta_mu_FeH;
	double H_mu_FeH;
};


// A model of the galaxy, interpolated along one line of sight
class TGalacticLOSModel : public TGalacticModel {
public:
	// Set default model parameters
	TGalacticLOSModel(double l, double b);
	
	// Set custom model parameters
	TGalacticLOSModel(double l, double b, double _R0, double _Z0, double _H1, double _L1, double _f_thick, double _H2, double _L2, double _fh, double _qh, double _nh, double _R_br, double _nh_outer, double _mu_FeH_inf, double _delta_mu_FeH, double _H_mu_FeH);
	
	~TGalacticLOSModel();
	
	// Stars per unit solid angle per unit distance modulus (up to a normalizing factor)
	double log_dNdmu(double DM) const;
	double log_dNdmu_full(double DM) const;
	
	// Fraction of stars in the halo at a given distance modulus (rho_halo / rho_disk)
	double f_halo(double DM) const;
	double f_halo_full(double DM) const;
	
	// Probability density of star being at given distance modulus with given metallicity
	double log_p_FeH(double DM, double FeH) const;
	
	// Convert from distance modulus to R and Z
	void DM_to_RZ(double DM, double &R, double &Z) const;
	
private:
	double cos_l, sin_l, cos_b, sin_b;
	double DM_min, DM_max, DM_samples, log_dNdmu_norm;
	TLinearInterp *log_dNdmu_arr, *f_halo_arr, *mu_FeH_disk_arr;
	
	void init(double l, double b);
	
	// Stellar density
	double rho_halo_interp(double DM) const;
	double rho_disk_interp(double DM) const;
	
	// Stellar metallicity
	double mu_FeH_disk_interp(double DM) const;
};


// Spectral energy distribution object, with operators necessary for interpolation
struct TSED {
	//double Mr, FeH;
	double absmag[NBANDS];
	
	TSED();			// Data left unitialized
	TSED(bool initialize);	// Overloaded version initializes data to zero
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
	
	TSED get_sed(double Mr, double FeH);
	bool in_model(double Mr, double FeH);
	double get_log_lf(double Mr);
	
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
	TStellarAbundance();
	~TStellarAbundance();
	
	double IMF(double logM);	// Initial mass function
	double SFR(double tau);		// Star formation rate
	
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
	
	bool get_sed(double *MtZ, TSED &sed) const;
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
	
	double get_A(double RV, unsigned int i);
	bool in_model(double RV);
	
private:
	double RV_min, RV_max;
	gsl_spline **A_spl;
	gsl_interp_accel **acc;
};


#endif // _MODEL_H__