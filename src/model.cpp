/*
 * model.cpp
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

#include "model.h"

#include <vector>
#include <string>
#include <sstream>
#include <math.h>
#include <iostream>
#include <fstream>



/****************************************************************************************************************************
 *
 * TGalStructParams
 *
 ****************************************************************************************************************************/

TGalStructParams::TGalStructParams() {
	// Solar position
	R0 = 8000;
	Z0 = 25;

	// Thin disk
	L1 = 2150;
	H1 = 245;

	// Thick disk
	f_thick = 0.13;
	L2 = 3261;
	H2 = 743;

	// Smoothing radial scale of disk
	L_epsilon = 500;

	// Halo
	fh = 0.0030; //0.0051;
	qh = 0.70;
	nh = -2.62;
	R_br = 27800;
	nh_outer = -3.8;

	// Smoothing of the halo core
	R_epsilon = 500;

	// Drimmel & Spergel (2001)
	H_ISM = 134.4;
	L_ISM = 2260.;
	dH_dR_ISM = 0.0148;
	R_flair_ISM = 4400.;

	// Metallicity
	mu_FeH_inf = -0.82;
	delta_mu_FeH = 0.55;
	H_mu_FeH = 500;
}


/****************************************************************************************************************************
 *
 * TGalacticModel
 *
 ****************************************************************************************************************************/

TGalacticModel::TGalacticModel() {
	TGalStructParams gal_struct_params;
	set_struct_params(gal_struct_params);

	// IMF and SFR
	disk_abundance = new TStellarAbundance(0);
	halo_abundance = new TStellarAbundance(1);
}

TGalacticModel::TGalacticModel(const TGalStructParams& gal_struct_params) {
	set_struct_params(gal_struct_params);

	disk_abundance = new TStellarAbundance(0);
	halo_abundance = new TStellarAbundance(1);
}

TGalacticModel::~TGalacticModel() {
	delete disk_abundance;
	delete halo_abundance;
	//if(lf != NULL) { delete lf; }
}

void TGalacticModel::set_struct_params(const TGalStructParams& gal_struct_params) {
	// Solar position
	R0 = gal_struct_params.R0;
	Z0 = gal_struct_params.Z0;

	// Thin disk
	L1 = gal_struct_params.L1;
	H1 = gal_struct_params.H1;

	// Thick disk
	f_thick = gal_struct_params.f_thick;
	L2 = gal_struct_params.L2;
	H2 = gal_struct_params.H2;

	// Smoothing radial scale of disk
	L_epsilon = gal_struct_params.L_epsilon;

	// Halo
	fh = gal_struct_params.fh;
	qh = gal_struct_params.qh;
	nh = gal_struct_params.nh;
	R_br = gal_struct_params.R_br;
	nh_outer = gal_struct_params.nh_outer;
	fh_outer = fh * pow(R_br/R0, nh - nh_outer);
	R_epsilon2 = gal_struct_params.R_epsilon * gal_struct_params.R_epsilon;

	// Smooth ISM disk
	H_ISM = gal_struct_params.H_ISM;
	L_ISM = gal_struct_params.L_ISM;
	dH_dR_ISM = gal_struct_params.dH_dR_ISM;
	R_flair_ISM = gal_struct_params.R_flair_ISM;

	// Metallicity
	mu_FeH_inf = gal_struct_params.mu_FeH_inf;
	delta_mu_FeH = gal_struct_params.delta_mu_FeH;
	H_mu_FeH = gal_struct_params.H_mu_FeH;
}

double TGalacticModel::rho_halo(double R, double Z) const {
	double r_eff2 = R*R + (Z/qh)*(Z/qh) + R_epsilon2;

	if(r_eff2 <= R_br*R_br) {
		return fh*pow(r_eff2/(R0*R0), nh/2.);
	} else {
		return fh_outer*pow(r_eff2/(R0*R0), nh_outer/2.);
	}
}

double TGalacticModel::rho_disk(double R, double Z) const {
	double R_eff = sqrt(R*R + L_epsilon*L_epsilon);

	double rho_thin = exp(-(fabs(Z+Z0) - fabs(Z0))/H1 - (R_eff-R0)/L1);
	double rho_thick = f_thick * exp(-(fabs(Z+Z0) - fabs(Z0))/H2 - (R_eff-R0)/L2);
	return rho_thin + rho_thick;
}

double TGalacticModel::rho_ISM(double R, double Z) const {
	double H = H_ISM;
	if(R > R_flair_ISM) { H += (R - R_flair_ISM) * dH_dR_ISM; }

	double L_term;
	if(R > 0.5 * R0) {
		L_term = exp(-R / L_ISM);
	} else {
		L_term = exp(-0.5*R0 / L_ISM - (R - 0.5*R0)*(R - 0.5*R0)/(0.25 * R0*R0));
	}

	double sqrt_H_term = cosh((Z+Z0) / H);

	return L_term / (sqrt_H_term * sqrt_H_term);
}

// Mean disk metallicity at given position in space
double TGalacticModel::mu_FeH_disk(double Z) const {
	return mu_FeH_inf + delta_mu_FeH * exp(-fabs(Z+Z0)/H_mu_FeH);
}

double TGalacticModel::log_p_FeH(double FeH, double R, double Z) const {
	double f_H = rho_halo(R, Z) / rho_disk(R, Z);

	// Halo
	double mu_H = -1.46;
	double sigma_H = 0.3;
	double P_tmp = f_H * exp(-(FeH-mu_H)*(FeH-mu_H)/(2.*sigma_H*sigma_H)) / (SQRT2PI*sigma_H);

	// Metal-poor disk
	double mu_D = mu_FeH_disk(Z) - 0.067;
	double sigma_D = 0.2;
	P_tmp += 0.63 * (1-f_H) * exp(-(FeH-mu_D)*(FeH-mu_D)/(2.*sigma_D*sigma_D)) / (SQRT2PI*sigma_D);

	// Metal-rich disk
	double mu_D_poor = mu_D + 0.14;
	double sigma_D_poor = 0.2;
	P_tmp += 0.37 * (1-f_H) * exp(-(FeH-mu_D_poor)*(FeH-mu_D_poor)/(2.*sigma_D_poor*sigma_D_poor)) / (SQRT2PI*sigma_D_poor);

	return log(P_tmp);
}

double TGalacticModel::p_FeH(double FeH, double R, double Z, int component) const {
	if(component == 0) {	// Disk
		// Metal-poor disk
		double mu_D = mu_FeH_disk(Z) - 0.067;
		const double sigma_D = 0.2;
		double P_tmp = 0.63 * exp(-(FeH-mu_D)*(FeH-mu_D)/(2.*sigma_D*sigma_D)) / (SQRT2PI*sigma_D);

		// Metal-rich disk
		const double mu_D_poor = mu_D + 0.14;
		const double sigma_D_poor = 0.2;
		return P_tmp + 0.37 * exp(-(FeH-mu_D_poor)*(FeH-mu_D_poor)/(2.*sigma_D_poor*sigma_D_poor)) / (SQRT2PI*sigma_D_poor);
	} else {		// Halo
		const double mu_H = -1.46;
		const double sigma_H = 0.3;
		return exp(-(FeH-mu_H)*(FeH-mu_H)/(2.*sigma_H*sigma_H)) / (SQRT2PI*sigma_H);
	}
}

double TGalacticModel::IMF(double logM, int component) const {
	if(component == 0) {
		return disk_abundance->IMF(logM);
	} else {
		return halo_abundance->IMF(logM);
	}
}

double TGalacticModel::SFR(double tau, int component) const {
	if(component == 0) {
		return disk_abundance->SFR(tau);
	} else {
		return halo_abundance->SFR(tau);
	}
}

//double TGalacticModel::lnp_Mr(double Mr) const {
//	assert(lf != NULL);
//	return (*lf)(Mr);
//}


//void TGalacticModel::load_lf(std::string lf_fname) {
//	lf = new TLuminosityFunc(lf_fname);
//}




/****************************************************************************************************************************
 *
 * TGalacticLOSModel
 *
 ****************************************************************************************************************************/

TGalacticLOSModel::TGalacticLOSModel(double _l, double _b)
	: TGalacticModel()
{
	init(_l, _b);
}

TGalacticLOSModel::TGalacticLOSModel(double _l, double _b, const TGalStructParams& gal_struct_params)
	: TGalacticModel(gal_struct_params)
{
	init(_l, _b);
}

TGalacticLOSModel::~TGalacticLOSModel() {
	delete log_dNdmu_arr;
	delete f_halo_arr;
	delete mu_FeH_disk_arr;
}

void TGalacticLOSModel::init(double _l, double _b) {
	// Precompute trig functions
	l = _l;
	b = _b;
	cos_l = cos(0.0174532925*l);
	sin_l = sin(0.0174532925*l);
	cos_b = cos(0.0174532925*b);
	sin_b = sin(0.0174532925*b);

	// Precompute interpolation anchors for log(dN/dDM), f_halo and mu_FeH_disk
	DM_min = 0.;
	DM_max = 25.;
	DM_samples = 10000;

	log_dNdmu_arr = new TLinearInterp(DM_min, DM_max, DM_samples);
	f_halo_arr = new TLinearInterp(DM_min, DM_max, DM_samples);
	mu_FeH_disk_arr = new TLinearInterp(DM_min, DM_max, DM_samples);
	double DM_i, log_dNdmu_tmp, Z;
	double log_dNdmu_0 = log_dNdmu_full(13.);
	log_dNdmu_norm = 0.;
	for(unsigned int i=0; i<DM_samples; i++) {
		DM_i = log_dNdmu_arr->get_x(i);
		log_dNdmu_tmp = log_dNdmu_full(DM_i);
		(*log_dNdmu_arr)[i] = log_dNdmu_tmp;
		(*f_halo_arr)[i] = f_halo_full(DM_i);
		Z = sin_b * pow10(DM_i/5. + 1.);
		(*mu_FeH_disk_arr)[i] = mu_FeH_disk(Z);
		log_dNdmu_norm += exp(log_dNdmu_tmp - log_dNdmu_0);
	}
	log_dNdmu_norm = log_dNdmu_0 + log(log_dNdmu_norm);
	log_dNdmu_norm += log((log_dNdmu_arr->get_x(DM_samples-1) - log_dNdmu_arr->get_x(0)) / DM_samples);
}

void TGalacticLOSModel::DM_to_RZ(double DM, double& R, double& Z) const {
	double d = pow10(DM/5. + 1.);
	double X = R0 - cos_l*cos_b*d;
	double Y = -sin_l*cos_b*d;

	Z = sin_b*d;
	R = sqrt(X*X + Y*Y);
}

double TGalacticLOSModel::rho_disk_los(double DM) const {
	double R, Z;
	DM_to_RZ(DM, R, Z);

	return rho_disk(R, Z);
}

double TGalacticLOSModel::rho_halo_los(double DM) const {
	double R, Z;
	DM_to_RZ(DM, R, Z);

	return rho_halo(R, Z);
}

double TGalacticLOSModel::rho_ISM_los(double DM) const {
	double R, Z;
	DM_to_RZ(DM, R, Z);

	return rho_ISM(R, Z);
}

double TGalacticLOSModel::log_dNdmu_full(double DM) const {
	double R, Z;
	DM_to_RZ(DM, R, Z);

	double log_rho = log(rho_disk(R,Z) + rho_halo(R,Z));
	double log_V = 3.*2.30258509/5. * DM;

	return log_rho + log_V;
}

double TGalacticLOSModel::f_halo_full(double DM) const {
	double R, Z;
	DM_to_RZ(DM, R, Z);

	double rho_halo_tmp = rho_halo(R, Z);
	double rho_disk_tmp = rho_disk(R, Z);
	double f_h_tmp = rho_halo_tmp / (rho_disk_tmp + rho_halo_tmp);

	return f_h_tmp;
}

double TGalacticLOSModel::log_p_FeH_fast(double DM, double FeH, double f_H) const {
	if(f_H < 0.) { f_H = f_halo(DM); }

	// Halo
	double mu_H = -1.46;
	double sigma_H = 0.3;
	double P_tmp = f_H * exp(-(FeH-mu_H)*(FeH-mu_H)/(2.*sigma_H*sigma_H)) / (SQRT2PI*sigma_H);

	// Metal-poor disk
	double mu_D = mu_FeH_disk_interp(DM) - 0.067;
	double sigma_D = 0.2;
	P_tmp += 0.63 * (1-f_H) * exp(-(FeH-mu_D)*(FeH-mu_D)/(2.*sigma_D*sigma_D)) / (SQRT2PI*sigma_D);

	// Metal-rich disk
	double mu_D_poor = mu_D + 0.14;
	double sigma_D_poor = 0.2;
	P_tmp += 0.37 * (1-f_H) * exp(-(FeH-mu_D_poor)*(FeH-mu_D_poor)/(2.*sigma_D_poor*sigma_D_poor)) / (SQRT2PI*sigma_D_poor);

	return log(P_tmp);
}

double TGalacticLOSModel::log_dNdmu(double DM) const {
	if((DM < DM_min) || (DM > DM_max)) { return log_dNdmu_full(DM) - log_dNdmu_norm; }
	return (*log_dNdmu_arr)(DM) - log_dNdmu_norm;
}

double TGalacticLOSModel::f_halo(double DM) const {
	if((DM < DM_min) || (DM > DM_max)) { return f_halo_full(DM); }
	return (*f_halo_arr)(DM);
}

double TGalacticLOSModel::mu_FeH_disk_interp(double DM) const {
	if((DM < DM_min) || (DM > DM_max)) {
		double Z = pow10(DM/5. + 1.) * sin_b;
		return mu_FeH_disk(Z);
	}
	return (*mu_FeH_disk_arr)(DM);
}

double TGalacticLOSModel::p_FeH_fast(double DM, double FeH, int component) const {
	if(component == 0) {	// Disk
		// Metal-poor disk
		double mu_D = mu_FeH_disk_interp(DM) - 0.067;
		const double sigma_D = 0.2;
		double P_tmp = 0.63 * exp(-(FeH-mu_D)*(FeH-mu_D)/(2.*sigma_D*sigma_D)) / (SQRT2PI*sigma_D);

		// Metal-rich disk
		double mu_D_poor = mu_D + 0.14;
		const double sigma_D_poor = 0.2;
		return P_tmp + 0.37 * exp(-(FeH-mu_D_poor)*(FeH-mu_D_poor)/(2.*sigma_D_poor*sigma_D_poor)) / (SQRT2PI*sigma_D_poor);
	} else {		// Halo
		const double mu_H = -1.46;
		const double sigma_H = 0.3;
		return exp(-(FeH-mu_H)*(FeH-mu_H)/(2.*sigma_H*sigma_H)) / (SQRT2PI*sigma_H);
	}
}

double TGalacticLOSModel::log_prior_synth(double DM, double logM, double logtau, double FeH) const {
	double f_H = f_halo(DM);
	double tau = pow(10, logtau);
	double p = (1. - f_H) * IMF(logM, 0) * tau * SFR(tau, 0) * p_FeH_fast(DM, FeH, 0);
	p += f_H * IMF(logM, 1) * SFR(tau, 1) * p_FeH_fast(DM, FeH, 1);
	return log_dNdmu(DM) + log(p);
}

double TGalacticLOSModel::log_prior_synth(const double *x) const {
	double f_H = f_halo(x[_DM]);
	double tau = pow(10, x[_LOGTAU]);
	double p = (1. - f_H) * IMF(x[_LOGMASS], 0) * tau * SFR(tau, 0) * p_FeH_fast(x[_DM], x[_FEH], 0);
	p += f_H * IMF(x[_LOGMASS], 1) * SFR(tau, 1) * p_FeH_fast(x[_DM], x[_FEH], 1);
	return log_dNdmu(x[_DM]) + log(p);
}

double TGalacticLOSModel::log_prior_emp(double DM, double Mr, double FeH) const {
	double f_H = f_halo(DM);
	double p = (1. - f_H) * p_FeH_fast(DM, FeH, 0);
	p += f_H * p_FeH_fast(DM, FeH, 1);
	// return log(p);
	return log_dNdmu(DM) + log(p);// + lnp_Mr(Mr);
}

double TGalacticLOSModel::log_prior_emp(const double *x) const {
	double f_H = f_halo(x[0]);
	double p = (1. - f_H) * p_FeH_fast(x[0], x[2], 0);
	p += f_H * p_FeH_fast(x[0], x[2], 1);
	return log_dNdmu(x[0]) + log(p);// + lnp_Mr(x[1]);
}

double TGalacticLOSModel::get_log_dNdmu_norm() const { return log_dNdmu_norm; }

double TGalacticLOSModel::dA_dmu(double DM) const {
	return rho_ISM_los(DM) * pow10(DM / 5.);
}


void TGalacticLOSModel::get_lb(double &_l, double &_b) const {
	_l = l;
	_b = b;
}



/****************************************************************************************************************************
 *
 * TSED
 *
 ****************************************************************************************************************************/

TSED::TSED() {
	for(unsigned int i=0; i<NBANDS; i++) { absmag[i] = 0; }
}

TSED::TSED(bool uninitialized) { }

TSED::~TSED() { }

TSED& TSED::operator=(const TSED &rhs) {
	for(unsigned int i=0; i<NBANDS; i++) { absmag[i] = rhs.absmag[i]; }
	//Mr = rhs.Mr;
	//FeH = rhs.FeH;
	return *this;
}

TSED operator+(const TSED &sed1, const TSED &sed2) {
	TSED tmp;
	for(unsigned int i=0; i<NBANDS; i++) { tmp.absmag[i] = sed1.absmag[i] + sed2.absmag[i]; }
	//tmp.Mr = sed1.Mr + sed2.Mr;
	//tmp.FeH = sed1.FeH + sed2.FeH;
	return tmp;
}

TSED operator-(const TSED &sed1, const TSED &sed2) {
	TSED tmp;
	for(unsigned int i=0; i<NBANDS; i++) { tmp.absmag[i] = sed1.absmag[i] - sed2.absmag[i]; }
	//tmp.Mr = sed1.Mr - sed2.Mr;
	//tmp.FeH = sed1.FeH - sed2.FeH;
	return tmp;
}

TSED operator*(const TSED &sed, const double &a) {
	TSED tmp;
	for(unsigned int i=0; i<NBANDS; i++) { tmp.absmag[i] = a*sed.absmag[i]; }
	//tmp.Mr = a*sed.Mr;
	//tmp.FeH = a*sed.FeH;
	return tmp;
}

TSED operator*(const double &a, const TSED &sed) {
	TSED tmp;
	for(unsigned int i=0; i<NBANDS; i++) { tmp.absmag[i] = a*sed.absmag[i]; }
	//tmp.Mr = a*sed.Mr;
	//tmp.FeH = a*sed.FeH;
	return tmp;
}

TSED operator/(const TSED &sed, const double &a) {
	TSED tmp;
	for(unsigned int i=0; i<NBANDS; i++) { tmp.absmag[i] = sed.absmag[i]/a; }
	//tmp.Mr = sed.Mr / a;
	//tmp.FeH = sed.FeH / a;
	return tmp;
}

TSED& TSED::operator*=(double a) {
	for(unsigned int i=0; i<NBANDS; i++) { absmag[i] *= a; }
	return *this;
}

TSED& TSED::operator/=(double a) {
	for(unsigned int i=0; i<NBANDS; i++) { absmag[i] /= a; }
	return *this;
}

TSED& TSED::operator+=(double a) {
	for(unsigned int i=0; i<NBANDS; i++) { absmag[i] += a; }
	return *this;
}

TSED& TSED::operator+=(const TSED &rhs) {
	for(unsigned int i=0; i<NBANDS; i++) { absmag[i] += rhs.absmag[i]; }
	return *this;
}


/****************************************************************************************************************************
 *
 * TStellarModel
 *
 ****************************************************************************************************************************/

TStellarModel::TStellarModel(std::string lf_fname, std::string seds_fname)
	: sed_interp(NULL), log_lf_interp(NULL)
{
	load_lf(lf_fname);
	load_seds(seds_fname);
}

TStellarModel::~TStellarModel() {
	if(log_lf_interp != NULL) { delete log_lf_interp; }
	if(sed_interp != NULL) { delete sed_interp; }
}

bool TStellarModel::load_lf(std::string lf_fname) {
	std::ifstream in(lf_fname.c_str());
	if(!in) { std::cerr << "Could not read LF from '" << lf_fname << "'\n"; return false; }

	double Mr0;
	double dMr = -1;
	log_lf_norm = 0.;
	std::vector<double> lf;
	lf.reserve(3000);

	// Determine length of file
	std::string line;
	double Mr, Phi;
	while(std::getline(in, line)) {
		if(!line.size()) { continue; }		// empty line
		if(line[0] == '#') { continue; }	// comment

		std::istringstream ss(line);
		ss >> Mr >> Phi;

		if(dMr == -1) {
			Mr0 = Mr; dMr = 0;
		} else if(dMr == 0) {
			dMr = Mr - Mr0;
		}

		lf.push_back(log(Phi));
		log_lf_norm += Phi;
	}

	double Mr1 = Mr0 + dMr*(lf.size()-1);
	log_lf_interp = new TLinearInterp(Mr0, Mr1, lf.size());
	for(unsigned int i=0; i<lf.size(); i++) { (*log_lf_interp)[i] = lf[i]; }

	log_lf_norm *= Mr1 / (double)(lf.size());
	log_lf_norm = log(log_lf_norm);

	std::cout << "# Loaded Phi(" << Mr0 << " <= Mr <= " <<  Mr1 << ") LF from " << lf_fname << "\n";

	return true;
}

bool TStellarModel::load_seds(std::string seds_fname) {
	double Mr, FeH, dMr_tmp, dFeH_tmp;
	double Mr_last = inf_replacement;
	double FeH_last = inf_replacement;
	double Mr_min = inf_replacement;
	double Mr_max = neg_inf_replacement;
	double FeH_min = inf_replacement;
	double FeH_max = neg_inf_replacement;
	double dMr = inf_replacement;
	double dFeH = inf_replacement;

	// Do a first pass through the file to get the grid spacing and size
	std::ifstream in(seds_fname.c_str());
	if(!in) { std::cerr << "Could not read SEDs from '" << seds_fname << std::endl; return false; }
	std::string line;
	while(std::getline(in, line)) {
		if(!line.size()) { continue; }		// empty line
		if(line[0] == '#') { continue; }	// comment

		std::istringstream ss(line);
		ss >> Mr >> FeH;

		// Keep track of values needed to get grid spacing and size
		if(Mr < Mr_min) { Mr_min = Mr; }
		if(Mr > Mr_max) { Mr_max = Mr; }
		if(FeH < FeH_min) { FeH_min = FeH; }
		if(FeH > FeH_max) { FeH_max = FeH; }

		dMr_tmp = fabs(Mr_last - Mr);
		dFeH_tmp = fabs(FeH_last - FeH);
		if((dMr_tmp != 0) && (dMr_tmp < dMr)) { dMr = dMr_tmp; }
		if((dFeH_tmp != 0) && (dFeH_tmp < dFeH)) { dFeH = dFeH_tmp; }
		Mr_last = Mr;
		FeH_last = FeH;
	}

	unsigned int N_Mr = (unsigned int)(round((Mr_max - Mr_min) / dMr)) + 1;
	unsigned int N_FeH = (unsigned int)(round((FeH_max - FeH_min) / dFeH)) + 1;

	// Construct the SED interpolation grid
	sed_interp = new TBilinearInterp<TSED>(Mr_min, Mr_max, N_Mr, FeH_min, FeH_max, N_FeH);
	unsigned int idx;
	double colors[NBANDS-1];
	unsigned int r_index = 1; // TODO: indicate r_index in the template file

	// Now do a second pass to load the SEDs
	in.clear();
	in.seekg(0, std::ios_base::beg);
	if(!in) { std::cerr << "# Could not seek back to beginning of SED file!" << std::endl; }
	unsigned int count=0;
	while(std::getline(in, line)) {
		if(!line.size()) { continue; }		// empty line
		if(line[0] == '#') { continue; }	// comment

		std::istringstream ss(line);
		ss >> Mr >> FeH;

		idx = sed_interp->get_index(Mr, FeH);

		TSED &sed_tmp = (*sed_interp)[idx];
		//sed_tmp.Mr = Mr;
		//sed_tmp.FeH = FeH;
		for(unsigned int i=0; i<NBANDS-1; i++) { ss >> colors[i]; }

		// Transform colors into absolute magnitudes
		sed_tmp.absmag[r_index] = Mr;
		for(int i=r_index-1; i>=0; i--) { sed_tmp.absmag[i] = sed_tmp.absmag[i+1] + colors[i]; }
		for(int i=r_index+1; i<NBANDS; i++) { sed_tmp.absmag[i] = sed_tmp.absmag[i-1] - colors[i-1]; }

		count++;
	}
	in.close();

	if(count != N_FeH*N_Mr) {
		std::cerr << "# Incomplete SED library provided (grid is sparse, i.e. missing some values of (Mr,FeH)). This may cause problems." << std::endl;
	}
	std::cout << "# Loaded " << N_FeH*N_Mr << " SEDs from " << seds_fname << std::endl;

	Mr_min_seds = Mr_min;
	Mr_max_seds = Mr_max;
	FeH_min_seds = FeH_min;
	FeH_max_seds = FeH_max;

	N_Mr_seds = N_Mr;
	N_FeH_seds = N_FeH;

	std::cout << "# " << Mr_min_seds << " < Mr < " << Mr_max_seds << std::endl;
	std::cout << "# " << FeH_min_seds << " < FeH < " << FeH_max_seds << std::endl;

	return true;
}


TSED TStellarModel::get_sed(double Mr, double FeH) {
	return (*sed_interp)(Mr, FeH);
}

// x = {M_r, Fe/H}
bool TStellarModel::get_sed(const double* x, TSED& sed) const {
	if((x[0] <= Mr_min_seds) || (x[0] >= Mr_max_seds) || (x[1] <= FeH_min_seds) || (x[1] >= FeH_max_seds)) {
		return false;
	}
	sed = (*sed_interp)(x[0], x[1]);
	return true;
}

bool TStellarModel::get_sed(double Mr, double FeH, TSED& sed) const {
	if((Mr <= Mr_min_seds) || (Mr >= Mr_max_seds) || (FeH <= FeH_min_seds) || (FeH >= FeH_max_seds)) {
		return false;
	}
	sed = (*sed_interp)(Mr, FeH);
	return true;
}

// Access by grid index, and set FeH and Mr to their values at that index
bool TStellarModel::get_sed(unsigned int Mr_idx, unsigned int FeH_idx,
						    TSED& sed, double& Mr, double& FeH) const {
	if((Mr_idx >= N_Mr_seds) || (FeH_idx >= N_FeH_seds)) {
		std::cerr << " ! Mr_idx = " << Mr_idx
				  << " , N_Mr_seds = " << N_Mr_seds << std::endl;
  		std::cerr << " ! FeH_idx = " << FeH_idx
  				  << " , N_FeH_seds = " << N_FeH_seds << std::endl;
		return false;
	}

	unsigned int flat_idx = sed_interp->get_flat_index(Mr_idx, FeH_idx);
	sed = (*sed_interp)[flat_idx];
	sed_interp->get_xy(Mr_idx, FeH_idx, Mr, FeH);

	return true;
}

unsigned int TStellarModel::get_N_FeH() const {
	return N_FeH_seds;
}

unsigned int TStellarModel::get_N_Mr() const {
	return N_Mr_seds;
}

bool TStellarModel::in_model(double Mr, double FeH) {
	return (Mr > Mr_min_seds) && (Mr < Mr_max_seds) && (FeH > FeH_min_seds) && (FeH < FeH_max_seds);
}

double TStellarModel::get_log_lf(double Mr) const {
	return (*log_lf_interp)(Mr) - log_lf_norm;
}



/****************************************************************************************************************************
 *
 * TStellarAbundance
 *
 ****************************************************************************************************************************/

// Defaults are from Chabrier (2003)
TStellarAbundance::TStellarAbundance(int component) {
	if(component == 0) {	// Disk defaults
		set_IMF(0., -1.10, 0.692, 1.3);
		set_SFR(1., 5.e9, 2.e9, 12.e9);
	} else {		// Halo defaults
		set_IMF(-0.155, -0.658, 0.33, 1.3);
		set_SFR(1.e2, 10.e9, 1.5e9, 12.5e9);
	}
}

TStellarAbundance::~TStellarAbundance() { }

double TStellarAbundance::IMF(double logM) const {
	if(logM <= logM_norm) {
		return IMF_norm * exp( -(logM - logM_c)*(logM - logM_c) / (2.*sigma_logM_2) );
	} else {
		return IMF_norm * A_21 * pow(10, -x * logM);
	}
}

double TStellarAbundance::SFR(double tau) const {
	if(tau >= tau_max) {
		return 0.;
	} else {
		//double tmp =  -(tau - tau_burst)*(tau - tau_burst) / (2.*sigma_tau_2);
		return SFR_norm * ( 1. + A_burst * exp( -(tau - tau_burst)*(tau - tau_burst) / (2.*sigma_tau_2) ) );
	}
}

void TStellarAbundance::set_IMF(double _logM_norm, double _logM_c, double _sigma_logM, double _x) {
	logM_norm = _logM_norm;
	logM_c = _logM_c;
	sigma_logM_2 = _sigma_logM * _sigma_logM;
	x = _x;
	A_21 = pow(10, x * logM_norm) * exp( -(logM_norm - logM_c)*(logM_norm - logM_c) / (2.*sigma_logM_2) );

	// Determine normalization constant s.t. the IMF integrates to unity
	IMF_norm = sqrt(PI) / 2. * ( 1. + erf( (logM_norm - logM_c) / (SQRT2 * _sigma_logM) ) );
	IMF_norm += A_21 * exp( -(x * LN10) * logM_norm );
	IMF_norm = 1. / IMF_norm;

	//std::cerr << "# IMF(logM = logM_norm-) = " << IMF(logM_norm - 0.0001) << std::endl;
	//std::cerr << "# IMF(logM = logM_norm+) = " << IMF(logM_norm + 0.0001) << std::endl;
}

void TStellarAbundance::set_SFR(double _A_burst, double _tau_burst, double _sigma_tau, double _tau_max) {
	A_burst = _A_burst;
	tau_burst = _tau_burst;
	sigma_tau_2 = _sigma_tau * _sigma_tau;
	tau_max = _tau_max;

	// Determine normalization constant s.t. the SFR integrates to unity
	SFR_norm = 1. + A_burst * sqrt(PI) / 2. * erf((tau_max - tau_burst) / (SQRT2 * _sigma_tau));
	SFR_norm = 1. / SFR_norm;
}






/****************************************************************************************************************************
 *
 * TSyntheticStellarModel
 *
 ****************************************************************************************************************************/

TSyntheticStellarModel::TSyntheticStellarModel(std::string seds_fname) {
	H5::H5File file(seds_fname.c_str(), H5F_ACC_RDONLY);
	//cout << "File opened." << endl;

	H5::DataSet dataset = file.openDataSet("PARSEC PS1 Templates");
	//cout << "Dataset opened." << endl;

	/*
	 *  Memory datatype
	 */
	H5::CompType mtype(sizeof(TSynthSED));
	mtype.insertMember("Z", HOFFSET(TSynthSED, Z), H5::PredType::NATIVE_FLOAT);
	mtype.insertMember("logtau", HOFFSET(TSynthSED, logtau), H5::PredType::NATIVE_FLOAT);
	mtype.insertMember("logMass_init", HOFFSET(TSynthSED, logMass_init), H5::PredType::NATIVE_FLOAT);
	mtype.insertMember("logTeff", HOFFSET(TSynthSED, logTeff), H5::PredType::NATIVE_FLOAT);
	mtype.insertMember("logg", HOFFSET(TSynthSED, logg), H5::PredType::NATIVE_FLOAT);
	mtype.insertMember("M_g", HOFFSET(TSynthSED, M_g), H5::PredType::NATIVE_FLOAT);
	mtype.insertMember("M_r", HOFFSET(TSynthSED, M_r), H5::PredType::NATIVE_FLOAT);
	mtype.insertMember("M_i", HOFFSET(TSynthSED, M_i), H5::PredType::NATIVE_FLOAT);
	mtype.insertMember("M_z", HOFFSET(TSynthSED, M_z), H5::PredType::NATIVE_FLOAT);
	mtype.insertMember("M_y", HOFFSET(TSynthSED, M_y), H5::PredType::NATIVE_FLOAT);

	/*
	 *  Dataspace
	 */
	hsize_t length;
	H5::DataSpace dataspace = dataset.getSpace();
	dataspace.getSimpleExtentDims(&length);
	std::cerr << "# # of elements: " << length << std::endl;

	/*
	 *  Read in data
	 */
	TSynthSED *data = new TSynthSED[length];
	dataset.read(data, mtype);
	std::cerr << "# Read in " << length << " stellar templates." << std::endl;

	H5::DataSet dim_dataset = file.openDataSet("Dimensions");
	std::cerr << "# Opened 'Dimensions' dataset." << std::endl;

	/*
	 *  Memory datatype
	 */
	H5::CompType dim_mtype(sizeof(TGridDim));
	dim_mtype.insertMember("N_Z", HOFFSET(TGridDim, N_Z), H5::PredType::NATIVE_UINT32);
	dim_mtype.insertMember("N_logtau", HOFFSET(TGridDim, N_logtau), H5::PredType::NATIVE_UINT32);
	dim_mtype.insertMember("N_logMass_init", HOFFSET(TGridDim, N_logMass_init), H5::PredType::NATIVE_UINT32);
	dim_mtype.insertMember("Z_min", HOFFSET(TGridDim, Z_min), H5::PredType::NATIVE_FLOAT);
	dim_mtype.insertMember("Z_max", HOFFSET(TGridDim, Z_max), H5::PredType::NATIVE_FLOAT);
	dim_mtype.insertMember("logtau_min", HOFFSET(TGridDim, logtau_min), H5::PredType::NATIVE_FLOAT);
	dim_mtype.insertMember("logtau_max", HOFFSET(TGridDim, logtau_max), H5::PredType::NATIVE_FLOAT);
	dim_mtype.insertMember("logMass_init_min", HOFFSET(TGridDim, logMass_init_min), H5::PredType::NATIVE_FLOAT);
	dim_mtype.insertMember("logMass_init_max", HOFFSET(TGridDim, logMass_init_max), H5::PredType::NATIVE_FLOAT);

	hsize_t dim_length;
	H5::DataSpace dim_dataspace = dim_dataset.getSpace();
	dim_dataspace.getSimpleExtentDims(&dim_length);
	std::cerr << "# # of elements: " << dim_length << std::endl;

	/*
	 *  Read in dimensions
	 */
	dim_dataset.read(&grid_dim, dim_mtype);
	std::cerr << "# Read in dimensions." << std::endl;

	/*
	 *  Construct trilinear interpolator
	 */
	unsigned int N_points[3] = {grid_dim.N_logMass_init, grid_dim.N_logtau, grid_dim.N_Z};
	double min[3] = {grid_dim.logMass_init_min, grid_dim.logtau_min, log10(grid_dim.Z_min/0.019)};
	double max[3] = {grid_dim.logMass_init_max, grid_dim.logtau_max, log10(grid_dim.Z_max/0.019)};
	TSED empty;
	empty.absmag[0] = std::numeric_limits<double>::quiet_NaN();
	empty.absmag[1] = std::numeric_limits<double>::quiet_NaN();
	empty.absmag[2] = std::numeric_limits<double>::quiet_NaN();
	empty.absmag[3] = std::numeric_limits<double>::quiet_NaN();
	empty.absmag[4] = std::numeric_limits<double>::quiet_NaN();
	sed_interp = new TMultiLinearInterp<TSED>(&min[0], &max[0], &N_points[0], 3, empty);

	std::cerr << "# Constructing interpolating grid over synthetic magnitudes." << std::endl;
	TSED tmp;
	bool good_sed;
	unsigned int N_filtered = 0;
	for(unsigned int i=0; i<length; i++) {
		Theta[0] = data[i].logMass_init;
		Theta[1] = data[i].logtau;
		Theta[2] = log10(data[i].Z/0.019);

		tmp.absmag[0] = data[i].M_g;
		tmp.absmag[1] = data[i].M_r;
		tmp.absmag[2] = data[i].M_i;
		tmp.absmag[3] = data[i].M_z;
		tmp.absmag[4] = data[i].M_y;

		good_sed = true;
		for(size_t k=0; k<5; k++) {
			if((tmp.absmag[k] < -6.) || (tmp.absmag[k] > 25.)) {
				good_sed = false;
				N_filtered++;
				//std::cerr << "# erroneous magnitude: " << data[i].logMass_init << " " << data[i].logtau << " " << data[i].Z << " ==> " << tmp.absmag[k] << std::endl;
				break;
			}
		}
		if(good_sed) { sed_interp->set(&Theta[0], tmp); }
	}

	std::cerr << "# Done constructing stellar library. " << N_filtered << " stellar templates rejected." << std::endl;

	delete[] data;
}

TSyntheticStellarModel::~TSyntheticStellarModel() {
	delete sed_interp;
}

bool TSyntheticStellarModel::get_sed(const double* MtZ, TSED& sed) const {
	return (*sed_interp)(MtZ, sed);
}

bool TSyntheticStellarModel::get_sed(double logMass, double logtau, double FeH, TSED &sed) {
	Theta[0] = logMass;
	Theta[1] = logtau;
	Theta[2] = FeH;

	return (*sed_interp)(&Theta[0], sed);
}



/****************************************************************************************************************************
 *
 * TExtinctionModel
 *
 ****************************************************************************************************************************/

TExtinctionModel::TExtinctionModel(std::string A_RV_fname) {
	std::vector<double> Acoeff;
	std::vector<double> RV;
	double tmp;

	// Load in A coefficients for each R_V
	std::ifstream in(A_RV_fname.c_str());
	if(!in) { std::cerr << "Could not read extinction coefficients from '" << A_RV_fname << std::endl; abort(); }
	std::string line;
	RV_min = inf_replacement;
	RV_max = neg_inf_replacement;
	while(std::getline(in, line)) {
		if(!line.size()) { continue; }		// empty line
		if(line[0] == '#') { continue; }	// comment

		std::istringstream ss(line);
		ss >> tmp;
		RV.push_back(tmp);
		if(tmp < RV_min) { RV_min = tmp; }
		if(tmp > RV_max) { RV_max = tmp; }

		for(unsigned int i=0; i<NBANDS; i++) {
			ss >> tmp;
			if(ss.fail()) { std::cerr << "Not enough bands in line. Expected " << NBANDS << ". Got " << i << " instead." << std::endl; abort(); }
			Acoeff.push_back(tmp);
		}
	}

	A_spl = new gsl_spline*[NBANDS];
	acc = new gsl_interp_accel*[NBANDS];

	unsigned int N = RV.size();
	double Acoeff_i[N];
	double RV_arr[N];
	for(unsigned int k=0; k<N; k++) { RV_arr[k] = RV[k]; }
	for(unsigned int i=0; i<NBANDS; i++) {
		for(unsigned int k=0; k<N; k++) { Acoeff_i[k] = Acoeff[NBANDS*k + i]; }
		A_spl[i] = gsl_spline_alloc(gsl_interp_cspline, N);
		acc[i] = gsl_interp_accel_alloc();
		gsl_spline_init(A_spl[i], RV_arr, Acoeff_i, N);
	}
}

TExtinctionModel::~TExtinctionModel() {
	for(unsigned int i=0; i<NBANDS; i++) {
		gsl_spline_free(A_spl[i]);
		gsl_interp_accel_free(acc[i]);
	}
	delete[] A_spl;
	delete[] acc;
}

double TExtinctionModel::get_A(double RV, unsigned int i) {
	if(!in_model(RV)) { return std::numeric_limits<double>::quiet_NaN(); }
	return gsl_spline_eval(A_spl[i], RV, acc[i]);
}

bool TExtinctionModel::in_model(double RV) {
	return (RV >= RV_min) && (RV <= RV_max);
}


/****************************************************************************************************************************
 *
 * TEBVSmoothing
 *
 ****************************************************************************************************************************/

TEBVSmoothing::TEBVSmoothing(double alpha_coeff[2], double beta_coeff[2],
							 double pct_smoothing_min, double pct_smoothing_max) {
	_alpha_coeff[0] = alpha_coeff[0];
	_alpha_coeff[1] = alpha_coeff[1];

	_beta_coeff[0] = beta_coeff[0];
	_beta_coeff[1] = beta_coeff[1];

	_pct_smoothing_min = pct_smoothing_min;
	_pct_smoothing_max = pct_smoothing_max;

	_healpix_scale = 60.0 * 180.0 / (SQRTPI * SQRT3);
}

TEBVSmoothing::~TEBVSmoothing() {}

double TEBVSmoothing::nside_2_arcmin(unsigned int nside) const {
	return _healpix_scale / ((double)nside);
}

// Calculate the percent smoothing (in the E(B-V) direction) to apply to
// individual stellar probability density surfaces.
void TEBVSmoothing::calc_pct_smoothing(unsigned int nside,
                                       double EBV_min, double EBV_max, int n_samples,
                                       std::vector<double>& sigma_pct) const {
    double log_scale = log10(nside_2_arcmin(nside));

    //std::cerr << "scale = " << pow(10., log_scale) << "'" << std::endl;

	//std::cerr << "a_coeff = " << _alpha_coeff[0] << ", " << _alpha_coeff[1] << std::endl;
	//std::cerr << "b_coeff = " << _beta_coeff[0] << ", " << _beta_coeff[1] << std::endl;

	double alpha = pow10(_alpha_coeff[0] * log_scale + _alpha_coeff[1]);
	double beta = pow10(_beta_coeff[0] * log_scale + _beta_coeff[1]);

	//std::cerr << "alpha = " << alpha << std::endl;
	//std::cerr << "beta = " << beta << std::endl;

	double dE = (EBV_max - EBV_min) / (double)(n_samples - 1);
	double sigma_pct_tmp;

	sigma_pct.clear();

	double EBV = EBV_min;

	for(int i=0; i<n_samples; i++, EBV+=dE) {
		sigma_pct_tmp = alpha * EBV + beta;

		if(sigma_pct_tmp < _pct_smoothing_min) {
			sigma_pct_tmp = _pct_smoothing_min;
		} else if(sigma_pct_tmp > _pct_smoothing_max) {
			sigma_pct_tmp = _pct_smoothing_max;
		}

		//std::cerr << i << " (" << EBV << "): " << sigma_pct_tmp << std::endl;

		sigma_pct.push_back(sigma_pct_tmp);
	}
}

double TEBVSmoothing::get_pct_smoothing_min() const { return _pct_smoothing_min; }

double TEBVSmoothing::get_pct_smoothing_max() const { return _pct_smoothing_max; }




/****************************************************************************************************************************
 *
 * TLuminosityFunc
 *
 ****************************************************************************************************************************/

/*
void TLuminosityFunc::load(const std::string &fn) {
	std::ifstream in(fn.c_str());
	if(!in) { std::cerr << "Could not read LF from '" << fn << "'\n"; abort(); }

	dMr = -1;
	log_lf_norm = 0.;
	lf.clear();

	std::string line;
	double Mr, Phi;
	while(std::getline(in, line))
	{
		if(!line.size()) { continue; }		// empty line
		if(line[0] == '#') { continue; }	// comment

		std::istringstream ss(line);
		ss >> Mr >> Phi;

		if(dMr == -1) {
			Mr0 = Mr; dMr = 0;
		} else if(dMr == 0) {
			dMr = Mr - Mr0;
		}

		lf.push_back(log(Phi));
		log_lf_norm += Phi;
	}

	double Mr1 = Mr0 + dMr*(lf.size()-1);
	lf_interp = new TLinearInterp(Mr0, Mr1, lf.size());
	for(unsigned int i=0; i<lf.size(); i++) { (*lf_interp)[i] = lf[i]; }

	log_lf_norm *= Mr1 / (double)(lf.size());
	log_lf_norm = log(log_lf_norm);

	std::cerr << "# Loaded Phi(" << Mr0 << " <= Mr <= " <<  Mr0 + dMr*(lf.size()-1) << ") LF from " << fn << "\n";
}
*/
