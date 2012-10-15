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
 * TGalacticModel
 * 
 ****************************************************************************************************************************/

TGalacticModel::TGalacticModel() {
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
	
	// Halo
	fh = 0.0051;
	qh = 0.70;
	nh = -2.62;
	R_br = 27800;
	nh_outer = -3.8;
	fh_outer = fh * pow(R_br/R0, nh - nh_outer);
	
	// Metallicity
	mu_FeH_inf = -0.82;
	delta_mu_FeH = 0.55;
	H_mu_FeH = 500;
}

TGalacticModel::TGalacticModel(double _R0, double _Z0, double _H1, double _L1, double _f_thick, double _H2, double _L2, double _fh, double _qh, double _nh, double _R_br, double _nh_outer, double _mu_FeH_inf, double _delta_mu_FeH, double _H_mu_FeH)
	: R0(_R0), Z0(_Z0), H1(_H1), L1(_L1), f_thick(_f_thick), H2(_H2), L2(_L2), fh(_fh), qh(_qh), nh(_nh), R_br(_R_br), nh_outer(_nh_outer), mu_FeH_inf(_mu_FeH_inf), delta_mu_FeH(_delta_mu_FeH), H_mu_FeH(_H_mu_FeH)
{
	fh_outer = fh * pow(R_br/R0, nh-nh_outer);
}

TGalacticModel::~TGalacticModel() {
	
}

double TGalacticModel::rho_halo(double R, double Z) const {
	double r_eff2 = R*R + (Z/qh)*(Z/qh);
	if(r_eff2 <= R_br*R_br) {
		return fh*pow(r_eff2/(R0*R0), nh/2.);
	} else {
		return fh_outer*pow(r_eff2/(R0*R0), nh_outer/2.);
	}
}

double TGalacticModel::rho_disk(double R, double Z) const {
	double rho_thin = exp(-(fabs(Z+Z0) - fabs(Z0))/H1 - (R-R0)/L1);
	double rho_thick = f_thick * exp(-(fabs(Z+Z0) - fabs(Z0))/H2 - (R-R0)/L2);
	return rho_thin + rho_thick;
}

// Mean disk metallicity at given position in space
double TGalacticModel::mu_FeH_disk(double Z) const {
	return mu_FeH_inf + delta_mu_FeH * exp(-fabs(Z+Z0)/H_mu_FeH);
}

double TGalacticModel::log_p_FeH(double FeH, double R, double Z) const {
	#define sqrttwopi 2.50662827
	double f_H = rho_halo(R, Z) / rho_disk(R, Z);
	
	// Halo
	double mu_H = -1.46;
	double sigma_H = 0.3;
	double P_tmp = f_H * exp(-(FeH-mu_H)*(FeH-mu_H)/(2.*sigma_H*sigma_H)) / (sqrttwopi*sigma_H);
	
	// Metal-poor disk
	double mu_D = mu_FeH_disk(Z) - 0.067;
	double sigma_D = 0.2;
	P_tmp += 0.63 * (1-f_H) * exp(-(FeH-mu_D)*(FeH-mu_D)/(2.*sigma_D*sigma_D)) / (sqrttwopi*sigma_D);
	
	// Metal-rich disk
	double mu_D_poor = mu_D + 0.14;
	double sigma_D_poor = 0.2;
	P_tmp += 0.37 * (1-f_H) * exp(-(FeH-mu_D_poor)*(FeH-mu_D_poor)/(2.*sigma_D_poor*sigma_D_poor)) / (sqrttwopi*sigma_D_poor);
	#undef sqrttwopi
	
	return log(P_tmp);
}


/****************************************************************************************************************************
 * 
 * TGalacticLOSModel
 * 
 ****************************************************************************************************************************/

TGalacticLOSModel::TGalacticLOSModel(double l, double b) 
	: TGalacticModel()
{
	init(l, b);
}

TGalacticLOSModel::TGalacticLOSModel(double l, double b, double _R0, double _Z0, double _H1, double _L1, double _f_thick, double _H2, double _L2, double _fh, double _qh, double _nh, double _R_br, double _nh_outer, double _mu_FeH_inf, double _delta_mu_FeH, double _H_mu_FeH)
	: TGalacticModel(_R0, _Z0, _H1, _L1, _f_thick, _H2, _L2, _fh, _qh, _nh, _R_br, _nh_outer, _mu_FeH_inf, _delta_mu_FeH, _H_mu_FeH)
{
	fh_outer = fh * pow(R_br/R0, nh-nh_outer);
	init(l, b);
}

TGalacticLOSModel::~TGalacticLOSModel() {
	delete log_dNdmu_arr;
	delete f_halo_arr;
	delete mu_FeH_disk_arr;
}

void TGalacticLOSModel::init(double l, double b) {
	// Precompute trig functions
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

double TGalacticLOSModel::log_p_FeH(double DM, double FeH) const {
	#define sqrttwopi 2.50662827
	double f_H = f_halo(DM);
	
	// Halo
	double mu_H = -1.46;
	double sigma_H = 0.3;
	double P_tmp = f_H * exp(-(FeH-mu_H)*(FeH-mu_H)/(2.*sigma_H*sigma_H)) / (sqrttwopi*sigma_H);
	
	// Metal-poor disk
	double mu_D = mu_FeH_disk_interp(DM) - 0.067;
	double sigma_D = 0.2;
	P_tmp += 0.63 * (1-f_H) * exp(-(FeH-mu_D)*(FeH-mu_D)/(2.*sigma_D*sigma_D)) / (sqrttwopi*sigma_D);
	
	// Metal-rich disk
	double mu_D_poor = mu_D + 0.14;
	double sigma_D_poor = 0.2;
	P_tmp += 0.37 * (1-f_H) * exp(-(FeH-mu_D_poor)*(FeH-mu_D_poor)/(2.*sigma_D_poor*sigma_D_poor)) / (sqrttwopi*sigma_D_poor);
	#undef sqrttwopi
	
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
	
	std::cerr << "# Loaded Phi(" << Mr0 << " <= Mr <= " <<  Mr1 << ") LF from " << lf_fname << "\n";
	
	return true;
}

bool TStellarModel::load_seds(std::string seds_fname) {
	double Mr, FeH, dMr_tmp, dFeH_tmp;
	double Mr_last = std::numeric_limits<double>::infinity();
	double FeH_last = std::numeric_limits<double>::infinity();
	double Mr_min = std::numeric_limits<double>::infinity();
	double Mr_max = -std::numeric_limits<double>::infinity();
	double FeH_min = std::numeric_limits<double>::infinity();
	double FeH_max = -std::numeric_limits<double>::infinity();
	double dMr = std::numeric_limits<double>::infinity();
	double dFeH = std::numeric_limits<double>::infinity();
	
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
	
	if(count != N_FeH*N_Mr) { std::cerr << "# Incomplete SED library provided (grid is sparse, i.e. missing some values of (Mr,FeH)). This may cause problems." << std::endl; }
	std::cerr << "# Loaded " << N_FeH*N_Mr << " SEDs from " << seds_fname << std::endl;
	
	return true;
}


TSED TStellarModel::get_sed(double Mr, double FeH) {
	return (*sed_interp)(Mr, FeH);
}

bool TStellarModel::in_model(double Mr, double FeH) {
	return (Mr > Mr_min_seds) && (Mr < Mr_max_seds) && (FeH > FeH_min_seds) && (FeH < FeH_max_seds);
}

double TStellarModel::get_log_lf(double Mr) {
	return (*log_lf_interp)(Mr) - log_lf_norm;
}



/****************************************************************************************************************************
 * 
 * TStellarAbundance
 * 
 ****************************************************************************************************************************/

double TStellarAbundance::IMF(double logM) {
	if(logM <= logM_norm) {
		return IMF_norm * exp( -(logM - logM_c)*(logM - logM_c) / (2.*sigma_logM_2) );
	} else {
		return IMF_norm * A_21 * pow(10, -x * logM);
	}
}

double TStellarAbundance::SFR(double tau) {
	if(tau >= tau_max) {
		return 0.;
	} else {
		return SFR_norm * ( 1. + exp( -(tau - tau_burst)*(tau - tau_burst) / (2.*sigma_tau_2) ) );
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
	std::cout << "# of elements: " << length << std::endl;
	
	/*
	 *  Read in data
	 */
	TSynthSED *data = new TSynthSED[length];
	dataset.read(data, mtype);
	
	H5::DataSet dim_dataset = file.openDataSet("Dimensions");
	
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
	
	/*
	 *  Read in dimensions
	 */
	dataset.read(&grid_dim, dim_mtype);
	
	/*
	 *  Construct trilinear interpolator
	 */
	unsigned int N_points[3] = {grid_dim.N_logMass_init, grid_dim.N_logtau, grid_dim.N_Z};
	double min[3] = {grid_dim.logMass_init_min, grid_dim.logtau_min, grid_dim.Z_min};
	double max[3] = {grid_dim.logMass_init_max, grid_dim.logtau_max, grid_dim.Z_max};
	TSED empty;
	empty.absmag[0] = std::numeric_limits<double>::quiet_NaN();
	empty.absmag[1] = std::numeric_limits<double>::quiet_NaN();
	empty.absmag[2] = std::numeric_limits<double>::quiet_NaN();
	empty.absmag[3] = std::numeric_limits<double>::quiet_NaN();
	empty.absmag[4] = std::numeric_limits<double>::quiet_NaN();
	sed_interp = new TMultiLinearInterp<TSED>(&min[0], &max[0], &N_points[0], 3, empty);
	
	TSED tmp;
	for(unsigned int i=0; i<length; i++) {
		Theta[0] = data[i].logMass_init;
		Theta[1] = data[i].logtau;
		Theta[2] = data[i].Z;
		
		tmp.absmag[0] = data[i].M_g;
		tmp.absmag[1] = data[i].M_r;
		tmp.absmag[2] = data[i].M_i;
		tmp.absmag[3] = data[i].M_z;
		tmp.absmag[4] = data[i].M_y;
		
		sed_interp->set(&Theta[0], tmp);
	}
	
	delete[] data;
}

TSyntheticStellarModel::~TSyntheticStellarModel() {
	delete sed_interp;
}

bool TSyntheticStellarModel::get_sed(double *MtZ, TSED &sed) const {
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
	RV_min = std::numeric_limits<double>::infinity();
	RV_max = -std::numeric_limits<double>::infinity();
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
}

double TExtinctionModel::get_A(double RV, unsigned int i) {
	if(!in_model(RV)) { return std::numeric_limits<double>::quiet_NaN(); }
	return gsl_spline_eval(A_spl[i], RV, acc[i]);
}

bool TExtinctionModel::in_model(double RV) {
	return (RV >= RV_min) && (RV <= RV_max);
}

