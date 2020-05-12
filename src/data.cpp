/*
 * data.cpp
 *
 * Defines class for stellar data.
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

#include "data.h"


TStellarData::TStellarData(const std::string& infile, std::string _pix_name, double err_floor)
	: pix_name(_pix_name)
{
	std::string group = "photometry";
	load(infile, group, pix_name, err_floor);
	if(EBV <= 0.) {
		EBV = 4.;
	} else {
		// Floor on E(B-V)_SFD
		EBV = sqrt(EBV*EBV + 0.02*0.02);
	}
}



TStellarData::TStellarData(uint64_t _healpix_index, uint32_t _nside, bool _nested, double _l, double _b) {
    healpix_index = _healpix_index;
	nside = _nside;
	nested = _nested;
	l = _l;
	b = _b;
	EBV = -1.;

	std::stringstream tmp_name;
	tmp_name << "pixel " << nside << "-" << healpix_index;
	pix_name = tmp_name.str();
}


bool TStellarData::save(const std::string& fname, const std::string& group, const std::string &dset, int compression) {
	if((compression < 0) || (compression > 9)) {
		std::cerr << "! Invalid gzip compression level: " << compression << std::endl;
		return false;
	}

	hsize_t nstars = star.size();
	if(nstars == 0) {
		std::cerr << "! No stars to write." << std::endl;
		return false;
	}

	H5::Exception::dontPrint();

	std::unique_ptr<H5::H5File> file = H5Utils::openFile(fname);
	if(!file) { return false; }

	std::unique_ptr<H5::Group> gp = H5Utils::openGroup(*file, group);
	if(!gp) { return false; }

	/*
	 *  Photometry
	 */

	// Datatype
	hsize_t nbands = NBANDS;
	H5::ArrayType f4arr(H5::PredType::NATIVE_FLOAT, 1, &nbands);
	H5::ArrayType u4arr(H5::PredType::NATIVE_FLOAT, 1, &nbands);
	H5::CompType dtype(sizeof(TFileData));
	dtype.insertMember("obj_id", HOFFSET(TFileData, obj_id), H5::PredType::NATIVE_UINT64);
	dtype.insertMember("l", HOFFSET(TFileData, l), H5::PredType::NATIVE_DOUBLE);
	dtype.insertMember("b", HOFFSET(TFileData, b), H5::PredType::NATIVE_DOUBLE);
	dtype.insertMember("pi", HOFFSET(TFileData, pi), H5::PredType::NATIVE_FLOAT);
	dtype.insertMember("pi_err", HOFFSET(TFileData, pi_err), H5::PredType::NATIVE_FLOAT);
	dtype.insertMember("mag", HOFFSET(TFileData, mag), f4arr);
	dtype.insertMember("err", HOFFSET(TFileData, err), f4arr);
	dtype.insertMember("maglimit", HOFFSET(TFileData, maglimit), f4arr);
	dtype.insertMember("nDet", HOFFSET(TFileData, N_det), u4arr);
	dtype.insertMember("EBV", HOFFSET(TFileData, EBV), H5::PredType::NATIVE_FLOAT);

	// Dataspace
	hsize_t dim = nstars;
	H5::DataSpace dspace(1, &dim);

	// Property List
	H5::DSetCreatPropList plist;
	plist.setChunk(1, &nstars);
	plist.setDeflate(compression);

	// Dataset
	H5::DataSet dataset = gp->createDataSet(dset, dtype, dspace, plist);

	// Write dataset
	TFileData* data = new TFileData[nstars];
	for(size_t i=0; i<nstars; i++) {
		data[i].obj_id = star[i].obj_id;
		data[i].l = star[i].l;
		data[i].b = star[i].b;
		data[i].pi = star[i].pi;
		data[i].pi_err = star[i].pi_err;
		for(size_t k=0; k<NBANDS; k++) {
			data[i].mag[k] = star[i].m[k];
			data[i].err[k] = star[i].err[k];
			data[i].maglimit[k] = star[i].maglimit[k];
		}
		data[i].EBV = star[i].EBV;
	}
	dataset.write(data, dtype);

	/*
	 *  Attributes
	 */

	dim = 1;
	H5::DataSpace att_dspace(1, &dim);

	H5::PredType att_dtype = H5::PredType::NATIVE_UINT64;
	H5::Attribute att_healpix_index = dataset.createAttribute("healpix_index", att_dtype, att_dspace);
	att_healpix_index.write(att_dtype, &healpix_index);

	att_dtype = H5::PredType::NATIVE_UINT32;
	H5::Attribute att_nside = dataset.createAttribute("nside", att_dtype, att_dspace);
	att_nside.write(att_dtype, &nside);

	att_dtype = H5::PredType::NATIVE_UCHAR;
	H5::Attribute att_nested = dataset.createAttribute("nested", att_dtype, att_dspace);
	att_nested.write(att_dtype, &nested);

	att_dtype = H5::PredType::NATIVE_DOUBLE;
	H5::Attribute att_l = dataset.createAttribute("l", att_dtype, att_dspace);
	att_l.write(att_dtype, &l);

	att_dtype = H5::PredType::NATIVE_DOUBLE;
	H5::Attribute att_b = dataset.createAttribute("b", att_dtype, att_dspace);
	att_b.write(att_dtype, &b);

	att_dtype = H5::PredType::NATIVE_DOUBLE;
	H5::Attribute att_EBV = dataset.createAttribute("EBV", att_dtype, att_dspace);
	att_EBV.write(att_dtype, &EBV);

	file->close();

	delete[] data;

	return true;
}


void TStellarData::TMagnitudes::set(const TStellarData::TFileData& dat, double err_floor) {
	obj_id = dat.obj_id;
	l = dat.l;
	b = dat.b;
	pi = dat.pi;
	pi_err = dat.pi_err;
	lnL_norm = 0.;
	for(unsigned int i=0; i<NBANDS; i++) {
		m[i] = dat.mag[i];
		err[i] = sqrt(dat.err[i]*dat.err[i] + err_floor*err_floor);
		maglimit[i] = dat.maglimit[i];
		maglim_width[i] = 0.20;
		if(err[i] < 9.e9) {	// Ignore missing bands (otherwise, they affect evidence)
			lnL_norm += 0.9189385332 + log(err[i]);
		}
		N_det[i] = dat.N_det[i];

		// Specific tweaks to PS1 / 2MASS
		if(i < 5) {
			maglimit[i] += 0.16;	// Fix PS1 magnitude limit
		} else {
			maglimit[i] += 0.20;	// Increase 2MASS magnitude limit, to be conservative
			maglim_width[i] = 0.30;	// Widen 2MASS magnitude cutoff
		}
	}
	EBV = dat.EBV;
}


bool TStellarData::load(const std::string& fname, const std::string& group, const std::string& dset,
			double err_floor, double default_EBV) {
	std::unique_ptr<H5::H5File> file = H5Utils::openFile(fname);
	if(!file) { return false; }

	std::unique_ptr<H5::Group> gp = H5Utils::openGroup(*file, group);
	if(!gp) { return false; }

	H5::DataSet dataset = gp->openDataSet(dset);

	/*
	 *  Photometry
	 */

	// Datatype
	hsize_t nbands = NBANDS;
	H5::ArrayType f4arr(H5::PredType::NATIVE_FLOAT, 1, &nbands);
	H5::ArrayType u4arr(H5::PredType::NATIVE_UINT32, 1, &nbands);
	H5::CompType dtype(sizeof(TFileData));
	dtype.insertMember("obj_id", HOFFSET(TFileData, obj_id), H5::PredType::NATIVE_UINT64);
	dtype.insertMember("l", HOFFSET(TFileData, l), H5::PredType::NATIVE_DOUBLE);
	dtype.insertMember("b", HOFFSET(TFileData, b), H5::PredType::NATIVE_DOUBLE);
	dtype.insertMember("pi", HOFFSET(TFileData, pi), H5::PredType::NATIVE_FLOAT);
	dtype.insertMember("pi_err", HOFFSET(TFileData, pi_err), H5::PredType::NATIVE_FLOAT);
	dtype.insertMember("mag", HOFFSET(TFileData, mag), f4arr);
	dtype.insertMember("err", HOFFSET(TFileData, err), f4arr);
	dtype.insertMember("maglimit", HOFFSET(TFileData, maglimit), f4arr);
	dtype.insertMember("nDet", HOFFSET(TFileData, N_det), u4arr);
	dtype.insertMember("EBV", HOFFSET(TFileData, EBV), H5::PredType::NATIVE_FLOAT);

	// Dataspace
	hsize_t length;
	H5::DataSpace dataspace = dataset.getSpace();
	dataspace.getSimpleExtentDims(&length);

	// Read in dataset
	TFileData* data_buf = new TFileData[length];
	dataset.read(data_buf, dtype);
	//std::cerr << "# Read in dimensions." << std::endl;

	// Fix magnitude limits
	for(int n=0; n<nbands; n++) {
		float tmp;
		float maglim_replacement = 25.;

		// Find the 95th percentile of valid magnitude limits
		std::vector<float> maglimit;
		for(hsize_t i=0; i<length; i++) {
			tmp = data_buf[i].maglimit[n];

			if((tmp > 10.) && (tmp < 40.) && (!std::isnan(tmp))) {
				maglimit.push_back(tmp);
			}
		}

		//std::sort(maglimit.begin(), maglimit.end());
		if(maglimit.size() != 0) {
			maglim_replacement = percentile(maglimit, 95.);
		}

		// Replace missing magnitude limits with the 95th percentile magnitude limit
		for(hsize_t i=0; i<length; i++) {
			tmp = data_buf[i].maglimit[n];

			if(!((tmp > 10.) && (tmp < 40.)) || std::isnan(tmp)) {
				//std::cout << i << ", " << n << ":  " << tmp << std::endl;
				data_buf[i].maglimit[n] = maglim_replacement;
			}
		}
	}

	//int n_filtered = 0;
	//int n_M_dwarfs = 0;

	TMagnitudes mag_tmp;
	for(size_t i=0; i<length; i++) {
		mag_tmp.set(data_buf[i], err_floor);
		star.push_back(mag_tmp);

		//int n_informative = 0;

		// Remove g-band
		//mag_tmp.m[0] = 0.;
		//mag_tmp.err[0] = 1.e10;

		//double g_err = mag_tmp.err[0];
		//mag_tmp.err[0] = sqrt(g_err*g_err + 0.1*0.1);
		//star.push_back(mag_tmp);

		// Filter bright end
                // TODO: Put this into query_lsd.py
		/*for(int j=0; j<NBANDS; j++) {
			if((mag_tmp.err[j] < 1.e9) && (mag_tmp.m[j] < 14.)) {
				mag_tmp.err[j] = 1.e10;
				mag_tmp.m[j] = 0.;
			}

			if(mag_tmp.err[j] < 1.e9) {
				n_informative++;
			}
		}*/

		// Filter M dwarfs based on color cut
		//bool M_dwarf = false;
		/*bool M_dwarf = true;

		double A_g = 3.172;
		double A_r = 2.271;
		double A_i = 1.682;

		if(mag_tmp.m[0] - A_g / (A_g - A_r) * (mag_tmp.m[0] - mag_tmp.m[1] - 1.2) > 20.) {
			M_dwarf = false;
		} else if(mag_tmp.m[1] - mag_tmp.m[2] - (A_r - A_i) / (A_g - A_r) * (mag_tmp.m[0] - mag_tmp.m[1]) < 0.) {
			M_dwarf = false;
		} else {
			n_M_dwarfs++;
		}
		*/

		/*if(n_informative >= 4) { //&& (!M_dwarf)) {
			star.push_back(mag_tmp);
		} else {
			n_filtered++;
		}*/
	}

	//std::cerr << "# of stars filtered: " << n_filtered << std::endl;
	//std::cerr << "# of M dwarfs: " << n_M_dwarfs << std::endl;

	/*
	 *  Attributes
	 */

	H5::Attribute att = dataset.openAttribute("healpix_index");
	H5::DataType att_dtype = H5::PredType::NATIVE_UINT64;
	att.read(att_dtype, reinterpret_cast<void*>(&healpix_index));

	att = dataset.openAttribute("nested");
	att_dtype = H5::PredType::NATIVE_UCHAR;
	att.read(att_dtype, reinterpret_cast<void*>(&nested));

	att = dataset.openAttribute("nside");
	att_dtype = H5::PredType::NATIVE_UINT32;
	att.read(att_dtype, reinterpret_cast<void*>(&nside));

	att = dataset.openAttribute("l");
	att_dtype = H5::PredType::NATIVE_DOUBLE;
	att.read(att_dtype, reinterpret_cast<void*>(&l));

	att = dataset.openAttribute("b");
	att_dtype = H5::PredType::NATIVE_DOUBLE;
	att.read(att_dtype, reinterpret_cast<void*>(&b));

	att = dataset.openAttribute("EBV");
	att_dtype = H5::PredType::NATIVE_DOUBLE;
	att.read(att_dtype, reinterpret_cast<void*>(&EBV));

	// TEST: Force l, b to anticenter
	//l = 180.;
	//b = 0.;

	if((EBV <= 0.) || (EBV > default_EBV) || std::isnan(EBV)) { EBV = default_EBV; }

	delete[] data_buf;

	return true;
}






TDraw1D::TDraw1D(func_ptr_t func, double _x_min, double _x_max, void* _params, unsigned int samples, bool is_log)
	: x_of_P(NULL), r(NULL), params(_params)
{
	assert(samples > 1);

	x_min = _x_min;
	x_max = _x_max;
	double dx = (x_max - x_min) / (double)(samples - 1);

	// Construct an interpolator for P(x)
	double fill = -1.;
	TMultiLinearInterp<double> P_of_x(&x_min, &x_max, &samples, 1, fill);
	double x;
	double P = 0.;
	for(unsigned int i=0; i<samples; i++) {
		x = x_min + (double)i * dx;
		P_of_x.set(&x, P);
		if(i < samples - 1) {
			if(is_log) { P += dx * exp(func(x, params)); } else { P += dx * func(x, params); }
		}
	}
	double P_norm = P;

	// Invert the interpolator for get x(P)
	double P_min = 0.;
	double P_max = 1.;
	double dP = 1. / (double)(samples - 1);
	x_of_P = new TMultiLinearInterp<double>(&P_min, &P_max, &samples, 1, fill);
	unsigned int k_last = 0;
	double P_tmp, dPdx;
	for(unsigned int i=0; i<samples; i++) {
		P = (double)i * dP;
		for(unsigned int k=k_last+1; k<samples; k++) {
			x = x_min + (double)k * dx;
			P_tmp = P_of_x(&x) / P_norm;
			if(P_tmp >= P) {
				dPdx = (P_tmp - (double)(i-1)*dP) / dx;
				x = x_min + (double)(k-1) * dx + dP / dPdx;
				k_last = k - 1;
				break;
			}
		}

		x_of_P->set(&P, x);
	}
	P_tmp = 1.;
	x_of_P->set(&P_tmp, x_max);

	seed_gsl_rng(&r);
}

TDraw1D::~TDraw1D() {
	delete x_of_P;
	gsl_rng_free(r);
}


double TDraw1D::operator()() {
	double P = gsl_rng_uniform(r);
	return (*x_of_P)(&P);
}



double log_dNdmu_draw(double DM, void* params) {
	TGalacticLOSModel *gal_model = static_cast<TGalacticLOSModel*>(params);
	return gal_model->log_dNdmu(DM);
}

double disk_IMF_draw(double logMass, void* params) {
	TGalacticLOSModel *gal_model = static_cast<TGalacticLOSModel*>(params);
	return gal_model->IMF(logMass, 0);
}

double halo_IMF_draw(double logMass, void* params) {
	TGalacticLOSModel *gal_model = static_cast<TGalacticLOSModel*>(params);
	return gal_model->IMF(logMass, 1);
}

double disk_SFR_draw(double tau, void* params) {
	TGalacticLOSModel *gal_model = static_cast<TGalacticLOSModel*>(params);
	return gal_model->SFR(tau, 0);
}

double halo_SFR_draw(double tau, void* params) {
	TGalacticLOSModel *gal_model = static_cast<TGalacticLOSModel*>(params);
	return gal_model->SFR(tau, 1);
}

double disk_FeH_draw(double FeH, void* params) {
	TGalacticLOSModel *gal_model = static_cast<TGalacticLOSModel*>(params);
	return gal_model->p_FeH_fast(5., FeH, 0);
}

double halo_FeH_draw(double FeH, void* params) {
	TGalacticLOSModel *gal_model = static_cast<TGalacticLOSModel*>(params);
	return gal_model->p_FeH_fast(23., FeH, 1);
}

double Mr_draw(double Mr, void* params) {
	TStellarModel *stellar_model = static_cast<TStellarModel*>(params);
	return stellar_model->get_log_lf(Mr);
}

void draw_from_synth_model(size_t nstars, double RV, TGalacticLOSModel& gal_model, TSyntheticStellarModel& stellar_model,
                     TStellarData& stellar_data, TExtinctionModel& ext_model, double (&mag_limit)[5]) {
	unsigned int samples = 1000;
	void* gal_model_ptr = static_cast<void*>(&gal_model);

	double DM_min = 0.;
	double DM_max = 25.;
	TDraw1D draw_DM(&log_dNdmu_draw, DM_min, DM_max, gal_model_ptr, samples, true);

	double logMass_min = -0.9;
	double logMass_max = 1.1;
	TDraw1D draw_logMass_disk(&disk_IMF_draw, logMass_min, logMass_max, gal_model_ptr, samples, false);
	TDraw1D draw_logMass_halo(&halo_IMF_draw, logMass_min, logMass_max, gal_model_ptr, samples, false);

	double tau_min = 1.e6;
	double tau_max = 13.e9;
	TDraw1D draw_tau_disk(&disk_SFR_draw, tau_min, tau_max, gal_model_ptr, samples, false);
	TDraw1D draw_tau_halo(&halo_SFR_draw, tau_min, tau_max, gal_model_ptr, samples, false);

	double FeH_min = -2.5;
	double FeH_max = 1.;
	TDraw1D draw_FeH_disk(&disk_FeH_draw, FeH_min, FeH_max, gal_model_ptr, samples, false);
	TDraw1D draw_FeH_halo(&halo_FeH_draw, FeH_min, FeH_max, gal_model_ptr, samples, false);

	stellar_data.clear();
	gal_model.get_lb(stellar_data.l, stellar_data.b);

	gsl_rng *r;
	seed_gsl_rng(&r);
	double EBV, DM, logtau, logMass, FeH;
	double f_halo;
	bool halo, in_lib, observed;
	TSED sed;
	double mag[NBANDS];
	double err[NBANDS];
	std::cout << "Component E(B-V)    DM        log(Mass) log(tau)  [Fe/H]    g         r         i         z         y        " << std::endl;
	std::cout << "=============================================================================================================" << std::endl;
	std::cout.flags(std::ios::left);
	std::cout.precision(3);
	for(size_t i=0; i<nstars; i++) {
		observed = false;
		while(!observed) {
			// Draw E(B-V)
			EBV = gsl_ran_chisq(r, 1.);

			// Draw DM
			DM = draw_DM();

			// Draw stellar type
			f_halo = gal_model.f_halo(DM);
			halo = (gsl_rng_uniform(r) < f_halo);
			in_lib = false;
			while(!in_lib) {
				if(halo) {
					logMass = draw_logMass_halo();
					logtau = log10(draw_tau_halo());
					FeH = draw_FeH_halo();
				} else {
					logMass = draw_logMass_disk();
					logtau = log10(draw_tau_disk());
					FeH = draw_FeH_disk();
				}
				in_lib = stellar_model.get_sed(logMass, logtau, FeH, sed);
			}

			// Generate magnitudes
			observed = true;
			unsigned int N_nonobs = 0;
			for(size_t k=0; k<NBANDS; k++) {
				mag[k] = sed.absmag[k] + DM + EBV * ext_model.get_A(RV, k);
				err[k] = 0.02 + 0.1*exp(mag[i]-mag_limit[i]-1.5);
				mag[k] += gsl_ran_gaussian_ziggurat(r, err[k]);

				// Require detection in g band and 3 other bands
				if(mag[k] > mag_limit[k]) {
					N_nonobs++;
					if((k == 0) || N_nonobs > 1) {
						observed = false;
						break;
					}
				}
			}
		}

		std::cout << (halo ? "halo" : "disk") << "      ";
		std::cout << std::setw(9) << EBV << " ";
		std::cout << std::setw(9) << DM << " ";
		std::cout << std::setw(9) << logMass << " ";
		std::cout << std::setw(9) << logtau << " ";
		std::cout << std::setw(9) << FeH << " ";
		for(size_t k=0; k<NBANDS; k++) {
			std::cout << std::setw(9) << mag[k] << " ";
		}
		std::cout << std::endl;

		TStellarData::TMagnitudes mag_tmp(mag, err);
		mag_tmp.obj_id = i;
		mag_tmp.l = stellar_data.l;
		mag_tmp.b = stellar_data.b;
		stellar_data.star.push_back(mag_tmp);

	}
	std::cout << std::endl;

	gsl_rng_free(r);

	/*std::vector<bool> filled;
	DM_of_P.get_filled(filled);
	for(std::vector<bool>::iterator it = filled.begin(); it != filled.end(); ++it) {
		std::cout << *it << std::endl;
	}
	*/

}



void draw_from_emp_model(size_t nstars, double RV, TGalacticLOSModel& gal_model, TStellarModel& stellar_model,
                     TStellarData& stellar_data, TExtinctionModel& ext_model, double (&mag_limit)[5]) {
	unsigned int samples = 1000;
	void* gal_model_ptr = static_cast<void*>(&gal_model);
	void* stellar_model_ptr = static_cast<void*>(&stellar_model);

	double DM_min = 0.;
	double DM_max = 25.;
	TDraw1D draw_DM(&log_dNdmu_draw, DM_min, DM_max, gal_model_ptr, samples, true);

	double FeH_min = -2.5;
	double FeH_max = 1.;
	TDraw1D draw_FeH_disk(&disk_FeH_draw, FeH_min, FeH_max, gal_model_ptr, samples, false);
	TDraw1D draw_FeH_halo(&halo_FeH_draw, FeH_min, FeH_max, gal_model_ptr, samples, false);

	double Mr_min = -1.;
	double Mr_max = mag_limit[1];
	TDraw1D draw_Mr(&Mr_draw, Mr_min, Mr_max, stellar_model_ptr, samples, true);

	stellar_data.clear();
	gal_model.get_lb(stellar_data.l, stellar_data.b);

	gsl_rng *r;
	seed_gsl_rng(&r);
	double EBV, DM, Mr, FeH;
	double f_halo;
	bool halo, in_lib, observed;
	TSED sed;
	double mag[NBANDS];
	double err[NBANDS];
	std::cout << "#         Component E(B-V)    DM        Mr        [Fe/H]    g         r         i         z         y        " << std::endl;
	std::cout << "=============================================================================================================" << std::endl;
	std::cout.flags(std::ios::left);
	std::cout.precision(3);
	for(size_t i=0; i<nstars; i++) {
		observed = false;
		while(!observed) {
			// Draw DM
			DM = draw_DM();

			// Draw E(B-V)
			//EBV = gsl_ran_chisq(r, 1.);

			EBV = 0.;
			//if(DM > 5.) { EBV += 0.05; }
			if(DM > 10.) { EBV += 2.5; }

			// Draw stellar type
			f_halo = gal_model.f_halo(DM);
			halo = (gsl_rng_uniform(r) < f_halo);
			in_lib = false;
			while(!in_lib) {
				if(halo) {
					FeH = draw_FeH_halo();
				} else {
					FeH = draw_FeH_disk();
				}
				Mr = draw_Mr();
				in_lib = stellar_model.get_sed(Mr, FeH, sed);
			}

			// Generate magnitudes
			observed = true;
			unsigned int N_nonobs = 0;
			double p_det;
			for(size_t k=0; k<NBANDS; k++) {
				mag[k] = sed.absmag[k] + DM + EBV * ext_model.get_A(RV, k);
				err[k] = 0.02 + 0.3*exp(mag[k]-mag_limit[k]);
				if(err[k] > 1.) { err[k] = 1.; }
				mag[k] += gsl_ran_gaussian_ziggurat(r, err[k]);

				// Require detection in g band and 3 other bands
				p_det = 0.5 - 0.5 * erf((mag[k] - mag_limit[k] + 0.5) / 0.25);
				if(gsl_rng_uniform(r) > p_det) {
					mag[k] = 0.;
					err[k] = 1.e10;

					N_nonobs++;
					if((k == 0) || N_nonobs > 1) {
						observed = false;
						break;
					}
				}
			}
		}

		std::cout << std::setw(9) << i+1 << " ";
		std::cout << (halo ? "halo" : "disk") << "      ";
		std::cout << std::setw(9) << EBV << " ";
		std::cout << std::setw(9) << DM << " ";
		std::cout << std::setw(9) << Mr << " ";
		std::cout << std::setw(9) << FeH << " ";
		for(size_t k=0; k<NBANDS; k++) {
			std::cout << std::setw(9) << mag[k] << " ";
		}
		std::cout << std::endl;

		TStellarData::TMagnitudes mag_tmp(mag, err);
		mag_tmp.obj_id = i;
		mag_tmp.l = stellar_data.l;
		mag_tmp.b = stellar_data.b;
		stellar_data.star.push_back(mag_tmp);

	}
	std::cout << std::endl;

	gsl_rng_free(r);

	/*std::vector<bool> filled;
	DM_of_P.get_filled(filled);
	for(std::vector<bool>::iterator it = filled.begin(); it != filled.end(); ++it) {
		std::cout << *it << std::endl;
	}
	*/

}



herr_t fetch_pixel_name(hid_t loc_id, const char *name, void *opdata) {
	std::vector<std::string> *pix_name = reinterpret_cast<std::vector<std::string>*>(opdata);

	std::string group_name(name);
	//group_name << name;

	//std::string tmp_name;
	try {
		//group_name >> tmp_name;
		pix_name->push_back(group_name);
	} catch(...) {
		// pass
	}

	return 0;
}

void get_input_pixels(
        std::string fname,
        std::vector<std::string> &pix_name,
        const std::string &base
) {
	std::unique_ptr<H5::H5File> file = H5Utils::openFile(fname, H5Utils::READ);

	file->iterateElems(base, NULL, fetch_pixel_name, reinterpret_cast<void*>(&pix_name));
}


void get_pixel_props(
        const std::string& fname,
        const std::vector<std::string>& pix_name,
        std::vector<double>& l,
        std::vector<double>& b,
        std::vector<double>& EBV,
        std::vector<uint32_t>& nside,
        std::vector<uint64_t>& healpix_index,
        const std::string& base
) {
	std::unique_ptr<H5::H5File> f = H5Utils::openFile(fname, H5Utils::READ);
    
    for(auto& name : pix_name) {
        std::stringstream path;
        path << base << "/" << name;
        std::unique_ptr<H5::Group> g = H5Utils::openGroup(*f, path.str(), H5Utils::READ);
        if(!g) { std::cerr << "Could not load " << path.str() << " !" << std::endl; }
        l.push_back(H5Utils::read_attribute<double>(*g, "l"));
        b.push_back(H5Utils::read_attribute<double>(*g, "b"));
        EBV.push_back(H5Utils::read_attribute<double>(*g, "EBV"));
        nside.push_back(H5Utils::read_attribute<uint32_t>(*g, "nside"));
        healpix_index.push_back(H5Utils::read_attribute<uint64_t>(*g, "healpix_index"));
    }
}


/*************************************************************************
 *
 *   Auxiliary Functions
 *
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
