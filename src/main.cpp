/*
 * main.cpp
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


#include <iostream>
#include <iomanip>

#include "model.h"
#include "data.h"
#include "sampler.h"
#include "los_sampler.h"

using namespace std;

void print_model() {
	double l = 180.;
	double b = 90.;
	TGalacticLOSModel los_model(l, b);
	double R, Z;
	cout << "(l, b) = (" << l << ", " << b << ")" << endl << endl;
	cout << "DM        ln(dNdmu) rho_disk  rho_halo  f_halo    R         Z" << endl;
	cout << "=====================================================================" << endl;
	for(double DM=5; DM<20; DM += 0.5) {
		los_model.DM_to_RZ(DM, R, Z);
		cout.flags(ios::left);
		cout.precision(4);
		cout << setw(9) << DM << " ";
		cout.precision(3);
		cout << setw(9) << los_model.log_dNdmu(DM) << " ";
		cout << setw(9) << los_model.rho_disk(R, Z) << " ";
		cout << setw(9) << los_model.rho_halo(R, Z) << " ";
		cout << setw(9) << los_model.f_halo(DM) << " ";
		cout << setw(9) << R << " ";
		cout << setw(9) << Z << endl;
	}
	cout << endl;
	
	TStellarModel stellar_model("/home/greg/projects/bayestar/data/PSMrLF.dat", "/home/greg/projects/bayestar/data/PScolors.dat");
	cout << endl;
	cout << "Mr        ln(dn/dMr) Mg     Mr     Mi     Mz     My" << endl;
	cout << "=======================================================" << endl;
	for(double Mr=-1; Mr < 28; Mr += 1) {
		cout.flags(ios::left);
		cout.precision(3);
		cout << setw(9) << Mr << " ";
		cout << setw(10) << stellar_model.get_log_lf(Mr) << " ";
		TSED sed = stellar_model.get_sed(Mr, -0.5);
		for(unsigned int i=0; i<5; i++) {
			cout << setw(6) << sed.absmag[i] << " ";
		}
		cout << endl;
	}
	cout << endl;
	
	TSyntheticStellarModel synthlib("/home/greg/projects/bayestar/data/PS1templates.h5");
	double logtau = 9.0;
	double FeH = -0.5;
	cout << endl;
	cout << "logM      IMF       SFR       Mg      Mr      Mi      Mz      My" << endl;
	cout << "===================================================================" << endl;
	for(double logM=-1.; logM < 1.; logM += 0.1) {
		cout.flags(ios::left);
		cout.precision(3);
		cout << setw(9) << logM << " ";
		cout << setw(9) << los_model.IMF(logM, 0) << " ";
		cout << setw(9) << los_model.SFR(pow(10, logtau), 0) << " ";
		
		cout.precision(4);
		TSED sed;
		if(synthlib.get_sed(logM, logtau, FeH, sed)) {
			for(unsigned int i=0; i<5; i++) {
				cout << setw(7) << sed.absmag[i] << " ";
			}
		} else {
			for(unsigned int i=0; i<5; i++) {
				cout << setw(7) << "-----   ";
			}
		}
		cout << endl;
	}
	cout << endl;
	
	cout << "          -----SFR-----------" << endl;
	cout << "logtau    Disk      Halo     " << endl;
	cout << "=============================" << endl;
	double tau;
	for(logtau = 6.; logtau < log10(16.e9); logtau += 0.05) {
		tau = pow(10, logtau);
		cout.flags(ios::left);
		cout.precision(3);
		cout << setw(9) << logtau << " ";
		cout << setw(9) << tau * los_model.SFR(pow(10, logtau), 0) << " ";
		cout << setw(9) << tau * los_model.SFR(pow(10, logtau), 1);
		cout << endl;
	}
	cout << endl;
	
	double min[2] = {-1., -1.};
	double max[2] = {1., 1.};
	unsigned int N_bins[2] = {20, 20};
	TSparseBinner binner(&(min[0]), &(max[0]), &(N_bins[0]), 2);
	double coord[2];
	for(double x=-0.95; x<1.; x+=0.1) {
		coord[0] = x;
		for(double y=-0.95; y<1.; y+=0.1) {
			coord[1] = y;
			binner(coord, x*y);
		}
	}
	
	cout << "         || ";
	for(double x=-0.95; x<1.; x+=0.1) {
		cout << setprecision(4) << setw(8) << x << " ";
	}
	cout << endl << "============";
	for(double x=-0.95; x<1.; x+=0.1) {
		cout << "=========";
	}
	cout << endl;
	for(double y=-0.95; y<1.; y+=0.1) {
		coord[1] = y;
		cout << setprecision(4) << setw(8) << y << " || ";
		for(double x=-0.95; x<1.; x+=0.1) {
			coord[0] = x;
			cout << setprecision(4) << setw(8) << binner.get_bin(coord) << " ";
		}
		cout << endl;
	}
	cout << endl;
	
	TExtinctionModel ext_model("/home/greg/projects/bayestar/data/PSExtinction.dat");
	cout << "R_V      g       r       i       z       y    " << endl;
	for(double RV = 2.1; RV < 6.; RV += 0.1) {
		cout << setprecision(2) << setw(6) << RV;
		for(unsigned int i=0; i<NBANDS; i++) {
			cout << " " << setprecision(4) << setw(7) << ext_model.get_A(RV, i);
		}
		cout << endl;
	}
	cout << endl;
}

void los_test() {
	double EBV_SFD = 0.5;
	
	TSyntheticStellarModel synthlib("/home/greg/projects/bayestar/data/PS1templates.h5");
	TExtinctionModel ext_model("/home/greg/projects/bayestar/data/PSExtinction.dat");
	TStellarData stellar_data("/home/greg/projects/bayestar/input/input_0.in", 0);
	TGalacticLOSModel los_model(stellar_data.l, stellar_data.b);
	
	sample_model_affine_synth(los_model, synthlib, ext_model, stellar_data, EBV_SFD);
}

void indiv_test() {
	double EBV_SFD = 1.5;
	
	TStellarModel emplib("/home/greg/projects/bayestar/data/PSMrLF.dat", "/home/greg/projects/bayestar/data/PScolors.dat");
	TSyntheticStellarModel synthlib("/home/greg/projects/bayestar/data/PS1templates.h5");
	TExtinctionModel ext_model("/home/greg/projects/bayestar/data/PSExtinction.dat");
	TStellarData stellar_data("/home/greg/projects/bayestar/input/input_0.in", 0);
	TGalacticLOSModel los_model(stellar_data.l, stellar_data.b);
	
	//sample_indiv_synth(los_model, synthlib, ext_model, stellar_data, EBV_SFD);
	//sample_indiv_emp(los_model, emplib, ext_model, stellar_data, EBV_SFD);
}

void mock_test() {
	size_t nstars = 25;
	double EBV_SFD = 1.5;
	double RV = 3.3;
	double l = 90.;
	double b = 10.;
	uint64_t healpix_index = 1519628;
	uint32_t nside = 512;
	bool nested = true;
	
	TStellarModel emplib("/home/greg/projects/bayestar/data/PSMrLF.dat", "/home/greg/projects/bayestar/data/PScolors.dat");
	TSyntheticStellarModel synthlib("/home/greg/projects/bayestar/data/PS1templates.h5");
	TExtinctionModel ext_model("/home/greg/projects/bayestar/data/PSExtinction.dat");
	TGalacticLOSModel los_model(l, b);
	//los_model.load_lf("/home/greg/projects/bayestar/data/PSMrLF.dat");
	TStellarData stellar_data(healpix_index, nside, nested, l, b);
	
	std::cout << std::endl;
	double mag_lim[5];
	for(size_t i=0; i<5; i++) { mag_lim[i] = 22.5; }
	draw_from_emp_model(nstars, RV, los_model, emplib, stellar_data, ext_model, mag_lim);
	
	std::stringstream group_name;
	group_name << healpix_index;
	remove("mock.hdf5");
	stellar_data.save("mock.hdf5", group_name.str());
	
	//sample_indiv_synth(los_model, synthlib, ext_model, stellar_data, EBV_SFD);
	
	TImgStack img_stack(stellar_data.star.size());
	
	sample_indiv_emp(los_model, emplib, ext_model, stellar_data, EBV_SFD, img_stack);
	
	// Fit line-of-sight extinction profile
	unsigned int N_regions = 20;
	sample_los_extinction(img_stack, N_regions, 1.e-50, 5.);
	
	TLOSMCMCParams params(&img_stack, 1.e-15, -1.);
	
	/*
	double Delta_EBV[6] = {10000.01, 10000.02, 10000.05, 1.0, 0.05, 10000000000.02};
	
	gsl_rng *r;
	seed_gsl_rng(&r);
	gen_rand_los_extinction(&(Delta_EBV[0]), N_regions+1, r, params);
	for(size_t i=0; i<=N_regions; i++) {
		std::cerr << i << ": " << Delta_EBV[i] << std::endl;
	}
	
	double *line_int = new double[img_stack.N_images];
	los_integral(img_stack, line_int, &(Delta_EBV[0]), N_regions);
	for(size_t i=0; i<img_stack.N_images; i++) {
		std::cerr << i << " --> " << line_int[i] << std::endl;
	}
	delete[] line_int;
	
	std::cerr << "ln(p) = " << lnp_los_extinction(&(Delta_EBV[0]), N_regions, params) << std::endl;
	*/
	
}

int main(int argc, char **argv) {
	mock_test();
	//indiv_test();
	//print_model();
	
	return 0;
}
