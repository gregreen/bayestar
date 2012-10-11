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

using namespace std;


int main(int argc, char **argv) {
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
	
	return 0;
}
