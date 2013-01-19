/*
 * data.h
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

#ifndef _STELLAR_DATA_H__
#define _STELLAR_DATA_H__

#include "model.h"

#include <vector>
#include <string>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

//#define __STDC_LIMIT_MACROS
#include <stdint.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "h5utils.h"


// Auxiliary functions
void seed_gsl_rng(gsl_rng **r);


struct TStellarData {
	struct TFileData {
		uint64_t obj_id;
		double l, b;
		float mag[NBANDS];
		float err[NBANDS];
		float maglimit[NBANDS];
		uint32_t N_det[NBANDS];
	};
	
	struct TMagnitudes {
		uint64_t obj_id;
		double l, b;
		double m[NBANDS];
		double err[NBANDS];
		double maglimit[NBANDS];
		unsigned int N_det[NBANDS];
		double lnL_norm;
		
		TMagnitudes() {}
		
		TMagnitudes(double (&_m)[NBANDS], double (&_err)[NBANDS]) {
			lnL_norm = NBANDS * 0.9189385332;
			for(unsigned int i=0; i<NBANDS; i++) {
				m[i] = _m[i];
				err[i] = _err[i];
				maglimit[i] = 0.;	// TODO
				lnL_norm += log(err[i]);
			}
		}
		
		TMagnitudes& operator=(const TMagnitudes& rhs) {
			obj_id = rhs.obj_id;
			l = rhs.l;
			b = rhs.b;
			for(unsigned int i=0; i<NBANDS; i++) {
				m[i] = rhs.m[i];
				err[i] = rhs.err[i];
				maglimit[i] = rhs.maglimit[i];
				N_det[i] = rhs.N_det[i];
			}
			lnL_norm = rhs.lnL_norm;
			return *this;
		}
		
		void set(const TStellarData::TFileData& dat, double err_floor = 0.02);
	};
	
	uint64_t healpix_index;
	uint32_t nside;
	bool nested;
	double l, b, EBV;
	std::vector<TMagnitudes> star;
	
	TStellarData(const std::string& infile, uint32_t _healpix_index, double err_floor = 0.02);
	TStellarData(uint64_t _healpix_index, uint32_t _nside, bool _nested, double _l, double _b);
	TStellarData() {}
	
	TMagnitudes& operator[](unsigned int index) { return star.at(index); }
	
	void clear() { star.clear(); }
	
	// Read/write stellar photometry from/to HDF5 files
	bool save(const std::string& fname, const std::string& group, const std::string& dset, int compression=9);
	bool load(const std::string& fname, const std::string& group, const std::string& dset, double err_floor=0.02);
	
	// Load magnitudes and errors of stars along one line of sight, along with (l,b) for the given l.o.s. Same as load_data, but for binary files.
	// Each binary file contains magnitudes and errors for stars along multiple lines of sight. The stars are grouped into lines of sight, called
	// pixels. The file begins with the number of pixels:
	//
	// 	N_pixels (uint32)
	// 
	// Each pixel has the form
	// 
	// 	Header:
	// 		healpix_index	(uint32)
	// 		l_mean		(double)
	// 		b_mean		(double)
	// 		N_stars		(uint32)
	// 	Data - For each star:
	// 		obj_id		(uint64)
	// 		l_star		(double)
	// 		b_star		(double)
	// 		mag[NBANDS]	(double)
	// 		err[NBANDS]	(double)
	// 
	bool load_data(std::string infile, uint32_t pix_index, double err_floor=0.02) {
		std::fstream f(infile.c_str(), std::ios::in | std::ios::binary);
		if(!f) { f.close(); return false; }
		
		// Read in number of pixels (sets of stars) in file
		uint32_t N_pix;
		f.read(reinterpret_cast<char*>(&N_pix), sizeof(N_pix));
		if(pix_index > N_pix) {
			std::cerr << "Pixel requested (" << pix_index << ") greater than number of pixels in file (" << N_pix << ")." << std::endl;
			f.close();
			return false;
		}
		
		// Seek to beginning of requested pixel
		uint32_t N_stars, healpixnum;
		for(uint32_t i=0; i<=pix_index; i++) {
			f.read(reinterpret_cast<char*>(&healpixnum), sizeof(healpixnum));
			f.read(reinterpret_cast<char*>(&l), sizeof(l));
			f.read(reinterpret_cast<char*>(&b), sizeof(b));
			f.read(reinterpret_cast<char*>(&N_stars), sizeof(N_stars));
			if(i < pix_index) { f.seekg(N_stars * (3 + 2*NBANDS)*sizeof(double), std::ios::cur); }
		}
		
		if(f.eof()) {
			std::cerr << "End of file reached before requested pixel (" << pix_index << ") reached. File corrupted." << std::endl;
			f.close();
			return false;
		}
		
		// Exit if N_stars is unrealistically large
		if(N_stars > 1e7) {
			std::cerr << "Error reading " << infile << ". Header absurdly indicates " << N_stars << " stars. Aborting attempt to read file." << std::endl;
			f.close();
			return false;
		}
		
		// Read in each star
		star.reserve(N_stars);
		for(uint32_t i=0; i<N_stars; i++) {
			TMagnitudes tmp;
			f.read(reinterpret_cast<char*>(&(tmp.obj_id)), sizeof(uint64_t));
			f.read(reinterpret_cast<char*>(&(tmp.l)), sizeof(double));
			f.read(reinterpret_cast<char*>(&(tmp.b)), sizeof(double));
			f.read(reinterpret_cast<char*>(&(tmp.m[0])), NBANDS*sizeof(double));
			f.read(reinterpret_cast<char*>(&(tmp.err[0])), NBANDS*sizeof(double));
			tmp.lnL_norm = 0.9189385332;
			for(unsigned int i=0; i<NBANDS; i++) {
				tmp.err[i] = sqrt(tmp.err[i]*tmp.err[i] + err_floor*err_floor);
				tmp.lnL_norm += log(tmp.err[i]);
			}
			star.push_back(tmp);
			
			//for(unsigned int i=0; i<NBANDS; i++) { std::cout << tmp.m[i] << "\t"; }
			//for(unsigned int i=0; i<NBANDS; i++) { std::cout << tmp.err[i] << "\t"; }
			//std::cout << std::endl;
		}
		
		if(f.fail()) { f.close(); return false; }
		
		std::cerr << "# Loaded " << N_stars << " stars from pixel " << pix_index << " of " << infile << "." << std::endl;
		
		f.close();
		return true;
	}
};


class TDraw1D {
public:
	typedef double (*func_ptr_t)(double x, void* params);
	
	TDraw1D(func_ptr_t func, double _x_min, double _x_max, void* _params, unsigned int samples, bool is_log = false);
	~TDraw1D();
	
	double operator()();
	
private:
	double x_min, x_max;
	TMultiLinearInterp<double>* x_of_P;
	gsl_rng *r;
	void *params;
};

// Generate mock photometry from the given stellar and Galactic model, and magnitude limits
void draw_from_synth_model(size_t nstars, double RV, TGalacticLOSModel& gal_model, TSyntheticStellarModel& stellar_model,
                           TStellarData& stellar_data, TExtinctionModel& ext_model, double (&mag_limit)[NBANDS]);
void draw_from_emp_model(size_t nstars, double RV, TGalacticLOSModel& gal_model, TStellarModel& stellar_model,
                           TStellarData& stellar_data, TExtinctionModel& ext_model, double (&mag_limit)[NBANDS]);

// Return healpix indices of pixels in input file
void get_input_pixels(std::string fname, std::vector<unsigned int> &healpix_index);


#endif // _STELLAR_DATA_H__