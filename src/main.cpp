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
#include <ctime>

#include <boost/program_options.hpp>

#include "cpp_utils.h"
#include "model.h"
#include "data.h"
#include "sampler.h"
#include "los_sampler.h"
#include "bayestar_config.h"

using namespace std;


void mock_test() {
	size_t nstars = 50;
	unsigned int N_regions = 20;
	double RV = 3.1;
	double l = 90.;
	double b = 10.;
	uint64_t healpix_index = 1519628;
	uint32_t nside = 512;
	bool nested = true;
	
	TStellarModel emplib(DATADIR "PSMrLF.dat", DATADIR "PScolors.dat");
	//TSyntheticStellarModel synthlib(DATADIR "PS1templates.h5");
	TExtinctionModel ext_model(DATADIR "PSExtinction.dat");
	TGalacticLOSModel los_model(l, b);
	//los_model.load_lf("/home/greg/projects/bayestar/data/PSMrLF.dat");
	TStellarData stellar_data(healpix_index, nside, nested, l, b);
	
	std::cout << std::endl;
	double mag_lim[5];
	for(size_t i=0; i<5; i++) { mag_lim[i] = 22.5; }
	draw_from_emp_model(nstars, RV, los_model, emplib, stellar_data, ext_model, mag_lim);
	
	std::string group = "photometry";
	std::stringstream dset;
	dset << "pixel " << healpix_index;
	remove("mock.h5");
	stellar_data.save("mock.h5", group, dset.str());
	
	// Prepare data structures for stellar parameters
	TImgStack img_stack(stellar_data.star.size());
	std::vector<bool> conv;
	std::vector<double> lnZ;
	
	std::string out_fname = "emp_out.h5";
	remove(out_fname.c_str());
	TMCMCOptions star_options(500, 20, 0.2, 4);
	sample_indiv_emp(out_fname, star_options, los_model, emplib, ext_model, stellar_data, img_stack, conv, lnZ);
	
	// Fit line-of-sight extinction profile
	img_stack.cull(conv);
	TMCMCOptions los_options(250, 15, 0.1, 4);
	//sample_los_extinction(out_fname, los_options, img_stack, N_regions, 1.e-50, 5., healpix_index);
	
	/*
	TLOSMCMCParams params(&img_stack, 1.e-100, -1.);
	
	
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
	gsl_set_error_handler_off();
	
	
	/*
	 *  Default commandline arguments
	 */
	
	std::string input_fname = "NONE";
	std::string output_fname = "NONE";
	
	bool saveSurfs = false;
	
	double err_floor = 20;
	
	bool synthetic = false;
	unsigned int star_steps = 1000;
	unsigned int star_samplers = 5;
	double star_p_replacement = 0.2;
	double sigma_RV = -1.;
	double minEBV = 0.;
	
	unsigned int N_regions = 20;
	unsigned int los_steps = 3000;
	unsigned int los_samplers = 2;
	double los_p_replacement = 0.0;
	
	unsigned int N_clouds = 1;
	unsigned int cloud_steps = 1000;
	unsigned int cloud_samplers = 80;
	double cloud_p_replacement = 0.2;
	
	bool disk_prior = false;
	bool SFDPrior = false;
	bool SFDsubpixel = false;
	double evCut = 15.;
	
	unsigned int N_runs = 4;
	unsigned int N_threads = 4;
	
	int verbosity = 0;
	
	
	/*
	 *  Parse commandline arguments
	 */
	
	namespace po = boost::program_options;
	po::options_description desc(std::string("Usage: ") + argv[0] + " [Input filename] [Output filename] \n\nOptions");
	desc.add_options()
		("help", "Display this help message")
		("version", "Display version number")
		("input", po::value<std::string>(&input_fname), "Input HDF5 filename (contains stellar photometry)")
		("output", po::value<std::string>(&output_fname), "Output HDF5 filename (MCMC output and smoothed probability surfaces)")
		
		("save-surfs", "Save probability surfaces.")
		
		("err-floor", po::value<double>(&err_floor), "Error to add in quadrature (in millimags)")
		("synthetic", "Use synthetic photometric library (default: use empirical library)")
		("star-steps", po::value<unsigned int>(&star_steps), "# of MCMC steps per star (per sampler)")
		("star-samplers", po::value<unsigned int>(&star_samplers), "# of samplers per dimension (stellar fit)")
		("star-p-replacement", po::value<double>(&star_p_replacement), "Probability of taking replacement step (stellar fit)")
		("sigma-RV", po::value<double>(&sigma_RV), "Variation in R_V (per star) (default: -1, interpreted as no variance)")
		("minEBV", po::value<double>(&minEBV), "Minimum stellar E(B-V) (default: 0)")
		
		("regions", po::value<unsigned int>(&N_regions), "# of piecewise-linear regions in l.o.s. extinction profile (default: 20)")
		("los-steps", po::value<unsigned int>(&los_steps), "# of MCMC steps in l.o.s. fit (per sampler)")
		("los-samplers", po::value<unsigned int>(&los_samplers), "# of samplers per dimension (l.o.s. fit)")
		("los-p-replacement", po::value<double>(&los_p_replacement), "Probability of taking replacement step (l.o.s. fit)")
		
		("clouds", po::value<unsigned int>(&N_clouds), "# of clouds along the line of sight (default: 0).\n"
		                                               "Setting this option causes the sampler to use a discrete\n"
		                                               "cloud model for the l.o.s. extinction profile.")
		("cloud-steps", po::value<unsigned int>(&cloud_steps), "# of MCMC steps in cloud fit (per sampler)")
		("cloud-samplers", po::value<unsigned int>(&cloud_samplers), "# of samplers per dimension (cloud fit)")
		("cloud-p-replacement", po::value<double>(&cloud_p_replacement), "Probability of taking replacement step (cloud fit)")
		
		("disk-prior", "Assume that dust density roughly traces stellar disk density.")
		("SFD-prior", "Use SFD E(B-V) as a prior on the total extinction in each pixel.")
		("SFD-subpixel", "Use SFD E(B-V) as a subpixel template for the angular variation in reddening.")
		("evidence-cut", po::value<double>(&evCut), "Delta lnZ to use as threshold for including star\n"
		                                            "in l.o.s. fit (default: 15).")
		
		("runs", po::value<unsigned int>(&N_runs), "# of times to run each chain (to check for non-convergence) (default: 4)")
		("threads", po::value<unsigned int>(&N_threads), "# of threads to run on (default: 4)")
		
		("verbosity", po::value<int>(&verbosity), "Level of verbosity (0 = minimal, 2 = highest)")
	;
	po::positional_options_description pd;
	pd.add("input", 1).add("output", 1);
	
	po::variables_map vm;
	po::store(po::command_line_parser(argc, argv).options(desc).positional(pd).run(), vm);
	po::notify(vm);
	
	if(vm.count("help")) { cout << desc << endl; return 0; }
	if(vm.count("version")) { cout << "git commit " << GIT_BUILD_VERSION << endl; return 0; }
	
	if(vm.count("synthetic")) { synthetic = true; }
	if(vm.count("save-surfs")) { saveSurfs = true; }
	if(vm.count("disk-prior")) { disk_prior = true; }
	if(vm.count("SFD-prior")) { SFDPrior = true; }
	if(vm.count("SFD-subpixel")) { SFDsubpixel = true; }
	
	// Convert error floor to mags
	err_floor /= 1000.;
	
	if(input_fname == "NONE") {
		cerr << "Input filename required." << endl << endl;
		cerr << desc << endl;
		return -1;
	}
	if(output_fname == "NONE") {
		cerr << "Output filename required." << endl << endl;
		cerr << desc << endl;
		return -1;
	}
	
	if(N_regions != 0) {
		if(120 % N_regions != 0) {
			cerr << "# of regions in extinction profile must divide 120 without remainder." << endl;
			return -1;
		}
	}
	
	time_t tmp_time = time(0);
	char * dt = ctime(&tmp_time);
	cout << "# Start time: " << dt;
	
	timespec prog_start_time;
	clock_gettime(CLOCK_MONOTONIC, &prog_start_time);
	
	
	/*
	 *  MCMC Options
	 */
	
	TMCMCOptions star_options(star_steps, star_samplers, star_p_replacement, N_runs);
	TMCMCOptions cloud_options(cloud_steps, cloud_samplers, cloud_p_replacement, N_runs);
	TMCMCOptions los_options(los_steps, los_samplers, los_p_replacement, N_runs);
	
	
	/*
	 *  Construct models
	 */
	
	TStellarModel *emplib = NULL;
	TSyntheticStellarModel *synthlib = NULL;
	if(synthetic) {
		synthlib = new TSyntheticStellarModel(DATADIR "PS1templates.h5");
	} else {
		emplib = new TStellarModel(DATADIR "PSMrLF.dat", DATADIR "PScolors.dat");
	}
	TExtinctionModel ext_model(DATADIR "PSExtinction.dat");
	
	
	/*
	 *  Execute
	 */
	
	omp_set_num_threads(N_threads);
	
	// Get list of pixels in input file
	vector<unsigned int> healpix_index;
	get_input_pixels(input_fname, healpix_index);
	cout << "# " << healpix_index.size() << " pixels in input file." << endl << endl;
	
	// Remove the output file
	remove(output_fname.c_str());
	H5::Exception::dontPrint();
	
	// Run each pixel
	timespec t_start, t_mid, t_end;
	
	double t_tot, t_star;
	unsigned int pixel_list_no = 0;
	
	for(vector<unsigned int>::iterator it = healpix_index.begin(); it != healpix_index.end(); ++it, pixel_list_no++) {
		clock_gettime(CLOCK_MONOTONIC, &t_start);
		
		cout << "# Healpix pixel " << *it << " (" << pixel_list_no + 1 << " of " << healpix_index.size() << ")" << endl;
		
		TStellarData stellar_data(input_fname, *it, err_floor);
		TGalacticLOSModel los_model(stellar_data.l, stellar_data.b);
		
		cout << "# (l, b) = " << stellar_data.l << ", " << stellar_data.b << endl;
		if(SFDPrior) { cout << "# E(B-V)_SFD = " << stellar_data.EBV << endl; }
		cout << "# " << stellar_data.star.size() << " stars in pixel" << endl;
		
		// Prepare data structures for stellar parameters
		TImgStack img_stack(stellar_data.star.size());
		vector<bool> conv;
		vector<double> lnZ;
		
		bool gatherSurfs = (N_regions || N_clouds || saveSurfs);
		
		// Sample individual stars
		if(synthetic) {
			sample_indiv_synth(output_fname, star_options, los_model, *synthlib, ext_model,
			                   stellar_data, img_stack, conv, lnZ, sigma_RV, minEBV, saveSurfs, gatherSurfs, verbosity);
		} else {
			sample_indiv_emp(output_fname, star_options, los_model, *emplib, ext_model,
			                 stellar_data, img_stack, conv, lnZ, sigma_RV, minEBV, saveSurfs, gatherSurfs, verbosity);
		}
		
		clock_gettime(CLOCK_MONOTONIC, &t_mid);
		
		// Filter based on convergence and lnZ
		assert(conv.size() == lnZ.size());
		vector<bool> keep;
		vector<double> lnZ_filtered;
		for(vector<double>::iterator it_lnZ = lnZ.begin(); it_lnZ != lnZ.end(); ++it_lnZ) {
			if(!isnan(*it_lnZ) && !is_inf_replacement(*it_lnZ)) {
				lnZ_filtered.push_back(*it_lnZ);
			}
		}
		double lnZmax = percentile_const(lnZ_filtered, 95.0);
		if(verbosity >= 2) { cout << "# ln(Z)_95pct = " << lnZmax << endl; }
		
		bool tmpFilter;
		size_t nFiltered = 0;
		std::vector<double> subpixel;
		lnZ_filtered.clear();
		for(size_t n=0; n<conv.size(); n++) {
			tmpFilter = conv[n] && (lnZ[n] > lnZmax - (20. + evCut)) && !isnan(lnZ[n]) && !is_inf_replacement(lnZ[n]);
			keep.push_back(tmpFilter);
			if(tmpFilter) {
				subpixel.push_back(stellar_data.star[n].EBV);
				lnZ_filtered.push_back(lnZ[n] - lnZmax);
			} else {
				nFiltered++;
			}
		}
		if(gatherSurfs) { img_stack.cull(keep); }
		
		// Fit line-of-sight extinction profile
		if((nFiltered < conv.size()) && ((N_clouds != 0) || (N_regions != 0))) {
			cout << "# of stars filtered: " << nFiltered << " of " << conv.size();
			cout << " (" << 100. * (double)nFiltered / (double)(conv.size()) << " %)" << endl;
			
			double p0 = exp(-5. - evCut);
			double EBV_max = -1.;
			if(SFDPrior) {
				if(SFDsubpixel) {
					EBV_max = 1.;
				} else {
					EBV_max = stellar_data.EBV;
				}
			}
			TLOSMCMCParams params(&img_stack, lnZ_filtered, p0, N_runs, N_threads, N_regions, EBV_max);
			if(SFDsubpixel) { params.set_subpixel_mask(subpixel); }
			if(N_clouds != 0) {
				sample_los_extinction_clouds(output_fname, cloud_options, params, N_clouds, *it, verbosity);
			}
			if(N_regions != 0) {
				params.gen_guess_covariance(1.);	// Covariance matrix for guess has (anti-)correlation length of 1 distance bin
				if(disk_prior) {
					params.calc_Delta_EBV_prior(los_model, stellar_data.EBV, verbosity);
				}
				sample_los_extinction(output_fname, los_options, params, *it, verbosity);
			}
		}
		
		stringstream group_name;
		group_name << "/pixel " << *it;
		H5Utils::add_watermark<uint64_t>(output_fname, group_name.str(), "nside", stellar_data.nside);
		
		clock_gettime(CLOCK_MONOTONIC, &t_end);
		t_tot = (t_end.tv_sec - t_start.tv_sec) + 1.e-9 * (t_end.tv_nsec - t_start.tv_nsec);
		t_star = (t_mid.tv_sec - t_start.tv_sec) + 1.e-9 * (t_mid.tv_nsec - t_start.tv_nsec);
		
		if(verbosity >= 1) {
			cout << endl;
			cout << "===================================================" << endl;
		}
		cout << "# Time elapsed for pixel: ";
		cout << setprecision(2) << t_tot;
		cout << " s (" << setprecision(2) << t_tot / (double)(stellar_data.star.size()) << " s / star)" << endl;
		cout << "# Percentage of time spent on l.o.s. fit: ";
		cout << setprecision(2) << 100. * (t_tot - t_star) / t_tot << " %" << endl;
		if(verbosity >= 1) {
			cout << "===================================================" << endl;
		}
		cout << endl;
	}
	
	
	/*
	 *  Add additional metadata to output file
	 */
	
	string watermark = GIT_BUILD_VERSION;
	H5Utils::add_watermark<string>(output_fname, "/", "bayestar git commit", watermark);
	
	stringstream commandline_args;
	for(int i=0; i<argc; i++) {
		commandline_args << argv[i] << " ";
	}
	string commandline_args_str(commandline_args.str());
	H5Utils::add_watermark<string>(output_fname, "/", "commandline invocation", commandline_args_str);
	
	
	/*
	 *  Cleanup
	 */
	
	if(synthlib != NULL) { delete synthlib; }
	if(emplib != NULL) { delete emplib; }
	
	tmp_time = time(0);
	dt = ctime(&tmp_time);
	cout << "# End time: " << dt;
	
	timespec prog_end_time;
	clock_gettime(CLOCK_MONOTONIC, &prog_end_time);
	double prog_ss = prog_end_time.tv_sec - prog_start_time.tv_sec + 1.e-9 * (prog_end_time.tv_nsec - prog_start_time.tv_nsec);
	int prog_mm = floor(prog_ss / 60.);
	int prog_hh = floor(prog_mm / 60.);
	int prog_dd = floor(prog_hh / 24.);
	prog_hh = prog_hh % 24;
	prog_mm = prog_mm % 60;
	prog_ss -= 60. * prog_mm + 3600. * prog_hh + 3600.*24. * prog_dd;
	cout << "# Elapsed time: " << prog_dd << " d " << prog_hh << " h " << prog_mm << " m " << prog_ss << " s" << endl;
	
	
	return 0;
}
