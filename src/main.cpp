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
#include "star_exact.h"
#include "bayestar_config.h"

using namespace std;


struct TProgramOpts {
	string input_fname;
	string output_fname;

	bool save_surfs;

	double err_floor;

	bool synthetic;
	unsigned int star_steps;
	unsigned int star_samplers;
	double star_p_replacement;
	double min_EBV;
	bool star_priors;

	double sigma_RV;
	double mean_RV;

	//double smoothing_slope;
	double smoothing_alpha_coeff[2];
	double smoothing_beta_coeff[2];
	double pct_smoothing_min;
	double pct_smoothing_max;

	unsigned int N_regions;
	unsigned int los_steps;
	unsigned int los_samplers;
	double los_p_replacement;

	unsigned int N_clouds;
	unsigned int cloud_steps;
	unsigned int cloud_samplers;
	double cloud_p_replacement;

	bool disk_prior;
	double log_Delta_EBV_floor;
	double log_Delta_EBV_ceil;

	bool SFD_prior;
	bool SFD_subpixel;
	double subpixel_max;
	double ev_cut;

	unsigned int N_runs;
	unsigned int N_threads;

	bool clobber;

	bool test_mode;

	int verbosity;

	string LF_fname;
	string template_fname;
	string ext_model_fname;

	TGalStructParams gal_struct_params;

	TProgramOpts() {
		input_fname = "NONE";
		output_fname = "NONE";

		save_surfs = false;

		err_floor = 20;

		synthetic = false;
		star_steps = 1000;
		star_samplers = 5;
		star_p_replacement = 0.2;
		min_EBV = 0.;
		star_priors = true;

		sigma_RV = -1.;
		mean_RV = 3.1;

		//smoothing_slope = 0.05;
		smoothing_alpha_coeff[0] = 0.880;
		smoothing_alpha_coeff[1] = -2.963;
		smoothing_beta_coeff[0] = 0.578;
		smoothing_beta_coeff[1] = -1.879;
		pct_smoothing_min = 0.;
		pct_smoothing_max = -1.;

		N_regions = 30;
		los_steps = 4000;
		los_samplers = 2;
		los_p_replacement = 0.0;

		N_clouds = 1;
		cloud_steps = 2000;
		cloud_samplers = 80;
		cloud_p_replacement = 0.2;

		disk_prior = false;
		log_Delta_EBV_floor = -10.;
		log_Delta_EBV_ceil = -3.;

		SFD_prior = false;
		SFD_subpixel = false;
		subpixel_max = 1.e9;
		ev_cut = 10.;

		N_runs = 4;
		N_threads = 1;

		clobber = false;

		test_mode = false;

		verbosity = 0;

		LF_fname = DATADIR "PSMrLF.dat";
		template_fname = DATADIR "PS1_2MASS_colors.dat";
		ext_model_fname = DATADIR "PS1_2MASS_Extinction.dat";
	}
};


template<typename T>
string to_string(const T& x) {
	stringstream ss;
	ss << x;
	return ss.str();
}


int get_program_opts(int argc, char **argv, TProgramOpts &opts) {
	namespace po = boost::program_options;

	std::string config_fname = "NONE";

	po::options_description config_desc("Configuration-file options");
	config_desc.add_options()
		("err-floor", po::value<double>(&(opts.err_floor)), ("Error to add in quadrature (in millimags) (default: " + to_string(opts.err_floor) + ")").c_str())
		("synthetic", "Use synthetic photometric library (default: use empirical library)")
		("star-steps", po::value<unsigned int>(&(opts.star_steps)), ("# of MCMC steps per star (per sampler) (default: " + to_string(opts.star_steps) + ")").c_str())
		("star-samplers", po::value<unsigned int>(&(opts.star_samplers)), ("# of samplers per dimension (stellar fit) (default: " + to_string(opts.star_samplers) + ")").c_str())
		("star-p-replacement", po::value<double>(&(opts.star_p_replacement)), ("Probability of taking replacement step (stellar fit) (default: " + to_string(opts.star_p_replacement) + ")").c_str())
		("no-stellar-priors", "Turn off priors for individual stars.")
		("min-EBV", po::value<double>(&(opts.min_EBV)), ("Minimum stellar E(B-V) (default: " + to_string(opts.min_EBV) + ")").c_str())

		("mean-RV", po::value<double>(&(opts.mean_RV)), ("Mean R_V (per star) (default: " + to_string(opts.mean_RV) + ")").c_str())
		("sigma-RV", po::value<double>(&(opts.sigma_RV)), ("Variation in R_V (per star) (default: " + to_string(opts.sigma_RV) + ", interpreted as no variance)").c_str())

		//("smoothing-pct", po::value<double>(&(opts.smoothing_slope)), ("Degree of smoothing (sigma/EBV) of per-star surfaces (default: " + to_string(opts.smoothing_slope) + ")").c_str())
		("pct-smoothing-coeffs", po::value< vector<double> >()->multitoken(), ("Coefficients for alpha and beta smoothing parameters (default: " + to_string(opts.smoothing_alpha_coeff[0]) + " "
		                                                                                                                                         + to_string(opts.smoothing_alpha_coeff[1]) + " "
		                                                                                                                                         + to_string(opts.smoothing_beta_coeff[0]) + " "
		                                                                                                                                         + to_string(opts.smoothing_beta_coeff[1]) + ")").c_str())
		("pct-smoothing-min", po::value<double>(&(opts.pct_smoothing_min)), ("Minimum smoothing percent of per-star surfaces (default: " + to_string(opts.pct_smoothing_min) + ")").c_str())
		("pct-smoothing-max", po::value<double>(&(opts.pct_smoothing_max)), ("Maximum smoothing percent of per-star surfaces (default: " + to_string(opts.pct_smoothing_max) + ")").c_str())

		("regions", po::value<unsigned int>(&(opts.N_regions)), ("# of piecewise-linear regions in l.o.s. extinction profile (default: " + to_string(opts.N_regions) + ")").c_str())
		("los-steps", po::value<unsigned int>(&(opts.los_steps)), ("# of MCMC steps in l.o.s. fit (per sampler) (default: " + to_string(opts.los_steps) + ")").c_str())
		("los-samplers", po::value<unsigned int>(&(opts.los_samplers)), ("# of samplers per dimension (l.o.s. fit) (default: " + to_string(opts.los_samplers) + ")").c_str())
		("los-p-replacement", po::value<double>(&(opts.los_p_replacement)), ("Probability of taking replacement step (l.o.s. fit) (default: " + to_string(opts.los_p_replacement) + ")").c_str())

		("clouds", po::value<unsigned int>(&(opts.N_clouds)), ("# of clouds along the line of sight (default: " + to_string(opts.N_clouds) + ")\n"
		                                                       "Setting this option causes the sampler to also fit a discrete "
		                                                       "cloud model of the l.o.s. extinction profile.").c_str())
		("cloud-steps", po::value<unsigned int>(&(opts.cloud_steps)), ("# of MCMC steps in cloud fit (per sampler) (default: " + to_string(opts.cloud_steps) + ")").c_str())
		("cloud-samplers", po::value<unsigned int>(&(opts.cloud_samplers)), ("# of samplers per dimension (cloud fit) (default: " + to_string(opts.cloud_samplers) + ")").c_str())
		("cloud-p-replacement", po::value<double>(&(opts.cloud_p_replacement)), ("Probability of taking replacement step (cloud fit) (default: " + to_string(opts.cloud_p_replacement) + ")").c_str())

		("disk-prior", "Assume that dust density roughly traces stellar disk density.")
		("log-Delta-EBV-min", po::value<double>(&(opts.log_Delta_EBV_floor)), ("Minimum log(Delta EBV) in l.o.s. reddening prior (default: " + to_string(opts.log_Delta_EBV_floor) + ")").c_str())
		("log-Delta-EBV-max", po::value<double>(&(opts.log_Delta_EBV_ceil)), ("Maximum log(Delta EBV) in l.o.s. reddening prior (default: " + to_string(opts.log_Delta_EBV_floor) + ")").c_str())

		("SFD-prior", "Use SFD E(B-V) as a prior on the total extinction in each pixel.")
		("SFD-subpixel", "Use SFD E(B-V) as a subpixel template for the angular variation in reddening.")
		("subpixel-max", po::value<double>(&(opts.subpixel_max)), ("Maximum subpixel value (above this values, stars will be filtered out). (default: " + to_string(opts.subpixel_max) + ")").c_str())
		("evidence-cut", po::value<double>(&(opts.ev_cut)), ("Delta lnZ to use as threshold for including star "
		                                                    "in l.o.s. fit (default: " + to_string(opts.ev_cut) + ")").c_str())

		("runs", po::value<unsigned int>(&(opts.N_runs)), ("# of times to run each chain (to check\n"
		                                                  "for non-convergence) (default: " + to_string(opts.N_runs) + ")").c_str())

		("LF-file", po::value<string>(&(opts.LF_fname)), "File containing stellar luminosity function.")
		("template-file", po::value<string>(&(opts.template_fname)), "File containing stellar color templates.")
		("ext-file", po::value<string>(&(opts.ext_model_fname)), "File containing extinction coefficients.")
	;

	po::options_description gal_desc("Galactic Structural Parameters (all distances in pc)");
	gal_desc.add_options()
		("R0", po::value<double>(&(opts.gal_struct_params.R0)), ("Solar Galactocentric distance (default: " + to_string(opts.gal_struct_params.R0) + ")").c_str())
		("Z0", po::value<double>(&(opts.gal_struct_params.Z0)), ("Solar height above Galactic midplane (default: " + to_string(opts.gal_struct_params.Z0) + ")").c_str())

		("H_thin", po::value<double>(&(opts.gal_struct_params.H1)), ("Thin-disk scale height (default: " + to_string(opts.gal_struct_params.H1) + ")").c_str())
		("L_thin", po::value<double>(&(opts.gal_struct_params.L1)), ("Thin-disk scale length (default: " + to_string(opts.gal_struct_params.L1) + ")").c_str())

		("f_thick", po::value<double>(&(opts.gal_struct_params.f_thick)), ("Thick-disk fraction, defined locally (default: " + to_string(opts.gal_struct_params.f_thick) + ")").c_str())
		("H_thick", po::value<double>(&(opts.gal_struct_params.H2)), ("Thick-disk scale height (default: " + to_string(opts.gal_struct_params.H2) + ")").c_str())
		("L_thick", po::value<double>(&(opts.gal_struct_params.L2)), ("Thin-disk scale length (default: " + to_string(opts.gal_struct_params.L2) + ")").c_str())

		("L_epsilon", po::value<double>(&(opts.gal_struct_params.L_epsilon)), ("Disk softening scale (default: " + to_string(opts.gal_struct_params.L_epsilon) + ")").c_str())

		("f_halo", po::value<double>(&(opts.gal_struct_params.fh)), ("Halo fraction, defined locally (default: " + to_string(opts.gal_struct_params.fh) + ")").c_str())
		("q_halo", po::value<double>(&(opts.gal_struct_params.qh)), ("Halo flattening parameter (default: " + to_string(opts.gal_struct_params.qh) + ")").c_str())
		("n_halo", po::value<double>(&(opts.gal_struct_params.nh)), ("Halo density slope (default: " + to_string(opts.gal_struct_params.nh) + ")").c_str())
		("R_break", po::value<double>(&(opts.gal_struct_params.R_br)), ("Halo break radius (default: " + to_string(opts.gal_struct_params.R_br) + ")").c_str())
		("n_halo_outer", po::value<double>(&(opts.gal_struct_params.nh_outer)), ("Halo outer density slope, past break (default: " + to_string(opts.gal_struct_params.nh_outer) + ")").c_str())

		("H_ISM", po::value<double>(&(opts.gal_struct_params.H_ISM)), ("Dust scale height (default: " + to_string(opts.gal_struct_params.H_ISM) + ")").c_str())
		("L_ISM", po::value<double>(&(opts.gal_struct_params.L_ISM)), ("Dust scale length (default: " + to_string(opts.gal_struct_params.L_ISM) + ")").c_str())
		("dH_dR_ISM", po::value<double>(&(opts.gal_struct_params.dH_dR_ISM)), ("Dust flare slope (default: " + to_string(opts.gal_struct_params.dH_dR_ISM) + ")").c_str())
		("R_flair_ISM", po::value<double>(&(opts.gal_struct_params.R_flair_ISM)), ("Dust flair slope (default: " + to_string(opts.gal_struct_params.R_flair_ISM) + ")").c_str())

		("mu_FeH_inf", po::value<double>(&(opts.gal_struct_params.mu_FeH_inf)), ("Disk metallicity at large elevation above midplane (default: " + to_string(opts.gal_struct_params.mu_FeH_inf) + ")").c_str())
		("delta_mu_FeH", po::value<double>(&(opts.gal_struct_params.delta_mu_FeH)), ("Disk metallicity at midplane, minus metallicity at large elevation (default: " + to_string(opts.gal_struct_params.delta_mu_FeH) + ")").c_str())
		("H_mu_FeH", po::value<double>(&(opts.gal_struct_params.H_mu_FeH)), ("Disk metallicity scale height (default: " + to_string(opts.gal_struct_params.H_mu_FeH) + ")").c_str())
	;
	config_desc.add(gal_desc);

	po::options_description generic_desc(std::string("Usage: ") + argv[0] + " [Input filename] [Output filename] \n\nCommandline Options");
	generic_desc.add_options()
		("help", "Display this help message")
		("show-config", "Display configuration-file options")
		("version", "Display version number")

		("input", po::value<std::string>(&(opts.input_fname)), "Input HDF5 filename (contains stellar photometry)")
		("output", po::value<std::string>(&(opts.output_fname)), "Output HDF5 filename (MCMC output and smoothed probability surfaces)")

		("config", po::value<std::string>(&config_fname), "Configuration file containing additional options.")

		("test-los", "Allow user to test specific line-of-sight profiles manually.")
	;

	po::options_description dual_desc("Dual Options (both commandline and configuration file)");
	dual_desc.add_options()
		("save-surfs", "Save probability surfaces.")
		("clobber", "Overwrite existing output. Otherwise, will\n"
		            "only process pixels with incomplete output.")
		("verbosity", po::value<int>(&(opts.verbosity)), ("Level of verbosity (0 = minimal, 2 = highest) (default: " + to_string(opts.verbosity) + ")").c_str())
		("threads", po::value<unsigned int>(&(opts.N_threads)), ("# of threads to run on (default: " + to_string(opts.N_threads) + ")").c_str())
	;

	po::positional_options_description pd;
	pd.add("input", 1).add("output", 1);


	// Agglomerate different categories of options
	po::options_description cmdline_desc;
	cmdline_desc.add(generic_desc).add(dual_desc);

	po::options_description config_all_desc;
	config_all_desc.add(config_desc).add(dual_desc);


	// Parse options

	po::variables_map vm;
	po::store(po::command_line_parser(argc, argv).options(cmdline_desc).positional(pd).run(), vm);
	po::notify(vm);

	if(config_fname != "NONE") {
		std::ifstream f_config(config_fname.c_str());
		if(!f_config) {
			cerr << "Could not open " << config_fname << endl;
			cerr << "Quitting." << endl;
			return -1;
		}
		po::store(po::parse_config_file(f_config, config_all_desc, false), vm);
		f_config.close();

		po::notify(vm);
	}

	if(vm.count("help")) {
		cout << cmdline_desc << endl;
		return 0;
	}

	if(vm.count("show-config")) {
		cout << config_all_desc << endl;
		return 0;
	}

	if(vm.count("version")) {
		cout << "git commit " << GIT_BUILD_VERSION << endl;
		return 0;
	}

	if(vm.count("synthetic")) { opts.synthetic = true; }
	if(vm.count("save-surfs")) { opts.save_surfs = true; }
	if(vm.count("no-stellar-priors")) { opts.star_priors = false; }
	if(vm.count("disk-prior")) { opts.disk_prior = true; }
	if(vm.count("SFD-prior")) { opts.SFD_prior = true; }
	if(vm.count("SFD-subpixel")) { opts.SFD_subpixel = true; }
	if(vm.count("clobber")) { opts.clobber = true; }
	if(vm.count("test-los")) { opts.test_mode = true; }

	// Read percent smoothing coefficients
	vector<double> pct_smoothing_coeffs;
	if(!vm["pct-smoothing-coeffs"].empty()) {
		if((pct_smoothing_coeffs = vm["pct-smoothing-coeffs"].as< vector<double> >()).size() != 4) {
			cerr << "'pct-smoothing-coeffs' takes exactly 4 numbers." << endl;
			return -1;
		}

		opts.smoothing_alpha_coeff[0] = pct_smoothing_coeffs[0];
		opts.smoothing_alpha_coeff[1] = pct_smoothing_coeffs[1];
		opts.smoothing_beta_coeff[0] = pct_smoothing_coeffs[2];
		opts.smoothing_beta_coeff[1] = pct_smoothing_coeffs[3];
	}


	// Convert error floor to mags
	opts.err_floor /= 1000.;

	if(opts.input_fname == "NONE") {
		cerr << "Input filename required." << endl << endl;
		cerr << cmdline_desc << endl;
		return -1;
	}
	if(opts.output_fname == "NONE") {
		cerr << "Output filename required." << endl << endl;
		cerr << cmdline_desc << endl;
		return -1;
	}

	if(opts.N_regions != 0) {
		if(120 % (opts.N_regions) != 0) {
			cerr << "# of regions in extinction profile must divide 120 without remainder." << endl;
			return -1;
		}
	}

	return 1;
}


int main(int argc, char **argv) {
	gsl_set_error_handler_off();

	/*
	 *  Parse commandline arguments
	 */

	TProgramOpts opts;
	int parse_res = get_program_opts(argc, argv, opts);
	if(parse_res <= 0) { return parse_res; }

	time_t tmp_time = time(0);
	char * dt = ctime(&tmp_time);
	cout << "# Start time: " << dt;

	timespec prog_start_time;
	clock_gettime(CLOCK_MONOTONIC, &prog_start_time);


	/*
	 *  MCMC Options
	 */

	TMCMCOptions star_options(opts.star_steps, opts.star_samplers, opts.star_p_replacement, opts.N_runs);
	TMCMCOptions cloud_options(opts.cloud_steps, opts.cloud_samplers, opts.cloud_p_replacement, opts.N_runs);
	TMCMCOptions los_options(opts.los_steps, opts.los_samplers, opts.los_p_replacement, opts.N_runs);


	/*
	 *  Construct models
	 */

	TStellarModel *emplib = NULL;
	TSyntheticStellarModel *synthlib = NULL;
	if(opts.synthetic) {
		synthlib = new TSyntheticStellarModel(DATADIR "PS1templates.h5");
	} else {
		emplib = new TStellarModel(opts.LF_fname, opts.template_fname);
	}
	TExtinctionModel ext_model(opts.ext_model_fname);

	TEBVSmoothing EBV_smoothing(opts.smoothing_alpha_coeff,
	                            opts.smoothing_beta_coeff,
	                            opts.pct_smoothing_min,
	                            opts.pct_smoothing_max);

	/*
	 *  Execute
	 */

	omp_set_num_threads(opts.N_threads);

	// Get list of pixels in input file
	vector<string> pix_name;
	get_input_pixels(opts.input_fname, pix_name);
	cout << "# " << pix_name.size() << " pixels in input file." << endl << endl;

	// Remove the output file
	if(opts.clobber) {
		remove(opts.output_fname.c_str());
	}

	H5::Exception::dontPrint();

	// Run each pixel
	timespec t_start, t_mid, t_end;

	double t_tot, t_star;
	unsigned int pixel_list_no = 0;

	for(vector<string>::iterator it = pix_name.begin(); it != pix_name.end(); ++it, pixel_list_no++) {
		clock_gettime(CLOCK_MONOTONIC, &t_start);

		cout << "# Pixel: " << *it << " (" << pixel_list_no + 1 << " of " << pix_name.size() << ")" << endl;

		TStellarData stellar_data(opts.input_fname, *it, opts.err_floor);
		TGalacticLOSModel los_model(stellar_data.l, stellar_data.b, opts.gal_struct_params);

		cout << "# HEALPix index: " << stellar_data.healpix_index << " (nside = " << stellar_data.nside << ")" << endl;
		cout << "# (l, b) = " << stellar_data.l << ", " << stellar_data.b << endl;
		if(opts.SFD_prior) { cout << "# E(B-V)_SFD = " << stellar_data.EBV << endl; }
		cout << "# " << stellar_data.star.size() << " stars in pixel" << endl;


		// Check if this pixel has already been fully processed
		if(!(opts.clobber)) {
			bool process_pixel = false;

			H5::H5File *out_file = H5Utils::openFile(opts.output_fname, H5Utils::READ | H5Utils::WRITE | H5Utils::DONOTCREATE);

			if(out_file == NULL) {
				process_pixel = true;

				//cout << "File does not exist" << endl;
			} else {
				//cout << "File exists" << endl;
				//stringstream group_name;
				//group_name << stellar_data.healpix_index;
				//group_name << stellar_data.nside << "-" << stellar_data.healpix_index;

				H5::Group *pix_group = H5Utils::openGroup(out_file, *it, H5Utils::READ | H5Utils::WRITE | H5Utils::DONOTCREATE);

				if(pix_group == NULL) {
					process_pixel = true;
				} else {
					//cout << "Group exists" << endl;

					if(!H5Utils::dataset_exists("stellar chains", pix_group)) {
						process_pixel = true;
					} else {
						if(opts.save_surfs) {
							if(!H5Utils::dataset_exists("stellar pdfs", pix_group)) {
								process_pixel = true;
							}
						}

						if(!process_pixel) {
							if(opts.N_clouds != 0) {
								if(!H5Utils::dataset_exists("clouds", pix_group)) {
									process_pixel = true;
								}
							}
						}

						if(!process_pixel) {
							if(opts.N_regions != 0) {
								if(!H5Utils::dataset_exists("los", pix_group)) {
									process_pixel = true;
								}
							}
						}
					}

					delete pix_group;

					// If pixel is missing data, remove it, so that it can be regenerated
					if(process_pixel) {
						try {
							out_file->unlink(*it);
						} catch(H5::FileIException unlink_err) {
							cout << "Unable to remove group: '" << *it << "'" << endl;
						}
					}
				}

				delete out_file;
			}

			if(!process_pixel) {
				cout << "# Pixel is already present in output. Skipping." << endl << endl;

				continue;	// All information is already present in output file
			}
		}

		// Prepare data structures for stellar parameters
		TImgStack img_stack(stellar_data.star.size());
		vector<bool> conv;
		vector<double> lnZ;

		bool gatherSurfs = (opts.N_regions || opts.N_clouds || opts.save_surfs);

		// Grid evaluation of stellar models
		grid_eval_stars(los_model, ext_model, *emplib, stellar_data, img_stack,
						opts.save_surfs, opts.output_fname, opts.mean_RV);

		// Sample individual stars
		if(opts.synthetic) {
			sample_indiv_synth(opts.output_fname, star_options, los_model, *synthlib, ext_model,
			                   stellar_data, img_stack, conv, lnZ, opts.sigma_RV,
			                   opts.min_EBV, opts.save_surfs, gatherSurfs, opts.verbosity);
		} else {
			sample_indiv_emp_pt(opts.output_fname, star_options, los_model, *emplib, ext_model, EBV_smoothing,
			                    stellar_data, img_stack, conv, lnZ, opts.mean_RV, opts.sigma_RV, opts.min_EBV,
			                    opts.save_surfs, gatherSurfs, opts.star_priors, opts.verbosity);
		}

		clock_gettime(CLOCK_MONOTONIC, &t_mid);

		// Tag output pixel with HEALPix nside and index
		stringstream group_name;
		group_name << "/" << *it;

		try {
			H5Utils::add_watermark<uint32_t>(opts.output_fname, group_name.str(), "nside", stellar_data.nside);
			H5Utils::add_watermark<uint64_t>(opts.output_fname, group_name.str(), "healpix_index", stellar_data.healpix_index);
		} catch(H5::AttributeIException err_att_exists) { }

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
		if(opts.verbosity >= 2) { cout << "# ln(Z)_95pct = " << lnZmax << endl; }

		bool tmpFilter;
		size_t nFiltered = 0;
		std::vector<double> subpixel;
		lnZ_filtered.clear();
		for(size_t n=0; n<conv.size(); n++) {
			tmpFilter = conv[n] && (lnZ[n] > lnZmax - (25. + opts.ev_cut)) && !isnan(lnZ[n]) && !is_inf_replacement(lnZ[n]) && (stellar_data.star[n].EBV < opts.subpixel_max);
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
		if((nFiltered < conv.size()) && ((opts.N_clouds != 0) || (opts.N_regions != 0))) {
			cout << "# of stars filtered: " << nFiltered << " of " << conv.size();
			cout << " (" << 100. * (double)nFiltered / (double)(conv.size()) << " %)" << endl;

			double p0 = exp(-5. - opts.ev_cut);
			double EBV_max = -1.;
			if(opts.SFD_prior) {
				if(opts.SFD_subpixel) {
					EBV_max = 1.;
				} else {
					EBV_max = stellar_data.EBV;
				}
			}
			TLOSMCMCParams params(&img_stack, lnZ_filtered, p0, opts.N_runs, opts.N_threads, opts.N_regions, EBV_max);
			if(opts.SFD_subpixel) { params.set_subpixel_mask(subpixel); }

			if(opts.test_mode) {
				test_extinction_profiles(params);
			}

			if(opts.N_clouds != 0) {
				sample_los_extinction_clouds(opts.output_fname, *it, cloud_options, params, opts.N_clouds, opts.verbosity);
			}
			if(opts.N_regions != 0) {
				params.gen_guess_covariance(1.);	// Covariance matrix for guess has (anti-)correlation length of 1 distance bin
				if(opts.disk_prior) {
					params.calc_Delta_EBV_prior(los_model, opts.log_Delta_EBV_floor,
					                            opts.log_Delta_EBV_ceil,
					                            stellar_data.EBV, opts.verbosity);
				}
				sample_los_extinction(opts.output_fname, *it, los_options, params, opts.verbosity);
			}
		}

		clock_gettime(CLOCK_MONOTONIC, &t_end);
		t_tot = (t_end.tv_sec - t_start.tv_sec) + 1.e-9 * (t_end.tv_nsec - t_start.tv_nsec);
		t_star = (t_mid.tv_sec - t_start.tv_sec) + 1.e-9 * (t_mid.tv_nsec - t_start.tv_nsec);

		if(opts.verbosity >= 1) {
			cout << endl;
			cout << "===================================================" << endl;
		}
		cout << "# Time elapsed for pixel: ";
		cout << setprecision(2) << t_tot;
		cout << " s (" << setprecision(2) << t_tot / (double)(stellar_data.star.size()) << " s / star)" << endl;
		cout << "# Percentage of time spent on l.o.s. fit: ";
		cout << setprecision(2) << 100. * (t_tot - t_star) / t_tot << " %" << endl;
		if(opts.verbosity >= 1) {
			cout << "===================================================" << endl;
		}
		cout << endl;
	}


	/*
	 *  Add additional metadata to output file
	 */
	try {
		string watermark = GIT_BUILD_VERSION;
		H5Utils::add_watermark<string>(opts.output_fname, "/", "bayestar git commit", watermark);
	} catch(H5::AttributeIException err_att_exists) { }

	stringstream commandline_args;
	for(int i=0; i<argc; i++) {
		commandline_args << argv[i] << " ";
	}
	try {
		string commandline_args_str(commandline_args.str());
		H5Utils::add_watermark<string>(opts.output_fname, "/", "commandline invocation", commandline_args_str);
	} catch(H5::AttributeIException err_att_exists) { }


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
