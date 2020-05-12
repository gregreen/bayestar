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

#include "cpp_utils.h"
#include "model.h"
#include "data.h"
#include "sampler.h"
#include "los_sampler.h"
#include "star_exact.h"
#include "neighbor_pixels.h"
#include "bayestar_config.h"
#include "program_opts.h"

using namespace std;


int full_workflow(TProgramOpts &opts, int argc, char **argv) {
    /*
     * Determines stellar posterior densities,
     * then determines the l.o.s. reddening.
     */


    /*
     *  MCMC Options
     */

    TMCMCOptions star_options(opts.star_steps, opts.star_samplers, opts.star_p_replacement, opts.N_runs);
    TMCMCOptions cloud_options(opts.cloud_steps, opts.cloud_samplers, opts.cloud_p_replacement, opts.N_runs);
    TMCMCOptions los_options(opts.los_steps, opts.los_samplers, opts.los_p_replacement, opts.N_runs);

    TMCMCOptions discrete_los_options(opts.discrete_steps, 1, 0., opts.N_runs);    // TODO: Create commandline options for this


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

        cout << "# Pixel: " << *it
            << " (" << pixel_list_no + 1 << " of " << pix_name.size() << ")"
            << endl;

        // Load input photometry
        TStellarData stellar_data(opts.input_fname, *it, opts.err_floor);
        TGalacticLOSModel los_model(
            stellar_data.l,
            stellar_data.b,
            opts.gal_struct_params
        );

        cout << "# HEALPix index: " << stellar_data.healpix_index
             << " (nside = " << stellar_data.nside << ")" << endl;
        cout << "# (l, b) = "
             << stellar_data.l << ", " << stellar_data.b << endl;
        if(opts.SFD_prior) {
            cout << "# E(B-V)_SFD = " << stellar_data.EBV << endl;
        }
        cout << "# " << stellar_data.star.size() << " stars in pixel" << endl;


        // Check if this pixel has already been fully processed
        if(!(opts.clobber)) {
            bool process_pixel = false;

            std::unique_ptr<H5::H5File> out_file = H5Utils::openFile(
                opts.output_fname,
                H5Utils::READ | H5Utils::WRITE | H5Utils::DONOTCREATE
            );

            if(!out_file) {
                process_pixel = true;

                //cout << "File does not exist" << endl;
            } else {
                //cout << "File exists" << endl;
                //stringstream group_name;
                //group_name << stellar_data.healpix_index;
                //group_name << stellar_data.nside << "-" << stellar_data.healpix_index;

                std::unique_ptr<H5::Group> pix_group = H5Utils::openGroup(
                    *out_file,
                    *it,
                    H5Utils::READ | H5Utils::WRITE | H5Utils::DONOTCREATE
                );

                if(!pix_group) {
                    process_pixel = true;
                } else {
                    //cout << "Group exists" << endl;
                    
                    if(opts.force_pix.size() != 0) {
                        std::stringstream pix_spec_ss;
                        pix_spec_ss << stellar_data.nside
                                    << "-" << stellar_data.healpix_index;
                        std::string pix_spec_str = pix_spec_ss.str();
                        for(auto const &s : opts.force_pix) {
                            if(pix_spec_str == s) {
                                std::cerr << "Force-reprocessing pixel " << s << std::endl;
                                process_pixel = true;
                                break;
                            }
                        }
                    }
                    
                    if(opts.sample_stars) {
                        if(!H5Utils::dataset_exists("stellar chains", *pix_group)) {
                            process_pixel = true;
                        }
                    }
                    if(opts.save_surfs) {
                        if(!H5Utils::dataset_exists("stellar pdfs", *pix_group)) {
                            process_pixel = true;
                        }
                    }

                    if((!process_pixel) && (opts.N_clouds != 0)) {
                        if(!H5Utils::dataset_exists("clouds", *pix_group)) {
                            process_pixel = true;
                        }
                    }

                    if((!process_pixel) && (opts.N_regions != 0)) {
                        if(!H5Utils::dataset_exists("los", *pix_group)) {
                            process_pixel = true;
                        }
                    }
                    
                    if((!process_pixel) && (opts.discrete_los)) {
                        if(!H5Utils::dataset_exists("discrete-los", *pix_group)) {
                            process_pixel = true;
                        }
                    }

                    // If pixel is missing data, remove it, so that it can be regenerated
                    if(process_pixel) {
                        try {
                            out_file->unlink(*it);
                        } catch(H5::FileIException unlink_err) {
                            cout << "Unable to remove group: '" << *it << "'"
                                 << endl;
                        }
                    }
                }
            }

            if(!process_pixel) {
                cout << "# Pixel is already present in output. Skipping."
                     << endl << endl;

                continue; // All information is already present in output file
            }
        }

        // Prepare data structures for stellar parameters
        unsigned int n_stars = stellar_data.star.size();
        std::unique_ptr<TImgStack> img_stack(new TImgStack(n_stars));
        vector<bool> conv;
        vector<double> lnZ;
        vector<double> chi2;

        bool gatherSurfs = (opts.N_regions || opts.N_clouds || opts.save_surfs);

        // Sample individual stars
        if(!opts.sample_stars) {
            // Grid evaluation of stellar models
            grid_eval_stars(los_model, ext_model, *emplib,
                            stellar_data, EBV_smoothing,
                            *img_stack, chi2,
                            opts.save_surfs,
                            opts.save_gridstars,
                            opts.output_fname,
                            opts.star_priors,
                            opts.use_gaia,
                            opts.mean_RV, opts.verbosity);
        } else if(opts.synthetic) {
            // MCMC sampling of synthetic stellar model
            sample_indiv_synth(opts.output_fname, star_options, los_model, *synthlib, ext_model,
                               stellar_data, *img_stack, conv, lnZ, opts.sigma_RV,
                               opts.min_EBV, opts.save_surfs, gatherSurfs, opts.verbosity);
        } else {
            #ifdef _USE_PARALLEL_TEMPERING__
            // MCMC sampling of empirical stellar model
            sample_indiv_emp_pt(opts.output_fname, star_options, los_model,
                                *emplib, ext_model, EBV_smoothing,
                                stellar_data, *img_stack, conv, lnZ,
                                opts.mean_RV, opts.sigma_RV, opts.min_EBV,
                                opts.save_surfs, gatherSurfs, opts.star_priors,
                                opts.verbosity);
            #else // _USE_PARALLEL_TEMPERING
            // MCMC sampling of empirical stellar model
            sample_indiv_emp(opts.output_fname, star_options, los_model,
                             *emplib, ext_model, EBV_smoothing,
                             stellar_data, *img_stack, conv, lnZ,
                             opts.mean_RV, opts.sigma_RV, opts.min_EBV,
                             opts.save_surfs, gatherSurfs, opts.star_priors,
                             opts.verbosity);
            #endif // _USE_PARALLEL_TERMPERING
        }

        clock_gettime(CLOCK_MONOTONIC, &t_mid);

        // Tag output pixel with HEALPix nside and index
        stringstream group_name;
        group_name << "/" << *it;

        try {
            H5Utils::add_watermark<uint32_t>(opts.output_fname, group_name.str(), "nside", stellar_data.nside);
            H5Utils::add_watermark<uint64_t>(opts.output_fname, group_name.str(), "healpix_index", stellar_data.healpix_index);
            H5Utils::add_watermark<double>(opts.output_fname, group_name.str(), "l", stellar_data.l);
            H5Utils::add_watermark<double>(opts.output_fname, group_name.str(), "b", stellar_data.b);
            H5Utils::add_watermark<uint32_t>(opts.output_fname, group_name.str(), "n_stars", stellar_data.star.size());
        } catch(H5::AttributeIException err_att_exists) { }

        // Filter based on goodness-of-fit and convergence
        vector<bool> keep;
        bool filter_tmp;
        size_t n_filtered = 0;

        std::vector<double> subpixel;
        vector<double> lnZ_filtered;

        if(opts.sample_stars) {
            // For sampled stars, use convergence and lnZ
            assert(conv.size() == lnZ.size());
            for(vector<double>::iterator it_lnZ = lnZ.begin(); it_lnZ != lnZ.end(); ++it_lnZ) {
                if(!std::isnan(*it_lnZ) && !is_inf_replacement(*it_lnZ)) {
                    lnZ_filtered.push_back(*it_lnZ);
                }
            }
            double lnZmax = percentile_const(lnZ_filtered, 95.0);
            if(opts.verbosity >= 2) {
                cout << "# ln(Z)_95pct = " << lnZmax << endl;
            }

            lnZ_filtered.clear();
            for(size_t n=0; n<conv.size(); n++) {
                filter_tmp = conv[n]
                             && (lnZ[n] > lnZmax - (25. + opts.ev_cut))
                             && !std::isnan(lnZ[n])
                             && !is_inf_replacement(lnZ[n])
                             && (stellar_data.star[n].EBV < opts.subpixel_max);
                keep.push_back(filter_tmp);
                if(filter_tmp) {
                    subpixel.push_back(stellar_data.star[n].EBV);
                    lnZ_filtered.push_back(lnZ[n] - lnZmax);
                } else {
                    n_filtered++;
                }
            }
        } else {
            // For grid-evaluated stars, use chi^2 / passband
            for(size_t n=0; n<chi2.size(); n++) {
                filter_tmp = (chi2[n] < opts.chi2_cut)
                             && !std::isnan(chi2[n])
                             && !is_inf_replacement(chi2[n])
                             && (stellar_data.star[n].EBV < opts.subpixel_max);
                keep.push_back(filter_tmp);
                if(filter_tmp) {
                    subpixel.push_back(stellar_data.star[n].EBV);
                    lnZ_filtered.push_back(0.); // Dummy value
                } else {
                    n_filtered++;
                }
            }
            // Save # of rejected stars
            try {
                H5Utils::add_watermark<uint32_t>(opts.output_fname, group_name.str(), "n_stars_rejected", n_filtered);
            } catch(H5::AttributeIException err_att_exists) { }
            // Save rejection fraction
            double reject_frac = (double)n_filtered / chi2.size();
            try {
                H5Utils::add_watermark<double>(opts.output_fname, group_name.str(), "reject_frac", reject_frac);
            } catch(H5::AttributeIException err_att_exists) { }
        }
        if(gatherSurfs) { img_stack->cull(keep); }

        cout << "# of stars filtered: "
             << n_filtered << " of " << n_stars;
        cout << " (" << 100. * (double)n_filtered / n_stars
             << " %)" << endl;

        // Fit line-of-sight extinction profile
        if((opts.N_clouds != 0) || (opts.N_regions != 0) || opts.discrete_los) {
            if(n_filtered >= n_stars) {
                cout << "Every star was rejected!" << endl;
            } else {
                double p0 = exp(-5. - opts.ev_cut);
                double EBV_max = -1.;
                if(opts.SFD_prior) {
                    if(opts.SFD_subpixel) {
                        EBV_max = 1.;
                    } else {
                        EBV_max = stellar_data.EBV;
                    }
                }

                TLOSMCMCParams params(
                    img_stack.get(), lnZ_filtered, p0,
                    opts.N_runs, opts.N_threads,
                    opts.N_regions, EBV_max
                );
                if(opts.SFD_subpixel) { params.set_subpixel_mask(subpixel); }

                if(opts.test_mode) {
                    test_extinction_profiles(params);
                }

                // Sample discrete l.o.s. model
                if(opts.discrete_los) {
                    std::unique_ptr<TNeighborPixels> neighbor_pixels;

                    if((opts.neighbor_lookup_fname != "NONE") &&
                       (opts.pixel_lookup_fname != "NONE") &&
                       (opts.output_fname_pattern != "NONE"))
                    {
                        // Load information on neighboring pixels
                        cout << "Loading information on neighboring pixels ..." << endl;
                        neighbor_pixels = std::make_unique<TNeighborPixels>(
                            stellar_data.nside,
                            stellar_data.healpix_index,
                            opts.neighbor_lookup_fname,
                            opts.pixel_lookup_fname,
                            opts.output_fname_pattern,
                            1000);
                        
                        if(!neighbor_pixels->data_loaded()) {
                            cerr << "Failed to load neighboring pixels! Aborting."
                                 << endl;
                            return 1;
                        }
                        
                        // Calculate covariance matrices tying
                        // pixels together at each distance
                        neighbor_pixels->init_covariance(
                            opts.correlation_scale,
                            opts.d_soft,
                            opts.gamma_soft);
                    }
                    
                    TDiscreteLosMcmcParams discrete_los_params(
                        std::move(img_stack),
                        std::move(neighbor_pixels),
                        1, 1,
                        opts.verbosity);
                    discrete_los_params.initialize_priors(
                        los_model,
                        opts.log_Delta_EBV_floor,
                        opts.log_Delta_EBV_ceil,
                        opts.sigma_log_Delta_EBV,
                        opts.verbosity
                    );
                    
                    std::vector<uint16_t> neighbor_sample;

                    if(discrete_los_params.neighbor_pixels) {
                        cout << "Initializing dominant distances ..." << endl;
                        discrete_los_params.neighbor_pixels->init_dominant_dist(opts.verbosity);

                        //cout << "Resampling neighboring pixels ..." << endl;
                        //sample_neighbors(*(discrete_los_params.neighbor_pixels), opts.verbosity);
                        //sample_neighbors_pt(
                        //    *(discrete_los_params.neighbor_pixels),
                        //    neighbor_sample,
                        //    opts.verbosity
                        //);
                    }

                    cout << "Sampling line of sight discretely ..." << endl;
                    sample_los_extinction_discrete(
                        opts.output_fname,
                        *it,
                        discrete_los_options,
                        discrete_los_params,
                        neighbor_sample,
                        opts.dsc_samp_settings,
                        opts.verbosity
                    );
                    cout << "Done with discrete sampling." << endl;
                }

                if(opts.N_clouds != 0) {
                    sample_los_extinction_clouds(
                        opts.output_fname, *it,
                        cloud_options, params,
                        opts.N_clouds, opts.verbosity
                    );
                }
                if(opts.N_regions != 0) {
                    // Covariance matrix for guess has (anti-)correlation
                    // length of 1 distance bin
                    params.gen_guess_covariance(1.);

                    if(opts.disk_prior) {
                        params.alpha_skew = 1.;
                        params.calc_Delta_EBV_prior(
                            los_model,
                            opts.log_Delta_EBV_floor,
                            opts.log_Delta_EBV_ceil,
                            stellar_data.EBV,
                            1.4,
                            opts.verbosity
                        );
                    }

                    sample_los_extinction(
                        opts.output_fname, *it,
                        los_options, params, opts.verbosity
                    );
                }
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &t_end);
        t_tot = (t_end.tv_sec - t_start.tv_sec)
                + 1.e-9 * (t_end.tv_nsec - t_start.tv_nsec);
        t_star = (t_mid.tv_sec - t_start.tv_sec)
                + 1.e-9 * (t_mid.tv_nsec - t_start.tv_nsec);
        
        try {
            H5Utils::add_watermark<float>(opts.output_fname, group_name.str(), "t_tot", (float)t_tot);
            H5Utils::add_watermark<float>(opts.output_fname, group_name.str(), "t_star", (float)t_star);
        } catch(H5::AttributeIException err_att_exists) { }

        if(opts.verbosity >= 1) {
            cout << endl
                 << "==================================================="
                 << endl;
        }
        cout << "# Time elapsed for pixel: "
             << setprecision(2) << t_tot
             << " s (" << setprecision(2)
             << t_tot / (double)(stellar_data.star.size())
             << " s / star)" << endl;
        cout << "# Percentage of time spent on l.o.s. fit: "
             << setprecision(2) << 100. * (t_tot - t_star) / t_tot
             << " %" << endl;
        if(opts.verbosity >= 1) {
            cout << "==================================================="
                 << endl;
        }
        cout << endl;
    }


    /*
     *  Add additional metadata to output file
     */
    try {
        string watermark = GIT_BUILD_VERSION;
        H5Utils::add_watermark<string>(
            opts.output_fname, "/",
            "bayestar git commit",
            watermark
        );
    } catch(H5::AttributeIException err_att_exists) { }

    stringstream commandline_args;
    for(int i=0; i<argc; i++) {
        commandline_args << argv[i] << " ";
    }
    try {
        string commandline_args_str(commandline_args.str());
        H5Utils::add_watermark<string>(
            opts.output_fname, "/",
            "commandline invocation",
            commandline_args_str
        );
    } catch(H5::AttributeIException err_att_exists) { }


    /*
     *  Cleanup
     */

    if(synthlib != NULL) { delete synthlib; }
    if(emplib != NULL) { delete emplib; }
    
    return 0;
}


int los_workflow(TProgramOpts &opts, int argc, char **argv) {
    /*
     * Uses pre-computed stellar posterior densities
     * to determine the l.o.s. reddening.
     */


    /*
     *  MCMC Options
     */

    TMCMCOptions cloud_options(opts.cloud_steps, opts.cloud_samplers, opts.cloud_p_replacement, opts.N_runs);
    TMCMCOptions los_options(opts.los_steps, opts.los_samplers, opts.los_p_replacement, opts.N_runs);

    TMCMCOptions discrete_los_options(opts.discrete_steps, 1, 0., opts.N_runs);    // TODO: Create commandline options for this


    /*
     *  Construct models
     */

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
    get_input_pixels(opts.input_fname, pix_name, "/stellar_pdfs");
    cout << "# " << pix_name.size() << " pixels in input file." << endl << endl;
    
    // Get (l,b), (nside, healpix index), EBV estimate of each pixel
    vector<double> pix_l, pix_b, pix_EBV;
    vector<uint32_t> pix_nside;
    vector<uint64_t> pix_idx;
    get_pixel_props(
        opts.input_fname,
        pix_name,
        pix_l, pix_b, pix_EBV,
        pix_nside, pix_idx,
        "/stellar_pdfs"
    );

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

        cout << "# Pixel: " << *it
            << " (" << pixel_list_no + 1 << " of " << pix_name.size() << ")"
            << endl;

        double l = pix_l.at(pixel_list_no);
        double b = pix_b.at(pixel_list_no);
        double EBV = pix_EBV.at(pixel_list_no);
        uint32_t nside = pix_nside.at(pixel_list_no);
        uint32_t hpidx = pix_idx.at(pixel_list_no);
        
        // Load input photometry
        TGalacticLOSModel los_model(
            l, b,
            opts.gal_struct_params
        );
        
        cout << "# HEALPix index: " << hpidx
             << " (nside = " << nside << ")" << endl;
        cout << "# (l, b) = "
             << l << ", " << b << endl;
        if(opts.SFD_prior) {
            cout << "# E(B-V)_SFD = " << EBV << endl;
        }
        
        // Load surfaces
        std::stringstream dset_name;
        dset_name << "/stellar_pdfs/" << *it << "/stellar_pdfs";
        std::unique_ptr<TImgStack> img_stack = read_img_stack(
            opts.input_fname,
            dset_name.str()
        );
        
        unsigned int n_stars = img_stack->N_images;
        
        cout << "# " << n_stars << " stars in pixel" << endl;

        // Check if this pixel has already been fully processed
        if(!(opts.clobber)) {
            bool process_pixel = false;

            std::unique_ptr<H5::H5File> out_file = H5Utils::openFile(
                opts.output_fname,
                H5Utils::READ | H5Utils::WRITE | H5Utils::DONOTCREATE
            );

            if(!out_file) {
                process_pixel = true;

                //cout << "File does not exist" << endl;
            } else {
                //cout << "File exists" << endl;
                //stringstream group_name;
                //group_name << stellar_data.healpix_index;
                //group_name << stellar_data.nside << "-" << stellar_data.healpix_index;

                std::unique_ptr<H5::Group> pix_group = H5Utils::openGroup(
                    *out_file,
                    *it,
                    H5Utils::READ | H5Utils::WRITE | H5Utils::DONOTCREATE
                );

                if(!pix_group) {
                    process_pixel = true;
                } else {
                    //cout << "Group exists" << endl;
                    
                    if(opts.force_pix.size() != 0) {
                        std::stringstream pix_spec_ss;
                        pix_spec_ss << nside << "-" << hpidx;
                        std::string pix_spec_str = pix_spec_ss.str();
                        for(auto const &s : opts.force_pix) {
                            if(pix_spec_str == s) {
                                std::cerr << "Force-reprocessing pixel " << s << std::endl;
                                process_pixel = true;
                                break;
                            }
                        }
                    }
                    
                    if((!process_pixel) && (opts.discrete_los)) {
                        if(!H5Utils::dataset_exists("discrete-los", *pix_group)) {
                            process_pixel = true;
                        }
                    }

                    // If pixel is missing data, remove it, so that it can be regenerated
                    if(process_pixel) {
                        try {
                            out_file->unlink(*it);
                        } catch(H5::FileIException unlink_err) {
                            cout << "Unable to remove group: '" << *it << "'"
                                 << endl;
                        }
                    }
                }
            }

            if(!process_pixel) {
                cout << "# Pixel is already present in output. Skipping."
                     << endl << endl;

                continue; // All information is already present in output file
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &t_mid);

        // Tag output pixel with HEALPix nside and index
        stringstream group_name;
        group_name << "/" << *it;

        try {
            H5Utils::add_watermark<uint32_t>(opts.output_fname, group_name.str(), "nside", nside);
            H5Utils::add_watermark<uint64_t>(opts.output_fname, group_name.str(), "healpix_index", hpidx);
            H5Utils::add_watermark<double>(opts.output_fname, group_name.str(), "l", l);
            H5Utils::add_watermark<double>(opts.output_fname, group_name.str(), "b", b);
            H5Utils::add_watermark<uint32_t>(opts.output_fname, group_name.str(), "n_stars", n_stars);
        } catch(H5::AttributeIException err_att_exists) { }

        // Sample discrete l.o.s. model
        if(opts.discrete_los) {
            std::unique_ptr<TNeighborPixels> neighbor_pixels;

            if((opts.neighbor_lookup_fname != "NONE") &&
               (opts.pixel_lookup_fname != "NONE") &&
               (opts.output_fname_pattern != "NONE"))
            {
                // Load information on neighboring pixels
                cout << "Loading information on neighboring pixels ..." << endl;
                neighbor_pixels = std::make_unique<TNeighborPixels>(
                    nside,
                    hpidx,
                    opts.neighbor_lookup_fname,
                    opts.pixel_lookup_fname,
                    opts.output_fname_pattern,
                    1000);
                
                if(!neighbor_pixels->data_loaded()) {
                    cerr << "Failed to load neighboring pixels! Aborting."
                         << endl;
                    return 1;
                }
                
                // Calculate covariance matrices tying
                // pixels together at each distance
                neighbor_pixels->init_covariance(
                    opts.correlation_scale,
                    opts.d_soft,
                    opts.gamma_soft);
            }
            
            TDiscreteLosMcmcParams discrete_los_params(
                std::move(img_stack),
                std::move(neighbor_pixels),
                1, 1,
                opts.verbosity
            );
            discrete_los_params.initialize_priors(
                los_model,
                opts.log_Delta_EBV_floor,
                opts.log_Delta_EBV_ceil,
                opts.sigma_log_Delta_EBV,
                opts.verbosity
            );
            
            std::vector<uint16_t> neighbor_sample;

            if(discrete_los_params.neighbor_pixels) {
                cout << "Initializing dominant distances ..." << endl;
                discrete_los_params.neighbor_pixels->init_dominant_dist(opts.verbosity);

                //cout << "Resampling neighboring pixels ..." << endl;
                //sample_neighbors(*(discrete_los_params.neighbor_pixels), opts.verbosity);
                //sample_neighbors_pt(
                //    *(discrete_los_params.neighbor_pixels),
                //    neighbor_sample,
                //    opts.verbosity
                //);
            }

            cout << "Sampling line of sight discretely ..." << endl;
            sample_los_extinction_discrete(
                opts.output_fname,
                *it,
                discrete_los_options,
                discrete_los_params,
                neighbor_sample,
                opts.dsc_samp_settings,
                opts.verbosity
            );
            cout << "Done with discrete sampling." << endl;
        }

        clock_gettime(CLOCK_MONOTONIC, &t_end);
        t_tot = (t_end.tv_sec - t_start.tv_sec)
                + 1.e-9 * (t_end.tv_nsec - t_start.tv_nsec);
        t_star = (t_mid.tv_sec - t_start.tv_sec)
                + 1.e-9 * (t_mid.tv_nsec - t_start.tv_nsec);
        
        try {
            H5Utils::add_watermark<float>(opts.output_fname, group_name.str(), "t_tot", (float)t_tot);
            H5Utils::add_watermark<float>(opts.output_fname, group_name.str(), "t_star", (float)t_star);
        } catch(H5::AttributeIException err_att_exists) { }

        if(opts.verbosity >= 1) {
            cout << endl
                 << "==================================================="
                 << endl;
        }
        cout << "# Time elapsed for pixel: "
             << setprecision(2) << t_tot
             << " s (" << setprecision(2)
             << t_tot / (double)(n_stars)
             << " s / star)" << endl;
        cout << "# Percentage of time spent on l.o.s. fit: "
             << setprecision(2) << 100. * (t_tot - t_star) / t_tot
             << " %" << endl;
        if(opts.verbosity >= 1) {
            cout << "==================================================="
                 << endl;
        }
        cout << endl;
    }

    /*
     *  Add additional metadata to output file
     */
    try {
        string watermark = GIT_BUILD_VERSION;
        H5Utils::add_watermark<string>(
            opts.output_fname, "/",
            "bayestar git commit",
            watermark
        );
    } catch(H5::AttributeIException err_att_exists) { }

    stringstream commandline_args;
    for(int i=0; i<argc; i++) {
        commandline_args << argv[i] << " ";
    }
    try {
        string commandline_args_str(commandline_args.str());
        H5Utils::add_watermark<string>(
            opts.output_fname, "/",
            "commandline invocation",
            commandline_args_str
        );
    } catch(H5::AttributeIException err_att_exists) { }

    return 0;
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
    
    int res;
    if(opts.load_surfs) {
        // Use pre-computed stellar posterior densities
        // to determine l.o.s. reddening
        res = los_workflow(opts, argc, argv);
    } else {
        // Determine stellar posterior densities,
        // then determine l.o.s. reddening
        res = full_workflow(opts, argc, argv);
    }

    tmp_time = time(0);
    dt = ctime(&tmp_time);
    cout << "# End time: " << dt;

    timespec prog_end_time;
    clock_gettime(CLOCK_MONOTONIC, &prog_end_time);
    double prog_ss = prog_end_time.tv_sec - prog_start_time.tv_sec
                    + 1.e-9 * (prog_end_time.tv_nsec - prog_start_time.tv_nsec);
    int prog_mm = floor(prog_ss / 60.);
    int prog_hh = floor(prog_mm / 60.);
    int prog_dd = floor(prog_hh / 24.);
    prog_hh = prog_hh % 24;
    prog_mm = prog_mm % 60;
    prog_ss -= 60. * prog_mm + 3600. * prog_hh + 3600.*24. * prog_dd;
    cout << "# Elapsed time: "
         << prog_dd << " d "
         << prog_hh << " h "
         << prog_mm << " m "
         << prog_ss << " s" << endl;

    return res;
}
