#ifndef _PROGRAM_OPTS_H__
#define _PROGRAM_OPTS_H__


#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>

#include <boost/program_options.hpp>

#include "model.h"
#include "bayestar_config.h"
#include "los_sampler.h"


using namespace std;


struct TProgramOpts {
    string input_fname;
    string output_fname;

    bool save_surfs;
    bool save_gridstars;
    bool load_surfs;

    double err_floor;  // in millimags

    bool synthetic;
    bool sample_stars;
    unsigned int star_steps;
    unsigned int star_samplers;
    double star_p_replacement;
    double min_EBV;    // in mags
    bool star_priors;
    bool use_gaia;

    double sigma_RV;
    double mean_RV;

    //double smoothing_slope;
    double smoothing_alpha_coeff[2];
    double smoothing_beta_coeff[2];
    double pct_smoothing_min;
    double pct_smoothing_max;

    bool discrete_los;
    unsigned int discrete_steps;

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
    double sigma_log_Delta_EBV;

    bool SFD_prior;
    bool SFD_subpixel;
    double subpixel_max;
    double ev_cut;
    double chi2_cut;

    unsigned int N_runs;
    unsigned int N_threads;

    bool clobber;

    bool test_mode;

    int verbosity;

    string LF_fname;
    string template_fname;
    string ext_model_fname;

    TGalStructParams gal_struct_params;

    string neighbor_lookup_fname;
    string pixel_lookup_fname;
    string output_fname_pattern;

    double correlation_scale;
    double d_soft;
    double gamma_soft;
    
    TDiscreteLOSSamplingSettings dsc_samp_settings;
    
    std::vector<std::string> force_pix;

    TProgramOpts();
};


int get_program_opts(int argc, char **argv, TProgramOpts &opts);


#endif // _PROGRAM_OPTS_H__
