
#include "program_opts.h"


template<typename T>
string to_string(const T& x) {
    stringstream ss;
    ss << x;
    return ss.str();
}


TProgramOpts::TProgramOpts() {
    input_fname = "NONE";
    output_fname = "NONE";

    save_surfs = false;
    save_gridstars = false;
    load_surfs = false;

    err_floor = 20;

    synthetic = false;
    sample_stars = false;
    star_steps = 1000;
    star_samplers = 5;
    star_p_replacement = 0.2;
    min_EBV = 0.;
    star_priors = true;
    use_gaia = false;

    sigma_RV = -1.;
    mean_RV = 3.1;

    //smoothing_slope = 0.05;
    smoothing_alpha_coeff[0] = 0.880;
    smoothing_alpha_coeff[1] = -2.963;
    smoothing_beta_coeff[0] = 0.578;
    smoothing_beta_coeff[1] = -1.879;
    pct_smoothing_min = 0.;
    pct_smoothing_max = -1.;

    discrete_los = false;
    discrete_steps = 10000;

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
    sigma_log_Delta_EBV = 0.75;

    SFD_prior = false;
    SFD_subpixel = false;
    subpixel_max = 1.e9;
    ev_cut = 10.;
    chi2_cut = 5.;

    N_runs = 4;
    N_threads = 1;

    clobber = false;

    test_mode = false;

    verbosity = 0;

    LF_fname = DATADIR "PSMrLF.dat";
    template_fname = DATADIR "PS1_2MASS_colors.dat";
    ext_model_fname = DATADIR "PS1_2MASS_Extinction.dat";
    
    neighbor_lookup_fname = "NONE";
    pixel_lookup_fname = "NONE";
    output_fname_pattern = "NONE";

    correlation_scale = 1.0; // in pc
    d_soft = 0.25; // in pc
    gamma_soft = 4.0; // unitless
}


int get_program_opts(int argc, char **argv, TProgramOpts &opts) {
    namespace po = boost::program_options;

    std::string config_fname = "NONE";

    po::options_description config_desc("Configuration-file options");
    config_desc.add_options()
        ("err-floor",
            po::value<double>(&(opts.err_floor)),
            ("Error to add in quadrature (in millimags) (default: " +
                to_string(opts.err_floor) + ")").c_str())

        ("synthetic",
            "Use synthetic photometric library "
                "(default: use empirical library)")
        ("sample-stars",
            "Use MCMC to calculate individual stellar posteriors, "
                "rather than approximate grid evaluation.")
        ("star-steps",
            po::value<unsigned int>(&(opts.star_steps)),
            ("# of MCMC steps per star (per sampler) (default: " +
                to_string(opts.star_steps) + ")").c_str())
        ("star-samplers",
            po::value<unsigned int>(&(opts.star_samplers)),
            ("# of samplers per dimension (stellar fit) (default: " +
                to_string(opts.star_samplers) + ")").c_str())
        ("star-p-replacement",
            po::value<double>(&(opts.star_p_replacement)),
            ("Probability of taking replacement step (stellar fit) "
                "(default: " +
                to_string(opts.star_p_replacement) + ")").c_str())
        ("no-stellar-priors",
            "Turn off priors for individual stars.")
        ("use-gaia",
            "Use gaia parallax likelihood for individual stars.")
        ("min-EBV",
            po::value<double>(&(opts.min_EBV)),
            ("Minimum stellar E(B-V) (default: " +
                to_string(opts.min_EBV) + ")").c_str())

        ("mean-RV",
            po::value<double>(&(opts.mean_RV)),
            ("Mean R_V (per star) (default: " +
                to_string(opts.mean_RV) + ")").c_str())
        ("sigma-RV",
            po::value<double>(&(opts.sigma_RV)),
            ("Variation in R_V (per star) (default: " +
                to_string(opts.sigma_RV) +
                ", interpreted as no variance)").c_str())

        // ("smoothing-pct",
        //     po::value<double>(&(opts.smoothing_slope)),
        //         ("Degree of smoothing (sigma/EBV) of per-star "
        //             "surfaces (default: " +
        //             to_string(opts.smoothing_slope) + ")").c_str())
        ("pct-smoothing-coeffs",
            po::value< vector<double> >()->multitoken(),
            ("Coefficients for alpha and beta smoothing parameters "
                "(default: " +
                to_string(opts.smoothing_alpha_coeff[0]) + " " +
                to_string(opts.smoothing_alpha_coeff[1]) + " " +
                to_string(opts.smoothing_beta_coeff[0]) + " " +
                to_string(opts.smoothing_beta_coeff[1]) + ")").c_str())
        ("pct-smoothing-min",
            po::value<double>(&(opts.pct_smoothing_min)),
            ("Minimum smoothing percent of per-star surfaces "
                "(default: " +
                to_string(opts.pct_smoothing_min) + ")").c_str())
        ("pct-smoothing-max",
            po::value<double>(&(opts.pct_smoothing_max)),
            ("Maximum smoothing percent of per-star surfaces (default: " +
                    to_string(opts.pct_smoothing_max) + ")").c_str())

        ("discrete-los",
            "Use the discrete line-of-sight model.")
        ("discrete-steps",
            po::value<unsigned int>(&(opts.discrete_steps)),
            ("# of steps to take for the discrete l.o.s. sampler "
                "(default: " +
                to_string(opts.discrete_steps) + ")").c_str())

        ("regions",
            po::value<unsigned int>(&(opts.N_regions)),
            ("# of piecewise-linear regions in l.o.s. extinction profile "
                "(default: " +
                to_string(opts.N_regions) + ")").c_str())
        ("los-steps",
            po::value<unsigned int>(&(opts.los_steps)),
            ("# of MCMC steps in l.o.s. fit (per sampler) (default: " +
                to_string(opts.los_steps) + ")").c_str())
        ("los-samplers",
            po::value<unsigned int>(&(opts.los_samplers)),
            ("# of samplers per dimension (l.o.s. fit) (default: " +
                to_string(opts.los_samplers) + ")").c_str())
        ("los-p-replacement",
            po::value<double>(&(opts.los_p_replacement)),
            ("Probability of taking replacement step (l.o.s. fit) "
                "(default: " +
                to_string(opts.los_p_replacement) + ")").c_str())

        ("clouds",
            po::value<unsigned int>(&(opts.N_clouds)),
            ("# of clouds along the line of sight (default: " +
                to_string(opts.N_clouds) + ")\n"
                "Setting this option causes the sampler to "
                "also fit a discrete cloud model of "
                "the l.o.s. extinction profile.").c_str())
        ("cloud-steps",
            po::value<unsigned int>(&(opts.cloud_steps)),
            ("# of MCMC steps in cloud fit (per sampler) (default: " +
                to_string(opts.cloud_steps) + ")").c_str())
        ("cloud-samplers",
            po::value<unsigned int>(&(opts.cloud_samplers)),
            ("# of samplers per dimension (cloud fit) (default: " +
                to_string(opts.cloud_samplers) + ")").c_str())
        ("cloud-p-replacement",
            po::value<double>(&(opts.cloud_p_replacement)),
            ("Probability of taking replacement step (cloud fit) "
                "(default: " +
                to_string(opts.cloud_p_replacement) + ")").c_str())

        ("disk-prior",
            "Assume that dust density roughly traces "
            "stellar disk density.")
        ("log-Delta-EBV-min",
            po::value<double>(&(opts.log_Delta_EBV_floor)),
            ("Minimum log(Delta EBV) in l.o.s. reddening prior "
                "(default: " +
                to_string(opts.log_Delta_EBV_floor) + ")").c_str())
        ("log-Delta-EBV-max",
            po::value<double>(&(opts.log_Delta_EBV_ceil)),
            ("Maximum log(Delta EBV) in l.o.s. reddening prior "
                "(default: " +
                to_string(opts.log_Delta_EBV_ceil) + ")").c_str())
        ("sigma-log-Delta-EBV",
            po::value<double>(&(opts.sigma_log_Delta_EBV)),
            ("Std. dev. of log(Delta EBV) in one distance bin "
                "in l.o.s. reddening prior (default: " +
                to_string(opts.sigma_log_Delta_EBV) + ")").c_str())
        ("SFD-prior",
            "Use SFD E(B-V) as a prior on the total "
            "extinction in each pixel.")
        ("SFD-subpixel",
            "Use SFD E(B-V) as a subpixel template for the "
            "angular variation in reddening.")
        ("subpixel-max",
            po::value<double>(&(opts.subpixel_max)),
            ("Maximum subpixel value (above this values, stars will "
                "be filtered out). (default: " +
                to_string(opts.subpixel_max) + ")").c_str())
        ("evidence-cut",
            po::value<double>(&(opts.ev_cut)),
            ("Delta lnZ to use as threshold for including star "
                  "in l.o.s. fit. Used with MCMC sampling. "
                  "(default: " +
                  to_string(opts.ev_cut) + ")").c_str())
        ("chi2-cut",
            po::value<double>(&(opts.chi2_cut)),
            ("chi^2 / passband to use as threshold for "
                "including star in l.o.s. fit.\n"
                "Used with grid evaluation. (default: " +
                to_string(opts.chi2_cut) + ")").c_str())
        ("runs",
            po::value<unsigned int>(&(opts.N_runs)),
            ("# of times to run each chain (to check\n"
                "for non-convergence) (default: " +
                to_string(opts.N_runs) + ")").c_str())

        ("LF-file",
            po::value<string>(&(opts.LF_fname)),
            "File containing stellar luminosity function.")
        ("template-file",
            po::value<string>(&(opts.template_fname)),
            "File containing stellar color templates.")
        ("ext-file",
            po::value<string>(&(opts.ext_model_fname)),
            "File containing extinction coefficients.")
        
        ("neighbor-lookup-file",
            po::value<string>(&(opts.neighbor_lookup_fname)),
            "Lookup file that maps pixels -> neighboring pixels.")
        ("pixel-lookup-file",
            po::value<string>(&(opts.pixel_lookup_fname)),
            "Lookup file that maps pixels -> files.")
        ("output-fname-pattern",
            po::value<string>(&(opts.output_fname_pattern)),
            ("Filename pattern for previous iteration of \n"
                "output files. E.g., output.@@@@@.h5."))
        
        ("correlation-scale",
            po::value<double>(&(opts.correlation_scale)),
            ("Dust log density correlation scale, in pc "
                "(default: " +
                to_string(opts.correlation_scale) + ")").c_str())
        ("corr-softening-scale",
            po::value<double>(&(opts.d_soft)),
            ("Softening length for the dust log density "
             "correlation scale, in pc (default: " +
                to_string(opts.d_soft) + ")").c_str())
        ("corr-softening-gamma",
            po::value<double>(&(opts.gamma_soft)),
            ("Shape of the correlation softening. Higher -> sharper "
             "(default: " +
                to_string(opts.d_soft) + ")").c_str())
    ;

    po::options_description dsc_samp_settings_desc(
        "Settings for discrete l.o.s. model sampler.");
    dsc_samp_settings_desc.add_options()
        ("dsc-n-temperatures",
            po::value<unsigned int>(&(opts.dsc_samp_settings.n_temperatures)),
                ("# of temperatures to use when sampling discrete \n"
                 "l.o.s. model (default: " +
                    to_string(opts.dsc_samp_settings.n_temperatures) +
                 ")").c_str())
        ("dsc-beta-spacing",
            po::value<double>(&(opts.dsc_samp_settings.beta_spacing)),
                ("Spacing of temperature ladder for discrete l.o.s. \n"
                 "(1=degenerate, 0=maximal spacing). (default: " +
                    to_string(opts.dsc_samp_settings.beta_spacing) +
                 ")").c_str())
        ("dsc-central-steps-per-update",
            po::value<unsigned int>(&(opts.dsc_samp_settings.central_steps_per_update)),
                ("Discrete l.o.s. sampler: # of steps to take in central \n"
                 "pixel per distance per update. (default: " +
                    to_string(opts.dsc_samp_settings.central_steps_per_update) +
                 ")").c_str())
        ("dsc-neighbor-steps-per-update",
            po::value<unsigned int>(&(opts.dsc_samp_settings.neighbor_steps_per_update)),
                ("Discrete l.o.s. sampler: # of steps to take in each \n"
                 "neighboring pixel per update. (default: " +
                    to_string(opts.dsc_samp_settings.neighbor_steps_per_update) +
                 ")").c_str())
        ("dsc-updates-per-swap",
            po::value<unsigned int>(&(opts.dsc_samp_settings.updates_per_swap)),
                ("Discrete l.o.s. sampler: # of update rounds of pixels \n"
                 "per attemped swap between temperatures (default: " +
                    to_string(opts.dsc_samp_settings.updates_per_swap) +
                 ")").c_str())
        ("dsc-n-swaps",
            po::value<unsigned int>(&(opts.dsc_samp_settings.n_swaps)),
                ("Discrete l.o.s. sampler: total # of swaps between \n"
                 "temperatures to attempt (default: " +
                    to_string(opts.dsc_samp_settings.n_swaps) +
                 ")").c_str())
        ("dsc-burnin-fraction",
            po::value<double>(&(opts.dsc_samp_settings.burnin_frac)),
                ("Discrete l.o.s. sampler: Length of burn-in, as a \n"
                 "fraction of the main sampling phase (default: " +
                    to_string(opts.dsc_samp_settings.burnin_frac) +
                 ")").c_str())
        ("dsc-n-save",
            po::value<unsigned int>(&(opts.dsc_samp_settings.n_save)),
                ("Discrete l.o.s. sampler: # of samples to save \n"
                 "(default: " +
                    to_string(opts.dsc_samp_settings.n_save) +
                 ")").c_str())
        ("dsc-log-shift-weight-min",
            po::value<double>(&(opts.dsc_samp_settings.log_shift_weight_min)),
                ("Discrete l.o.s. sampler: Parameter that controls \n"
                 "strength of correlations between neighboring \n"
                 "distances in the temperature=1 sampler, \n"
                 "as a fraction of inv. cov. between neighboring \n"
                 "pixels (default: " +
                    to_string(opts.dsc_samp_settings.log_shift_weight_min) +
                 ")").c_str())
        ("dsc-log-shift-weight-max",
            po::value<double>(&(opts.dsc_samp_settings.log_shift_weight_max)),
                ("Discrete l.o.s. sampler: Parameter that controls \n"
                 "strength of correlations between neighboring \n"
                 "distances in the highest-temperature sampler, \n"
                 "as a fraction of inv. cov. between neighboring \n"
                 "pixels (default: " +
                    to_string(opts.dsc_samp_settings.log_shift_weight_max) +
                 ")").c_str())
        ("dsc-shift-weight-ladder-logarithmic",
            po::value<bool>(&(opts.dsc_samp_settings.shift_weight_ladder_logarithmic)),
                ("If this flag is set, then the shift weights will \n"
                 "be spaced logarithmically, instead of linearly \n"
                 "(default: " +
                    to_string(opts.dsc_samp_settings.shift_weight_ladder_logarithmic) +
                 ")").c_str())
        ("dsc-save-all-temperatures",
            po::value<bool>(&(opts.dsc_samp_settings.save_all_temperatures)),
                ("Discrete l.o.s. sampler: If true, samples from higher \n"
                 "temperature samplers will be saved as well (default: " +
                    to_string(opts.dsc_samp_settings.save_all_temperatures) +
                 ")").c_str())
        ("dsc-p-badstar",
            po::value<double>(&(opts.dsc_samp_settings.p_badstar)),
                ("Stellar outlier fraction: larger values mean less \n"
                 "weight to outliers (default: " +
                    to_string(opts.dsc_samp_settings.p_badstar) +
                 ")").c_str())
    ;
    config_desc.add(dsc_samp_settings_desc);
    
    po::options_description gal_desc(
        "Galactic Structural Parameters (all distances in pc)");
    gal_desc.add_options()
        ("R0",
            po::value<double>(&(opts.gal_struct_params.R0)),
                ("Solar Galactocentric distance (default: " +
                    to_string(opts.gal_struct_params.R0) + ")").c_str())
        ("Z0",
            po::value<double>(&(opts.gal_struct_params.Z0)),
                ("Solar height above Galactic midplane (default: " +
                    to_string(opts.gal_struct_params.Z0) + ")").c_str())

        ("H_thin",
            po::value<double>(&(opts.gal_struct_params.H1)),
                ("Thin-disk scale height (default: " +
                    to_string(opts.gal_struct_params.H1) + ")").c_str())
        ("L_thin",
            po::value<double>(&(opts.gal_struct_params.L1)),
                ("Thin-disk scale length (default: " +
                    to_string(opts.gal_struct_params.L1) + ")").c_str())

        ("f_thick",
            po::value<double>(&(opts.gal_struct_params.f_thick)),
                ("Thick-disk fraction, defined locally (default: " +
                    to_string(opts.gal_struct_params.f_thick) +
                    ")").c_str())
        ("H_thick",
            po::value<double>(&(opts.gal_struct_params.H2)),
                ("Thick-disk scale height (default: " +
                    to_string(opts.gal_struct_params.H2) + ")").c_str())
        ("L_thick",
            po::value<double>(&(opts.gal_struct_params.L2)),
                ("Thin-disk scale length (default: " +
                    to_string(opts.gal_struct_params.L2) + ")").c_str())

        ("L_epsilon",
            po::value<double>(&(opts.gal_struct_params.L_epsilon)),
                ("Disk softening scale (default: " +
                    to_string(opts.gal_struct_params.L_epsilon) +
                    ")").c_str())

        ("f_halo",
            po::value<double>(&(opts.gal_struct_params.fh)),
                ("Halo fraction, defined locally (default: " +
                    to_string(opts.gal_struct_params.fh) + ")").c_str())
        ("q_halo",
            po::value<double>(&(opts.gal_struct_params.qh)),
                ("Halo flattening parameter (default: " +
                    to_string(opts.gal_struct_params.qh) + ")").c_str())
        ("n_halo",
            po::value<double>(&(opts.gal_struct_params.nh)),
                ("Halo density slope (default: " +
                    to_string(opts.gal_struct_params.nh) + ")").c_str())
        ("R_break",
            po::value<double>(&(opts.gal_struct_params.R_br)),
                ("Halo break radius (default: " +
                    to_string(opts.gal_struct_params.R_br) +
                    ")").c_str())
        ("n_halo_outer",
            po::value<double>(&(opts.gal_struct_params.nh_outer)),
                ("Halo outer density slope, past break (default: " +
                    to_string(opts.gal_struct_params.nh_outer) +
                    ")").c_str())

        ("H_ISM",
            po::value<double>(&(opts.gal_struct_params.H_ISM)),
                ("Dust scale height (default: " +
                    to_string(opts.gal_struct_params.H_ISM) +
                    ")").c_str())
        ("L_ISM",
            po::value<double>(&(opts.gal_struct_params.L_ISM)),
                ("Dust scale length (default: " +
                    to_string(opts.gal_struct_params.L_ISM) +
                    ")").c_str())
        ("dH_dR_ISM",
            po::value<double>(&(opts.gal_struct_params.dH_dR_ISM)),
                ("Dust flare slope (default: " +
                    to_string(opts.gal_struct_params.dH_dR_ISM) +
                    ")").c_str())
        ("R_flair_ISM",
            po::value<double>(&(opts.gal_struct_params.R_flair_ISM)),
                ("Dust flair slope (default: " +
                    to_string(opts.gal_struct_params.R_flair_ISM) +
                    ")").c_str())

        ("mu_FeH_inf",
            po::value<double>(&(opts.gal_struct_params.mu_FeH_inf)),
                ("Disk metallicity at large elevation above "
                    "midplane (default: " +
                    to_string(opts.gal_struct_params.mu_FeH_inf) +
                    ")").c_str())
        ("delta_mu_FeH",
            po::value<double>(&(opts.gal_struct_params.delta_mu_FeH)),
                ("Disk metallicity at midplane, minus metallicity "
                    "at large elevation (default: " +
                    to_string(opts.gal_struct_params.delta_mu_FeH) +
                    ")").c_str())
        ("H_mu_FeH",
            po::value<double>(&(opts.gal_struct_params.H_mu_FeH)),
                ("Disk metallicity scale height (default: " +
                    to_string(opts.gal_struct_params.H_mu_FeH) +
                    ")").c_str())
    ;
    config_desc.add(gal_desc);

    po::options_description generic_desc(
        std::string("Usage: ") + argv[0] +
        " [Input filename] [Output filename] \n\n"
        "Commandline Options");
    generic_desc.add_options()
        ("help", "Display this help message")
        ("show-config", "Display configuration-file options")
        ("version", "Display version number")

        ("input",
            po::value<std::string>(&(opts.input_fname)),
            "Input HDF5 filename (contains stellar photometry)")
        ("output",
            po::value<std::string>(&(opts.output_fname)),
            "Output HDF5 filename (MCMC output and smoothed "
                "probability surfaces)")

        ("config",
            po::value<std::string>(&config_fname),
            "Configuration file containing additional options.")

        ("test-los",
            "Allow user to test specific line-of-sight profiles manually.")
    ;

    po::options_description dual_desc(
        "Dual Options (both commandline and configuration file)");
    dual_desc.add_options()
        ("save-surfs", "Save probability surfaces.")
        ("load-surfs", "Use pre-computed probability surfaces from output file.")
        ("save-gridstars", "Save grid-evaluated stellar inferences.")
        ("clobber", "Overwrite existing output. Otherwise, will\n"
                    "only process pixels with incomplete output.")
        ("verbosity",
            po::value<int>(&(opts.verbosity)),
            ("Level of verbosity (0 = minimal, 2 = highest) (default: " +
                to_string(opts.verbosity) + ")").c_str())
        ("threads",
            po::value<unsigned int>(&(opts.N_threads)),
            ("# of threads to run on (default: " +
                to_string(opts.N_threads) + ")").c_str())
        ("force-pix",
            po::value<std::vector<std::string>>()->multitoken(),
            ("Force the given pixels to run. E.g., \"1024-0 512-50\\n "
             "would force (nside,healpix_idx) = (1024,0) and (512,50)\n "
             "to run, even if they are already present in the output."))
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
    po::store(
        po::command_line_parser(argc, argv).options(cmdline_desc)
            .positional(pd).run(),
        vm
    );
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
    if(vm.count("sample-stars")) { opts.sample_stars = true; }
    if(vm.count("save-surfs")) { opts.save_surfs = true; }
    if(vm.count("load-surfs")) { opts.load_surfs = true; }
    if(vm.count("save-gridstars")) { opts.save_gridstars = true; }
    if(vm.count("no-stellar-priors")) { opts.star_priors = false; }
    if(vm.count("use-gaia")) { opts.use_gaia = true; }
    if(vm.count("disk-prior")) { opts.disk_prior = true; }
    if(vm.count("SFD-prior")) { opts.SFD_prior = true; }
    if(vm.count("SFD-subpixel")) { opts.SFD_subpixel = true; }
    if(vm.count("clobber")) { opts.clobber = true; }
    if(vm.count("test-los")) { opts.test_mode = true; }
    if(vm.count("discrete-los")) { opts.discrete_los = true; }

    // Read percent smoothing coefficients
    if(!vm["pct-smoothing-coeffs"].empty()) {
        vector<double> pct_smoothing_coeffs =
            vm["pct-smoothing-coeffs"].as<vector<double> >();

        if(pct_smoothing_coeffs.size() != 4) {
            cerr << "'pct-smoothing-coeffs' takes exactly 4 numbers." << endl;
            return -1;
        }

        opts.smoothing_alpha_coeff[0] = pct_smoothing_coeffs[0];
        opts.smoothing_alpha_coeff[1] = pct_smoothing_coeffs[1];
        opts.smoothing_beta_coeff[0] = pct_smoothing_coeffs[2];
        opts.smoothing_beta_coeff[1] = pct_smoothing_coeffs[3];
    }
    
    // Read forced pixels
    if(!vm["force-pix"].empty()) {
        opts.force_pix =
            vm["force-pix"].as<std::vector<std::string> >();
    }

    // Convert error floor from mmags to mags
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
            cerr << "# of regions in extinction profile must divide "
                    "120 without remainder." << endl;
            return -1;
        }
    }

    return 1;
}
