/*
 * sampler.cpp
 *
 * Samples from posterior distribution of line-of-sight model.
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



#include "sampler.h"


/****************************************************************************************************************************
 *
 * TMCMCParams
 *
 ****************************************************************************************************************************/

TMCMCParams::TMCMCParams(TGalacticLOSModel *_gal_model, TSyntheticStellarModel *_synth_stellar_model, TStellarModel *_emp_stellar_model,
			  TExtinctionModel *_ext_model, TStellarData *_data, unsigned int _N_DM, double _DM_min, double _DM_max)
	: gal_model(_gal_model), synth_stellar_model(_synth_stellar_model), emp_stellar_model(_emp_stellar_model),
          ext_model(_ext_model), data(_data), N_DM(_N_DM), DM_min(_DM_min), DM_max(_DM_max)
{
    N_stars = data->star.size();
    EBV_interp = new TLinearInterp(DM_min, DM_max, N_DM);

    // Defaults
    lnp0 = -10.;
    idx_star = 0;

    EBV_floor = 0.;

    vary_RV = false;
    RV_mean = 3.1;
    RV_variance = 0.2*0.2;

    use_priors = true;
}

TMCMCParams::~TMCMCParams() {
    delete EBV_interp;
}


void TMCMCParams::update_EBV_interp(const double *x) {
    double EBV = 0.;
    for(unsigned int i=0; i<N_DM; i++) {
        EBV += exp(x[1+i]);
        (*EBV_interp)[i] = EBV;
        if(i == 0) { EBV_min = EBV; }
    }
    EBV_max = EBV;
}

double TMCMCParams::get_EBV(double DM) {
    if(DM <= DM_min) { return EBV_min; } else if(DM >= DM_max) { return EBV_max; }
    return (*EBV_interp)(DM);
}



/****************************************************************************************************************************
 *
 * Probability density functions
 *
 ****************************************************************************************************************************/


// Natural logarithm of posterior probability density for one star, given parameters x, where
//
//     x = {DM, Log_10(Mass_init), Log_10(Age), [Fe/H]}
double logP_single_star_synth(const double *x, double EBV, double RV,
                              const TGalacticLOSModel &gal_model, const TSyntheticStellarModel &stellar_model,
                              TExtinctionModel &ext_model, const TStellarData::TMagnitudes &d, TSED *tmp_sed) {
    double logP = 0.;

    /*
     *  Likelihood
     */
    bool del_sed = false;
    if(tmp_sed == NULL) {
        del_sed = true;
        tmp_sed = new TSED(true);
    }
    if(!stellar_model.get_sed(x+1, *tmp_sed)) {
        if(del_sed) { delete tmp_sed; }
        return neg_inf_replacement;
    }

    double logL = 0.;
    double tmp;
    for(unsigned int i=0; i<NBANDS; i++) {
        if(d.err[i] < 1.e9) {
            tmp = tmp_sed->absmag[i] + x[_DM] + EBV * ext_model.get_A(RV, i);	// Model apparent magnitude
            logL -= log( 1. + exp((tmp - d.maglimit[i]) / d.maglim_width[i]) );
            //logL += log( 0.5 - 0.5 * erf((tmp - d.maglimit[i] + 0.1) / 0.25) );	// Completeness fraction
            tmp = (d.m[i] - tmp) / d.err[i];
            logL -= 0.5*tmp*tmp;
        }
    }
    logP += logL - d.lnL_norm;

    if(del_sed) { delete tmp_sed; }

    /*
     *  Priors
     */
    logP += gal_model.log_prior_synth(x);

    return logP;
}

// Natural logarithm of posterior probability density for one star, given parameters x, where
//
//     x = {DM, M_r, [Fe/H]}
double logP_single_star_emp(const double *x, double EBV, double RV,
                            const TGalacticLOSModel &gal_model, const TStellarModel &stellar_model,
                            TExtinctionModel &ext_model, const TStellarData::TMagnitudes &d, TSED *tmp_sed) {
    double logP = 0.;

    /*
     * Don't allow NaN parameters
     */
    if(std::isnan(x[0]) || std::isnan(x[1]) || std::isnan(x[2])) {
        /*#pragma omp critical (cout)
        {
        std::cerr << "Encountered NaN parameter value!" << std::endl;
        std::cerr << "  " << x[0] << std::endl;
        std::cerr << "  " << x[1] << std::endl;
        std::cerr << "  " << x[2] << std::endl;
        }*/
        return neg_inf_replacement;
    }

    /*
     *  Likelihood
     */
    bool del_sed = false;
    if(tmp_sed == NULL) {
        del_sed = true;
        tmp_sed = new TSED(true);
    }
    if(!stellar_model.get_sed(x+1, *tmp_sed)) {
        if(del_sed) { delete tmp_sed; }
        return neg_inf_replacement;
    }

    double logL = 0.;
    double tmp;
    for(unsigned int i=0; i<NBANDS; i++) {
        if(d.err[i] < 1.e9) {
            tmp = tmp_sed->absmag[i] + x[_DM] + EBV * ext_model.get_A(RV, i);	// Model apparent magnitude
            logL -= log( 1. + exp((tmp - d.maglimit[i]) / d.maglim_width[i]) );
            //logL += log( 0.5 - 0.5 * erf((tmp - d.maglimit[i] + 0.1) / 0.25) );	// Completeness fraction
            //std::cout << tmp << ", " << d.maglimit[i] << std::endl;
            tmp = (d.m[i] - tmp) / d.err[i];
            logL -= 0.5*tmp*tmp;
        }
    }
    logP += logL - d.lnL_norm;

    if(del_sed) { delete tmp_sed; }

    /*
     *  Priors
     */
    logP += gal_model.log_prior_emp(x) + stellar_model.get_log_lf(x[1]);

    return logP;
}


// Natural logarithm of posterior probability density for one star, given parameters x, where
//
//     x = {DM, M_r, [Fe/H]}
double logP_single_star_emp_noprior(const double *x, double EBV, double RV,
                                    const TGalacticLOSModel &gal_model, const TStellarModel &stellar_model,
                                    TExtinctionModel &ext_model, const TStellarData::TMagnitudes &d, TSED *tmp_sed) {
    double logP = 0.;

    /*
     *  Likelihood
     */
    bool del_sed = false;
    if(tmp_sed == NULL) {
        del_sed = true;
        tmp_sed = new TSED(true);
    }
    if(!stellar_model.get_sed(x+1, *tmp_sed)) {
        if(del_sed) { delete tmp_sed; }
        return neg_inf_replacement;
    }

    double logL = 0.;
    double tmp;
    for(unsigned int i=0; i<NBANDS; i++) {
        if(d.err[i] < 1.e9) {
            tmp = tmp_sed->absmag[i] + x[_DM] + EBV * ext_model.get_A(RV, i);	// Model apparent magnitude
            tmp = (d.m[i] - tmp) / d.err[i];
            logL -= 0.5*tmp*tmp;
        }
    }
    logP += logL - d.lnL_norm;

    if(del_sed) { delete tmp_sed; }

    return logP;
}


double logP_EBV(TMCMCParams &p) {
    double logP = 0.;

    // SFD prior
    logP -= (p.EBV_max * p.EBV_max) / (2. * p.EBV_SFD * p.EBV_SFD);

    // Lognormal
    double tmp;
    for(double DM = p.DM_min; DM <= p.DM_max; DM += (p.DM_max-p.DM_min)/10.) {
        tmp = p.get_EBV(DM);
        logP -= tmp*tmp / 2.;
    }

    return logP;
}

double logP_los_synth(const double *x, unsigned int N, TMCMCParams &p, double *lnP_star) {
    double logP = 0.;

    // Prior on RV
    double RV = x[0];
    if(p.ext_model->in_model(RV)) {
        logP -= (RV - 3.1)*(RV - 3.1) / (2. * 0.1 * 0.1);
    } else {
        return neg_inf_replacement;
    }

    // Prior on extinction
    logP += logP_EBV(p);

    // Probabilities of stars
    double tmp;
    const double *x_star;
    for(unsigned int i=0; i<p.N_stars; i++) {
        x_star = &(x[1 + p.N_DM + 4*i]);
        tmp = logP_single_star_synth(x_star, p.get_EBV(x_star[_DM]), RV, *p.gal_model, *p.synth_stellar_model, *p.ext_model, p.data->star[i]);
        logP += tmp;
        if(lnP_star != NULL) { lnP_star[i] = tmp; }
    }

    return logP;
}



/****************************************************************************************************************************
 *
 * Sampling functions
 *
 ****************************************************************************************************************************/

void sample_model_synth(TGalacticLOSModel &galactic_model, TSyntheticStellarModel &stellar_model, TExtinctionModel &extinction_model, TStellarData &stellar_data) {
    unsigned int N_DM = 20;
    double DM_min = 4.;
    double DM_max = 20.;
    TMCMCParams params(&galactic_model, &stellar_model, NULL, &extinction_model, &stellar_data, N_DM, DM_min, DM_max);
    TMCMCParams params_tmp(&galactic_model, &stellar_model, NULL, &extinction_model, &stellar_data, N_DM, DM_min, DM_max);

    // Random number generator
    gsl_rng *r;
    seed_gsl_rng(&r);

    // Vector describing position in probability space
    size_t length = 1 + params.N_DM + 4*params.N_stars;
    // x = {RV, Delta_EBV_1, ..., Delta_EBV_M, Theta_1, ..., Theta_N}, where Theta = {DM, logMass, logtau, FeH}.
    double *x = new double[length];

    // Random starting point for reddening profile
    x[0] = 3.1;// + gsl_ran_gaussian_ziggurat(r, 0.2);	// RV
    for(size_t i=0; i<params.N_DM; i++) { x[i+1] = params.data->EBV / (double)N_DM * gsl_ran_chisq(r, 1.); }		// Delta_EBV

    // Random starting point for each star
    TSED sed_tmp(true);
    for(size_t i = 1 + params.N_DM; i < 1 + params.N_DM + 4*params.N_stars; i += 4) {
        x[i] = 5. + 13.*gsl_rng_uniform(r);
        double logMass, logtau, FeH, tau;
        bool in_lib = false;
        while(!in_lib) {
            logMass = gsl_ran_gaussian_ziggurat(r, 0.5);
            tau = -1.;
            while(tau <= 0.) {
                tau = 1.e9 * (5. + gsl_ran_gaussian_ziggurat(r, 2.));
            }
            logtau = log10(tau);
            FeH = -1.0 + gsl_ran_gaussian_ziggurat(r, 1.);

            in_lib = stellar_model.get_sed(logMass, logtau, FeH, sed_tmp);
        }
        x[i+1] = logMass;
        x[i+2] = logtau;
        x[i+3] = FeH;
    }

    params.update_EBV_interp(x);
    double *lnp_star = new double[params.N_stars];
    double lnp_los = logP_los_synth(x, length, params, lnp_star);
    std::cerr << "# ln p(x_0) = " << lnp_los << std::endl;

    double *x_tmp = new double[length];
    double Theta_tmp[4];
    double sigma_Theta[4] = {0.1, 0.1, 0.1, 0.1};
    double sigma_RV = 0.05;
    double sigma_lnEBV = 0.1;
    double lnp_tmp;
    double *lnp_star_tmp = new double[params.N_stars];
    double p;

    unsigned int N_steps = 1000000;

    TChain chain(length, N_steps);
    TStats EBV_stats(N_DM);

    // In each step
    unsigned int N_star = 0;
    unsigned int N_accept_star = 0;
    unsigned int N_los = 0;
    unsigned int N_accept_los = 0;
    bool accept;
    bool burn_in = true;
    for(unsigned int i=0; i<N_steps; i++) {
        if(i == N_steps/2) {
            sigma_Theta[0] = 0.05;
            sigma_Theta[1] = 0.05;
            sigma_Theta[2] = 0.05;
            sigma_Theta[3] = 0.05;
            sigma_RV = 0.005;
            sigma_lnEBV = 0.05;
            burn_in = false;
        }

        // Step each star
        for(unsigned int n=0; n<params.N_stars; n++) {
            if(!burn_in) { N_star++; }

            rand_gaussian_vector(&Theta_tmp[0], &x[1+N_DM+4*n], &sigma_Theta[0], 4, r);
            lnp_tmp = logP_single_star_synth(&Theta_tmp[0], params.get_EBV(Theta_tmp[_DM]), x[0], galactic_model, stellar_model, extinction_model, stellar_data.star[n]);

            accept = false;
            if(lnp_tmp > lnp_star[n]) {
                accept = true;
            } else {
                p = gsl_rng_uniform(r);
                if((p > 0.) && (log(p) < lnp_tmp - lnp_star[n])) {
                    accept = true;
                }
            }

            if(accept) {
                if(!burn_in) { N_accept_star++; }
                for(size_t k=0; k<4; k++) { x[1+N_DM+4*n+k] = Theta_tmp[k]; }
                lnp_los += lnp_tmp - lnp_star[n];
                lnp_star[n] = lnp_tmp;
            }
        }

        // Step reddening profile
        if(!burn_in) { N_los++; }
        for(size_t k=0; k<length; k++) { x_tmp[k] = x[k]; }
        for(unsigned int m=0; m<params.N_DM; m++) { x_tmp[1+m] += gsl_ran_gaussian_ziggurat(r, sigma_lnEBV); }

        params_tmp.update_EBV_interp(x_tmp);
        lnp_tmp = logP_los_synth(x_tmp, length, params_tmp, lnp_star_tmp);

        accept = false;
        if(lnp_tmp > lnp_los) {
            accept = true;
        } else if(log(gsl_rng_uniform(r)) < lnp_tmp - lnp_los) {
            accept = true;
        }

        if(accept) {
            if(!burn_in) { N_accept_los++; }
            for(size_t k=0; k<1+N_DM; k++) { x[k] = x_tmp[k]; }
            for(size_t k=0; k<params.N_stars; k++) { lnp_star[k] = lnp_star_tmp[k]; }
            lnp_los = lnp_tmp;
            params.update_EBV_interp(x);
            //std::cerr << "# ln p(x) = " << lnp_los << std::endl;
        }

        if(!burn_in) {
            chain.add_point(x, lnp_los, 1.);

            x_tmp[0] = exp(x[1]);
            for(size_t k=1; k<N_DM; k++) {
                x_tmp[k] = x_tmp[k-1] + exp(x[k]);
            }
            EBV_stats(x_tmp, 1);
        }
    }

    std::cerr << "# ln p(x) = " << lnp_los << std::endl;
    std::cout.precision(4);
    std::cerr << std::endl;
    std::cerr << "# % acceptance: " << 100. * (double)N_accept_star / (double)N_star << " (stars)" << std::endl;
    std::cerr << "                " << 100. * (double)N_accept_los / (double)N_los << " (extinction)" << std::endl;
    std::cerr << "# R_V = " << x[0] << std::endl << std::endl;
    std::cerr << "#  DM   E(B-V)" << std::endl;
    std::cerr << "# =============" << std::endl;
    for(double DM=5.; DM<20.; DM+=1.) {
        std::cerr << "#  " << DM << " " << params.get_EBV(DM) << std::endl;
    }
    std::cerr << std::endl;
    EBV_stats.print();
    std::cerr << std::endl;

    delete[] x;
    delete[] x_tmp;
    delete[] lnp_star;
    delete[] lnp_star_tmp;
}

void gen_rand_state_synth(double *const x, unsigned int N, gsl_rng *r, TMCMCParams &params) {
    assert(N == 1 + params.N_DM + 4*params.N_stars);

    // R_V
    x[0] = 3.1 + gsl_ran_gaussian_ziggurat(r, 0.2);

    // Delta_EBV
    for(size_t i=0; i<params.N_DM; i++) { x[i+1] = params.data->EBV / (double)params.N_DM * gsl_ran_chisq(r, 1.); }

    // Stars
    TSED sed_tmp(true);
    for(size_t i = 1 + params.N_DM; i < 1 + params.N_DM + 4*params.N_stars; i += 4) {
        // DM
        x[i] = 5. + 13. * gsl_rng_uniform(r);

        // Stellar type
        double logMass, logtau, FeH, tau;
        bool in_lib = false;
        while(!in_lib) {
            logMass = gsl_ran_gaussian_ziggurat(r, 0.5);
            tau = -1.;
            while(tau <= 0.) {
                tau = 1.e9 * (5. + gsl_ran_gaussian_ziggurat(r, 2.));
            }
            logtau = log10(tau);
            FeH = -1.0 + gsl_ran_gaussian_ziggurat(r, 1.);

            in_lib = params.synth_stellar_model->get_sed(logMass, logtau, FeH, sed_tmp);
        }
        x[i+1] = logMass;
        x[i+2] = logtau;
        x[i+3] = FeH;
    }
}

double logP_los_simple_synth(const double *x, unsigned int N, TMCMCParams &params) {
    params.update_EBV_interp(x);
    return logP_los_synth(x, N, params, NULL);
}

void sample_model_affine_synth(TGalacticLOSModel &galactic_model, TSyntheticStellarModel &stellar_model, TExtinctionModel &extinction_model, TStellarData &stellar_data) {
    unsigned int N_DM = 20;
    double DM_min = 4.;
    double DM_max = 19.;
    TMCMCParams params(&galactic_model, &stellar_model, NULL, &extinction_model, &stellar_data, N_DM, DM_min, DM_max);
    TStats EBV_stats(N_DM);

    unsigned int N_steps = 100;
    unsigned int N_samplers = 4;

    TAffineSampler<TMCMCParams, TStats>::pdf_t f_pdf = &logP_los_simple_synth;
    TAffineSampler<TMCMCParams, TStats>::rand_state_t f_rand_state = &gen_rand_state_synth;

    std::cerr << "# Setting up sampler" << std::endl;
    unsigned int ndim = 1 + params.N_DM + 4*params.N_stars;
    TParallelAffineSampler<TMCMCParams, TStats> sampler(f_pdf, f_rand_state, ndim, 10*ndim, params, EBV_stats, N_samplers);

    std::cerr << "# Burn-in" << std::endl;
    sampler.set_scale(1.1);
    sampler.set_replacement_bandwidth(0.5);
    sampler.step(N_steps, false, 0, 0.01);
    sampler.clear();
    std::cerr << "# Main run" << std::endl;
    sampler.step(N_steps, true, 0, 0.01);

    std::cout << "Sampler stats:" << std::endl;
    sampler.print_stats();
    std::cout << std::endl;

    std::cout << "E(B-V) statistics:" << std::endl;
    EBV_stats.print();
    std::cout << std::endl;

    /*
    std::cerr << "# ln p(x) = " << lnp_los << std::endl;
    std::cout.precision(4);
    std::cerr << std::endl;
    std::cerr << "# % acceptance: " << 100. * (double)N_accept_star / (double)N_star << " (stars)" << std::endl;
    std::cerr << "                " << 100. * (double)N_accept_los / (double)N_los << " (extinction)" << std::endl;
    std::cerr << "# R_V = " << x[0] << std::endl << std::endl;
    std::cerr << "#  DM   E(B-V)" << std::endl;
    std::cerr << "# =============" << std::endl;
    for(double DM=5.; DM<20.; DM+=1.) {
            std::cerr << "#  " << DM << " " << params.get_EBV(DM) << std::endl;
    }
    std::cerr << std::endl;
    EBV_stats.print();
    std::cerr << std::endl;
    */
}

void gen_rand_state_indiv_synth(double *const x, unsigned int N, gsl_rng *r, TMCMCParams &params) {
    assert(N == 5);

    // Stars
    TSED sed_tmp(true);

    // E(B-V)
    x[0] = params.EBV_floor + (1.5 * params.data->EBV - params.EBV_floor) * (0.05 + 0.9 * gsl_rng_uniform(r));

    // DM
    x[1] = params.DM_min + (params.DM_max - params.DM_min) * (0.05 + 0.9 * gsl_rng_uniform(r));

    // Stellar type
    double logMass, logtau, FeH, tau;
    bool in_lib = false;
    while(!in_lib) {
        logMass = gsl_ran_gaussian_ziggurat(r, 0.5);
        //tau = -1.;
        //while(tau <= 0.) {
        //	tau = 1.e9 * (5. + gsl_ran_gaussian_ziggurat(r, 2.));
        //}
        //logtau = log10(tau);
        logtau = 8. + gsl_ran_gaussian_ziggurat(r, 1.);
        FeH = -1.0 + gsl_ran_gaussian_ziggurat(r, 1.);

        in_lib = params.synth_stellar_model->get_sed(logMass, logtau, FeH, sed_tmp);
    }
    x[2] = logMass;
    x[3] = logtau;
    x[4] = FeH;

    if(params.vary_RV) {
        double RV = -1.;
        while((RV <= 2.1) || (RV >= 5.)) {
            RV = params.RV_mean + gsl_ran_gaussian_ziggurat(r, 1.5*params.RV_variance*params.RV_variance);
        }
        x[5] = RV;
    }
}

void gen_rand_state_indiv_emp(double *const x, unsigned int N, gsl_rng *r, TMCMCParams &params) {
    if(params.vary_RV) { assert(N == 5); } else { assert(N == 4); }

    // Stars
    TSED sed_tmp(true);

    // Stellar type
    double Mr, FeH;
    Mr = -0.5 + 15.5 * gsl_rng_uniform(r);
    FeH = -2.45 + 2.4 * gsl_rng_uniform(r);

    x[2] = Mr;
    x[3] = FeH;

    double RV = params.RV_mean;;
    if(params.vary_RV) {
        RV = -1.;
        while((RV <= 2.1) || (RV >= 5.)) {
            RV = params.RV_mean + gsl_ran_gaussian_ziggurat(r, 1.5*params.RV_variance*params.RV_variance);
        }
        x[4] = RV;
    }

    // Guess E(B-V) on the basis of other parameters

    // Choose first two bands that have been observed
    /*int b1, b2;
    for(b1=0; b1<NBANDS-1; b1++) {
            if(params.data->star[params.idx_star].err[b1] < 1.e9) {
                    break;
            }
    }
    for(b2=b1+1; b2<NBANDS; b2++) {
            if(params.data->star[params.idx_star].err[b2] < 1.e9) {
                    break;
            }
    }

    // Color excess
    TSED * tmp_sed = new TSED(true);
    params.emp_stellar_model->get_sed(Mr, FeH, *tmp_sed);

    double mod_color = tmp_sed->absmag[b2] - tmp_sed->absmag[b1];
    double obs_color = params.data->star[params.idx_star].m[b2] - params.data->star[params.idx_star].m[b1];

    // Reddening vector
    double R_XY = params.ext_model->get_A(RV, b2) - params.ext_model->get_A(RV, b1);

    // E(B-V)
    for(int i=0; i<5; i++) {
            x[0] = (obs_color - mod_color) / R_XY + gsl_ran_gaussian_ziggurat(r, 0.1);
            if((x[0] > params.EBV_floor) && (x[0] < 8.)) {	// Accept first guess above E(B-V) floor
                    break;
            } else if(i == 4) {	// Revert to dumb, uniform guess
                    //#pragma omp critical
                    //{
                    //std::cout << "  <E(B-V)> = " << x[0] << " >~ " << 8. << std::endl;
                    //}
                    x[0] = fabs(gsl_ran_gaussian_ziggurat(r, 0.1));
                    //x[0] = params.EBV_floor + (1.5 * params.data->EBV - params.EBV_floor) * (0.05 + 0.9 * gsl_rng_uniform(r));
            }
    }*/

    TSED * tmp_sed = new TSED(true);
    params.emp_stellar_model->get_sed(Mr, FeH, *tmp_sed);

    double mod_color, obs_color, R_XY;

    double inv_sigma2_sum = 0.;
    double weighted_sum = 0.;
    double sigma1, sigma2;

    for(int b1=0; b1<NBANDS-1; b1++) {
        for(int b2=b1+1; b2<NBANDS; b2++) {
            mod_color = tmp_sed->absmag[b2] - tmp_sed->absmag[b1];
            obs_color = params.data->star[params.idx_star].m[b2] - params.data->star[params.idx_star].m[b1];
            R_XY = params.ext_model->get_A(RV, b2) - params.ext_model->get_A(RV, b1);

            sigma1 = params.data->star[params.idx_star].err[b1];
            sigma2 = params.data->star[params.idx_star].err[b2];

            weighted_sum += (obs_color - mod_color) / R_XY / (sigma1*sigma1 + sigma2*sigma2);
            inv_sigma2_sum += 1. / (sigma1*sigma1 + sigma2*sigma2);
        }
    }

    double EBV_est = weighted_sum / inv_sigma2_sum;

    for(int i=0; i<5; i++) {
        x[0] = EBV_est + gsl_ran_gaussian_ziggurat(r, 0.1);
        if((x[0] > params.EBV_floor) && (x[0] < 8.)) {	// Accept first guess above E(B-V) floor
            break;
        } else if(i == 4) {	// Revert to dumber guess
            //#pragma omp critical
            //{
            //std::cout << "  <E(B-V)> = " << x[0] << " >~ " << 8. << std::endl;
            //}
            if(EBV_est > 8.) {
                x[0] = 8. - fabs(gsl_ran_gaussian_ziggurat(r, 0.1));
            } else {
                x[0] = fabs(gsl_ran_gaussian_ziggurat(r, 0.1));
            }
            //x[0] = params.EBV_floor + (1.5 * params.data->EBV - params.EBV_floor) * (0.05 + 0.9 * gsl_rng_uniform(r));
        }
    }

    // Guess distance on the basis of model magnitudes vs. observed apparent magnitudes
    inv_sigma2_sum = 0.;
    weighted_sum = 0.;

    double sigma;
    double reddened_mag, obs_mag, maglim;
    double max_DM = inf_replacement;

    for(int i=0; i<NBANDS; i++) {
        sigma = params.data->star[params.idx_star].err[i];
        reddened_mag = tmp_sed->absmag[i] + x[0] * params.ext_model->get_A(RV, i);
        obs_mag = params.data->star[params.idx_star].m[i];
        maglim = params.data->star[params.idx_star].maglimit[i];
        if(obs_mag > maglim) {
            obs_mag = maglim;
        }
        weighted_sum += (obs_mag - reddened_mag) / (sigma * sigma);
        inv_sigma2_sum += 1. / (sigma * sigma);

        // Update maximum allowable distance modulus
        if(maglim - reddened_mag < max_DM) {
            max_DM = maglim - reddened_mag;
        }
    }

    double DM_est = weighted_sum / inv_sigma2_sum;

    x[1] = DM_est + gsl_ran_gaussian_ziggurat(r, 0.1);

    // Adjust distance to ensure that star is observable
    if(x[1] > max_DM + 0.25) {
        //#pragma omp critical (cout)
        //{
        //std::cerr << "DM: " << x[1] << " --> ";
        x[1] = max_DM + gsl_ran_gaussian_ziggurat(r, 0.1);
        //std::cerr << x[1] << std::endl;
        //}
    }

    /*#pragma omp critical (cout)
    {
    //std::cerr << " " << x[0] << " " << max_DM;
    for(int i=0; i<NBANDS; i++) {
            std::cerr << " " << tmp_sed->absmag[i] + x[0] * params.ext_model->get_A(RV, i) + x[1];
    }
    std::cerr << std::endl;
    }*/

    // Don't allow the distance guess to be crazy
    /*if((x[1] < params.DM_min - 2.) || (x[1] > params.DM_max)) {
            #pragma omp critical
            {
                    std::cerr << "!!! DM = " << x[1];
                    x[1] = params.DM_min + (params.DM_max - params.DM_min) * (0.05 + 0.9 * gsl_rng_uniform(r));
                    std::cerr << " --> " << x[1] << " !!!" << std::endl;
            }
    }*/

    //#pragma omp critical (cout)
    //{
    //std::cout << "E(B-V) guess: " << x[0] << " = " << "E(" << b2 << " - " << b1 << ") / (R_" << b2 << " - R_" << b1 << ")" << std::endl;
    //std::cout << "DM guess: " << x[1] << std::endl;
    //}

    /*#pragma omp critical (cout)
    {
    std::cerr << "Guess: " << x[0] << " " << x[1] << " " << x[2] << " " << x[3] << std::endl;
    }*/

    delete tmp_sed;
}

double logP_indiv_simple_synth(const double *x, unsigned int N, TMCMCParams &params) {
    if(x[0] < params.EBV_floor) { return neg_inf_replacement; }
    double RV;
    double logp = 0;
    if(params.vary_RV) {
        RV = x[5];
        if((RV <= 2.1) || (RV >= 5.)) {
            return neg_inf_replacement;
        }
        logp = -0.5*(RV-params.RV_mean)*(RV-params.RV_mean)/params.RV_variance;
    } else {
        RV = params.RV_mean;
    }
    logp += logP_single_star_synth(x+1, x[0], RV, *params.gal_model, *params.synth_stellar_model, *params.ext_model, params.data->star[params.idx_star], NULL);
    return logp;
}

double logP_indiv_simple_emp(const double *x, unsigned int N, TMCMCParams &params) {
    if(x[0] < params.EBV_floor) { return neg_inf_replacement; }
    double RV;
    double logp = 0;
    if(params.vary_RV) {
        RV = x[4];
        if((RV <= 2.1) || (RV >= 5.)) {
            return neg_inf_replacement;
        }
        logp = -0.5*(RV-params.RV_mean)*(RV-params.RV_mean)/params.RV_variance;
    } else {
        RV = params.RV_mean;
    }
    if(params.use_priors) {
        logp += logP_single_star_emp(x+1, x[0], RV, *params.gal_model, *params.emp_stellar_model, *params.ext_model, params.data->star[params.idx_star], NULL);
    } else {
        logp += logP_single_star_emp_noprior(x+1, x[0], RV, *params.gal_model, *params.emp_stellar_model, *params.ext_model, params.data->star[params.idx_star], NULL);
    }
    return logp;
}

void sample_indiv_synth(std::string &out_fname, TMCMCOptions &options, TGalacticLOSModel& galactic_model,
                        TSyntheticStellarModel& stellar_model, TExtinctionModel& extinction_model, TStellarData& stellar_data,
                        TImgStack& img_stack, std::vector<bool> &conv, std::vector<double> &lnZ,
                        double RV_sigma, double minEBV, const bool saveSurfs, const bool gatherSurfs, int verbosity) {
    // Parameters must be consistent - cannot save surfaces without gathering them
    assert(!(saveSurfs & (!gatherSurfs)));

    unsigned int N_DM = 20;
    double DM_min = 4.;
    double DM_max = 19.;
    TMCMCParams params(&galactic_model, &stellar_model, NULL, &extinction_model, &stellar_data, N_DM, DM_min, DM_max);
    params.EBV_floor = minEBV;

    if(RV_sigma > 0.) {
        params.vary_RV = true;
        params.RV_variance = RV_sigma*RV_sigma;
    }

    double min[2] = {0., DM_min};
    double max[2] = {7., DM_max};
    unsigned int N_bins[2] = {700, 120};
    TRect rect(min, max, N_bins);

    if(gatherSurfs) {
        img_stack.resize(params.N_stars);
        img_stack.set_rect(rect);
    }

    TImgWriteBuffer *imgBuffer = NULL;
    if(saveSurfs) { imgBuffer = new TImgWriteBuffer(rect, params.N_stars); }

    TNullLogger logger;

    unsigned int max_attempts = 3;
    unsigned int N_steps = options.steps;
    unsigned int N_samplers = options.samplers;
    unsigned int N_runs = options.N_runs;
    unsigned int ndim;

    if(params.vary_RV) { ndim = 6; } else { ndim = 5; }

    double *GR = new double[ndim];
    double GR_threshold = 1.1;

    TAffineSampler<TMCMCParams, TNullLogger>::pdf_t f_pdf = &logP_indiv_simple_synth;
    TAffineSampler<TMCMCParams, TNullLogger>::rand_state_t f_rand_state = &gen_rand_state_indiv_synth;

    if(verbosity >= 1) {
        std::cout << std::endl;
    }

    unsigned int N_nonconv = 0;

    TChainWriteBuffer chainBuffer(ndim, 100, params.N_stars);
    std::stringstream group_name;
    group_name << "/" << stellar_data.pix_name;

    timespec t_start, t_write, t_end;

    for(size_t n=0; n<params.N_stars; n++) {
        params.idx_star = n;

        clock_gettime(CLOCK_MONOTONIC, &t_start);

        if(verbosity >= 2) {
            std::cout << "Star #" << n+1 << " of " << params.N_stars << std::endl;
            std::cout << "====================================" << std::endl;
        }

        //std::cerr << "# Setting up sampler" << std::endl;
        TParallelAffineSampler<TMCMCParams, TNullLogger> sampler(f_pdf, f_rand_state, ndim, N_samplers*ndim, params, logger, N_runs);
        sampler.set_scale(1.2);
        sampler.set_replacement_bandwidth(0.2);
        sampler.set_sigma_min(0.02);

        //std::cerr << "# Burn-in" << std::endl;
        sampler.step(N_steps, false, 0., 0.2);
        sampler.clear();

        //std::cerr << "# Main run" << std::endl;
        bool converged = false;
        size_t attempt;
        for(attempt = 0; (attempt < max_attempts) && (!converged); attempt++) {
            sampler.step((1<<attempt)*N_steps, true, 0., 0.2);

            converged = true;
            sampler.get_GR_diagnostic(GR);
            for(size_t i=0; i<ndim; i++) {
                if(GR[i] > GR_threshold) {
                    converged = false;
                    if(attempt != max_attempts-1) {
                        sampler.clear();
                        //logger.clear();
                    }
                    break;
                }
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &t_write);

        // Compute evidence
        TChain chain = sampler.get_chain();
        double lnZ_tmp = chain.get_ln_Z_harmonic(true, 10., 0.25, 0.05);
        //if(isinf(lnZ_tmp)) { lnZ_tmp = neg_inf_replacement; }

        // Save thinned chain
        chainBuffer.add(chain, converged, lnZ_tmp, GR);

        // Save binned p(DM, EBV) surface
        if(gatherSurfs) {
            chain.get_image(*(img_stack.img[n]), rect, 0, 1, true, 1.0, 1.0, 30., true);
        }
        if(saveSurfs) { imgBuffer->add(*(img_stack.img[n])); }

        lnZ.push_back(lnZ_tmp);
        conv.push_back(converged);

        clock_gettime(CLOCK_MONOTONIC, &t_end);

        //std::cout << "Sampler stats:" << std::endl;
        if(verbosity >= 2) {
            sampler.print_stats();
            std::cout << std::endl;
        }

        if(!converged) {
            N_nonconv++;
            if(verbosity >= 2) {
                std::cout << "# Failed to converge." << std::endl;
            }
        }

        if(verbosity >= 2) {
            std::cout << "# Number of steps: " << (1<<(attempt-1))*N_steps << std::endl;
            std::cout << "# Time elapsed: " << std::setprecision(2) << (t_end.tv_sec - t_start.tv_sec) + 1.e-9*(t_end.tv_nsec - t_start.tv_nsec) << " s" << std::endl;
            std::cout << "# Sample time: " << std::setprecision(2) << (t_write.tv_sec - t_start.tv_sec) + 1.e-9*(t_write.tv_nsec - t_start.tv_nsec) << " s" << std::endl;
            std::cout << "# Write time: " << std::setprecision(2) << (t_end.tv_sec - t_write.tv_sec) + 1.e-9*(t_end.tv_nsec - t_write.tv_nsec) << " s" << std::endl << std::endl;
        }
    }

    chainBuffer.write(out_fname, group_name.str(), "stellar chains");
    if(saveSurfs) { imgBuffer->write(out_fname, group_name.str(), "stellar pdfs"); }

    if(verbosity >= 1) {
        std::cout << "====================================" << std::endl;
        std::cout << std::endl;
        std::cout << "# Failed to converge " << N_nonconv << " of " << params.N_stars << " times (" << std::setprecision(2) << 100.*(double)N_nonconv/(double)(params.N_stars) << " %)." << std::endl;
        std::cout << std::endl;
        std::cout << "====================================" << std::endl;
    }

    if(imgBuffer != NULL) { delete imgBuffer; }
    delete[] GR;
}

void sample_indiv_emp(std::string &out_fname, TMCMCOptions &options, TGalacticLOSModel& galactic_model,
                      TStellarModel& stellar_model, TExtinctionModel& extinction_model, TEBVSmoothing& EBV_smoothing,
					  TStellarData& stellar_data, TImgStack& img_stack, std::vector<bool> &conv, std::vector<double> &lnZ,
                      double RV_mean, double RV_sigma, double minEBV, const bool saveSurfs, const bool gatherSurfs, const bool use_priors,
                      int verbosity) {
    // Parameters must be consistent - cannot save surfaces without gathering them
    assert(!(saveSurfs & (!gatherSurfs)));

    unsigned int N_DM = 20;
    double DM_min = 4.;
    double DM_max = 19.;
    TMCMCParams params(&galactic_model, NULL, &stellar_model, &extinction_model, &stellar_data, N_DM, DM_min, DM_max);
    params.EBV_floor = minEBV;
    params.use_priors = use_priors;

    params.RV_mean = RV_mean;
    if(RV_sigma > 0.) {
        params.vary_RV = true;
        params.RV_variance = RV_sigma*RV_sigma;
    }

    //std::string dim_name[5] = {"E(B-V)", "DM", "Mr", "FeH", "R_V"};

    double min[2] = {minEBV, DM_min};
    double max[2] = {7., DM_max};
    unsigned int N_bins[2] = {700, 120};
    TRect rect(min, max, N_bins);

    if(gatherSurfs) {
        img_stack.resize(params.N_stars);
        img_stack.set_rect(rect);
    }
    TImgWriteBuffer *imgBuffer = NULL;
    if(saveSurfs) { imgBuffer = new TImgWriteBuffer(rect, params.N_stars); }

    unsigned int max_attempts = 3;
    unsigned int N_steps = options.steps;
    unsigned int N_samplers = options.samplers;
    unsigned int N_runs = options.N_runs;
    unsigned int ndim;

    if(params.vary_RV) { ndim = 5; } else { ndim = 4; }

    double *GR = new double[ndim];
    double GR_threshold = 1.1;

    TNullLogger logger;
    TAffineSampler<TMCMCParams, TNullLogger>::pdf_t f_pdf = &logP_indiv_simple_emp;
    TAffineSampler<TMCMCParams, TNullLogger>::rand_state_t f_rand_state = &gen_rand_state_indiv_emp;

    timespec t_start, t_write, t_end;

    if(verbosity >= 1) {
        std::cout << std::endl;
    }

    unsigned int N_nonconv = 0;

    TChainWriteBuffer chainBuffer(ndim, 100, params.N_stars);
    std::stringstream group_name;
    group_name << "/" << stellar_data.pix_name;

    for(size_t n=0; n<params.N_stars; n++) {
        params.idx_star = n;

        clock_gettime(CLOCK_MONOTONIC, &t_start);

        if(verbosity >= 2) {
            std::cout << "Star #" << n+1 << " of " << params.N_stars << std::endl;
            std::cout << "====================================" << std::endl;

            std::cout << "mags = ";
            for(unsigned int i=0; i<NBANDS; i++) {
                    std::cout << std::setprecision(4) << params.data->star[n].m[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "errs = ";
            for(unsigned int i=0; i<NBANDS; i++) {
                    std::cout << std::setprecision(3) << params.data->star[n].err[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "maglimit = ";
            for(unsigned int i=0; i<NBANDS; i++) {
                    std::cout << std::setprecision(3) << params.data->star[n].maglimit[i] << " ";
            }
            std::cout << std::endl << std::endl;
        }

        //std::cerr << "# Setting up sampler" << std::endl;
        TParallelAffineSampler<TMCMCParams, TNullLogger> sampler(f_pdf, f_rand_state, ndim, N_samplers*ndim, params, logger, N_runs);
        sampler.set_scale(1.5);
        sampler.set_replacement_bandwidth(0.30);
        sampler.set_replacement_accept_bias(1.e-5);
        sampler.set_sigma_min(0.02);

        //std::cerr << "# Burn-in" << std::endl;

        // Burn-in

        // Round 1 (3/6)
        sampler.step_MH(N_steps*(1./6.), false);
        sampler.step(N_steps*(2./6.), false, 0., options.p_replacement);

        if(verbosity >= 2) {
            std::cout << std::endl;
            std::cout << "scale: (";
            std::cout << std::setprecision(2);
            for(int k=0; k<sampler.get_N_samplers(); k++) {
                std::cout << sampler.get_sampler(k)->get_scale() << ((k == sampler.get_N_samplers() - 1) ? "" : ", ");
            }
        }

        // Remove spurious modes
        sampler.set_replacement_accept_bias(1.e-2);
        int N_steps_biased = N_steps*(1./6.);
        if(N_steps_biased > 20) { N_steps_biased = 20; }
        sampler.step(N_steps_biased, false, 0., 1.);

        sampler.tune_stretch(6, 0.30);
        sampler.tune_MH(6, 0.30);

        if(verbosity >= 2) {
            std::cout << ") -> (";
            for(int k=0; k<sampler.get_N_samplers(); k++) {
                std::cout << sampler.get_sampler(k)->get_scale() << ((k == sampler.get_N_samplers() - 1) ? "" : ", ");
            }
            std::cout << ")" << std::endl;
        }

        // Round 2 (3/6)
        sampler.set_replacement_accept_bias(0.);
        sampler.step_MH(N_steps*(1./6.), false);
        sampler.step(N_steps*(2./6.), false, 0., options.p_replacement);

        if(verbosity >= 2) {
            std::cout << "scale: (";
            std::cout << std::setprecision(2);
            for(int k=0; k<sampler.get_N_samplers(); k++) {
                std::cout << sampler.get_sampler(k)->get_scale() << ((k == sampler.get_N_samplers() - 1) ? "" : ", ");
            }
        }

        sampler.tune_stretch(6, 0.30);
        sampler.tune_MH(6, 0.30);

        if(verbosity >= 2) {
            std::cout << ") -> (";
            for(int k=0; k<sampler.get_N_samplers(); k++) {
                std::cout << sampler.get_sampler(k)->get_scale() << ((k == sampler.get_N_samplers() - 1) ? "" : ", ");
            }
            std::cout << ")" << std::endl;
            std::cout << std::endl;
        }

        sampler.clear();

        //std::cerr << "# Main run" << std::endl;

        // Main run
        bool converged = false;
        size_t attempt;
        for(attempt = 0; (attempt < max_attempts) && (!converged); attempt++) {
            sampler.step((1<<attempt)*N_steps, true, 0., options.p_replacement);
            //sampler.step_MH((1<<attempt)*N_steps*(1./3.), true);

            converged = true;
            sampler.get_GR_diagnostic(GR);
            for(size_t i=0; i<ndim; i++) {
                if(GR[i] > GR_threshold) {
                    converged = false;
                    if(attempt != max_attempts-1) {
                        sampler.clear();
                        //logger.clear();
                    }
                    break;
                }
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &t_write);

        // Compute evidence
        TChain chain = sampler.get_chain();
        double lnZ_tmp = chain.get_ln_Z_harmonic(true, 10., 0.25, 0.05);
        //if(isinf(lnZ_tmp)) { lnZ_tmp = neg_inf_replacement; }

        // Save thinned chain
        chainBuffer.add(chain, converged, lnZ_tmp, GR);

        // Save binned p(DM, EBV) surface
        if(gatherSurfs) {
            chain.get_image(*(img_stack.img[n]), rect, 0, 1, true, 1.0, 1.0, 30., true);
        }

        lnZ.push_back(lnZ_tmp);
        conv.push_back(converged);

        clock_gettime(CLOCK_MONOTONIC, &t_end);

        if(verbosity >= 2) {
            sampler.print_stats();
            std::cout << std::endl;
        }

        if(!converged) {
            N_nonconv++;
            if(verbosity >= 2) {
                std::cout << "# Failed to converge." << std::endl;
            }
        }

        if(verbosity >= 2) {
            std::cout << "# Number of steps: " << (1<<(attempt-1))*N_steps << std::endl;
            std::cout << "# ln Z: " << lnZ.back() << std::endl;
            std::cout << "# Time elapsed: " << std::setprecision(2) << (t_end.tv_sec - t_start.tv_sec) + 1.e-9*(t_end.tv_nsec - t_start.tv_nsec) << " s" << std::endl;
            std::cout << "# Sample time: " << std::setprecision(2) << (t_write.tv_sec - t_start.tv_sec) + 1.e-9*(t_write.tv_nsec - t_start.tv_nsec) << " s" << std::endl;
            std::cout << "# Write time: " << std::setprecision(2) << (t_end.tv_sec - t_write.tv_sec) + 1.e-9*(t_end.tv_nsec - t_write.tv_nsec) << " s" << std::endl << std::endl;
        }
    }

    // Smooth the individual stellar surfaces along E(B-V) axis, with
    // kernel that varies with E(B-V).
    if(EBV_smoothing.get_pct_smoothing_max() > 0.) {
        std::vector<double> sigma_pix;
        EBV_smoothing.calc_pct_smoothing(stellar_data.nside, min[0], max[0], N_bins[0], sigma_pix);
        for(int i=0; i<sigma_pix.size(); i++) { sigma_pix[i] *= (double)i; }
        img_stack.smooth(sigma_pix);
    }

    if(saveSurfs) {
        for(int n=0; n<params.N_stars; n++) {
            imgBuffer->add(*(img_stack.img[n]));
        }
    }

    chainBuffer.write(out_fname, group_name.str(), "stellar chains");
    if(saveSurfs) { imgBuffer->write(out_fname, group_name.str(), "stellar pdfs"); }

    if(verbosity >= 1) {
        if(verbosity >= 2) {
            std::cout << "====================================" << std::endl;
            std::cout << std::endl;
        }
        std::cout << "# Failed to converge " << N_nonconv << " of " << params.N_stars << " times (" << std::setprecision(2) << 100.*(double)N_nonconv/(double)(params.N_stars) << " %)." << std::endl;
        if(verbosity >= 2) {
            std::cout << std::endl;
            std::cout << "====================================" << std::endl << std::endl;
        }
    }

    if(imgBuffer != NULL) { delete imgBuffer; }
    delete[] GR;
}


#ifdef _USE_PARALLEL_TEMPERING__
// Sample individual star using parallel tempering
void sample_indiv_emp_pt(
        std::string &out_fname,
        TMCMCOptions &options,
        TGalacticLOSModel& galactic_model,
        TStellarModel& stellar_model,
        TExtinctionModel& extinction_model,
        TEBVSmoothing& EBV_smoothing,
        TStellarData& stellar_data,
        TImgStack& img_stack,
        std::vector<bool> &conv,
        std::vector<double> &lnZ,
        double RV_mean, double RV_sigma, double minEBV,
        const bool saveSurfs, const bool gatherSurfs,
        const bool use_priors, int verbosity)
{
    // Parameters must be consistent - cannot save surfaces without gathering them
    assert(!(saveSurfs & (!gatherSurfs)));

    unsigned int N_DM = 20;
    double DM_min = 4.;
    double DM_max = 19.;
    TMCMCParams params(&galactic_model, NULL, &stellar_model, &extinction_model, &stellar_data, N_DM, DM_min, DM_max);
    params.EBV_floor = minEBV;
    params.use_priors = use_priors;

    params.RV_mean = RV_mean;
    if(RV_sigma > 0.) {
        params.vary_RV = true;
        params.RV_variance = RV_sigma*RV_sigma;
    }

    //std::string dim_name[5] = {"E(B-V)", "DM", "Mr", "FeH", "R_V"};

    double min[2] = {minEBV, DM_min};
    double max[2] = {7., DM_max};
    unsigned int N_bins[2] = {700, 120};
    TRect rect(min, max, N_bins);

    if(gatherSurfs) {
        img_stack.resize(params.N_stars);
        img_stack.set_rect(rect);
    }
    TImgWriteBuffer *imgBuffer = NULL;
    if(saveSurfs) {
        imgBuffer = new TImgWriteBuffer(rect, params.N_stars);
    }

    unsigned int max_attempts = 3;
    unsigned int N_steps = options.steps;
    unsigned int N_samplers = options.samplers;
    unsigned int N_runs = options.N_runs;
    unsigned int ndim;

    if(params.vary_RV) { ndim = 5; } else { ndim = 4; }

    double *GR = new double[ndim];
    double GR_threshold = 1.1;

    // Parallel tempering parameters
    cppsampler::pdensity ln_prior = [](double* x) { return 0.; };
    cppsampler::pdensity lnL = [&params, ndim](double* x) {
        double res = logP_indiv_simple_emp(x, ndim, params);
        // std::cerr << "ln p( ";
        // for(int k=0; k<ndim; k++) {
        // 	std::cerr << x[k] << " ";
        // }
        // std::cerr << ") = " << res << std::endl;
        return res;
    };

    int n_temperatures = 4;
    double temperature_spacing = 5.;

    // Loop through the stars
    timespec t_start, t_write, t_end;

    if(verbosity >= 1) {
        std::cout << std::endl;
    }

    unsigned int N_nonconv = 0;

    TChainWriteBuffer chainBuffer(ndim, 100, params.N_stars);
    std::stringstream group_name;
    group_name << "/" << stellar_data.pix_name;

    for(size_t n=0; n<params.N_stars; n++) {
        params.idx_star = n;

        clock_gettime(CLOCK_MONOTONIC, &t_start);

        if(verbosity >= 2) {
            std::cout << "Star #" << n+1 << " of " << params.N_stars << std::endl;
            std::cout << "====================================" << std::endl;

            std::cout << "mags = ";
            for(unsigned int i=0; i<NBANDS; i++) {
                std::cout << std::setprecision(4) << params.data->star[n].m[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "errs = ";
            for(unsigned int i=0; i<NBANDS; i++) {
                std::cout << std::setprecision(3) << params.data->star[n].err[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "maglimit = ";
            for(unsigned int i=0; i<NBANDS; i++) {
                std::cout << std::setprecision(3) << params.data->star[n].maglimit[i] << " ";
            }
            std::cout << std::endl << std::endl;
        }

        // Set up the parallel tempering sampler
        cppsampler::PTSampler pt_sampler(lnL, ln_prior, ndim,
                                         n_temperatures, temperature_spacing);

        // Seed the sampler
        gsl_rng *r;
        seed_gsl_rng(&r);

        cppsampler::vector_generator rand_state = [ndim, &params, &r]() {
            cppsampler::shared_vector x0 = std::make_shared<std::vector<double> >(ndim, 0.);
            gen_rand_state_indiv_emp(x0->data(), ndim, r, params);
            return x0;
        };

        pt_sampler.set_state(rand_state);

        gsl_rng_free(r);

        // Burn-in
        cppsampler::PTTuningParameters tune_params;
        tune_params.n_rounds = 10;
        tune_params.n_swaps_per_round = 20;
        tune_params.n_steps_per_swap = 20;
        tune_params.step_accept = 0.25;

        pt_sampler.tune_all(tune_params);

        int n_steps_per_swap = 2;
        int n_swaps = N_steps / n_steps_per_swap;

        pt_sampler.step_multiple(n_swaps/4, n_steps_per_swap);

        tune_params.n_rounds = 20;
        tune_params.n_swaps_per_round = 20;
        tune_params.n_steps_per_swap = 20;

        pt_sampler.tune_all(tune_params);

        pt_sampler.step_multiple(n_swaps/4, n_steps_per_swap);

        std::cerr << std::endl
                  << "MH acceptance: "
                  << 100. * pt_sampler.get_sampler(0)->accept_frac()
                  << "%"
                  << std::endl << std::endl;

        std::cerr << std::endl
                  << "swap acceptance: "
                  << 100. * pt_sampler.swap_accept_frac()
                  << "%"
                  << std::endl << std::endl;

        std::cerr << "beta = ";
        for(auto beta : *pt_sampler.get_beta()) {
            std::cerr << beta << " ";
        }
        std::cerr << std::endl;

        pt_sampler.clear_chain();

        // Main sampling phase
        pt_sampler.step_multiple(n_swaps, n_steps_per_swap);

        std::cerr << std::endl
                  << "MH acceptance: "
                  << 100. * pt_sampler.get_sampler(0)->accept_frac()
                  << "%"
                  << std::endl << std::endl;

        std::cerr << std::endl
                  << "swap acceptance: "
                  << 100. * pt_sampler.swap_accept_frac()
                  << "%"
                  << std::endl << std::endl;

        clock_gettime(CLOCK_MONOTONIC, &t_write);

        std::cerr << "done sampling." << std::endl;

        // Copy over chain
        std::shared_ptr<const cppsampler::Chain> pt_chain = pt_sampler.get_chain(0);
        TChain chain(ndim, pt_chain->get_length()+1);
        cppsampler::shared_const_vector chain_el = pt_chain->get_elements();
        cppsampler::shared_const_vector chain_w = pt_chain->get_weights();
        cppsampler::shared_const_vector chain_lnp = pt_chain->get_lnL();
        for(int k=0; k<chain_w->size(); k++) {
            chain.add_point(
                chain_el->data()+ndim*k,
                chain_lnp->at(k),
                chain_w->at(k)
            );
        }

        std::cerr << "done copying chain." << std::endl;

        // Compute evidence
        // cppsampler::BasicRandGenerator rand_gen;
        // double lnZ_tmp = rand_gen.uniform();
        double lnZ_tmp = chain.get_ln_Z_harmonic(true, 10., 0.25, 0.05);

        std::cerr << "calculated lnZ" << std::endl;

        // Save thinned chain
        bool converged = true; // TODO: calculate convergence and GR diagnostic.
        chainBuffer.add(chain, converged, lnZ_tmp, GR);

        std::cerr << "added to chain buffer." << std::endl;

        // Save binned p(DM, EBV) surface
        if(gatherSurfs) {
            chain.get_image(*(img_stack.img[n]), rect, 0, 1, true, 1.0, 1.0, 30., true);
        }

        std::cerr << "calculated image." << std::endl;

        // Save convergence/goodness-of-fit statistics
        lnZ.push_back(lnZ_tmp);
        conv.push_back(converged);

        clock_gettime(CLOCK_MONOTONIC, &t_end);

        // Report timing
        if(verbosity >= 2) {
            std::cout << "# Number of steps: " << N_steps << std::endl;
            std::cout << "# ln Z: " << lnZ.back() << std::endl;
            std::cout << "# Time elapsed: " << std::setprecision(2) << (t_end.tv_sec - t_start.tv_sec) + 1.e-9*(t_end.tv_nsec - t_start.tv_nsec) << " s" << std::endl;
            std::cout << "# Sample time: " << std::setprecision(2) << (t_write.tv_sec - t_start.tv_sec) + 1.e-9*(t_write.tv_nsec - t_start.tv_nsec) << " s" << std::endl;
            std::cout << "# Write time: " << std::setprecision(2) << (t_end.tv_sec - t_write.tv_sec) + 1.e-9*(t_end.tv_nsec - t_write.tv_nsec) << " s" << std::endl << std::endl;
        }
    }

    // Smooth the individual stellar surfaces along E(B-V) axis, with
    // kernel that varies with E(B-V).
    if(EBV_smoothing.get_pct_smoothing_max() > 0.) {
        std::cerr << "Smoothing images along reddening axis." << std::endl;
        std::vector<double> sigma_pix;
        EBV_smoothing.calc_pct_smoothing(stellar_data.nside, min[0], max[0], N_bins[0], sigma_pix);
        for(int i=0; i<sigma_pix.size(); i++) { sigma_pix[i] *= (double)i; }
        img_stack.smooth(sigma_pix);
    }

    if(saveSurfs) {
        for(int n=0; n<params.N_stars; n++) {
            imgBuffer->add(*(img_stack.img[n]));
        }
    }

    chainBuffer.write(out_fname, group_name.str(), "stellar chains");
    if(saveSurfs) { imgBuffer->write(out_fname, group_name.str(), "stellar pdfs"); }

    std::cerr << "cleaning up." << std::endl;

    if(imgBuffer != NULL) { delete imgBuffer; }
    delete[] GR;
}
#endif // _USE_PARALLEL_TERMPERING


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


void rand_vector(double*const x, double* min, double* max, size_t N, gsl_rng* r) {
    for(size_t i=0; i<N; i++) {
        x[i] = min[i] + gsl_rng_uniform(r) * (max[i] - min[i]);
    }
}

void rand_vector(double*const x, size_t N, gsl_rng* r, double A) {
    for(size_t i=0; i<N; i++) { x[i] = A * gsl_rng_uniform(r); }
}

void rand_gaussian_vector(double*const x, double mu, double sigma, size_t N, gsl_rng* r) {
    for(size_t i=0; i<N; i++) { x[i] = mu + gsl_ran_gaussian_ziggurat(r, sigma); }
}

void rand_gaussian_vector(double*const x, double* mu, double* sigma, size_t N, gsl_rng* r) {
    for(size_t i=0; i<N; i++) { x[i] = mu[i] + gsl_ran_gaussian_ziggurat(r, sigma[i]); }
}
