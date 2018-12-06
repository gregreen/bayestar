
#include "star_exact.h"


/*
 * Pretty-print various objects
 */

void print_float(double x, std::ostream& o,
                 int width=10, int precision=5,
                 std::string pm="") {
    std::stringstream s;
    s << std::fixed << std::setw(width) << std::setprecision(precision) << x;
    o << s.str();
}


void print_matrix(cv::Mat& mat, std::ostream& s,
                  int width=10, int precision=5) {
    for(int j=0; j<mat.rows; j++) {
        for(int k=0; k<mat.cols; k++) {
            print_float(mat.at<floating_t>(j,k), s, width, precision);
        }
        s << std::endl;
    }
}


/*
 * LinearFitParams
 */

LinearFitParams::LinearFitParams(unsigned int n_dim)
    : _n_dim(n_dim), mean(n_dim), inv_cov(n_dim, n_dim),
      chi2(std::numeric_limits<double>::infinity())
{}


/*
 * Grid evaluation of stellar parameters (E, \mu, M_r, [Fe/H])
 */

void star_covariance(TStellarData::TMagnitudes& mags_obs,
                     TExtinctionModel& ext_model,
                     double& inv_cov_00, double& inv_cov_01, double& inv_cov_11,
                     double RV) {
    // Various useful terms
    double inv_sigma2 = 0.;         // 1 / sigma_i^2
    double A_over_sigma2 = 0.;      // A_i / sigma_i^2
    double A2_over_sigma2 = 0.;     // A_i^2 / sigma_i^2

    for(int i=0; i<NBANDS; i++) {
        double A = ext_model.get_A(RV, i);
        double ivar = 1. / (mags_obs.err[i] * mags_obs.err[i]);

        inv_sigma2 += ivar;
        A_over_sigma2 += A * ivar;
        A2_over_sigma2 += A*A * ivar;
    }

    // Set the inverse covariance terms
    inv_cov_00 = inv_sigma2;
    inv_cov_01 = A_over_sigma2;
    inv_cov_11 = A2_over_sigma2;
}

void star_max_likelihood(TSED& mags_model, TStellarData::TMagnitudes& mags_obs,
                         TExtinctionModel& ext_model,
                         double inv_cov_00, double inv_cov_01, double inv_cov_11,
                         double& mu, double& E, double& chi2,
                         double RV) {
    // Various useful terms
    double dm_over_sigma2 = 0.;     // (m_i - M_i) / sigma_i^2
    double dm_A_over_sigma2 = 0.;   // (m_i - M_i) A_i / sigma_i^2

    for(int i=0; i<NBANDS; i++) {
        double A = ext_model.get_A(RV, i);
        double ivar = 1. / (mags_obs.err[i] * mags_obs.err[i]);
        double dm = mags_obs.m[i] - mags_model.absmag[i];

        dm_over_sigma2 += dm * ivar;
        dm_A_over_sigma2 += dm * A * ivar;
    }

    double mu_0 = dm_over_sigma2 / inv_cov_00;
    double E_0 = dm_A_over_sigma2 / inv_cov_11;

    double C_01 = inv_cov_01 / inv_cov_00;
    double C_10 = inv_cov_01 / inv_cov_11;

    // Compute maximum-likelihood (mu, E) using the formula
    //   (1 + C) (mu E)^T = (mu_0 E_0)^T
    double C_det_inv = 1. / (1. - C_01 * C_10);
    mu = C_det_inv * (mu_0 - C_01 * E_0);
    E  = C_det_inv * (E_0  - C_10 * mu_0);

    // Compute best chi^2 by plugging in ML (mu, E)
    chi2 = 0.;

    for(int i=0; i<NBANDS; i++) {
        double A = ext_model.get_A(RV, i);
        double ivar = 1. / (mags_obs.err[i] * mags_obs.err[i]);
        double dm = mags_obs.m[i] - mags_model.absmag[i];

        double delta = (dm - E * A - mu);

        chi2 += delta*delta * ivar;
    }
}

// Calculate the chi^2 of a given stellar fit, parameterized by
// (spectral energy distribution, distance modulus, reddening),
// with a given reddening -> extinction mapping.
double calc_star_chi2(TStellarData::TMagnitudes& mags_obs,
                      TExtinctionModel& ext_model,
                      TSED& mags_model,
                      double mu, double E, double RV) {
    double chi2 = 0.;

    for(int i=0; i<NBANDS; i++) {
        double A = ext_model.get_A(RV, i);
        double ivar = 1. / (mags_obs.err[i] * mags_obs.err[i]);
        double dm = mags_obs.m[i] - mags_model.absmag[i];

        double delta = (dm - E * A - mu);

        chi2 += delta*delta * ivar;
    }

    return chi2;
}


std::shared_ptr<LinearFitParams> star_max_likelihood(
        TSED& mags_model,
        TStellarData::TMagnitudes& mags_obs,
        TExtinctionModel& ext_model,
        double RV) {
    // Create empty return class
    std::shared_ptr<LinearFitParams> ret = std::make_shared<LinearFitParams>(2);

    // Various useful terms
    double inv_sigma2 = 0.;         // 1 / sigma_i^2
    double A_over_sigma2 = 0.;      // A_i / sigma_i^2
    double A2_over_sigma2 = 0.;     // A_i^2 / sigma_i^2
    double dm_over_sigma2 = 0.;     // (m_i - M_i) / sigma_i^2
    double dm_A_over_sigma2 = 0.;   // (m_i - M_i) A_i / sigma_i^2

    for(int i=0; i<NBANDS; i++) {
        double A = ext_model.get_A(RV, i);
        double ivar = 1. / (mags_obs.err[i] * mags_obs.err[i]);
        double dm = mags_obs.m[i] - mags_model.absmag[i];

        inv_sigma2 += ivar;
        A_over_sigma2 += A * ivar;
        A2_over_sigma2 += A*A * ivar;
        dm_over_sigma2 += dm * ivar;
        dm_A_over_sigma2 += dm * A * ivar;
    }

    double mu_0 = dm_over_sigma2 / inv_sigma2;
    double E_0 = dm_A_over_sigma2 / A2_over_sigma2;

    double C_01 = A_over_sigma2 / inv_sigma2;
    double C_10 = A_over_sigma2 / A2_over_sigma2;

    // Compute maximum-likelihood (mu, E) using the formula
    //   (1 + C) (mu E)^T = (mu_0 E_0)^T
    double C_det_inv = 1. / (1. - C_01 * C_10);
    double mu = C_det_inv * (mu_0 - C_01 * E_0);
    double E  = C_det_inv * (E_0  - C_10 * mu_0);
    ret->mean(0) = mu;
    ret->mean(1) = E;

    // Compute inverse covariance
    ret->inv_cov(0,0) = inv_sigma2;
    ret->inv_cov(0,1) = A_over_sigma2;
    ret->inv_cov(1,0) = A_over_sigma2;
    ret->inv_cov(1,1) = A2_over_sigma2;

    // Compute best chi^2 by plugging in ML (mu, E)
    double chi2 = 0.;

    for(int i=0; i<NBANDS; i++) {
        double A = ext_model.get_A(RV, i);
        double ivar = 1. / (mags_obs.err[i] * mags_obs.err[i]);
        double dm = mags_obs.m[i] - mags_model.absmag[i];

        double delta = (dm - E * A - mu);

        chi2 += delta*delta * ivar;
    }

    ret->chi2 = chi2;

    return ret;
}


void gaussian_filter(std::shared_ptr<LinearFitParams> p,
                     TRect& grid, cv::Mat& img,
                     double n_sigma, int min_width) {
    // Determine sigma along each axis
    double det = p->inv_cov(0,0) * p->inv_cov(1,1) - p->inv_cov(0,1) * p->inv_cov(1,0) + 1.e-5;
    double sigma[2] = {
        sqrt(p->inv_cov(1,1) / det),
        sqrt(p->inv_cov(0,0) / det)
    };

    // Determine dimensions of filter
    int width[2];

    for(unsigned int i=0; i<2; i++) {
        width[i] = std::max(min_width, (int)(ceil(sigma[i] / grid.dx[i])));
    }

    // std::cerr << "width = (" << width[0] << ", " << width[1] << ")" << std::endl;

    // std::cerr << "initializing img" << std::endl;
    img = cv::Mat::zeros(2*width[0]+1, 2*width[1]+1, CV_FLOATING_TYPE);

    // Evaluate filter at each point
    // std::cerr << "evaluating image" << std::endl;
    double dx, dy;
    double cxx, cxy, cyy;
    for(int i=0; i<(int)(2*width[0]+1); i++) {
        dx = (i - width[0]) * grid.dx[0];
        cxx = p->inv_cov(0,0) * dx*dx;

        for(int j=0; j<2*width[1]+1; j++) {
            dy = (j - width[1]) * grid.dx[1];
            cxy = p->inv_cov(0,1) * dx*dy;
            cyy = p->inv_cov(1,1) * dy*dy;

            // std::cerr << " (" << i << ", " << j << ")" << std::endl;
            // std::cerr << " width = (" << 2*width[0]+1 << ", " << 2*width[1]+1 << ")" << std::endl;

            img.at<floating_t>(i, j) += exp(-0.5 * (cxx + 2*cxy + cyy));
        }
    }
    // std::cerr << "done creating filter" << std::endl;
}


void gaussian_filter(double inv_cov_00, double inv_cov_01, double inv_cov_11,
                     TRect& grid, cv::Mat& img,
                     double n_sigma, int min_width,
                     double add_diagonal=-1.,
                     int subsample=5, int verbosity=0) {
    // Add extra smoothing along each axis
    if(add_diagonal > 0.) {
        double diag[2] = {
            add_diagonal*grid.dx[0],
            add_diagonal*grid.dx[1]
        };

        // std::cerr << "diagonal = (" << diag[0] << ", " << diag[1] << ")" << std::endl;

        double det = inv_cov_00 * inv_cov_11 - inv_cov_01 * inv_cov_01;
        double cov_00 = inv_cov_11 / det;
        double cov_11 = inv_cov_00 / det;
        double cov_01 = -inv_cov_01 / det;

        cov_00 += diag[0] * diag[0];
        cov_11 += diag[1] * diag[1];

        det = cov_00 * cov_11 - cov_01 * cov_01;

        inv_cov_00 = cov_11 / det;
        inv_cov_11 = cov_00 / det;
        inv_cov_01 = -cov_01 / det;
    }

    // Determine sigma along each axis
    double det = inv_cov_00 * inv_cov_11 - inv_cov_01 * inv_cov_01 + 1.e-5;
    double sigma[2] = {
        sqrt(inv_cov_11 / det),// + diag[0]*diag[0]),
        sqrt(inv_cov_00 / det) // + diag[1]*diag[1])
    };


    // Determine dimensions of filter
    int width[2];

    for(unsigned int i=0; i<2; i++) {
        width[i] = std::max(min_width, (int)(ceil(n_sigma * sigma[i] / grid.dx[i])));
    }

    if(verbosity >= 2) {
        std::cerr << "sigma -> (" << sigma[0] << ", " << sigma[1] << ")" << std::endl;
        std::cerr << "width = (" << width[0] << ", " << width[1] << ")" << std::endl;
    }

    // std::cerr << "initializing img" << std::endl;
    int w = 2 * width[0] + 1;
    int h = 2 * width[1] + 1;

    // Size of sub-sampled image
    int w_sub = subsample * w;
    int h_sub = subsample * h;

    // Center of sub-sampled image
    double w0 = 0.5 * (double)(w_sub - 1);
    double h0 = 0.5 * (double)(h_sub - 1);

    // Create zeroed sub-sampled image
    cv::Mat img_sub = cv::Mat::zeros(w_sub, h_sub, CV_FLOATING_TYPE);

    // std::cerr << std::endl
    //           << "inv_cov_?? : "
    //           << inv_cov_00 << " "
    //           << inv_cov_01 << " "
    //           << inv_cov_11 << std::endl
    //           << std::endl;

    // Evaluate filter at each point
    // std::cerr << "evaluating image" << std::endl;
    // double sum = 0;
    double dx, dy;
    double cxx, cxy, cyy;
    for(int i=0; i<w_sub; i++) {
        dx = ((double)i - w0) * grid.dx[0] / (double)subsample;
        cxx = inv_cov_00 * dx*dx;

        for(int j=0; j<subsample*h; j++) {
            dy = ((double)j - h0) * grid.dx[1] / (double)subsample;
            cxy = inv_cov_01 * dx*dy;
            cyy = inv_cov_11 * dy*dy;

            // std::cerr << " (" << i << ", " << j << ")" << std::endl;
            // std::cerr << " width = (" << 2*width[0]+1 << ", " << 2*width[1]+1 << ")" << std::endl;
            double weight = exp(-0.5 * (cxx + 2*cxy + cyy));
            img_sub.at<floating_t>(i, j) += weight;
            // sum += weight;
        }
    }
    // std::cerr << "done creating filter" << std::endl;

    // img.resize(h, w)
    cv::Mat img_down;
    cv::resize(img_sub, img_down, cv::Size(h, w), 0, 0, cv::INTER_AREA);

    img = img_down;
    img /= img.at<floating_t>(width[0], width[1]);

    // std::cerr << "size = " << img_sub.rows << ", " << img_sub.cols << std::endl;
    // std::cerr << "size = " << img_down.rows << ", " << img_down.cols << std::endl;
}




double integrate_ML_solution(
        TStellarModel& stellar_model,
        TGalacticLOSModel& los_model,
        TStellarData::TMagnitudes& mags_obs,
        TExtinctionModel& ext_model,
        TImgStack& img_stack,
        unsigned int img_idx,
        bool save_gaussians,
        std::vector<TDMESaveData>& fit_centers,
        std::vector<float>& fit_icov,
        bool use_priors,
        bool use_gaia,
        double RV, int verbosity)
{
    //
    TSED sed;
    unsigned int N_Mr = stellar_model.get_N_Mr();
    unsigned int N_FeH = stellar_model.get_N_FeH();
    double Mr, FeH;

    // std::cerr << "N_Mr = " << N_Mr << std::endl;
    // std::cerr << "N_FeH = " << N_FeH << std::endl;

    // Calculate covariance of ML solution for (mu, E)
    double inv_cov_00, inv_cov_01, inv_cov_11;

    star_covariance(mags_obs, ext_model,
                    inv_cov_00, inv_cov_01, inv_cov_11,
                    RV);

    // Set image of p(mu, E) to zero
    if (!img_stack.initialize_to_zero(img_idx)) {
        std::cerr << "Failed to initialize image to zero!" << std::endl;
    }

    // Calculate ML (E, mu), chi^2 and prior for each (Mr, [Fe/H]) pair
    std::vector<double> E_ML;
    std::vector<double> mu_ML;
    std::vector<double> chi2_ML;
    std::vector<double> prior_ML;

    unsigned int N_reserve = N_Mr*N_FeH + 1;
    E_ML.reserve(N_reserve);
    mu_ML.reserve(N_reserve);
    chi2_ML.reserve(N_reserve);
    prior_ML.reserve(N_reserve);

    for(int Mr_idx=0; Mr_idx<N_Mr; Mr_idx++) {
        for(int FeH_idx=0; FeH_idx<N_FeH; FeH_idx++) {
            // Look up model absolute magnitudes of this stellar type
            bool success = stellar_model.get_sed(
                Mr_idx, FeH_idx,
                sed, Mr, FeH);
            if(!success) {
                std::cerr << "SED (" << Mr_idx << ", " << FeH_idx
                          << ") not in library!" << std::endl;
                continue;
            }

            // std::cerr << "Mr check: " << Mr << " == " << sed.absmag[1] << std::endl;

            // Calculate max. likelihood solution for (mu, E) given this fixed stellar type
            double mu, E, chi2;
            star_max_likelihood(sed, mags_obs, ext_model,
                                inv_cov_00, inv_cov_01, inv_cov_11,
                                mu, E, chi2,
                                RV);
            
            if(use_gaia) {
                chi2 += chi2_parallax(mu, mags_obs.pi, mags_obs.pi_err);
            }
            
            double prior = 0.;
            if(use_priors) {
                prior = los_model.log_prior_emp(mu, Mr, FeH)
                        + stellar_model.get_log_lf(Mr);
                if(std::isnan(prior) || std::isinf(prior)) {
                    prior = -std::numeric_limits<double>::infinity();
                }
            }
            
            if(std::isnan(chi2) || std::isinf(chi2)) {
                chi2 = std::numeric_limits<double>::infinity();
            }
            
            // std::cerr << "p(FeH = " << FeH << ") = " << prior << std::endl;

            E_ML.push_back(E);
            mu_ML.push_back(mu);
            chi2_ML.push_back(chi2);
            prior_ML.push_back(prior);
        }
    }
    
    // Calculate best prior and likelihood (minimum chi^2)
    double prior_max = 0.;
    if(use_priors) {
        prior_max = *std::max_element(prior_ML.begin(), prior_ML.end());
    }

    double chi2_min = *std::min_element(chi2_ML.begin(), chi2_ML.end());
    
    if(verbosity >= 2) {
        std::cerr << "prior_max = " << prior_max << std::endl;
        std::cerr << "chi2_min = " << chi2_min << std::endl;
    }

    assert( E_ML.size() == mu_ML.size() );
    assert( chi2_ML.size() == mu_ML.size() );
    assert( prior_ML.size() == mu_ML.size() );

    std::vector<int32_t> save_list; // List of Gaussians to save
    save_list.reserve(mu_ML.size());
    double delta_logp_threshold = -8.;
    
    // Interpolation information
    unsigned int img_idx0, img_idx1;
    double a0, a1;
    
    std::vector<double> log_p_all;
    log_p_all.reserve(mu_ML.size());
    
    double chi2_min_filtered = std::numeric_limits<double>::infinity();
    
    for(int k=0; k<mu_ML.size(); k++) {
        // Calculate ln(p/p_0)
        double log_p = -0.5 * (chi2_ML.at(k) - chi2_min);
        if(use_priors) {
            log_p += prior_ML.at(k) - prior_max;
        }
        
        if(save_gaussians) {
            log_p_all.push_back(log_p);
        }
        
        // Add single point to image at ML solution location (E, mu)
        bool in_bounds = img_stack.rect->get_interpolant(
            E_ML.at(k), mu_ML.at(k),
            img_idx0, img_idx1,
            a0, a1
        );

        // bool in_bounds = img_stack.rect->get_index(E_ML.at(k), mu_ML.at(k), img_idx0, img_idx1);

        if(in_bounds) {
            double p = exp(log_p);
            //if((a0 < 0.) || (a0 > 1.) || (a1 < 0.) || (a1 > 1.)) {
            //    std::cerr << "(a0, a1) = (" << a0 << ", " << a1 << ")" << std::endl;
            //}
            //if((p < 0.) || std::isnan(p)) {
            //    std::cerr << "p = " << p << std::endl;
            //}

            // Interpolate between bins
            img_stack.img[img_idx]->at<floating_t>(img_idx0, img_idx1) += (1-a0) * (1-a1) * p;
            img_stack.img[img_idx]->at<floating_t>(img_idx0+1, img_idx1) += a0 * (1-a1) * p;
            img_stack.img[img_idx]->at<floating_t>(img_idx0, img_idx1+1) += (1-a0) * a1 * p;
            img_stack.img[img_idx]->at<floating_t>(img_idx0+1, img_idx1+1) += a0 * a1 * p;
            
            if(chi2_ML.at(k) < chi2_min_filtered) {
                chi2_min_filtered = chi2_ML.at(k);
            }
            // Choose whether or not to save this point
            //if(log_p > delta_logp_threshold) {
            //    save_list.push_back(k);
            //}
        }
    }
    
    // TODO: Move this to after save_list is determined?
    // Filtered version of chi^2, excluding out-of-bounds (mu,E) values
    //std::vector<double> chi2_filtered;
    //chi2_filtered.reserve(save_list.size());
    //for(auto i : save_list) {
    //    chi2_filtered.push_back(chi2_ML.at(i));
    //}
    //double chi2_min_filtered = 1.e9;
    //if(save_list.size()) {
    //    chi2_min_filtered = *std::min_element(chi2_filtered.begin(), chi2_filtered.end());
    //}
    
    //for(int j=0; j<img_stack.rect->N_bins[0]; j++) {
    //    for(int k=0; k<img_stack.rect->N_bins[1]; k++) {
    //        double val_tmp = img_stack.img[img_idx]->at<floating_t>(j, k);
    //        if((val_tmp < 0.) || std::isnan(val_tmp)) {
    //            std::cerr << "img[" << j << ", " << k << "] = " << val_tmp << std::endl;
    //        }
    //    }
    //}

    // Smooth PDF with covariance of the ML solution
    cv::Mat cov_img;
    gaussian_filter(inv_cov_11, inv_cov_01, inv_cov_00,
                    *(img_stack.rect), cov_img, 5, 2, 1.0,
                    5, verbosity);
    
    //print_matrix(cov_img, std::cerr);

    cv::Mat filtered_img = cv::Mat::zeros(
        img_stack.rect->N_bins[0],
        img_stack.rect->N_bins[1],
        CV_FLOATING_TYPE
    );
    cv::filter2D(*img_stack.img[img_idx], filtered_img, CV_FLOATING_TYPE, cov_img);
    *img_stack.img[img_idx] = cv::max(filtered_img, 0.); // Copy over matrix, setting negative values to zero
    
    //for(int j=0; j<img_stack.rect->N_bins[0]; j++) {
    //    for(int k=0; k<img_stack.rect->N_bins[1]; k++) {
    //        double val_tmp = img_stack.img[img_idx]->at<floating_t>(j, k);
    //        if(val_tmp < 0.) {
    //            std::cerr << "img'[" << j << ", " << k << "] = " << val_tmp << std::endl;
    //        }
    //    }
    //}
    
    // Stores chosen Gaussians
    if(save_gaussians) {
        // Construct discrete distribution
        std::vector<double> p_all;
        p_all.reserve(mu_ML.size());
        double lnp_max = *std::max_element(
            log_p_all.begin(),
            log_p_all.end()
        );
        for(auto lnp : log_p_all) {
            p_all.push_back(std::exp(lnp-lnp_max));
            //std::cout << lnp - lnp_max << std::endl;
        }
        std::discrete_distribution<uint16_t> d(
            p_all.begin(),
            p_all.end()
        );
        
        // Create pseudorandom number generator
        std::mt19937 r;
        std::random_device rd;
        r.seed(rd());
        
        // Draw n samples
        int n_samples = 100;
        for(int i=0; i<n_samples; i++) {
            save_list.push_back(d(r));
        }
        
        //if(verbosity >= 2) {
        //    std::cerr << "Saving " << save_list.size()
        //              << " Gaussians." << std::endl;
        //}
        
        fit_centers.reserve(save_list.size());
        
        for(int32_t i : save_list) {
            fit_centers.push_back({
                (float)(mu_ML.at(i)),
                (float)(E_ML.at(i)),
                (float)(-0.5*(chi2_ML.at(i) - chi2_min_filtered)),
                (float)(prior_ML.at(i) - prior_max)
            });
        }
        
        fit_icov.push_back(inv_cov_00); // dm,dm
        fit_icov.push_back(inv_cov_01); // dm,E
        fit_icov.push_back(inv_cov_11); // E,E
    }

    // Return mininum chi^2 / passband
    int n_passbands = 0;
    for(int i=0; i<NBANDS; i++) {
        if(std::isnan(mags_obs.err[i]) || std::isinf(mags_obs.err[i]) || (mags_obs.err[i] > 1.e9)) {
            continue;
        }
        n_passbands++;
    }
    if(use_gaia) {
        if(mags_obs.pi_err < mags_obs.pi) {
            n_passbands++;
        }
    }

    if(verbosity >= 2) {
        std::cerr << "# of passbands: " << n_passbands
                  << std::endl;
        std::cerr << "chi^2 / passband: "
                  << chi2_min_filtered / n_passbands
                  << std::endl;
    }

    return chi2_min_filtered / n_passbands;
}

void grid_eval_stars(TGalacticLOSModel& los_model,
                     TExtinctionModel& ext_model,
                     TStellarModel& stellar_model,
                     TStellarData& stellar_data,
                     TEBVSmoothing& EBV_smoothing,
                     TImgStack& img_stack,
                     std::vector<double>& chi2,
                     bool save_surfs,
                     bool save_gaussians,
                     std::string out_fname,
                     bool use_priors,
                     bool use_gaia,
                     double RV, int verbosity) {
    // Timing
    auto t_start = std::chrono::steady_clock::now();

    // Set up image stack for stellar PDFs
    double min[2] = {-0.2,  3.75};   // (E, DM)
	double max[2] = { 7.2, 19.25};  // (E, DM)
	unsigned int N_bins[2] = {740, 124};
	TRect rect(min, max, N_bins);
    img_stack.set_rect(rect);

    // Loop over all stars and evaluate PDFs on grid in (mu, E)
    int n_stars = stellar_data.star.size();
    chi2.clear();
    
    // Name of group to save data to
    std::stringstream group_name;
    group_name << "/" << stellar_data.pix_name;
    
    // Create empty vector of stellar data to save
    std::vector<std::vector<TDMESaveData> > fit_centers;
    std::vector<float> fit_icovs;
    fit_centers.reserve(n_stars);
    fit_icovs.reserve(n_stars);

    for(int i=0; i<n_stars; i++) {
        if(verbosity >= 2) {
            std::cerr << "Star " << i+1 << " of " << n_stars << std::endl;
        }
        
        fit_centers.push_back(
            std::vector<TDMESaveData>()
        );

        double chi2_min = integrate_ML_solution(
            stellar_model, los_model,
            stellar_data[i], ext_model,
            img_stack, i,
            save_gaussians,
            fit_centers.at(i),
            fit_icovs,
            use_priors,
            use_gaia,
            RV,
            verbosity
        );
        chi2.push_back(chi2_min);
    }
    
    // Save individual Gaussians for each star
    if(save_gaussians) {
        save_gridstars(
            out_fname,
            group_name.str(),
            "gridstars",
            fit_centers,
            fit_icovs
        );
    }
    
    // Save chi^2/passband for each star
    //std::cerr << "Saving chi^2/passband for stars ..." << std::endl;
	std::unique_ptr<H5::H5File> file = H5Utils::openFile(out_fname);
	if(file) {
        std::unique_ptr<H5::DataSet> chi2_dset = H5Utils::createDataSet(
            *file,
            group_name.str(),
            "star_chi2",
            chi2
        );
        //std::cerr << "chi^2 values written." << std::endl;
    } else {
        std::cerr << "! Failed to open " << out_fname << " !" << std::endl;
    }

    // Crop to correct (E, DM) range
    img_stack.crop(0., 7., 4., 19.);

    // Smooth the individual stellar surfaces along E(B-V) axis, with
	// kernel that varies with E(B-V).
    auto t_smooth = std::chrono::steady_clock::now();

	if(EBV_smoothing.get_pct_smoothing_max() > 0.) {
		std::cerr << "Smoothing images along reddening axis." << std::endl;
		std::vector<double> sigma_pix;
		EBV_smoothing.calc_pct_smoothing(
            stellar_data.nside,
            img_stack.rect->min[0],
            img_stack.rect->max[0],
            img_stack.rect->N_bins[0],
            sigma_pix
        );
		for(int i=0; i<sigma_pix.size(); i++) {
            sigma_pix[i] *= (double)i;
        }
		img_stack.smooth(sigma_pix);
	}
    
    // Normalize PDFs to unity
    img_stack.normalize();

    // Save the PDFs to disk
    auto t_write = std::chrono::steady_clock::now();

    if(save_surfs) {
        TImgWriteBuffer img_buffer(*(img_stack.rect), n_stars);

		for(int n=0; n<n_stars; n++) {
            // std::cerr << "image[" << n << "].shape = ("
            //           << img_stack.img[n]->rows << ", "
            //           << img_stack.img[n]->cols << ")" << std::endl;
			img_buffer.add(*(img_stack.img[n]));
		}

        img_buffer.write(out_fname, group_name.str(), "stellar pdfs");
	}

    auto t_end = std::chrono::steady_clock::now();

    std::chrono::duration<double, std::milli> dt_sample = t_smooth - t_start;
    std::chrono::duration<double, std::milli> dt_smooth = t_write - t_smooth;
    std::chrono::duration<double, std::milli> dt_write = t_end - t_write;
    std::chrono::duration<double, std::milli> dt_total = t_end - t_start;

    if(verbosity >= 1) {
        std::cerr << "Done with grid evaluation for all stars."
                  << std::endl << std::endl;
        std::cerr << "Time elapsed / star:" << std::endl
                  << "  * sample: " << dt_sample.count() / n_stars << " ms"
                  << std::endl
                  << "  * smooth: " << dt_smooth.count() / n_stars << " ms"
                  << std::endl
                  << "  *  write: " << dt_write.count() / n_stars << " ms"
                  << std::endl
                  << "  *  total: " << dt_total.count() / n_stars << " ms"
                  << std::endl << std::endl;
    }
}


bool save_gridstars(
    const std::string& fname,
    const std::string& group,
    const std::string& dset,
    std::vector<std::vector<TDMESaveData> >& fit_centers,
    std::vector<float>& fit_icovs)
{
    // Number of stars to save
    uint32_t n_stars = fit_centers.size();
	if(n_stars == 0) {
		std::cerr << "! No stars to write." << std::endl;
		return false;
	}
    
    // # of covariance matrices should match # of stars
    assert(fit_icovs.size() == 3 * n_stars);

    // Open up file and create group
	H5::Exception::dontPrint();

	std::unique_ptr<H5::H5File> file = H5Utils::openFile(fname);
	if(!file) { return false; }

	std::unique_ptr<H5::Group> gp = H5Utils::openGroup(*file, group);
	if(!gp) { return false; }
    
    // Determine maximum number of Gaussians for one star
    uint32_t n_gaussians = 0;
    for(auto& v : fit_centers) {
        if(v.size() > n_gaussians) {
            n_gaussians = v.size();
        }
    }
    
    // Datatype
    H5::CompType dtype(sizeof(TDMESaveData));
    dtype.insertMember("dm", HOFFSET(TDMESaveData, dm), H5::PredType::NATIVE_FLOAT);
    dtype.insertMember("E", HOFFSET(TDMESaveData, E), H5::PredType::NATIVE_FLOAT);
    dtype.insertMember("ln_likelihood", HOFFSET(TDMESaveData, ln_likelihood), H5::PredType::NATIVE_FLOAT);
    dtype.insertMember("ln_prior", HOFFSET(TDMESaveData, ln_prior), H5::PredType::NATIVE_FLOAT);
    
	// Dataspace
	hsize_t dim[2] = {n_stars, n_gaussians};
	H5::DataSpace dspace(2, &(dim[0]));

	// Property List
	H5::DSetCreatPropList plist;
	
    // Target chunk size of 1 MB
	int64_t target_size = 1024*1024;
	int64_t size_per_star = n_gaussians * sizeof(TDMESaveData);
	hsize_t n_stars_chunk = target_size / size_per_star;
    if(n_stars_chunk < 1) {
        n_stars_chunk = 1;
    } else if(n_stars_chunk > n_stars) {
        n_stars_chunk = n_stars;
    }
	hsize_t chunk[2] = {n_stars_chunk, n_gaussians};
	plist.setChunk(2, &(chunk[0]));
    
	plist.setDeflate(3); // DEFLATE compression level (min=0, max=9)

	// Dataset
	H5::DataSet dataset = gp->createDataSet(dset, dtype, dspace, plist);
    
    // Copy data into one array
    TDMESaveData* data = new TDMESaveData[n_stars*n_gaussians];
    for(int i=0; i<fit_centers.size(); i++) {
        for(int j=0; j<fit_centers[i].size(); j++) {
            data[n_gaussians*i+j] = fit_centers[i].at(j);
        }
        for(int j=fit_centers[i].size(); j<n_gaussians; j++) {
            TDMESaveData* d = &(data[n_gaussians*i+j]);
            d->dm = 0.;
            d->E = 0.;
            d->ln_likelihood = -std::numeric_limits<float>::infinity();
            d->ln_prior = -std::numeric_limits<float>::infinity();
        }
    }
    
    // Write dataset
    dataset.write(data, dtype);
    
    // Write stellar covariances
    std::stringstream dset_cov;
    dset_cov << dset << "_icov";
    
    //std::cout << "dset_cov = " << dset_cov.str() << std::endl;
    
    hsize_t dim_cov[2] = {n_stars, 3}; // (dm,dm), (dm,E), (E,E)
    H5::DataSpace dspace_cov(2, &(dim_cov[0]));
    
    H5::DSetCreatPropList plist_cov;
	int64_t size_per_star_cov = dim_cov[1] * sizeof(float);
	hsize_t n_stars_chunk_cov = target_size / size_per_star_cov;
    if(n_stars_chunk_cov < 1) {
        n_stars_chunk_cov = 1;
    } else if(n_stars_chunk_cov > n_stars) {
        n_stars_chunk_cov = n_stars;
    }
	hsize_t chunk_cov[2] = {n_stars_chunk_cov, dim_cov[1]};
	plist_cov.setChunk(2, &(chunk_cov[0]));
	plist_cov.setDeflate(3); // DEFLATE compression level (min=0, max=9)
    //std::cout << "chunk length: " << n_stars_chunk_cov << std::endl;

    //std::cout << "Creating dataset_cov" << std::endl;
    
	H5::DataSet dataset_cov = gp->createDataSet(
        dset_cov.str(),
        H5::PredType::NATIVE_FLOAT,
        dspace_cov,
        plist_cov);
    
    //std::cout << "Writing dataset_cov" << std::endl;
    
    dataset_cov.write(fit_icovs.data(), H5::PredType::NATIVE_FLOAT);
    
    // Attribute noting meaning of inverse cov. matrix entries
    //std::cout << "Closing file" << std::endl;
    file->close();
    
    //std::cout << "Adding attribute" << std::endl;
    std::stringstream dset_cov_fullpath;
    dset_cov_fullpath << group << "/" << dset_cov.str();
    //std::cout << "Writing attribute to " << dset_cov_fullpath.str()
    //          << std::endl;
    std::string attr_name = "entries";
    std::string attr_value = "dd, dE, EE";
    H5Utils::add_watermark(
        fname,
        dset_cov_fullpath.str(),
        attr_name,
        attr_value
    );
    //std::cout << "Done." << std::endl;
}
