
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
            print_float(mat.at<float>(j,k), s, width, precision);
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
    img = cv::Mat::zeros(2*width[0]+1, 2*width[1]+1, CV_32F);

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

            img.at<float>(i, j) += exp(-0.5 * (cxx + 2*cxy + cyy));
        }
    }
    // std::cerr << "done creating filter" << std::endl;
}


void gaussian_filter(double inv_cov_00, double inv_cov_01, double inv_cov_11,
                     TRect& grid, cv::Mat& img,
                     double n_sigma, int min_width,
                     double add_diagonal=-1.,
                     int subsample=5) {
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

    std::cerr << "sigma -> (" << sigma[0] << ", " << sigma[1] << ")" << std::endl;

    // Determine dimensions of filter
    int width[2];

    for(unsigned int i=0; i<2; i++) {
        width[i] = std::max(min_width, (int)(ceil(n_sigma * sigma[i] / grid.dx[i])));
    }

    std::cerr << "width = (" << width[0] << ", " << width[1] << ")" << std::endl;

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
    cv::Mat img_sub = cv::Mat::zeros(w_sub, h_sub, CV_32F);

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
            img_sub.at<float>(i, j) += weight;
            // sum += weight;
        }
    }
    // std::cerr << "done creating filter" << std::endl;

    // img.resize(h, w)
    cv::Mat img_down;
    cv::resize(img_sub, img_down, cv::Size(h, w), 0, 0, cv::INTER_AREA);

    img = img_down;
    img /= img.at<float>(width[0], width[1]);

    // std::cerr << "size = " << img_sub.rows << ", " << img_sub.cols << std::endl;
    // std::cerr << "size = " << img_down.rows << ", " << img_down.cols << std::endl;
}


void add_fit_to_image(cv::Mat& img, TRect& grid, LinearFitParams& fit,
                      double weight, double n_sigma, int min_width) {
      // Determine sigma along each axis
    //   double det = p->inv_cov(0,0) * p->inv_cov(1,1) - p->inv_cov(0,1) * p->inv_cov(1,0);// + 1.e-5;
    //   double sigma[2] = {
    //       sqrt(p->inv_cov(1,1) / det),
    //       sqrt(p->inv_cov(0,0) / det)
    //   };
      //
    //   // Determine dimensions of filter
    //   int width[2];
      //
    //   for(unsigned int i=0; i<2; i++) {
    //       width[i] = std::max(min_width, (int)(ceil(sigma[i] / grid.dx[i])));
    //   }
      //
    //   // TODO: Finish this
}


double integrate_ML_solution(TStellarModel& stellar_model,
                             TGalacticLOSModel& los_model,
                             TStellarData::TMagnitudes& mags_obs,
                             TExtinctionModel& ext_model,
                             TImgStack& img_stack,
                             unsigned int img_idx,
                             double RV) {
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

    if (!img_stack.initialize_to_zero(img_idx)) {
        std::cerr << "Failed to initialize image to zero!" << std::endl;
    }

    // Arrays holding ML (E, mu), chi2 and prior
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
            bool success = stellar_model.get_sed(Mr_idx, FeH_idx, sed, Mr, FeH);
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

            // Check that the max-likelihood solution really is at a local
            // minimum in chi2
            // double eps = 1.e-2;
            // for(int dmu=-1; dmu<2; dmu++) {
            //     for(int dE=-1; dE<2; dE++) {
            //         if((dmu == 0) && (dE == 0)) { continue; }
            //
            //         double mu_test = mu + (double)dmu*eps;
            //         double E_test = E + (double)dE*eps;
            //
            //         double chi2_tmp = calc_star_chi2(
            //             mags_obs, ext_model, sed,
            //             mu_test, E_test, RV
            //         );
            //
            //         if(chi2_tmp < chi2) {
            //             std::cerr << std::endl
            //                       << "Not a trough in chi^2!" << std::endl
            //                       << "  chi^2(" << mu_test << ", " << E_test << ")"
            //                       << " = " << chi2_tmp << std::endl
            //                       << "  chi^2(" << mu << ", " << E << ")"
            //                       << " = " << chi2 << std::endl << std::endl;
            //             std::exit(1);
            //         } else {
            //             // std::cerr << " chi^2 : " << chi2_tmp << " > " << chi2 << std::endl;
            //         }
            //     }
            // }

            // std::shared_ptr<LinearFitParams> ML = star_max_likelihood(sed, mags_obs, ext_model, RV);
            // if((Mr_idx == 0) && (FeH_idx == 0)) {
            // // if(ML->chi2 < 3.) {
            //     double det = ML->inv_cov(0,0) * ML->inv_cov(1,1) - ML->inv_cov(0,1) * ML->inv_cov(1,0);
            //     std::cerr << "  (mu, E) = (" << ML->mean(0) << ", " << ML->mean(1) << ")" << std::endl;
            //     std::cerr << "  chi^2 = " << ML->chi2 << std::endl;
            //     std::cerr << "  sigma = (" << sqrt(ML->inv_cov(1,1) / det)
            //               << ", " << sqrt(ML->inv_cov(0,0) / det) << ")" << std::endl;
            //     std::cerr << "  Sigma^-1:" << std::endl;
            //     std::cerr << "    " << ML->inv_cov(0, 0) << "  " << ML->inv_cov(0, 1) << std::endl;
            //     std::cerr << "    " << ML->inv_cov(1, 0) << "  " << ML->inv_cov(1, 1) << std::endl;
            //     std::cerr << std::endl;
            // }
            // if((Mr_idx == 0) && (FeH_idx == 0)) {
            //     double det = inv_cov_00 * inv_cov_11 - inv_cov_01 * inv_cov_01;
            //     std::cerr << "  (E, mu) = (" << E << ", " << mu << ")" << std::endl;
            //     std::cerr << "  chi^2 = " << chi2 << std::endl;
            //     std::cerr << "  sigma = (" << sqrt(inv_cov_00 / det)
            //               << ", " << sqrt(inv_cov_11 / det) << ")" << std::endl;
            //     std::cerr << "  Sigma^-1:" << std::endl;
            //     std::cerr << "    " << inv_cov_11 << "  " << inv_cov_01 << std::endl;
            //     std::cerr << "    " << inv_cov_01 << "  " << inv_cov_00 << std::endl;
            //     std::cerr << std::endl;
            // }

            double prior = los_model.log_prior_emp(mu, Mr, FeH)
                         + stellar_model.get_log_lf(Mr);

            // std::cerr << "p(FeH = " << FeH << ") = " << prior << std::endl;

            E_ML.push_back(E);
            mu_ML.push_back(mu);
            chi2_ML.push_back(chi2);
            prior_ML.push_back(prior);

            // Add single point to image at ML solution location (mu, E)
            // bool in_bounds = img_stack.rect->get_index(E, mu, img_idx0, img_idx1);
            // if(in_bounds) {
            //     // std::cerr << ML->chi2 << " -> " << exp(-0.5 * (ML->chi2)) << std::endl;
            //     chi2 += los_model.log_prior_emp(mu, Mr, FeH) + stellar_model.get_log_lf(Mr);
            //     // std::cerr << chi2 << std::endl;
            //
            //     img_stack.img[img_idx]->at<float>(img_idx0, img_idx1) += exp(-0.5 * chi2);
            // } else {
            //     // std::cerr << "(mu, E) = (" << ML->mean(1) << ", " << ML->mean(0) << ") out of bounds." << std::endl;
            // }

            // gaussian_filter(ML, *(img_stack.rect), cov_img, 5, 2);
        }
    }

    double prior_max = *std::max_element(prior_ML.begin(), prior_ML.end());
    std::cerr << "prior_max = " << prior_max << std::endl;

    double chi2_min = *std::min_element(chi2_ML.begin(), chi2_ML.end());
    std::cerr << "chi2_min = " << chi2_min << std::endl;

    assert( E_ML.size() == mu_ML.size() );
    assert( chi2_ML.size() == mu_ML.size() );
    assert( prior_ML.size() == mu_ML.size() );

    unsigned int img_idx0, img_idx1;
    double a0, a1;

    for(int k=0; k<mu_ML.size(); k++) {
        // Add single point to image at ML solution location (E, mu)
        bool in_bounds = img_stack.rect->get_interpolant(
            E_ML.at(k), mu_ML.at(k),
            img_idx0, img_idx1,
            a0, a1
        );

        // bool in_bounds = img_stack.rect->get_index(E_ML.at(k), mu_ML.at(k), img_idx0, img_idx1);

        if(in_bounds) {
            double log_p = -0.5 * (chi2_ML.at(k) - chi2_min)
                           + prior_ML.at(k) - prior_max;
            double p = exp(log_p);
            // if(std::isnan(log_p)) {
            //     std::cerr << "NaN: (" << E_ML.at(arr_idx) << ", "
            //                           << mu_ML.at(arr_idx) << ", "
            //                           << Mr << ", "
            //                           << FeH << ")" << std::endl;
            // }
            // if(std::isinf(log_p)) {
            //     std::cerr << "Inf: (" << E_ML.at(arr_idx) << ", "
            //                           << mu_ML.at(arr_idx) << ", "
            //                           << Mr << ", "
            //                           << FeH << ")" << std::endl;
            //     std::cerr << "  * chi2 = " << chi2_ML.at(arr_idx) << std::endl
            //               << "  * prior = " << prior_ML.at(arr_idx)-prior_max << std::endl;
            //     std::cerr << "  * exp(-0.5*chi2 + prior) = "
            //               << exp(log_p) << std::endl;
            // }
            // std::cerr << log_p << std::endl;

            // img_stack.img[img_idx]->at<float>(img_idx0, img_idx1) += exp(log_p);


            // Interpolate between bins
            img_stack.img[img_idx]->at<float>(img_idx0, img_idx1) += (1-a0) * (1-a1) * p;
            img_stack.img[img_idx]->at<float>(img_idx0+1, img_idx1) += a0 * (1-a1) * p;
            img_stack.img[img_idx]->at<float>(img_idx0, img_idx1+1) += (1-a0) * a1 * p;
            img_stack.img[img_idx]->at<float>(img_idx0+1, img_idx1+1) += a0 * a1 * p;
        }
    }

    // for(int Mr_idx=0, arr_idx=0; Mr_idx<N_Mr; Mr_idx++) {
    //     for(int FeH_idx=0; FeH_idx<N_FeH; FeH_idx++, arr_idx++) {
    //         // Add single point to image at ML solution location (E, mu)
    //         bool in_bounds = img_stack.rect->get_index(E_ML.at(arr_idx), mu_ML.at(arr_idx), img_idx0, img_idx1);
    //         if(in_bounds) {
    //             double log_p = -0.5 * (chi2_ML.at(arr_idx) - chi2_min)
    //                            + prior_ML.at(arr_idx) - prior_max;
    //             // if(std::isnan(log_p)) {
    //             //     std::cerr << "NaN: (" << E_ML.at(arr_idx) << ", "
    //             //                           << mu_ML.at(arr_idx) << ", "
    //             //                           << Mr << ", "
    //             //                           << FeH << ")" << std::endl;
    //             // }
    //             // if(std::isinf(log_p)) {
    //             //     std::cerr << "Inf: (" << E_ML.at(arr_idx) << ", "
    //             //                           << mu_ML.at(arr_idx) << ", "
    //             //                           << Mr << ", "
    //             //                           << FeH << ")" << std::endl;
    //             //     std::cerr << "  * chi2 = " << chi2_ML.at(arr_idx) << std::endl
    //             //               << "  * prior = " << prior_ML.at(arr_idx)-prior_max << std::endl;
    //             //     std::cerr << "  * exp(-0.5*chi2 + prior) = "
    //             //               << exp(log_p) << std::endl;
    //             // }
    //             // std::cerr << log_p << std::endl;
    //             img_stack.img[img_idx]->at<float>(img_idx0, img_idx1) += exp(log_p);
    //         }
    //
    //         // gaussian_filter(ML, *(img_stack.rect), cov_img, 5, 2);
    //     }
    // }

    // std::cerr << "img.type = " << img_stack.img[img_idx]->type() << std::endl;
    // img_stack.img[img_idx]->at<float>(0, 0) = 1.;
    // img_stack.img[img_idx]->at<double>(0, 0) = 1.;
    // (*img_stack.img[img_idx])(0,0) = 1.;

    // cv::Mat filter_test;
    // gaussian_filter(inv_cov_11, inv_cov_01, inv_cov_00,
    //                 *(img_stack.rect), filter_test, 5, 2, 0.5);
    // std::cerr << std::endl << "filter_test:" << std::endl;
    // print_matrix(filter_test, std::cerr, 15, 12);
    // std::cerr << std::endl;

    // Smooth PDF with covariance of the ML solution
    cv::Mat cov_img;
    double expand = 1.;// / (10.*10.);
    gaussian_filter(expand*inv_cov_11, expand*inv_cov_01, expand*inv_cov_00,
                    *(img_stack.rect), cov_img, 5, 2, 1.0);

    cv::Mat filtered_img = cv::Mat::zeros(img_stack.rect->N_bins[0], img_stack.rect->N_bins[1], CV_32F);
    cv::filter2D(*img_stack.img[img_idx], filtered_img, CV_32F, cov_img);
    *img_stack.img[img_idx] = filtered_img;

    std::cerr << "Done with grid evaluation." << std::endl;

    // Return mininum chi^2 / passband
    int n_passbands = 0;
    for(int i=0; i<NBANDS; i++) {
        if(std::isnan(mags_obs.err[i]) || std::isinf(mags_obs.err[i]) || (mags_obs.err[i] > 1.e9)) {
            continue;
        }
        n_passbands++;
    }

    std::cerr << "# of passbands: " << n_passbands << std::endl;
    std::cerr << "chi^2 / passband: " << chi2_min / n_passbands << std::endl;

    return chi2_min / n_passbands;
}

void grid_eval_stars(TGalacticLOSModel& los_model, TExtinctionModel& ext_model,
                     TStellarModel& stellar_model, TStellarData& stellar_data,
                     TImgStack& img_stack, std::vector<double>& chi2,
                     bool save_surfs, std::string out_fname,
                     double RV) {
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

    for(int i=0; i<n_stars; i++) {
        double chi2_min = integrate_ML_solution(
            stellar_model, los_model,
            stellar_data[i], ext_model,
            img_stack, i, RV
        );
        chi2.push_back(chi2_min);
    }

    // Crop to correct (E, DM) range
    img_stack.crop(0., 7., 4., 19.);

    // Save the PDFs to disk
    auto t_write = std::chrono::steady_clock::now();

    if(save_surfs) {
        std::stringstream group_name;
        group_name << "/" << stellar_data.pix_name;

        TImgWriteBuffer img_buffer(*(img_stack.rect), n_stars);

		for(int n=0; n<n_stars; n++) {
            std::cerr << "image[" << n << "].shape = ("
                      << img_stack.img[n]->rows << ", "
                      << img_stack.img[n]->cols << ")" << std::endl;
			img_buffer.add(*(img_stack.img[n]));
		}

        img_buffer.write(out_fname, group_name.str(), "stellar pdfs");
	}

    auto t_end = std::chrono::steady_clock::now();

    std::chrono::duration<double, std::milli> dt_sample = t_write - t_start;
    std::chrono::duration<double, std::milli> dt_write = t_end - t_write;
    std::chrono::duration<double, std::milli> dt_total = t_end - t_start;

    std::cerr << "Done with grid evaluation for all stars."
              << std::endl << std::endl;
    std::cerr << "Time elapsed / star:" << std::endl
              << "  * sample: " << dt_sample.count() / n_stars << " ms"
              << std::endl
              << "  *  write: " << dt_write.count() / n_stars << " ms"
              << std::endl
              << "  *  total: " << dt_total.count() / n_stars << " ms"
              << std::endl << std::endl;
}
