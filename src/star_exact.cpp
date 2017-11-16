
#include "star_exact.h"


LinearFitParams::LinearFitParams(unsigned int n_dim)
    : _n_dim(n_dim), mean(n_dim), inv_cov(n_dim, n_dim),
      chi2(std::numeric_limits<double>::infinity())
{}


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
    double det = p->inv_cov(0,0) * p->inv_cov(1,1) - p->inv_cov(0,1) * p->inv_cov(1,0);// + 1.e-5;
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
    img = cv::Mat::zeros(2*width[0]+1, 2*width[1]+1, CV_64F);

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
            cyy = p->inv_cov(0,0) * dy*dy;

            // std::cerr << " (" << i << ", " << j << ")" << std::endl;
            // std::cerr << " width = (" << 2*width[0]+1 << ", " << 2*width[1]+1 << ")" << std::endl;

            img.at<double>(i, j) += exp(-0.5 * (cxx + 2*cxy + cyy));
        }
    }
    // std::cerr << "done creating filter" << std::endl;
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


void integrate_ML_solution(TStellarModel& stellar_model,
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

    std::cerr << "N_Mr = " << N_Mr << std::endl;
    std::cerr << "N_FeH = " << N_FeH << std::endl;

    // return;

    cv::Mat cov_img;
    unsigned int img_idx0, img_idx1;

    if (!img_stack.initialize_to_zero(img_idx)) {
        std::cerr << "Failed to initialize image to zero!" << std::endl;
    }

    for(int Mr_idx=0; Mr_idx<N_Mr; Mr_idx++) {
        for(int FeH_idx=0; FeH_idx<N_FeH; FeH_idx++) {
            // Look up model absolute magnitudes of this stellar type
            bool success = stellar_model.get_sed(Mr_idx, FeH_idx, sed, FeH, Mr);
            if(!success) {
                std::cerr << "SED (" << Mr_idx << ", " << FeH_idx
                          << ") not in library!" << std::endl;
                continue;
            }

            // Calculate max. likelihood solution for (mu, E) given this fixed stellar type
            std::shared_ptr<LinearFitParams> ML = star_max_likelihood(sed, mags_obs, ext_model, RV);

            if((Mr_idx == 0) && (FeH_idx == 0)) {
            // if(ML->chi2 < 3.) {
                double det = ML->inv_cov(0,0) * ML->inv_cov(1,1) - ML->inv_cov(0,1) * ML->inv_cov(1,0);
                std::cerr << "  (mu, E) = (" << ML->mean(0) << ", " << ML->mean(1) << ")" << std::endl;
                std::cerr << "  chi^2 = " << ML->chi2 << std::endl;
                std::cerr << "  sigma = (" << sqrt(ML->inv_cov(1,1) / det)
                          << ", " << sqrt(ML->inv_cov(0,0) / det) << ")" << std::endl;
                std::cerr << "  Sigma^-1:" << std::endl;
                std::cerr << "    " << ML->inv_cov(0, 0) << "  " << ML->inv_cov(0, 1) << std::endl;
                std::cerr << "    " << ML->inv_cov(1, 0) << "  " << ML->inv_cov(1, 1) << std::endl;
                std::cerr << std::endl;
            }

            // Add single point to image at ML solution location (mu, E)
            bool in_bounds = img_stack.rect->get_index(ML->mean(1), ML->mean(0), img_idx0, img_idx1);
            if(in_bounds) {
                // std::cerr << ML->chi2 << " -> " << exp(-0.5 * (ML->chi2)) << std::endl;
                img_stack.img[img_idx]->at<double>(img_idx0, img_idx1) += 1.;//exp(-0.5 * (ML->chi2));
            } else {
                // std::cerr << "(mu, E) = (" << ML->mean(1) << ", " << ML->mean(0) << ") out of bounds." << std::endl;
            }

            // gaussian_filter(ML, *(img_stack.rect), cov_img, 5, 2);
        }
    }

    std::cerr << "Done with grid evaluation." << std::endl;
}

void grid_eval_stars(TGalacticLOSModel& los_model, TExtinctionModel& ext_model,
                     TStellarModel& stellar_model, TStellarData& stellar_data,
                     TImgStack& img_stack, bool save_surfs, std::string out_fname,
                     double RV) {
    // Set up image stack for stellar PDFs
    double min[2] = {0., 4.};   // (E, DM)
	double max[2] = {7., 19.};  // (E, DM)
	unsigned int N_bins[2] = {700, 120};
	TRect rect(min, max, N_bins);
    img_stack.set_rect(rect);

    // Loop over all stars and evaluate PDFs on grid in (mu, E)
    int n_stars = stellar_data.star.size();

    for(int i=0; i<n_stars; i++) {
        integrate_ML_solution(stellar_model, stellar_data[i], ext_model,
                              img_stack, i, RV);
    }

    // Save the PDFs to disk
    if(save_surfs) {
        std::stringstream group_name;
        group_name << "/" << stellar_data.pix_name;

        TImgWriteBuffer img_buffer(rect, n_stars);

		for(int n=0; n<n_stars; n++) {
            // std::cerr << "image[" << n << "].shape = ("
            //           << img_stack.img[n]->rows << ", "
            //           << img_stack.img[n]->cols << ")" << std::endl;
			img_buffer.add(*(img_stack.img[n]));
		}

        img_buffer.write(out_fname, group_name.str(), "stellar pdfs");
	}
}
