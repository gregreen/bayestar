
#ifndef _STAR_EXACT_H__
#define _STAR_EXACT_H__


#include <iostream>
#include <limits>
#include <algorithm>
#include <memory>
#include <cstdlib>
#include <chrono>

#include <Eigen/Dense>

#include "model.h"
#include "data.h"
#include "chain.h"
#include "los_sampler.h"


class LinearFitParams {
public:
    LinearFitParams(unsigned int n_dim);

    Eigen::VectorXd mean;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> inv_cov;
    double chi2;

private:
    unsigned int _n_dim;
};


std::shared_ptr<LinearFitParams> star_max_likelihood(
    TSED& mags_model,
    TStellarData::TMagnitudes& mags_obs,
    TExtinctionModel& ext_model,
    double RV=3.1);


void star_covariance(TStellarData::TMagnitudes& mags_obs,
                     TExtinctionModel& ext_model,
                     double& inv_cov_00, double& inv_cov_01, double& inv_cov_11,
                     double RV=3.1);

void star_max_likelihood(TSED& mags_model, TStellarData::TMagnitudes& mags_obs,
                         TExtinctionModel& ext_model,
                         double inv_cov_00, double inv_cov_01, double inv_cov_11,
                         double& mu, double& E, double& chi2,
                         double RV=3.1);

struct TDMESaveData {
    float dm;
    float E;
    float Mr;
    float FeH;
    float ln_likelihood;
    float ln_prior;
};

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
    double RV, int verbosity);

void grid_eval_stars(TGalacticLOSModel& los_model, TExtinctionModel& ext_model,
                     TStellarModel& stellar_model, TStellarData& stellar_data,
                     TEBVSmoothing& EBV_smoothing,
                     TImgStack& img_stack, std::vector<double>& chi2,
                     bool save_surfs, bool save_gaussians,
                     std::string out_fname,
                     bool use_priors, bool use_gaia,
                     double RV, int verbosity);

bool save_gridstars(
    const std::string& fname,
    const std::string& group,
    const std::string& dset,
    std::vector<std::vector<TDMESaveData> >& fit_centers,
    std::vector<float>& fit_icovs);


#endif // _STAR_EXACT_H__
