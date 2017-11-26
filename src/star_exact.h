
#ifndef _STAR_EXACT_H__
#define _STAR_EXACT_H__


#include <iostream>
#include <limits>
#include <algorithm>
#include <memory>
#include <cstdlib>

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

void integrate_ML_solution(TStellarModel& stellar_model,
                           TGalacticLOSModel& los_model,
                           TStellarData::TMagnitudes& mags_obs,
                           TExtinctionModel& ext_model,
                           TImgStack& img_stack,
                           unsigned int img_idx,
                           double RV);


void grid_eval_stars(TGalacticLOSModel& los_model, TExtinctionModel& ext_model,
                     TStellarModel& stellar_model, TStellarData& stellar_data,
                     TImgStack& img_stack, bool save_surfs, std::string out_fname,
                     double RV);


#endif // _STAR_EXACT_H__
