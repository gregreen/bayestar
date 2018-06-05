/*
 * gaussian_process.h
 *
 * Gaussian-process-related functions for bayestar.
 *
 * This file is part of bayestar.
 * Copyright 2018 Gregory Green
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

#ifndef _GAUSSIAN_PROCESS_H__
#define __GAUSSIAN_PROCESS_H_


#include <iostream>
#include <algorithm>
#include <memory>
#include <vector>
#include <cmath>

#include <Eigen/Dense>

//typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXd;
typedef std::shared_ptr<Eigen::MatrixXd> SharedMatrixXd;
typedef std::unique_ptr<Eigen::MatrixXd> UniqueMatrixXd;
//typedef std::shared_ptr<std::vector<double> > SharedVectord;

//void conditional_gaussian(MatrixXd& C11, MatrixXd& C12, MatrixXd& C22, MatrixXd& C11_cond, MatrixXd& A_cond);


void conditional_gaussian_scalar(
        const Eigen::MatrixXd& C_inv, unsigned int idx,
        double& inv_var, Eigen::MatrixXd& A_cond);

void distance_matrix_lonlat(
        const std::vector<double>& lon,
        const std::vector<double>& lat,
        Eigen::MatrixXd& d2);

void inv_cov_lonlat(
        const std::vector<double>& lon,
        const std::vector<double>& lat,
        const std::vector<double>& dist,
        std::function<double(double)>& kernel,
        std::vector<UniqueMatrixXd>& inv_cov);


#endif // __GAUSSIAN_PROCESS_H_
