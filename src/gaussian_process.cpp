/*
 * gaussian_process.cpp
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


#include "gaussian_process.h"


//void conditional_gaussian_matrices(
//        MatrixXd& C11, MatrixXd& C12, MatrixXd& C22,
//        MatrixXd& C11_cond, MatrixXd& A_cond)
//{
//    // Calculates the matrices necessary to quickly
//    // compute conditional Gaussian distributions.
//    
//    A_cond = C12 * C22.inverse(); // \mu_{1|2} = A (y_2 - \mu_2)
//    C11_cond = C11 - A * C12.transpose(); // \Sigma_{11|2}
//}


void conditional_gaussian_scalar(
        const Eigen::MatrixXd& C_inv, unsigned int idx,
        double& inv_var, Eigen::MatrixXd& A_cond)
{
    // Calculates p(x_i | x_{\i}), where p(x) is a Gaussian,
    // and all the elements of the vector x except x_i are
    // given.
    // 
    // Inputs:
    //   C_inv (MatrixXd): Covariance matrix of the entire
    //                     vector x.
    //   idx (unsigned int): The element i in the vector x
    //                       to calculate the probability
    //                       density function of.
    //
    // Sets the following references:
    //   inv_var (double): the inverse variance of x_i
    //   A_cond (MatrixXd): A matrix that takes the values
    //                      of x_{\i} and gives the mean x_i.
    
    inv_var = C_inv(idx, idx);
    A_cond = (-1./inv_var) * C_inv.row(idx);
}


//void remove_matrix_rowcol(MatrixXd& m, unsigned int idx) {
//    // Removes the specified row and column from a matrix.
//    //
//    // Based on these StackOverflow answers by raahlb and Andrew:
//    //   * <https://stackoverflow.com/a/46303314/1103939>
//    //   * <https://stackoverflow.com/a/21068014/1103939>
//
//    unsigned int n_rows = matrix.rows();
//    unsigned int n_cols = matrix.cols();
//
//    if(idx < n_rows-1) {
//        matrix.block(idx, 0, n_rows-idx-1, n_cols) = matrix.bottomRows(n_rows-idx-1);
//    }
//    if(idx < n_cols-1) {
//        matrix.block(0, idx, n_rows-1, n_cols-idx-1) = matrix.rightCols(n_cols-idx-1);
//    }
//
//    matrix.conservativeResize(n_rows-1, n_cols-1);
//}

const double pi_over_180 = 3.141592653589793238 / 180.;


void distance_matrix_lonlat(
        const std::vector<double>& lon,
        const std::vector<double>& lat,
        Eigen::MatrixXd& d2)
{
    // Fills the matrix d2 with the pairwise squared angular
    // distances (in rad) between the given (lon, lat) pairs.
    //
    // These distances are approximate. They are actually 3D
    // distances between points on a unit sphere, rather than
    // great-circle distances.
    
    // Convert (lon, lat) to (x, y, z)
    uint32_t n_coords = lon.size();
    std::vector<double> xyz;
    xyz.reserve(3*n_coords);
    
    //std::cerr << "Calculating (x, y, z) of pixels ..." << std::endl;

    double cos_lon, cos_lat, sin_lon, sin_lat;
    for(int i=0; i<n_coords; i++) {
        cos_lon = std::cos(pi_over_180*lon.at(i));
        sin_lon = std::sin(pi_over_180*lon.at(i));
        cos_lat = std::cos(pi_over_180*lat.at(i));
        sin_lat = std::sin(pi_over_180*lat.at(i));
        
        xyz[3*i]   = cos_lon * cos_lat;
        xyz[3*i+1] = sin_lon * cos_lat;
        xyz[3*i+2] = sin_lat;
    }
    
    // Calculate pairwise squared distances
    //std::cerr << "Calculating distance^2 between pixels ..." << std::endl;
    d2.resize(n_coords, n_coords);
    double dx, dy, dz, d2_tmp;
    for(int j=0; j<n_coords; j++) {
        for(int k=0; k<=j; k++) {
            dx = (xyz[3*j] - xyz[3*k]);
            dy = (xyz[3*j+1] - xyz[3*k+1]);
            dz = (xyz[3*j+2] - xyz[3*k+2]);
            d2_tmp = dx*dx + dy*dy + dz*dz;
            d2(j,k) = d2_tmp;
            d2(k,j) = d2_tmp;
        }
    }

    //std::cerr << "Done calculating distance^2 matrix." << std::endl;
}


void inv_cov_lonlat(
        const std::vector<double>& lon,
        const std::vector<double>& lat,
        const std::vector<double>& dist,
        std::function<double(double)>& kernel,
        std::vector<UniqueMatrixXd>& inv_cov)
{
    // Calculates a set of inverse covariance matrices - one per
    // distance. Each matrix gives the inverse covariance between
    // the coordinate (specified by lon, lat) at a given distance
    // (specified by dist). Works with a custom kernel function,
    // which maps distance^2 -> covariance. The inverse
    // covariance matrices are stored in inv_cov.

    // Calculate pairwise angular distances
    Eigen::MatrixXd d2_mat;
    distance_matrix_lonlat(lon, lat, d2_mat);

    //std::cerr << "Transverse distances:" << std::endl
    //          << d2_mat << std::endl;
    
    // Generate one covariance matrix per physical distance
    for(double d : dist) {
        //std::cerr << "Calculating Cov^-1 at distance = " << d << " pc ..." << std::endl;
        UniqueMatrixXd C = std::make_unique<Eigen::MatrixXd>();
        //UniqueMatrixXd C = std::unique_ptr<Eigen::MatrixXd>();
        C->resize(d2_mat.rows(), d2_mat.cols());
        *C = d2_mat.unaryExpr([kernel, d](double d2) -> double {
            // Angular distance scaled by physical distance
            return kernel(d*d * d2); 
        }).inverse();
        
        //std::cerr << std::endl << "dist = " << d << " pc" << std::endl;
        //std::cerr << std::endl << *C << std::endl;

        inv_cov.push_back(std::move(C));
    }
}
