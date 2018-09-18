/*
 * neighbor_pixels.h
 *
 * Information on neighboring pixels, used to provide prior on
 * line-of-sight dust distribution.
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

#ifndef _NEIGHBOR_PIXELS_H__
#define _NEIGHBOR_PIXELS_H__


#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cmath>
#include <cassert>

#include "gaussian_process.h"
#include "healpix_tree.h"
#include "h5utils.h"


class TNeighborPixels {
    // Information on a set of nearby pixels
private:
    unsigned int n_pix, n_samples, n_dists;
    double dm_min, dm_max;
    
    // shape = (pix, sample, dist)
    std::vector<double> delta;
    std::vector<double> log_dy;

    // shape = (pix, sample). Summed over distance.
    std::vector<double> sum_log_dy;
    
    // Prior and likelihood stored for each neighbor
    std::vector<double> prior;
    std::vector<double> likelihood;
    
    // Locations of neibhoring pixels
    std::vector<double> lon, lat;
    std::vector<uint32_t> nside, pix_idx; // Pairs of (nside, pix_idx)
    
    // Inverse covariance matrix for each distance
    std::vector<UniqueMatrixXd> inv_cov;
    
    std::vector<double> A_i_given_noti;
    std::vector<double> inv_var; // shape = (pix, dist)

    // Dominant distance for each (pix, sample)
    std::vector<uint16_t> dominant_dist;

    // # of samples in given pix with dominant distance at given dist
    std::vector<uint16_t> n_dominant_dist_samples; // shape = (pix, dist)
    
    // True if data loaded successfully, else false
    bool loaded;

public:
    // Constructor/destructor
    TNeighborPixels(uint32_t nside_center,
                    uint32_t pix_idx_center,
                    const std::string& neighbor_lookup_fname,
                    const std::string& pixel_lookup_fname,
                    const std::string& output_fname_pattern,
                    int n_samples_max=-1);
    ~TNeighborPixels();
    
    // Getters
    double get_delta(
            unsigned int pix,
            unsigned int sample,
            unsigned int dist) const;
    
    double get_log_dy(
            unsigned int pix,
            unsigned int sample,
            unsigned int dist) const;
    
    double get_sum_log_dy(
            unsigned int pix,
            unsigned int sample) const;
    
    uint16_t get_dominant_dist(
            unsigned int pix,
            unsigned int sample) const;

    uint16_t get_n_dominant_dist_samples(
            unsigned int pix,
            unsigned int dist) const;

    const std::vector<double> get_prior() const;
    double get_prior(unsigned int pix,
                     unsigned int sample) const;
    double get_likelihood(unsigned int pix,
                          unsigned int sample) const;
    
    unsigned int get_n_pix() const;
    unsigned int get_n_samples() const;
    unsigned int get_n_dists() const;
    
    double get_inv_cov(
            unsigned int dist,
            unsigned int pix0,
            unsigned int pix1) const;
    
    bool data_loaded() const;
    
    // Setters
    void set_delta(
            double value,
            unsigned int pix,
            unsigned int sample,
            unsigned int dist);
    
    void set_log_dy(
            double value,
            unsigned int pix,
            unsigned int sample,
            unsigned int dist);
    
    void set_sum_log_dy(
            double value,
            unsigned int pix,
            unsigned int sample);
    
    // Calculate statistics
    double get_inv_var(unsigned int pix,
                       unsigned int dist) const;

    double calc_mean(
            unsigned int pix,
            unsigned int dist,
            const std::vector<uint16_t>& sample) const;
    
    double calc_mean_shifted(
            unsigned int pix,
            unsigned int dist,
            const std::vector<uint16_t>& sample,
            const double shift_weight,
            unsigned int start_pix=0) const;
    
    double calc_lnprob(const std::vector<uint16_t>& sample) const;

    double calc_lnprob_shifted(
            const std::vector<uint16_t>& sample,
            const double shift_weight,
            const bool add_eff_prior=true) const;
    
    // Initialization
    bool load_neighbor_list(
            uint32_t nside_center, uint32_t pix_idx_center,
            const std::string& neighbor_lookup_fname);
    
    bool lookup_pixel_files(
            const std::string& pixel_lookup_fname,
            std::vector<int32_t>& file_idx);
    
    bool load_neighbor_los(
            const std::string& output_fname_pattern,
            const std::vector<int32_t>& file_idx,
            int n_samples_max);
    
    void apply_priors(
            const std::vector<double>& mu,
            const std::vector<double>& sigma,
            double reddening_scale);
    
    void apply_priors(
            const std::vector<double>& mu,
            double sigma,
            double reddening_scale);

    void apply_priors_indiv(
            const std::vector<double>& mu,
            const std::vector<double>& sigma,
            double reddening_scale,
            int pix,
            int sample);
    
    void apply_priors_inner(
            int pix, int sample, int dist,
            double mu, double sigma,
            double log_scale);
    
    void init_covariance(
            double scale,
            double d_soft,
            double gamma_soft);

    void init_dominant_dist(int verbosity=0);
};


#endif // _NEIGHBOR_PIXELS_H__
