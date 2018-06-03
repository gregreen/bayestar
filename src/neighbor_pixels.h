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


#include <string>
#include <vector>
#include <memory>
#include <cmath>

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
    
    // Prior stored for each neighbor
    std::vector<double> prior;
    
    // Locations of neibhoring pixels
    std::vector<double> lon, lat;
    std::vector<uint32_t> nside_pix_idx; // Pairs of (nside, pix_idx)
    
    // Inverse covariance matrix for each distance
    std::vector<UniqueMatrixXd> inv_cov;
    
    std::vector<double> A_i_given_noti;
    std::vector<double> inv_var; // shape = (pix, dist)
    
public:
    // Constructor/destructor
    TNeighborPixels(uint32_t nside, uint32_t pix_idx,
                    const std::string& neighbor_lookup_fname,
                    const std::string& pixel_lookup_fname);
    ~TNeighborPixels();
    
    // Getters
    double get_delta(
            unsigned int pix,
            unsigned int sample,
            unsigned int dist) const;
    
    const std::vector<double> get_prior() const;
    
    unsigned int get_n_pix() const;
    unsigned int get_n_samples() const;
    unsigned int get_n_dists() const;
    
    // Setters
    
    // Calculate statistics
    double get_inv_var(unsigned int pix,
                       unsigned int dist) const;
    double calc_mean(
            unsigned int pix,
            unsigned int dist,
            const std::vector<uint32_t>& sample) const;
    
    // Initialization
   bool load_neighbor_list(
            uint32_t nside, uint32_t pix_idx,
            const std::string& neighbor_lookup_fname);
    
    void init_covariance(double scale);
};


#endif // _NEIGHBOR_PIXELS_H__
