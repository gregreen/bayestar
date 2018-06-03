/*
 * neighbor_pixels.cpp
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


#include "neighbor_pixels.h"


/****************************************************************************************************************************
 *
 * TNeighborPixels 
 *
 ****************************************************************************************************************************/


TNeighborPixels::TNeighborPixels(
        uint32_t nside, uint32_t pix_idx,
        const std::string& neighbor_lookup_fname,
        const std::string& pixel_lookup_fname)
{
    // TODO

    // Lookup neighboring pixels
    load_neighbor_list(nside, pix_idx, neighbor_lookup_fname);

    // Lookup pixel locations

    // Load in neighboring pixels
    
}


TNeighborPixels::~TNeighborPixels() {}


bool TNeighborPixels::load_neighbor_list(
        uint32_t nside, uint32_t pix_idx,
        const std::string& neighbor_lookup_fname)
{
	std::unique_ptr<H5::H5File> f = H5Utils::openFile(
            neighbor_lookup_fname,
            H5Utils::READ);
    if(!f) { return false; }
    
    std::unique_ptr<H5::DataSet> dataset = healtree_get_dataset(
            *f, nside, pix_idx);
    if(!dataset) { return false; }
    
    // Datatype
    H5::DataType dtype = H5::PredType::NATIVE_UINT32;
    
	// Dataspace
	hsize_t dims[3]; // (entry, nside, pix_idx)
	H5::DataSpace dataspace = dataset->getSpace();
	dataspace.getSimpleExtentDims(&(dims[0]));
    hsize_t length = dims[0] * dims[1] * dims[2];

	// Read in dataset
	uint32_t* buf = new uint32_t[length];
	dataset->read(buf, dtype);
    
    // Search for the correct entry
    int entry = 0;
    int idx = 0;
    int entry_length = dims[1] * dims[2];
    bool located = false;
    for(; entry<dims[0]; entry++, idx+=entry_length) {
        if((buf[idx] == nside) && (buf[idx+1] == pix_idx)) {
            located = true;
            break;
        }
    }
    
    if(!located) { return false; }
    
    // Read in (nside, pix_idx) pairs from entry
    nside_pix_idx.reserve(entry_length);
    for(int i=entry_length*entry; i<entry_length*(entry+1); i++) {
        nside_pix_idx.push_back(buf[i]);
    }
    
    return true;
}


//void TNeighborPixels::init_neighbor_info(
//        const std::vector<double>& lon,
//        const std::vector<double>& lat,
//        const TGalacticLOSModel& gal_los_model)
//{ //    n_neighbors = lon.size(); //    
//    double l, b;
//    gal_los_model.get_lb(l, b);
//    
//    neighbor_lon = std::make_shared<std::vector<double> >();
//    neighbor_lat = std::make_shared<std::vector<double> >();
//    neighbor_lon.reserve(n_neighbors+1);
//    neighbor_lat.reserve(n_neighbors+1);
//    
//    neighbor_lon->push_back(l);
//    neighbor_lat->push_back(b);
//    neighbor_lon->insert(neighbor_lon->end(), lon.begin(), lon.end());
//    neighbor_lat->insert(neighbor_lon->end(), lon.begin(), lon.end());
//    
//    // TODO: Load in real data 
//    
//}


void TNeighborPixels::init_covariance(
        double scale)
{
    //
    std::vector<double> dist;
	double dmu = (dm_max - dm_min) / (double)(n_dists);
    double mu;
    for(int i=0; i<n_dists; i++) {
        mu = dm_min + i * dmu;
        dist.push_back(std::pow(10., 0.2*mu + 1.));
    }
    
    double scale_coeff = -1. / scale;
    std::function<double(double)> kernel = [scale_coeff](double d2) -> double {
        return std::exp(scale_coeff * std::sqrt(d2));
    };
    
    inv_cov.clear();
    
    inv_cov_lonlat(lon, lat, dist, kernel, inv_cov);
    
    // TODO: Calculate A_cond for central and each neighbor,
    //       or for i and \i.
    //conditional_gaussian_scalar(
    //    SharedMatrixXd& C_inv, 0,
    //    inv_var, SharedMatrixXd& A_cond);
    
    inv_var.clear();
    inv_var.reserve(n_pix*n_dists);
    for(int pix=0; pix<n_pix; pix++) {
        for(int dist=0; dist<n_dists; dist++) {
            inv_var.push_back((*(inv_cov[dist]))(pix, pix));
        }
    }
}


double TNeighborPixels::get_inv_var(
        unsigned int pix,
        unsigned int dist) const
{
    return inv_var[pix*n_dists + dist];
}


double TNeighborPixels::get_delta(
        unsigned int pix,
        unsigned int sample,
        unsigned int dist) const
{
    return delta[(pix*n_samples + sample)*n_dists + dist];
}


double TNeighborPixels::calc_mean(
        unsigned int pix,
        unsigned int dist,
        const std::vector<uint32_t>& sample) const
{
    double mu = 0.;
    for(int i=0; i<pix; i++) {
        mu += (*(inv_cov[dist]))(pix, i) * get_delta(i, sample[i], dist);
    }
    for(int i=pix+1; i<n_pix; i++) {
        mu += (*(inv_cov[dist]))(pix, i) * get_delta(i, sample[i], dist);
    }
    mu *= -1. / get_inv_var(pix, dist);
    return mu;
}


