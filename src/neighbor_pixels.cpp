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
        uint32_t nside_center, uint32_t pix_idx_center,
        const std::string& neighbor_lookup_fname,
        const std::string& pixel_lookup_fname,
        const std::string& output_fname_pattern)
{
    // TODO

    // Lookup neighboring pixels
    bool status;
    status = load_neighbor_list(
                nside_center,
                pix_idx_center,
                neighbor_lookup_fname);
    if(!status) {
        std::cerr << "Failed to load list of neighbors!"
                  << std::endl;
    }

    // Lookup pixel locations
    std::vector<int32_t> file_idx;
    status = lookup_pixel_files(pixel_lookup_fname, file_idx);
    if(!status) {
        std::cerr << "Failed to load list of output files containing neighbors!"
                  << std::endl;
    }

    // Load in neighboring pixels
    status = load_neighbor_los(output_fname_pattern, file_idx);
    if(!status) {
        std::cerr << "Failed to load neighboring sightline data!"
                  << std::endl;
    }
}


TNeighborPixels::~TNeighborPixels() {}


bool TNeighborPixels::load_neighbor_list(
        uint32_t nside_center, uint32_t pix_idx_center,
        const std::string& neighbor_lookup_fname)
{
    //std::cerr << "Loading " << neighbor_lookup_fname << " ..."
    //          << std::endl;
	std::unique_ptr<H5::H5File> f = H5Utils::openFile(
            neighbor_lookup_fname,
            H5Utils::READ);
    //std::cerr << "Loaded." << std::endl;
    if(!f) {
        std::cerr << "Could not open neighbor lookup file!"
                  << std::endl;
        return false;
    }
    
    std::unique_ptr<H5::DataSet> dataset = healtree_get_dataset(
            *f, nside_center, pix_idx_center);
    if(!dataset) {
        std::cerr << "Could not locate neighbor lookup dataset!"
                  << std::endl;
        return false;
    }
    
    // Datatype
    H5::DataType dtype = H5::PredType::NATIVE_INT32;
    
	// Dataspace
	hsize_t dims[3]; // (entry, neighbors, nside pix_idx)
	H5::DataSpace dataspace = dataset->getSpace();
	dataspace.getSimpleExtentDims(&(dims[0]));
    hsize_t length = dims[0] * dims[1] * dims[2];
    
    if(dims[2] != 2) {
        std::cerr << neighbor_lookup_fname << ": ("
                  << nside_center << ", " << pix_idx_center
                  << ") does not have correct shape."
                  << std::endl;
    }
    
	// Read in dataset
	int32_t* buf = new int32_t[length];
	dataset->read(buf, dtype);
    
    // Search for the correct entry
    int entry = 0;
    int idx = 0;
    int entry_length = dims[1] * dims[2];
    bool located = false;
    for(; entry<dims[0]; entry++, idx+=entry_length) {
        if((buf[idx] == nside_center) &&
           (buf[idx+1] == pix_idx_center))
        {
            located = true;
            break;
        }
    }
    
    if(!located) {
        std::cerr << "Could not locate pixel in neighbor lookup!"
                  << std::endl;
        delete[] buf;
        return false;
    }
    
    // Read in (nside, pix_idx) pairs from entry
    nside.reserve(dims[1]);
    pix_idx.reserve(dims[1]);
    for(int i=entry_length*entry; i<entry_length*(entry+1); i+=2) {
        if(buf[i+1] < 0) { continue; } // Ignore invalid pixels
        nside.push_back((uint32_t)buf[i]);
        pix_idx.push_back((uint32_t)buf[i+1]);
    }
    
    std::cerr << "There are " << nside.size() - 1
              << " neighboring pixels." << std::endl;
    
    delete[] buf;
    return true;
}


bool TNeighborPixels::lookup_pixel_files(
        const std::string& pixel_lookup_fname,
        std::vector<int32_t>& file_idx)
{
    // Load the lookup table for (nside, pix_idx) -> file_idx
	std::unique_ptr<H5::H5File> f = H5Utils::openFile(
            pixel_lookup_fname,
            H5Utils::READ);
    if(!f) {
        std::cerr << "Could not open pixel lookup file!"
                  << std::endl;
        return false;
    }
    
    // Initialize file_idx vector
    file_idx.clear();
    file_idx.reserve(nside.size());
    
    // Datatype
    H5::DataType dtype = H5::PredType::NATIVE_INT32;
    
    // Loop through neighboring pixels
    for(int i=0; i<nside.size(); i+=2) {
        std::unique_ptr<H5::DataSet> dataset = healtree_get_dataset(
                *f, nside.at(i), pix_idx.at(i));
        if(!dataset) {
            std::cerr << "Could not locate neighbor lookup dataset ("
                      << nside.at(i) << ", " <<  pix_idx.at(i) << ")!"
                      << std::endl;
            return false;
        }
    
        // Dataspace
        hsize_t dims[2]; // (entry, nside pix_idx file_idx)
        H5::DataSpace dataspace = dataset->getSpace();
        dataspace.getSimpleExtentDims(&(dims[0]));
        hsize_t length = dims[0] * dims[1];
        
        if(dims[1] != 3) {
            std::cerr << pixel_lookup_fname << ": ("
                      << nside.at(i) << ", " << pix_idx.at(i)
                      << ") does not have correct shape."
                      << std::endl;
        }

        // Read in dataset
        int32_t* buf = new int32_t[length];
        dataset->read(buf, dtype);
    
        // Search for the correct entry
        int entry = 0;
        int idx = 0;
        int entry_length = dims[1];
        bool located = false;
        for(; entry<dims[0]; entry++, idx+=entry_length) {
            if((buf[idx] == nside.at(i)) && (buf[idx+1] == pix_idx.at(i))) {
                // Read in file_idx from entry
                file_idx.push_back(buf[idx+2]);
                located = true;
                break;
            }
        }
        
        if(!located) {
            std::cerr << "Could not locate ("
                      << nside.at(i) << ", " << pix_idx.at(i)
                      << ") in pixel lookup!"
                      << std::endl;
            delete[] buf;
            return false;
        }
        
        delete[] buf;
    }
    
    return true;
}


bool TNeighborPixels::load_neighbor_los(
        const std::string& output_fname_pattern,
        const std::vector<int32_t>& file_idx)
{
    // Datatype
    H5::DataType dtype = H5::PredType::NATIVE_DOUBLE;
    
    // Do an argsort of the file indices, so that all the pixels
    // that reside in the same file will be handled in a row
    std::vector<std::pair<int32_t,int32_t> > file_idx_sort;
    for(int32_t i=0; i<file_idx.size(); i++) {
        file_idx_sort.push_back(std::make_pair(file_idx.at(i), i));
    }
    std::sort(file_idx_sort.begin(), file_idx_sort.end());
    
    int32_t file_idx_current = -1;
    std::unique_ptr<H5::H5File> f = nullptr;
    
    // Loop through file indices
    for(auto p : file_idx_sort) {
        int32_t fidx = p.first;
        int32_t i = p.second;
        
        if(fidx < 0) { continue; } // Ignore invalid file indices
        
        // Only open a new file when necessary
        if(fidx != file_idx_current) {
            //std::cerr << "output_fname_pattern = " << output_fname_pattern << std::endl;
            //std::cerr << "fidx = " << std::to_string(fidx) << std::endl;
            auto fname_size = std::snprintf(nullptr, 0, output_fname_pattern.c_str(), fidx);
            std::string fname(fname_size+1, '\0');
            std::sprintf(&fname[0], output_fname_pattern.c_str(), fidx);
            std::cerr << "Opening output file " << fname
                      << " ..." << std::endl;
            //f.reset();
            f = H5Utils::openFile(fname, H5Utils::READ);
            if(!f) {
                std::cerr << "Could not open output file "
                          << fname << " !"
                          << std::endl;
                return false;
            }
            file_idx_current = fidx;
        }
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


