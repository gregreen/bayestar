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
        const std::string& output_fname_pattern,
        int n_samples_max)
{
    // Initialize some variables to dummy values
    n_pix = 0;
    n_samples = 0;
    n_dists = 0;
    dm_min = -9999.;
    dm_max = -9999.;
    
    // Lookup neighboring pixels
    bool status;
    status = load_neighbor_list(
                nside_center,
                pix_idx_center,
                neighbor_lookup_fname);
    if(!status) {
        std::cerr << "Failed to load list of neighbors!"
                  << std::endl;
        loaded = false;
    }

    // Lookup pixel locations
    std::vector<int32_t> file_idx;
    status = lookup_pixel_files(pixel_lookup_fname, file_idx);
    if(!status) {
        std::cerr << "Failed to load list of output files "
                  << "containing neighbors!"
                  << std::endl;
        loaded = false;
    }

    // Load in neighboring pixels
    status = load_neighbor_los(output_fname_pattern, file_idx, n_samples_max);
    if(!status) {
        std::cerr << "Failed to load neighboring sightline data!"
                  << std::endl;
        loaded = false;
    }
    
    // Successfully loaded data
    loaded = true;
}


TNeighborPixels::~TNeighborPixels() {}


bool TNeighborPixels::data_loaded() const {
    return loaded;
}


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
    for(int i=0; i<nside.size(); i++) {
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
        const std::vector<int32_t>& file_idx,
        int n_samples_max)
{
    // Set number of pixels
    n_pix = file_idx.size();
    
    // Do an argsort of the file indices, so that all the pixels
    // that reside in the same file will be handled in a row
    std::vector<std::pair<int32_t,int32_t> > file_idx_sort;
    for(int32_t i=0; i<file_idx.size(); i++) {
        file_idx_sort.push_back(std::make_pair(file_idx.at(i), i));
        //std::cerr << "file_idx[" << i << "] = " << file_idx.at(i) << std::endl;
    }
    std::sort(file_idx_sort.begin(), file_idx_sort.end());
    
    int32_t file_idx_current = -1;
    std::unique_ptr<H5::H5File> f = nullptr;

    // Clear priors and likelihoods
    prior.clear();
    likelihood.clear();
    
    // Loop through file indices
    for(auto p : file_idx_sort) {
        int32_t fidx = p.first;
        int32_t i = p.second;
        uint32_t nside_current = nside.at(i);
        uint32_t pix_idx_current = pix_idx.at(i);
        
        // TODO: Ignore invalid file indices before setting n_pix
        if(fidx < 0) { continue; } // Ignore invalid file indices
        
        // Only open a new file when necessary
        if(fidx != file_idx_current) {
            //std::cerr << "output_fname_pattern = " << output_fname_pattern << std::endl;
            //std::cerr << "fidx = " << std::to_string(fidx) << std::endl;
            auto fname_size = std::snprintf(
                nullptr,
                0,
                output_fname_pattern.c_str(),
                fidx
            );
            std::string fname(fname_size+1, '\0');
            std::sprintf(&fname[0], output_fname_pattern.c_str(), fidx);
            //std::cerr << "Opening output file " << fname
            //          << " ..." << std::endl;
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

        // Load l.o.s. dataset
        std::stringstream group_name;
        group_name << "/pixel " << nside_current
                   << "-" << pix_idx_current;
        std::stringstream dset_name;
        dset_name << group_name.str() << "/discrete-los";
        std::unique_ptr<H5::DataSet> dataset
            = H5Utils::openDataSet(*f, dset_name.str());
        if(!dataset) {
            std::cerr << "Failed to open dataset "
                      << dset_name.str()
                      << " !" << std::endl;
            return false;
        }

        // Dataspace
        hsize_t dims[3]; // (null, GR best samples, prob distances)
        H5::DataSpace dataspace = dataset->getSpace();
        dataspace.getSimpleExtentDims(&(dims[0]));
        hsize_t length =  dims[1] * dims[2];
        
        // Set dimensions
        if(n_samples == 0) {
            n_samples = dims[1] - 2; // (GR, best, samples)
            if((n_samples_max > 0) && (n_samples > n_samples_max)) {
                n_samples = n_samples_max;
            }
        }
        if(n_dists == 0) {
            n_dists = dims[2] - 2; // (likelihood, prior, distances)
        }
        
        // Check dimensions
        //if(dims[0] > 1) {
        //    std::cerr << "Ignoring " << dims[0]-1 << " higher-temperature samples "
        //              << "in neighboring pixels (" << dset_name.str() << ")"
        //              << std::endl;
        //}
        if(dims[1] < n_samples+2) {
            std::cerr << "Not enough samples in dataset "
                      << dset_name.str()
                      << " !" << std::endl;
            return false;
        }
        if(dims[2] < n_dists+2) {
            std::cerr << "Not enough distance bins in dataset "
                      << dset_name.str()
                      << " !" << std::endl;
            return false;
        }
        
        // Read in dataset
        hsize_t mem_shape[1] = {dims[1] * dims[2]};
        H5::DataSpace memspace(1, &(mem_shape[0]));
        
        hsize_t sel_shape[3] = {1, dims[1], dims[2]};
        hsize_t sel_offset[3] = {0, 0, 0};
        dataspace.selectHyperslab(H5S_SELECT_SET, &(sel_shape[0]), &(sel_offset[0]));
        
        float* buf = new float[length];
        dataset->read(buf, H5::PredType::NATIVE_FLOAT, memspace, dataspace);
        
        // Copy into class data structure
        if(delta.size() == 0) {
            delta.resize(n_pix * n_samples * n_dists);
        }
        
        if(prior.size() == 0) {
            prior.resize(n_pix*n_samples);
        }
        
        if(likelihood.size() == 0) {
            likelihood.resize(n_pix*n_samples);
        }

        if(log_dy.size() == 0) {
            log_dy.resize(n_pix*n_samples*n_dists);
        }

        if(sum_log_dy.size() == 0) {
            sum_log_dy.assign(n_pix*n_samples, 0.);
        }

        uint32_t buf_idx;
        for(int sample=0; sample<n_samples; sample++) {
            // Likelihood
            buf_idx = dims[2] * (sample+2);
            likelihood.at(n_samples*i + sample) = buf[buf_idx];
            
            // Prior
            buf_idx = dims[2] * (sample+2) + 1;
            prior.at(n_samples*i + sample) = buf[buf_idx];
            
            // Line-of-sight reddening
            double sum_log_dy_tmp = 0.;
            double y_last = 0.;
            double y, dy, log_dy_tmp;
            for(int dist=0; dist<n_dists; dist++) {
                buf_idx = dims[2] * (sample+2) + (dist+2);
                y = buf[buf_idx];
                set_delta(y, i, sample, dist);
                
                dy = y - y_last;
                log_dy_tmp = std::log(dy);
                //if(dy > 1.e-8) {
                //    log_dy_tmp = (dy+1.) * std::log(dy+1) - dy * std::log(dy) - 1;
                //} else {
                //    log_dy_tmp = -1.;
                //}
                set_log_dy(log_dy_tmp, i, sample, dist);
                sum_log_dy_tmp += get_log_dy(i, sample, dist);
                y_last = y;
            }

            set_sum_log_dy(sum_log_dy_tmp, i, sample);
        }
        
        //std::cerr << "Loaded output from " << dset_name.str()
        //          << std::endl;

        // Load attributes
        if(dm_min < -99.) {
            dm_min = H5Utils::read_attribute<double>(*dataset, "DM_min");
        }
        if(dm_max < -99.) {
            dm_max = H5Utils::read_attribute<double>(*dataset, "DM_max");
        }

        double lon_tmp, lat_tmp;
        H5::Group group = f->openGroup(group_name.str());
        H5::Attribute att_lon = group.openAttribute("l");
        att_lon.read(H5::PredType::NATIVE_DOUBLE, &lon_tmp);
        H5::Attribute att_lat = group.openAttribute("b");
        att_lat.read(H5::PredType::NATIVE_DOUBLE, &lat_tmp);
        lon.push_back(lon_tmp);
        lat.push_back(lat_tmp);
        
        //std::cerr << "Loaded attributes related to " << dset_name.str()
        //          << std::endl;
        
        delete[] buf;
    }
    
    return true;
}


void TNeighborPixels::apply_priors(
        const std::vector<double>& mu,
        const std::vector<double>& sigma,
        double reddening_scale)
{
    // Transforms the stored deltas from raw cumulative reddenings,
    // in units of pixels, to scores for the log-normal prior.
    // First, cumulative reddenings are transformed into differential
    // reddenings. Then, the log is taken, the mean of the log reddening
    // is subtracted out, and the result is divided by sigma.

    assert( mu.size() == sigma.size() );
    assert( mu.size() == n_dists );
    
    double log_scale = log(reddening_scale);
    
    for(int dist=n_dists-1; dist != -1; dist--) {
        for(int pix=0; pix<n_pix; pix++) {
            for(int sample=0; sample<n_samples; sample++) {
                apply_priors_inner(
                    pix, sample, dist,
                    mu.at(dist), sigma.at(dist),
                    log_scale);
            }
        }
    }
}


void TNeighborPixels::apply_priors_inner(
        int pix, int sample, int dist,
        double mu, double sigma,
        double log_scale)
{
    // Calculate the increase in reddening at this distance
    double d = get_delta(pix, sample, dist);
    if(dist != 0) {
        d -= get_delta(pix, sample, dist-1);
    }
    
    if((d < 1.e-5) && (mu < log_scale)) {
        // If the inferred reddening is zero, and the
        // prior on the mean of the log reddening is less
        // than one pixel, then no penalty.
        d = 0;
    } else if(d < 1.e-5) {
        // If the inferred reddening is zero, but the
        // prior on the mean of the log reddening is greater
        // than one pixel, then impose a penalty that
        // increases smoothly. This essentially assumes that
        // the inferred reddening is one pixel.
        d = (log_scale - mu) / sigma;
    } else {
        // The normal case: inferred reddening is greater than
        // zero.
        d = (log_scale + log(d) - mu) / sigma;
    }
    set_delta(d, pix, sample, dist);
}


void TNeighborPixels::apply_priors(
            const std::vector<double>& mu,
            double sigma,
            double reddening_scale)
{
    std::vector<double> s;
    s.reserve(mu.size());
    for(int i=0; i<mu.size(); i++) {
        s.push_back(sigma);
    }
    apply_priors(mu, s, reddening_scale);
}


void TNeighborPixels::apply_priors_indiv(
        const std::vector<double>& mu,
        const std::vector<double>& sigma,
        double reddening_scale,
        int pix,
        int sample)
{
    //if(pix == 0) {
    //    std::cerr << "0 : " << get_delta(pix, sample, 0) << std::endl;
    //    for(int dist=1; dist<n_dists; dist++) {
    //        std::cerr << dist << " : "
    //                  << get_delta(pix, sample, dist) - get_delta(pix, sample, dist-1)
    //                  << std::endl;
    //    }
    //}
    double log_scale = log(reddening_scale);
    for(int dist=n_dists-1; dist != -1; dist--) {
        apply_priors_inner(
            pix, sample, dist,
            mu.at(dist), sigma.at(dist),
            log_scale);
    }
    //if(pix == 0) {
    //    std::cerr << "-->" << std::endl;
    //    for(int dist=0; dist<n_dists; dist++) {
    //        std::cerr << dist << " : "
    //                  << get_delta(pix, sample, dist)
    //                  << std::endl;
    //    }
    //}
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
        double scale,
        double d_soft,
        double gamma_soft)
{
    // For each distance, initializes the covariance matrix
    // describing the correlations between neighboring pixels.
    //
    // Input:
    //     scale: Correlation scale, in pc.

    // Calculate the distances
    //std::cerr << "Calculating distances ..." << std::endl;
    
    std::vector<double> dist;   // In pc
    double dmu = (dm_max - dm_min) / (double)(n_dists);
    double mu;  // Distance modulus, in mag
    //std::cerr << "dm in (" << dm_min << ", " << dm_max << ")"
    //          << std::endl;
    for(int i=0; i<n_dists; i++) {
        mu = dm_min + i * dmu;
        dist.push_back(std::pow(10., 0.2*mu + 1.));
    }

    std::cerr << std::endl << "(lon,lat) of neighbors:" << std::endl;
    for(int i=0; i<lon.size(); i++) {
        std::cerr << "(" << lon.at(i) << ", "
                  << lat.at(i) << ")" << std::endl;
    }
    std::cerr << std::endl;
    
    double scale_coeff = -1. / scale;
    std::function<double(double)> kernel
        = [d_soft, gamma_soft, scale_coeff](double d2) -> double
    {
        if(d2 > 1.e-8) {
            double d_eff = std::pow(d2, gamma_soft/2.);
            d_eff += std::pow(d_soft, gamma_soft);
            d_eff = std::pow(d_eff, 1./gamma_soft);
            //std::cerr << "d: " << std::sqrt(d2) << " -> " << d_eff
            //          << std::endl;
            return std::exp(scale_coeff * d_eff);
        } else {
            return 1.;
        }
        //double xi = scale_coeff * std::sqrt(d2);
        //return 1. / (std::exp(xi) + std::exp(-xi));
        //return std::exp(scale_coeff * std::sqrt(d2));
    };
    
    //std::cerr << "Initializing covariance matrices ..." << std::endl;

    inv_cov.clear();
    inv_cov_lonlat(lon, lat, dist, kernel, inv_cov);
    
    // TODO: Calculate A_cond for central and each neighbor,
    //       or for i and \i.
    //conditional_gaussian_scalar(
    //    SharedMatrixXd& C_inv, 0,
    //    inv_var, SharedMatrixXd& A_cond);
    
    //std::cerr << "Reading off inverse variances ..." << std::endl;

    inv_var.clear();
    inv_var.reserve(n_pix*n_dists);
    for(int pix=0; pix<n_pix; pix++) {
        for(int dist=0; dist<n_dists; dist++) {
            inv_var.push_back((*(inv_cov[dist]))(pix, pix));
        }
    }
    
    //std::cerr << "Done initializing covariance matrices." << std::endl;
}


void TNeighborPixels::init_dominant_dist(int verbosity) {
    // For each sample of each pixel, calculates distance
    // with largest deviation from priors.
    
    dominant_dist.clear();
    dominant_dist.reserve(n_pix*n_samples);

    n_dominant_dist_samples.resize(n_pix*n_dists);
    std::fill(
        n_dominant_dist_samples.begin(),
        n_dominant_dist_samples.end(),
        0.
    );

    double delta_max, delta_tmp;
    uint16_t dist_max;
    
    for(int pix=0; pix<n_pix; pix++) {
        for(int samp=0; samp<n_samples; samp++) {
            delta_max = -1;
            for(int dist=0; dist<n_dists; dist++) {
                delta_tmp = std::fabs(get_delta(pix, samp, dist));

                if(delta_tmp > delta_max) {
                    dist_max = dist;
                    delta_max = delta_tmp;
                }
            }
            dominant_dist.push_back(dist_max);
            n_dominant_dist_samples.at(n_dists*pix + dist_max)++;
        }
    }
    
    if(verbosity >= 2) {
        std::cerr << std::endl
                  << "Dominant distance histograms:"
                  << std::endl << std::endl;
        
        int h_max = 20;
        for(int pix=0; pix<n_pix; pix++) {
            std::string msg(h_max*(n_dists+1), ' ');

            for(int j=0; j<h_max; j++) {
                msg[(j+1)*(n_dists+1) - 1] = '\n';
            }

            for(int dist=0; dist<n_dists; dist++) {
                int h = n_dominant_dist_samples[n_dists*pix + dist]/2;
                if(h > h_max) { h = h_max; }
                
                int base = (h_max-1)*(n_dists+1) + dist;

                for(int j=0; j<h; j++) {
                    int idx = base - j*(n_dists+1);
                    msg[idx] = '*';
                }
            }
            std::cerr << msg;

            for(int dist=0; dist<n_dists; dist++) {
                std::cerr << "-";
            }
            std::cerr << std::endl;

            for(int dist=0; dist<n_dists; dist++) {
                std::cerr << (dist % 10 ? ' ' : '|');
            }
            std::cerr << std::endl << std::endl;
        }
    }
}


uint16_t TNeighborPixels::get_dominant_dist(
        unsigned int pix,
        unsigned int sample) const
{
    return dominant_dist[n_samples*pix + sample];
}


uint16_t TNeighborPixels::get_n_dominant_dist_samples(
        unsigned int pix,
        unsigned int dist) const
{
    return dominant_dist[n_dists*pix + dist];
}


double TNeighborPixels::get_inv_var(
        unsigned int pix,
        unsigned int dist) const
{
    // Returns 1/sigma^2 for the specified (pixel, distance bin)
    // combination.

    return inv_var[pix*n_dists + dist];
}


double TNeighborPixels::get_delta(
        unsigned int pix,
        unsigned int sample,
        unsigned int dist) const
{
    return delta[(pix*n_samples + sample)*n_dists + dist];
}


void TNeighborPixels::set_delta(
        double value,
        unsigned int pix,
        unsigned int sample,
        unsigned int dist)
{
    delta[(pix*n_samples + sample)*n_dists + dist] = value;
}


double TNeighborPixels::get_log_dy(
        unsigned int pix,
        unsigned int sample,
        unsigned int dist) const
{
    return log_dy[(pix*n_samples + sample)*n_dists + dist];
}


void TNeighborPixels::set_log_dy(
        double value,
        unsigned int pix,
        unsigned int sample,
        unsigned int dist)
{
    if(!std::isfinite(value)) {
        value = -0.4054651; // -ln(1.5)
    }
    log_dy[(pix*n_samples + sample)*n_dists + dist] = value;
}


double TNeighborPixels::get_sum_log_dy(
        unsigned int pix,
        unsigned int sample) const
{
    return sum_log_dy[pix*n_samples + sample];
}


void TNeighborPixels::set_sum_log_dy(
        double value,
        unsigned int pix,
        unsigned int sample)
{
    sum_log_dy[pix*n_samples + sample] = value;
}


double TNeighborPixels::get_inv_cov(
        unsigned int dist,
        unsigned int pix0,
        unsigned int pix1) const
{
    //std::cerr << "pix0, pix1, dist = ("
    //          << pix0 << ", "
    //          << pix1 << ", "
    //          << dist << ")"
    //          << std::endl;
    return (*(inv_cov[dist]))(pix0, pix1);
}


double TNeighborPixels::calc_mean(
        unsigned int pix,
        unsigned int dist,
        const std::vector<uint16_t>& sample) const
{
    // Calculates the mean of the specified pixel, given that
    // the specified samples are chosen for the other pixels.
    //
    // Inputs:
    //     pix: index of pixel to compute mean for
    //     dist: distance bin to compute mean for
    //     sample: Which sample to choose for each pixel. Should
    //             have the same length as the total number of
    //             pixels. The pixel corresponding to `pix` will
    //             be ignored.
    
    //std::cerr << "(pix, dist) = (" << pix << ", " << dist << ")" << std::endl;
    //std::cerr << "inv_cov[dist].shape = (" << inv_cov[dist]->rows()
    //          << ", " << inv_cov[dist]->cols() << ")" << std::endl;

    //std::cerr << "Calculating mean of (pix, dist) = ("
    //          << pix << ", " << dist << ")" << std::endl;

    double mu = 0.;
    for(int i=0; i<pix; i++) {
        //std::cerr << "  pix = " << i << ":"
        //          << " inv_cov = " << (*(inv_cov[dist]))(pix, i) << ","
        //          << " delta = " << get_delta(i, sample[i], dist)
        //          << std::endl;
        mu += (*(inv_cov[dist]))(pix, i) * get_delta(i, sample[i], dist);
    }
    for(int i=pix+1; i<n_pix; i++) {
        //std::cerr << "  pix = " << i << ":"
        //          << " inv_cov = " << (*(inv_cov[dist]))(pix, i) << ","
        //          << " delta = " << get_delta(i, sample[i], dist) << std::endl;
        mu += (*(inv_cov[dist]))(pix, i) * get_delta(i, sample[i], dist);
    }
    mu *= -1. / get_inv_var(pix, dist);

    //std::cerr << " --> mu = " << mu << std::endl;

    return mu;
}


double TNeighborPixels::calc_mean_shifted(
        unsigned int pix,
        unsigned int dist,
        const std::vector<uint16_t>& sample,
        const double shift_weight,
        unsigned int start_pix) const
{
    // Calculates the mean of the specified pixel, given that
    // the specified samples are chosen for the other pixels.
    //
    // An additional "shift" term is added into the inverse covariance
    // matrix, which couples a given distance of the central pixel
    // with the neighboring distances of the neighboring pixels.
    // Neighboring distances of the central pixel are not coupled.
    // This shift term deforms the prior, encouraging transitions between
    // states in which a reddening jump occurs in neighboring distances.
    // Taking <shift_weight> to zero recovers the unmodified prior.
    //
    // Inputs:
    //     pix: index of pixel to compute mean for
    //     dist: distance bin to compute mean for
    //     sample: Which sample to choose for each pixel. Should
    //             have the same length as the total number of
    //             pixels. The pixel corresponding to `pix` will
    //             be ignored.
    //     shift_weight: A small positive constant (<< 1) which
    //                   parameterizes the strength of the coupling
    //                   between neighboring distances.
    
    double mu = 0.;
    
    Eigen::MatrixXd& inv_cov_0 = *(inv_cov[dist]);
    
    double norm = 1. + 2.*shift_weight;

    if(dist == 0) {
        Eigen::MatrixXd& inv_cov_p1 = *(inv_cov[dist+1]);
        //norm = 1. + shift_weight;
        for(int i=start_pix; i<pix; i++) {
            double icov_0 = inv_cov_0(pix, i);
            double icov_p1 = 0.5 * (icov_0 + inv_cov_p1(pix, i));
            mu += icov_0 * get_delta(i, sample[i], dist)
               + shift_weight * (
                   icov_p1 * get_delta(i, sample[i], dist+1)
               );
        }
        for(int i=pix+1; i<n_pix; i++) {
            double icov_0 = inv_cov_0(pix, i);
            double icov_p1 = 0.5 * (icov_0 + inv_cov_p1(pix, i));
            mu += icov_0 * get_delta(i, sample[i], dist)
               + shift_weight * (
                   icov_p1 * get_delta(i, sample[i], dist+1)
               );
        }
    } else if(dist == n_dists-1) {
        Eigen::MatrixXd& inv_cov_m1 = *(inv_cov[dist-1]);
        //norm = 1. + shift_weight;
        for(int i=start_pix; i<pix; i++) {
            double icov_0 = inv_cov_0(pix, i);
            double icov_m1 = 0.5 * (icov_0 + inv_cov_m1(pix, i));
            mu += icov_0 * get_delta(i, sample[i], dist)
               + shift_weight * (
                   icov_m1 * get_delta(i, sample[i], dist-1)
               );
        }
        for(int i=pix+1; i<n_pix; i++) {
            double icov_0 = inv_cov_0(pix, i);
            double icov_m1 = 0.5 * (icov_0 + inv_cov_m1(pix, i));
            mu += icov_0 * get_delta(i, sample[i], dist)
               + shift_weight * (
                   icov_m1 * get_delta(i, sample[i], dist-1)
               );
        }
    } else {
        Eigen::MatrixXd& inv_cov_m1 = *(inv_cov[dist-1]);
        Eigen::MatrixXd& inv_cov_p1 = *(inv_cov[dist+1]);
        //norm = 1. + 2. * shift_weight;
        for(int i=start_pix; i<pix; i++) {
            double icov_0 = inv_cov_0(pix, i);
            double icov_m1 = 0.5 * (icov_0 + inv_cov_m1(pix, i));
            double icov_p1 = 0.5 * (icov_0 + inv_cov_p1(pix, i));
            mu += icov_0 * get_delta(i, sample[i], dist)
               + shift_weight * (
                   icov_m1 * get_delta(i, sample[i], dist-1)
                 + icov_p1 * get_delta(i, sample[i], dist+1)
               );
        }
        for(int i=pix+1; i<n_pix; i++) {
            double icov_0 = inv_cov_0(pix, i);
            double icov_m1 = 0.5 * (icov_0 + inv_cov_m1(pix, i));
            double icov_p1 = 0.5 * (icov_0 + inv_cov_p1(pix, i));
            mu += icov_0 * get_delta(i, sample[i], dist)
               + shift_weight * (
                   icov_m1 * get_delta(i, sample[i], dist-1)
                 + icov_p1 * get_delta(i, sample[i], dist+1)
               );
        }
    }

    mu *= -1. / (norm * get_inv_var(pix, dist));

    return mu;
}


double TNeighborPixels::calc_lnprob(
    const std::vector<uint16_t>& sample) const
{
    double p = 0.;
    unsigned int s0, s1;

    for(int pix0=0; pix0<n_pix; pix0++) {
        s0 = sample[pix0];
        
        // Off-diagonal terms
        for(int pix1=pix0+1; pix1<n_pix; pix1++) {
            s1 = sample[pix1];

            for(int dist=0; dist<n_dists; dist++) {
                p += (*(inv_cov[dist]))(pix0, pix1)
                     * get_delta(pix0, s0, dist)
                     * get_delta(pix1, s1, dist);
            }
        }
        
        // Diagonal terms
        for(int dist=0; dist<n_dists; dist++) {
            p += 0.5 * (*(inv_cov[dist]))(pix0, pix0)
                 * get_delta(pix0, s0, dist)
                 * get_delta(pix0, s0, dist);
            //std::cerr << get_delta(pix0, s0, dist) << std::endl;
        }

        p -= get_prior(pix0, s0);
        //std::cerr << std::endl;
    }
    //std::cerr << std::endl;

    return -1. * p;
}


double TNeighborPixels::calc_lnprob_shifted(
        const std::vector<uint16_t>& sample,
        const double shift_weight,
        const bool add_eff_prior) const
{
    double p = 0.;
    unsigned int s0, s1;
    
    // Normalization factors related to distance shift.
    // The factors of 2 out front come from the fact that we are
    // only calculating one triangle of the pixel-pixel covariance matrix.
    double norm = 1. + 2.*shift_weight;
    double a = 2. / norm; // d,d
    double b = 2. * shift_weight / norm; // d,d-1 or d-1,d

    for(int pix0=0; pix0<n_pix; pix0++) {
        s0 = sample[pix0];
        
        // Off-diagonal terms
        for(int pix1=pix0+1; pix1<n_pix; pix1++) {
            s1 = sample[pix1];
            
            // Distance-0 case (no cross-distance term)
            p += a
                 * (*(inv_cov[0]))(pix0, pix1)
                 * get_delta(pix0, s0, 0)
                 * get_delta(pix1, s1, 0);
            
            // Each distance coupled to distance-1 bins
            for(int dist=1; dist<n_dists; dist++) {
                double cov = (*(inv_cov[dist]))(pix0, pix1);
                double cov_m1 = (*(inv_cov[dist-1]))(pix0, pix1);
                double cov_avg = 0.5 * (cov + cov_m1);
                
                p += 
                    // Same distance
                    a * cov * (
                        get_delta(pix0, s0, dist)
                      * get_delta(pix1, s1, dist)
                    )
                    // Cross-distance
                    + b * cov_avg * (
                        get_delta(pix0, s0, dist)
                      * get_delta(pix1, s1, dist-1)
                      +
                        get_delta(pix0, s0, dist-1)
                      * get_delta(pix1, s1, dist)
                    )
                ;
            }
        }

        // Diagonal terms (same pixel and distance)
        for(int dist=0; dist<n_dists; dist++) {
            p += (*(inv_cov[dist]))(pix0, pix0)
                 * get_delta(pix0, s0, dist)
                 * get_delta(pix0, s0, dist);
            // Introducing cross-distance terms between the same pixel would
            // cause each pixel to be a correlated multivariate Gaussian, instead
            // of an uncorrelated multivariate Gaussian.
        }

        p *= -0.5;
        
        if(add_eff_prior) {
            p -= get_prior(pix0, s0);
        }
        //std::cerr << std::endl;
    }
    //std::cerr << std::endl;

    return p;
}


double TNeighborPixels::get_prior(
        unsigned int pix,
        unsigned int sample) const
{
    return prior[n_samples*pix + sample];
}


double TNeighborPixels::get_likelihood(
        unsigned int pix,
        unsigned int sample) const
{
    return likelihood[n_samples*pix + sample];
}


unsigned int TNeighborPixels::get_n_pix() const {
    return n_pix;
}


unsigned int TNeighborPixels::get_n_samples() const {
    return n_samples;
}


unsigned int TNeighborPixels::get_n_dists() const {
    return n_dists;
}

