/*
 * h5utils.h
 * 
 * Useful functions for dealing with HDF5 files.
 * 
 * Provides capabilities that really should have been included in the
 * HDF5 C++ bindings to begin with.
 * 
 * This file is part of bayestar.
 * Copyright 2012 Gregory Green
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

#ifndef _H5UTILS_H__
#define _H5UTILS_H__

#include <iostream>
#include <string.h>
#include <sstream>
#include <memory>
#include <vector>
#include <cassert>
#include <H5Cpp.h>

namespace H5Utils {
	
	extern int READ;
	extern int WRITE;
	extern int DONOTCREATE;
	
	std::unique_ptr<H5::H5File> openFile(const std::string& fname, int accessmode = (READ | WRITE));
	std::unique_ptr<H5::Group> openGroup(H5::H5File& file, const std::string& name, int accessmode = 0);
	std::unique_ptr<H5::DataSet> openDataSet(H5::H5File& file, const std::string& name);
	
	H5::Attribute openAttribute(H5::Group& group, const std::string& name, H5::DataType& dtype, H5::DataSpace& dspace);
	H5::Attribute openAttribute(H5::DataSet& dataset, const std::string& name, H5::DataType& dtype, H5::DataSpace& dspace);
	H5::Attribute openAttribute(H5::Group& group, const std::string& name, H5::StrType& strtype, H5::DataSpace& dspace);
	H5::Attribute openAttribute(H5::DataSet& dataset, const std::string& name, H5::StrType& strtype, H5::DataSpace& dspace);

    // Read attribute from dataset
    template<class T>
    T read_attribute(H5::DataSet& dataset, const std::string& name);
    
    template<>
    double read_attribute<double>(H5::DataSet& dataset, const std::string& name);
    
    template<>
    float read_attribute<float>(H5::DataSet& dataset, const std::string& name);
    
    // Read attribute from group
    template<class T>
    T read_attribute(H5::Group& group, const std::string& name);
    
    template<>
    float read_attribute<float>(H5::Group& g, const std::string& name);
    
    template<>
    double read_attribute<double>(H5::Group& g, const std::string& name);
    
    template<>
    uint32_t read_attribute<uint32_t>(H5::Group& g, const std::string& name);
    
    template<>
    uint64_t read_attribute<uint64_t>(H5::Group& g, const std::string& name);
	
	bool group_exists(const std::string& name, H5::H5File& file);
	bool group_exists(const std::string& name, H5::Group& group);
	
	bool dataset_exists(const std::string& name, H5::H5File& file);
	bool dataset_exists(const std::string& name, H5::Group& group);
    
    // Read attribute directly
    template<class T>
    void read_attribute_1d_helper(H5::Attribute& attribute, std::vector<T> &ret);
    
    template<class T>
    std::vector<T> read_attribute_1d(H5::Attribute& attribute);
    
    template<>
    std::vector<double> read_attribute_1d<double>(H5::Attribute& attribute);

    template<>
    std::vector<uint32_t> read_attribute_1d<uint32_t>(H5::Attribute& attribute);
	
    // Write attribute to group or dataset in file
	template<class T>
	bool add_watermark(const std::string& filename, const std::string& group_name, const std::string& attribute_name, const T& value);
	
	template<>
	bool add_watermark<bool>(const std::string& filename, const std::string& group_name, const std::string& attribute_name, const bool& value);
	
	template<>
	bool add_watermark<float>(const std::string& filename, const std::string& group_name, const std::string& attribute_name, const float& value);
	
	template<>
	bool add_watermark<double>(const std::string& filename, const std::string& group_name, const std::string& attribute_name, const double& value);
	
	template<>
	bool add_watermark<uint32_t>(const std::string& filename, const std::string& group_name, const std::string& attribute_name, const uint32_t& value);
	
	template<>
	bool add_watermark<uint64_t>(const std::string& filename, const std::string& group_name, const std::string& attribute_name, const uint64_t& value);
	
	template<>
	bool add_watermark<std::string>(const std::string& filename, const std::string& group_name, const std::string& attribute_name, const std::string& value);
    
    // Convert C++ data types to HDF5 data types
    template<class T>
    H5::PredType get_dtype();
    
    template<>
    H5::PredType get_dtype<float>();

    template<>
    H5::PredType get_dtype<double>();

    template<>
    H5::PredType get_dtype<uint32_t>();

    template<>
    H5::PredType get_dtype<uint64_t>();

    template<>
    H5::PredType get_dtype<bool>();
    
    // Create dataset from 1D array.
    template<class T>
    std::unique_ptr<H5::DataSet> createDataSet(
            H5::H5File& file,
            const std::string& group,
            const std::string& dset,
            const std::vector<T>& data,
            uint8_t gzip=3, // 0 for no compression, 9 for max. compression
            uint64_t chunk_size=1048576); // in Bytes
    
}


/* 
 * Create dataset from 1D array.
 * 
 */
template<class T>
std::unique_ptr<H5::DataSet> H5Utils::createDataSet(
        H5::H5File& file,
        const std::string& group,
        const std::string& dset,
        const std::vector<T>& data,
        uint8_t gzip,
        uint64_t chunk_size)
{
    // Get group containing dataset
    //std::cerr << "Opening group " << group << std::endl;
    std::unique_ptr<H5::Group> g = H5Utils::openGroup(
        file,
        group,
        H5Utils::READ | H5Utils::WRITE);
    if(!g) {
        std::cerr << "Failed to open group!" << std::endl;
    }

    // Datatype
    //std::cerr << "Datatype" << std::endl;
    H5::DataType dtype = get_dtype<T>();
    
    // Dataspace
    //std::cerr << "Dataspace" << std::endl;
    hsize_t dim = data.size();
    H5::DataSpace dspace(1, &dim);
    //std::cerr << "dim = " << dim << std::endl;
    
    // Property List
    //std::cerr << "Chunking" << std::endl;
    H5::DSetCreatPropList plist;
    
    hsize_t n_per_chunk = chunk_size / sizeof(T);
    if(n_per_chunk < 1) {
        n_per_chunk = 1;
    } else if(n_per_chunk > data.size()) {
        n_per_chunk = data.size();
    }
    plist.setChunk(1, &n_per_chunk); // Chunk size
    //std::cerr << "n_per_chunk = " << n_per_chunk << std::endl;
    
    //std::cerr << "Deflate" << std::endl;
    plist.setDeflate(gzip); // DEFLATE compression level (min=0, max=9)
    //std::cerr << "gzip = " << (unsigned int)gzip << std::endl;
    
    // Create dataset
    //std::cerr << "Dataset" << std::endl;
    std::unique_ptr<H5::DataSet> dataset = std::unique_ptr<H5::DataSet>(
        new H5::DataSet(g->createDataSet(dset, dtype, dspace, plist))
    );

    // Write data
    //std::cerr << "write" << std::endl;
    dataset->write(data.data(), dtype);
    
    //std::cerr << "returning" << std::endl;
    return std::move(dataset);
}


#endif // _H5UTILS_H__
