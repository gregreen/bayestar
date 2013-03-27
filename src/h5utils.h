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
#include <H5Cpp.h>

namespace H5Utils {
	
	extern int READ;
	extern int WRITE;
	extern int DONOTCREATE;
	
	H5::H5File* openFile(const std::string &fname, int accessmode = (READ | WRITE));
	H5::Group* openGroup(H5::H5File* file, const std::string &name, int accessmode = 0);
	H5::DataSet* openDataSet(H5::H5File* file, const std::string &name);
	
	H5::Attribute openAttribute(H5::Group* group, const std::string &name, H5::DataType &dtype, H5::DataSpace &dspace);
	H5::Attribute openAttribute(H5::DataSet* dataset, const std::string &name, H5::DataType &dtype, H5::DataSpace &dspace);
	H5::Attribute openAttribute(H5::Group* group, const std::string &name, H5::StrType &strtype, H5::DataSpace &dspace);
	H5::Attribute openAttribute(H5::DataSet* dataset, const std::string &name, H5::StrType &strtype, H5::DataSpace &dspace);
	
	template<class T>
	bool add_watermark(const std::string &filename, const std::string &group_name, const std::string &attribute_name, const T &value);
	
	template<>
	bool add_watermark<bool>(const std::string &filename, const std::string &group_name, const std::string &attribute_name, const bool &value);
	
	template<>
	bool add_watermark<float>(const std::string &filename, const std::string &group_name, const std::string &attribute_name, const float &value);
	
	template<>
	bool add_watermark<double>(const std::string &filename, const std::string &group_name, const std::string &attribute_name, const double &value);
	
	template<>
	bool add_watermark<uint64_t>(const std::string &filename, const std::string &group_name, const std::string &attribute_name, const uint64_t &value);
	
	template<>
	bool add_watermark<std::string>(const std::string &filename, const std::string &group_name, const std::string &attribute_name, const std::string &value);
	
}

#endif // _H5UTILS_H__
