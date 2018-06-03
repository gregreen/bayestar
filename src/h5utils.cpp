/*
 * h5utils.cpp
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

#include "h5utils.h"

int H5Utils::READ = 1;
int H5Utils::WRITE = (1 << 1);
int H5Utils::DONOTCREATE = (1 << 2);

/* 
 * Opens a file, creating it if it does not exist.
 * 
 * The variable "accessmode" is by default set to
 *     H5Utils::READ | H5Utils::WRITE
 * providing read/write access to the file. If only
 *     H5Utils::READ
 * is provided, then the file is read-only. In this
 * case, if the file does not exist, NULL is returned.
 * The flag
 *     H5Utils::DONOTCREATE
 * can also be set. This flag specifies that NULL
 * should be returned if the file does not exist.
 * 
 * If an incorrect access mode is passed, NULL
 * is returned.
 * 
 */
std::unique_ptr<H5::H5File> H5Utils::openFile(const std::string& fname, int accessmode) {
	// Read/Write access
	if((accessmode & H5Utils::READ) && (accessmode & H5Utils::WRITE)) {
		try {
			return std::unique_ptr<H5::H5File>(new H5::H5File(fname.c_str(), H5F_ACC_RDWR));
		} catch(const H5::FileIException& err_does_not_exist) {
			if(accessmode& H5Utils::DONOTCREATE) {
				return std::unique_ptr<H5::H5File>(nullptr);
			} else {
				return std::unique_ptr<H5::H5File>(new H5::H5File(fname.c_str(), H5F_ACC_TRUNC));
			}
		}
	// Read-only access
	} else if(accessmode & H5Utils::READ) {
		try {
			return std::unique_ptr<H5::H5File>(new H5::H5File(fname.c_str(), H5F_ACC_RDONLY));
		} catch(const H5::FileIException& err_does_not_exist) {
			return std::unique_ptr<H5::H5File>(nullptr);
		}
	// Other (incorrect) access mode
	} else {
		std::cerr << "openFile: Invalid access mode." << std::endl;
	}
	
	return std::unique_ptr<H5::H5File>(nullptr);
}

/* 
 * Opens a group, creating it if it does not exist. Nonexistent parent groups are also
 * created. This works similarly to the Unix/Linux command
 *     mkdir -p /parent/subgroup/group
 * in that if /parent and /parent/subgroup do not exist, they will be created.
 * 
 * If no accessmode has H5Utils::DONOTCREATE flag set, then returns NULL if group
 * does not yet exist.
 * 
 */
std::unique_ptr<H5::Group> H5Utils::openGroup(H5::H5File& file, const std::string& name, int accessmode) {
	// User does not want to create group
	if(accessmode & H5Utils::DONOTCREATE) {
		try {
			return std::unique_ptr<H5::Group>(new H5::Group(file.openGroup(name.c_str())));
		} catch(const H5::FileIException& err_does_not_exist) {
			return std::unique_ptr<H5::Group>(nullptr);
		}
	}
	
	// Possibly create group and parent groups
	std::stringstream ss(name);
	std::stringstream path;
	std::string gp_name;
    std::unique_ptr<H5::Group> group;
	while(std::getline(ss, gp_name, '/')) {
		path << "/" << gp_name;
		try {
			group.reset(new H5::Group(file.openGroup(path.str().c_str())));
		} catch(const H5::FileIException& err_does_not_exist) {
			group.reset(new H5::Group(file.createGroup(path.str().c_str())));
		}
	}
	
	return std::move(group);
}

/* 
 * Opens an existing dataset.
 * 
 */
std::unique_ptr<H5::DataSet> H5Utils::openDataSet(H5::H5File& file, const std::string& name) {
	try {
		return std::unique_ptr<H5::DataSet>(new H5::DataSet(file.openDataSet(name.c_str())));
	} catch(const H5::FileIException& err_does_not_exist) {
		return std::unique_ptr<H5::DataSet>(nullptr);
	}
}

/* 
 * 
 * Opens an attribute, creating it if it does not exist.
 * 
 */

H5::Attribute H5Utils::openAttribute(H5::Group& group, const std::string& name, H5::DataType& dtype, H5::DataSpace& dspace) {
	try {
		return group.openAttribute(name);
	} catch(H5::AttributeIException err_att_does_not_exist) {
		return group.createAttribute(name, dtype, dspace);
	}
}

H5::Attribute H5Utils::openAttribute(H5::DataSet& dataset, const std::string& name, H5::DataType& dtype, H5::DataSpace& dspace) {
	try {
		return dataset.openAttribute(name);
	} catch(H5::AttributeIException err_att_does_not_exist) {
		return dataset.createAttribute(name, dtype, dspace);
	}
}

H5::Attribute H5Utils::openAttribute(H5::Group& group, const std::string& name, H5::StrType& strtype, H5::DataSpace& dspace) {
	try {
		return group.openAttribute(name);
	} catch(H5::AttributeIException err_att_does_not_exist) {
		return group.createAttribute(name, strtype, dspace);
	}
}

H5::Attribute H5Utils::openAttribute(H5::DataSet& dataset, const std::string& name, H5::StrType& strtype, H5::DataSpace& dspace) {
	try {
		return dataset.openAttribute(name);
	} catch(H5::AttributeIException err_att_does_not_exist) {
		return dataset.createAttribute(name, strtype, dspace);
	}
}

/*
 * 
 * Check existence of datasets, groups
 * 
 */

bool H5Utils::group_exists(const std::string& name, H5::H5File& file) {
	try {
		file.openGroup(name);
	} catch(H5::FileIException err_gp_does_not_exist) {
		return false;
	}
	
	return true;
}

bool H5Utils::group_exists(const std::string& name, H5::Group& group) {
	try {
		group.openGroup(name);
	} catch(H5::GroupIException err_gp_does_not_exist) {
		return false;
	}
	
	return true;
}

bool H5Utils::dataset_exists(const std::string& name, H5::H5File& file) {
	try {
		file.openDataSet(name);
	} catch(H5::FileIException err_dset_does_not_exist) {
		return false;
	}
	
	return true;
}

bool H5Utils::dataset_exists(const std::string& name, H5::Group& group) {
	try {
		group.openDataSet(name);
	} catch(H5::GroupIException err_dset_does_not_exist) {
		return false;
	}
	
	return true;
}





/*
 * 
 * Add an attribute to a group in the given file.
 * 
 */

template<class T>
bool add_watermark_helper(const std::string& filename, const std::string& group_name, const std::string& attribute_name, const T& value,
                          const H5::DataType* dtype, const H5::StrType* strtype, const H5::DataSpace& dspace) {
	if((strtype == NULL) && (dtype == NULL)) { return false; }
	
	std::unique_ptr<H5::H5File> file = H5Utils::openFile(filename);
	if(!file) { return false; }
	
	bool is_group = true;
    std::unique_ptr<H5::Group> group;
	try {
		group = H5Utils::openGroup(*file, group_name);
	} catch(H5::FileIException err_not_group) {
		is_group = false;
	}
	
	if(is_group) {
		if(!group) {
			return false;
		}
		
		if(strtype == NULL) {
			H5::Attribute att = group->createAttribute(attribute_name, *dtype, dspace);
			att.write(*dtype, reinterpret_cast<const void*>(&value));
		} else {
			H5::Attribute att = group->createAttribute(attribute_name, *strtype, dspace);
			att.write(*strtype, reinterpret_cast<const void*>(&value));
		}
	} else {
		std::unique_ptr<H5::DataSet> dataset = H5Utils::openDataSet(*file, group_name);
		if(!dataset) {
			return false;
		}
		
		if(strtype == NULL) {
			H5::Attribute att = dataset->createAttribute(attribute_name, *dtype, dspace);
			att.write(*dtype, reinterpret_cast<const void*>(&value));
		} else {
			H5::Attribute att = dataset->createAttribute(attribute_name, *strtype, dspace);
			att.write(*strtype, reinterpret_cast<const void*>(&value));
		}
	}
	
	return true;
}

template<class T>
bool add_watermark_helper(const std::string& filename, const std::string& group_name, const std::string& attribute_name, const T& value, const H5::DataType *dtype) {
	int rank = 1;
	hsize_t dim = 1;
	H5::DataSpace dspace(rank, &dim);
	
	return add_watermark_helper(filename, group_name, attribute_name, value, dtype, NULL, dspace);
}

template<>
bool H5Utils::add_watermark<bool>(const std::string& filename, const std::string& group_name, const std::string& attribute_name, const bool& value) {
	H5::DataType dtype = H5::PredType::NATIVE_UCHAR;
	return add_watermark_helper<bool>(filename, group_name, attribute_name, value, &dtype);
}

template<>
bool H5Utils::add_watermark<float>(const std::string& filename, const std::string& group_name, const std::string& attribute_name, const float& value) {
	H5::DataType dtype = H5::PredType::NATIVE_FLOAT;
	return add_watermark_helper<float>(filename, group_name, attribute_name, value, &dtype);
}

template<>
bool H5Utils::add_watermark<double>(const std::string& filename, const std::string& group_name, const std::string& attribute_name, const double& value) {
	H5::DataType dtype = H5::PredType::NATIVE_DOUBLE;
	return add_watermark_helper<double>(filename, group_name, attribute_name, value, &dtype);
}

template<>
bool H5Utils::add_watermark<uint32_t>(const std::string& filename, const std::string& group_name, const std::string& attribute_name, const uint32_t& value) {
	H5::DataType dtype = H5::PredType::NATIVE_UINT32;
	return add_watermark_helper<uint32_t>(filename, group_name, attribute_name, value, &dtype);
}

template<>
bool H5Utils::add_watermark<uint64_t>(const std::string& filename, const std::string& group_name, const std::string& attribute_name, const uint64_t& value) {
	H5::DataType dtype = H5::PredType::NATIVE_UINT64;
	return add_watermark_helper<uint64_t>(filename, group_name, attribute_name, value, &dtype);
}

template<>
bool H5Utils::add_watermark<std::string>(const std::string& filename, const std::string& group_name, const std::string& attribute_name, const std::string& value) {
	H5::StrType strtype(0, H5T_VARIABLE);
	H5::DataSpace dspace(H5S_SCALAR);
	return add_watermark_helper<std::string>(filename, group_name, attribute_name, value, NULL, &strtype, dspace);
}

template<class T>
bool H5Utils::add_watermark(const std::string& filename, const std::string& group_name, const std::string& attribute_name, const T& value) {
	// Unknown type
	return false;
}
