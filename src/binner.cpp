/*
 * binner.cpp
 * 
 * Defines classes to bin posterior densities.
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


#include "binner.h"


/****************************************************************************************************************************
 * 
 * TSparseBinner
 * 
 ****************************************************************************************************************************/

TSparseBinner::TSparseBinner(double *_min, double *_max, unsigned int *_N_bins, unsigned int _N)
	: min(NULL), max(NULL), N_bins(NULL), dx(NULL), multiplier(NULL)
{
	N = _N;
	min = new double[N];
	max = new double[N];
	N_bins = new unsigned int[N];
	dx = new double[N];
	multiplier = new uint64_t[N];
	multiplier[0] = 1;
	max_index = 1;
	for(unsigned int i=0; i<N; i++) {
		min[i] = _min[i];
		max[i] = _max[i];
		N_bins[i] = _N_bins[i];
		dx[i] = (max[i] - min[i]) / (double)(N_bins[i]);
		if(i != 0) { multiplier[i] = multiplier[i-1] * N_bins[i]; }
		max_index *= N_bins[i];
	}
}

TSparseBinner::TSparseBinner(std::string fname, std::string dset_name, std::string group_name)
	: min(NULL), max(NULL), N_bins(NULL), dx(NULL), multiplier(NULL)
{
	load(fname, dset_name, group_name);
}

TSparseBinner::~TSparseBinner() {
	if(min != NULL) { delete[] min; }
	if(max != NULL) { delete[] max; }
	if(dx != NULL) { delete[] dx; }
	if(N_bins != NULL) { delete[] N_bins; }
	if(multiplier != NULL) { delete[] multiplier; }
}

uint64_t TSparseBinner::coord_to_index(double* coord) {
	uint64_t index = 0;
	uint64_t k;
	for(unsigned int i=0; i<N; i++) {
		if((coord[i] >= max[i]) || (coord[i] < min[i])) { return UINT64_MAX; }
		k = (coord[i] - min[i]) / dx[i];
		index += multiplier[i] * k;
	}
	return index;
}

bool TSparseBinner::index_to_coord(uint64_t index, double* coord) {
	if(index >= max_index) { return false; }
	uint64_t k = index % N_bins[0];
	coord[0] = min[0] + ((double)k + 0.5) * dx[0];
	for(unsigned int i=1; i<N; i++) {
		index = (index - k) / N_bins[i-1];
		k = index % N_bins[i];
		coord[i] = min[i] + ((double)k + 0.5) * dx[i];
	}
	return true;
}

bool TSparseBinner::index_to_array_coord(uint64_t index, uint64_t* coord) {
	if(index >= max_index) { return false; }
	uint64_t k = index % N_bins[0];
	coord[0] = min[0] + ((double)k + 0.5) * dx[0];
	for(unsigned int i=1; i<N; i++) {
		index = (index - k) / N_bins[i-1];
		coord[i] = index % N_bins[i];
	}
	return true;
}


void TSparseBinner::add_point(double* x, double weight) {
	uint64_t index = coord_to_index(x);
	if(index != UINT64_MAX) { bins[index] += weight; }
}

void TSparseBinner::operator()(double* x, double weight) {
	add_point(x, weight);
}

double TSparseBinner::get_bin(double* x) {
	uint64_t index;
	if((index = coord_to_index(x)) != UINT64_MAX) {
		return bins[index];
	} else {
		return -1.;
	}
}

void TSparseBinner::clear() {
	bins.clear();
}

bool TSparseBinner::write(std::string fname, std::string group_name,
			  std::string dset_name, std::string *dim_name,
			  int compression, hsize_t chunk) {
	if((compression<0) || (compression > 9)) {
		std::cerr << "! Invalid gzip compression level: " << compression << std::endl;
		return false;
	}
	
	H5::Exception::dontPrint();
	H5::H5File *file = NULL;
	try {
		file = new H5::H5File(fname.c_str(), H5F_ACC_RDWR);
	} catch(H5::FileIException) {
		file = new H5::H5File(fname.c_str(), H5F_ACC_TRUNC);
	}
	H5::Group *group = NULL;
	try {
		group = new H5::Group(file->openGroup(group_name.c_str()));
	} catch(H5::FileIException not_found_error) {
		group = new H5::Group(file->createGroup(group_name.c_str()));
	}
	
	/*
	 *  Bins
	 */
	
	// Datatype
	H5::CompType mtype(sizeof(TIndexValue));
	mtype.insertMember("index", HOFFSET(TIndexValue, index), H5::PredType::NATIVE_UINT64);
	mtype.insertMember("value", HOFFSET(TIndexValue, value), H5::PredType::NATIVE_FLOAT);
	
	// Dataspace and dataset creation property list 
	int rank = 1;
	hsize_t fdim = bins.size();
	if(fdim < chunk) { chunk = fdim; }
	H5::DataSpace dspace(rank, &fdim);
	
	TIndexValue fillvalue;
	fillvalue.index = 0;
	fillvalue.value = -1.;
	
	H5::DSetCreatPropList plist;
	plist.setChunk(rank, &chunk);	// Chunking (required for compression)
	plist.setDeflate(compression);	// gzip compression level
	plist.setFillValue(mtype, &fillvalue);
	
	// Dataset
	
	//std::stringstream dset_path;
	//dset_path << group_name << "/" << dset_name;
	H5::DataSet* dataset = NULL;
	try {
		dataset = new H5::DataSet(group->createDataSet(dset_name.c_str(), mtype, dspace, plist));
	} catch(H5::GroupIException dset_creation_err) {
		std::cerr << "! Unable to create dataset '" << dset_name << "'." << std::endl;
		delete dataset;
		delete group;
		delete file;
		return false;
	}
	
	// Generate data
	TIndexValue *data = new TIndexValue[fdim];
	std::map<uint64_t, double>::iterator it;
	std::map<uint64_t, double>::iterator it_end = bins.end();
	size_t i = 0;
	for(it = bins.begin(); it != it_end; ++it, i++) {
		data[i].index = it->first;
		data[i].value = it->second;
		if(i > fdim) {
			std::cerr << "! Loop ran over end of data!" << std::endl;
			break;
		}
	}
	
	dataset->write(data, mtype);
	
	/*
	 *  Attributes
	 */
	
	// Datatype
	H5::CompType att_type(sizeof(TDimDesc));
	hid_t tid = H5Tcopy(H5T_C_S1);
	H5Tset_size(tid, H5T_VARIABLE);
	att_type.insertMember("name", HOFFSET(TDimDesc, name), tid);
	att_type.insertMember("min", HOFFSET(TDimDesc, min), H5::PredType::NATIVE_FLOAT);
	att_type.insertMember("max", HOFFSET(TDimDesc, max), H5::PredType::NATIVE_FLOAT);
	att_type.insertMember("N_bins", HOFFSET(TDimDesc, N_bins), H5::PredType::NATIVE_UINT64);
	
	// Dataspace
	int att_rank = 1;
	hsize_t att_dim = N;
	H5::DataSpace att_space(att_rank, &att_dim);
	
	// Dataset
	H5::Attribute att = dataset->createAttribute("dimensions", att_type, att_space);
	
	TDimDesc *att_data = new TDimDesc[N];
	for(size_t i=0; i<N; i++) {
		att_data[i].name = new char[sizeof(dim_name[i])];
		std::strcpy(att_data[i].name, dim_name[i].c_str());
		att_data[i].min = min[i];
		att_data[i].max = max[i];
		att_data[i].N_bins = N_bins[i];
	}
	
	att.write(att_type, att_data);
	
	
	for(size_t i=0; i<N; i++) { delete[] att_data[i].name; }
	delete[] att_data;
	
	delete[] data;
	delete dataset;
	delete group;
	delete file;
	
	return true;
}

bool TSparseBinner::load(std::string fname, std::string group_name, std::string dset_name) {
	//H5::Exception::dontPrint();
	H5::H5File *file = NULL;
	try {
		file = new H5::H5File(fname.c_str(), H5F_ACC_RDONLY);
	} catch(H5::FileIException) {
		std::cerr << "Could not open file " << fname << std::endl;
		return false;
	}
	H5::Group *group = NULL;
	try {
		group = new H5::Group(file->openGroup(group_name.c_str()));
	} catch(H5::FileIException not_found_error) {
		std::cerr << "Could not open group '" << group_name << "'." << std::endl;
		delete file;
		return false;
	}
	
	H5::DataSet* dataset = NULL;
	try {
		dataset = new H5::DataSet(group->openDataSet(dset_name.c_str()));
	} catch(H5::GroupIException dset_creation_err) {
		std::cerr << "Could not open dataset '" << dset_name << "'." << std::endl;
		delete group;
		delete file;
		return false;
	}
	
	/*
	 *  Attributes
	 */
	
	// Datatype
	H5::CompType att_type(sizeof(TDimDesc));
	hid_t tid = H5Tcopy(H5T_C_S1);
	H5Tset_size(tid, H5T_VARIABLE);
	att_type.insertMember("name", HOFFSET(TDimDesc, name), tid);
	att_type.insertMember("min", HOFFSET(TDimDesc, min), H5::PredType::NATIVE_FLOAT);
	att_type.insertMember("max", HOFFSET(TDimDesc, max), H5::PredType::NATIVE_FLOAT);
	att_type.insertMember("N_bins", HOFFSET(TDimDesc, N_bins), H5::PredType::NATIVE_UINT64);
	
	H5::Attribute att = dataset->openAttribute("dimensions");
	
	hsize_t length;
	att.getSpace().getSimpleExtentDims(&length);
	TDimDesc *att_buf = new TDimDesc[length];
	att.read(att_type, reinterpret_cast<void*>(&att_buf));
	
	if(min != NULL) { delete[] min; }
	if(max != NULL) { delete[] max; }
	if(dx != NULL) { delete[] dx; }
	if(N_bins != NULL) { delete[] N_bins; }
	if(multiplier != NULL) { delete[] multiplier; }
	
	N = length;
	min = new double[N];
	max = new double[N];
	N_bins = new unsigned int[N];
	dx = new double[N];
	multiplier = new uint64_t[N];
	multiplier[0] = 1;
	max_index = 1;
	for(size_t i=0; i<N; i++) {
		min[i] = att_buf[i].min;
		max[i] = att_buf[i].max;
		N_bins[i] = att_buf[i].N_bins;
		dx[i] = (max[i] - min[i]) / (double)(N_bins[i]);
		if(i != 0) { multiplier[i] = multiplier[i-1] * N_bins[i]; }
		max_index *= N_bins[i];
	}
	
	delete[] att_buf;
	
	/*
	 *  Bins
	 */
	
	// Datatype
	H5::CompType dtype(sizeof(TIndexValue));
	dtype.insertMember("index", HOFFSET(TIndexValue, index), H5::PredType::NATIVE_UINT64);
	dtype.insertMember("value", HOFFSET(TIndexValue, value), H5::PredType::NATIVE_FLOAT);
	
	// Dataspace
	dataset->getSpace().getSimpleExtentDims(&length);
	
	TIndexValue *bin_buf = new TIndexValue[length];
	dataset->read(bin_buf, dtype);
	for(size_t i=0; i<length; i++) {
		bins[bin_buf[i].index] += bin_buf[i].value;
	}
	
	delete[] bin_buf;
	
	delete dataset;
	delete group;
	delete file;
	
	return true;
}

void TSparseBinner::get_image(cv::Mat& mat, unsigned int dim1, unsigned int dim2, double sigma1, double sigma2) {
	assert((dim1 >= 0) && (dim1 < N) && (dim2 >= 0) && (dim2 < N) && (dim1 != dim2));
	
	//mat.create(N_bins[dim1], N_bins[dim2], CV_64F);
	mat = cv::Mat::zeros(N_bins[dim1], N_bins[dim2], CV_64F);
	
	std::map<uint64_t, double>::iterator it;
	std::map<uint64_t, double>::iterator it_end = bins.end();
	size_t i = 0;
	uint64_t coord[N];
	for(it = bins.begin(); it != it_end; ++it, i++) {
		if(index_to_array_coord(it->first, &(coord[0]))) {
			mat.at<double>(coord[dim1], coord[dim2]) += it->second;
		}
	}
	
	if((sigma1 >= 0.) && (sigma2 >= 0.)) {
		double s1 = sigma1 / dx[dim1];
		double s2 = sigma2 / dx[dim2];
		int w1 = ceil(5*s1);
		int w2 = ceil(5*s2);
		
		cv::GaussianBlur(mat, mat, cv::Size(w1,s2), s1, s2, cv::BORDER_REPLICATE);
	}
}
