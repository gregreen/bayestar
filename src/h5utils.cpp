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
H5::H5File* H5Utils::openFile(std::string fname, int accessmode) {
	H5::H5File* file = NULL;
	
	// Read/Write access
	if((accessmode & H5Utils::READ) && (accessmode & H5Utils::WRITE)) {
		try {
			file = new H5::H5File(fname.c_str(), H5F_ACC_RDWR);
		} catch(const H5::FileIException& err_does_not_exist) {
			if(accessmode & H5Utils::DONOTCREATE) {
				file = NULL;
			} else {
				file = new H5::H5File(fname.c_str(), H5F_ACC_TRUNC);
			}
		}
	// Read-only access
	} else if(accessmode & H5Utils::READ) {
		try {
			file = new H5::H5File(fname.c_str(), H5F_ACC_RDONLY);
		} catch(const H5::FileIException& err_does_not_exist) {
			file = NULL;
		}
	// Other (incorrect) access mode
	} else {
		std::cerr << "openFile: Invalid access mode." << std::endl;
	}
	
	return file;
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
H5::Group* H5Utils::openGroup(H5::H5File* file, std::string name, int accessmode) {
	H5::Group* group = NULL;
	
	// User does not want to create group
	if(accessmode & H5Utils::DONOTCREATE) {
		try {
			group = new H5::Group(file->openGroup(name.c_str()));
		} catch(const H5::FileIException& err_does_not_exist) {
			return group;
		}
	}
	
	// Possibly create group and parent groups
	std::stringstream ss(name);
	std::stringstream path;
	std::string gp_name;
	while(std::getline(ss, gp_name, '/')) {
		if(gp_name != "") {
			path << "/" << gp_name;
			if(group != NULL) { delete group; }
			try {
				group = new H5::Group(file->openGroup(path.str().c_str()));
			} catch(const H5::FileIException& err_does_not_exist) {
				group = new H5::Group(file->createGroup(path.str().c_str()));
			}
		}
	}
	
	return group;
}
