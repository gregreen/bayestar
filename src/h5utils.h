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

	H5::H5File* openFile(std::string fname, int accessmode = (READ | WRITE));
	H5::Group* openGroup(H5::H5File* file, std::string name, int accessmode = 0);
	
}

#endif // _H5UTILS_H__