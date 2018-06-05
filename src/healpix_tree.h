/*
 * healpix_tree.h
 * 
 * Traverse a tree structure representing a nested HEALPix map,
 * stored in an HDF5 file by nested groups.
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

#ifndef _HEALPIX_TREE_H__
#define _HEALPIX_TREE_H__


#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "h5utils.h"


std::unique_ptr<H5::DataSet> healtree_get_dataset(H5::H5File& file, uint32_t nside, uint32_t pix_idx);

void healpix_loc2digits(uint32_t nside, uint32_t pix_idx, std::vector<uint8_t>& digits);


#endif // _HEALPIX_TREE_H__
