#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       query_lsd.py
#       
#       Copyright 2012 Greg <greg@greg-G53JW>
#       
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later version.
#       
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#       
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.
#       
#       

import os, sys, argparse
from os.path import abspath

import healpy as hp
import numpy as np
import pyfits
import h5py

import lsd

import iterators

import matplotlib.pyplot as plt


def mapper(qresult, nside, nest, bounds):
	obj = lsd.colgroup.fromiter(qresult, blocks=True)
	
	if (obj != None) and (len(obj) > 0):
		# Determine healpix index of each star
		theta = np.pi/180. * (90. - obj['b'])
		phi = np.pi/180. * obj['l']
		pix_indices = hp.ang2pix(nside, theta, phi, nest=nest)
		
		# Group together stars having same index
		for pix_index, block_indices in iterators.index_by_key(pix_indices):
			# Filter out pixels by bounds
			if bounds != None:
				theta_0, phi_0 = hp.pix2ang(nside, pix_index, nest=nest)
				l_0 = 180./np.pi * phi_0
				b_0 = 90. - 180./np.pi * theta_0
				if (l_0 < bounds[0]) or (l_0 > bounds[1]) or (b_0 < bounds[2]) or (b_0 > bounds[3]):
					continue
			
			yield (pix_index, obj[block_indices])


def reducer(keyvalue):
	pix_index, obj = keyvalue
	obj = lsd.colgroup.fromiter(obj, blocks=True)
	
	# Find stars with bad detections
	mask_zero_mag = (obj['mean'] == 0.)
	mask_zero_err = (obj['err'] == 0.)
	mask_nan_mag = np.isnan(obj['mean'])
	mask_nan_err = np.isnan(obj['err'])
	
	# Set errors for nondetections to some large number
	obj['mean'][mask_nan_mag] = 0.
	obj['err'][mask_zero_err] = 1.e10
	obj['err'][mask_nan_err] = 1.e10
	obj['err'][mask_zero_mag] = 1.e10
	
	# Combine and apply the masks
	mask_detect = np.sum(obj['mean'], axis=1).astype(np.bool)
	mask_informative = (np.sum(obj['err'] > 1.e10, axis=1) < 3).astype(np.bool)
	mask_keep = np.logical_and(mask_detect, mask_informative)
	
	yield (pix_index, obj[mask_keep])


def start_file(base_fname, index):
	fout = open('%s_%d.in' % (base_fname, index), 'wb')
	f.write(np.array([0], dtype=np.uint32).tostring())
	return f


def to_file(f, pix_index, nside, nest, data):
	close_file = False
	if type(f) == str:
		f = h5py.File(fname, 'a')
		close_file = True
	
	ds_name = '/photometry/pixel %d' % pix_index
	ds = f.create_dataset(ds_name, data.shape, data.dtype, chunks=True,
	                      compression='gzip', compression_opts=9)
	ds[:] = data[:]
	
	N_stars = data.shape[0]
	t,p = hp.pixelfunc.pix2ang(nside, pix_index, nest=nest)
	t *= 180. / np.pi
	p *= 180. / np.pi
	gal_lb = np.array([p, 90. - t], dtype='f8')
	
	att_u8 = np.array([pix_index], dtype='u8')
	att_u4 = np.array([nside, N_stars], dtype='u4')
	
	ds.attrs['healpix_index'] = att_u8[0]
	ds.attrs['nested'] = att_u4[0]
	ds.attrs['nside'] = att_u4[1]
	#ds.attrs['N_stars'] = N_stars
	ds.attrs['l'] = gal_lb[0]
	ds.attrs['b'] = gal_lb[1]
	
	if close_file:
		f.close()
	
	return gal_lb


def main():
	parser = argparse.ArgumentParser(
	           prog='query_lsd.py',
	           description='Generate galstar input files from PanSTARRS data.',
	           add_help=True)
	parser.add_argument('out', type=str, help='Output filename.')
	parser.add_argument('-n', '--nside', type=int, default=512,
	                    help='Healpix nside parameter (default: 512).')
	parser.add_argument('-b', '--bounds', type=float, nargs=4, default=None,
	                    help='Restrict pixels to region enclosed by: l_min, l_max, b_min, b_max.')
	parser.add_argument('-min', '--min_stars', type=int, default=1,
	                    help='Minimum # of stars in pixel (default: 1).')
	parser.add_argument('-sdss', '--sdss', action='store_true',
	                    help='Only select objects identified in the SDSS catalog as stars.')
	parser.add_argument('-ext', '--maxAr', type=float, default=None,
	                    help='Maximum allowed A_r.')
	parser.add_argument('-r', '--ring', action='store_true',
	                    help='Use healpix ring ordering scheme (default: nested).')
	parser.add_argument('-vis', '--visualize', action='store_true',
	                    help='Show number of stars in each pixel when query is done')
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	values = parser.parse_args(sys.argv[offset:])
	
	# Determine the query bounds
	query_bounds = None
	if values.bounds != None:
		query_bounds = []
		query_bounds.append(0.)
		query_bounds.append(360.)
		pix_height = 90. / 2**np.sqrt(values.nside / 12)
		query_bounds.append(max(-90., values.bounds[2] - 5.*pix_height))
		query_bounds.append(min(90., values.bounds[3] + 5.*pix_height))
	else:
		query_bounds = [0., 360., -90., 90.]
	query_bounds = lsd.bounds.rectangle(query_bounds[0], query_bounds[2],
	                                    query_bounds[1], query_bounds[3],
	                                    coordsys='gal')
	query_bounds = lsd.bounds.make_canonical(query_bounds)
	
	# Set up the query
	db = lsd.DB(os.environ['LSD_DB'])
	query = None
	if values.sdss:
		if values.maxAr == None:
			query = ("select obj_id, equgal(ra, dec) as (l, b), "
			         "mean, err, mean_ap, nmag_ok "
			         "from sdss, ucal_magsqw_noref "
			         "where (numpy.sum(nmag_ok > 0, axis=1) >= 4) "
			         "& (nmag_ok[:,0] > 0) "
			         "& (numpy.sum(mean - mean_ap < 0.1, axis=1) >= 2) "
			         "& (type == 6)")
		else:
			query = ("select obj_id, equgal(ra, dec) as (l, b), "
			         "mean, err, mean_ap, nmag_ok from sdss, "
			         "ucal_magsqw_noref(matchedto=sdss,nmax=1,dmax=5) "
			         "where (numpy.sum(nmag_ok > 0, axis=1) >= 4) "
			         "& (nmag_ok[:,0] > 0) & "
			         "(numpy.sum(mean - mean_ap < 0.1, axis=1) >= 2) & "
			         "(type == 6) & (rExt <= %.4f)" % values.maxAr)
	else:
		query = ("select obj_id, equgal(ra, dec) as (l, b), mean, err, "
		         "mean_ap, nmag_ok, maglimit from ucal_magsqw_noref "
		         "where (numpy.sum(nmag_ok > 0, axis=1) >= 4) "
		         "& (nmag_ok[:,0] > 0) & "
		         "(numpy.sum(mean - mean_ap < 0.1, axis=1) >= 2)")
	
	query = db.query(query)
	
	# Initialize map to store number of stars in each pixel
	pix_map = None
	if values.visualize:
		pix_map = np.zeros(12 * values.nside**2, dtype=np.uint64)
	
	# Initialize stats on pixels, # of stars, etc.
	l_min = np.inf
	l_max = -np.inf
	b_min = np.inf
	b_max = -np.inf
	
	N_stars = 0
	N_pix = 0
	N_min = np.inf
	N_max = -np.inf
	
	# Open output file
	fname = abspath(values.out)
	f = h5py.File(fname, 'w')
	
	# Write each pixel to the same file
	nest = (not values.ring)
	for (pix_index, obj) in query.execute([(mapper, values.nside, nest, values.bounds), reducer],
	                                      group_by_static_cell=True,
	                                      bounds=query_bounds):
		if len(obj) < values.min_stars:
			continue
		
		# Write object to file
		outarr = np.empty(len(obj), dtype=[('obj_id','u8'),
		                                   ('l','f8'), ('b','f8'), 
		                                   ('mean','f4',5), ('err','f4',5),
		                                   ('nmag_ok','u4',5),
		                                   ('maglimit','f4',5)])
		outarr['obj_id'] = obj['obj_id']
		outarr['l'] = obj['l']
		outarr['b'] = obj['b']
		outarr['mean'] = obj['mean']
		outarr['err'] = obj['err']
		outarr['nmag_ok'] = obj['nmag_ok']
		outarr['maglimit'] = obj['maglimit']
		
		gal_lb = to_file(f, pix_index, values.nside, nest, outarr)
		
		# Update stats
		N_pix += 1
		stars_in_pix = len(obj)
		N_stars += stars_in_pix
		
		if values.visualize:
			pix_max[pix_index] += N_stars
		
		if gal_lb[0] < l_min:
			l_min = gal_lb[0]
		if gal_lb[0] > l_max:
			l_max = gal_lb[0]
		if gal_lb[1] < b_min:
			b_min = gal_lb[1]
		if gal_lb[1] > b_max:
			b_max = gal_lb[1]
		
		if stars_in_pix < N_min:
			N_min = stars_in_pix
		if stars_in_pix > N_max:
			N_max = stars_in_pix
	
	f.close()
	
	if N_pix != 0:
		print '# of stars in footprint: %d.' % N_stars
		print '# of pixels in footprint: %d.' % N_pix
		print 'Stars per pixel:'
		print '    min: %d' % N_min
		print '    mean: %d' % (N_stars / N_pix)
		print '    max: %d' % N_max
	else:
		print 'No pixels in specified bounds with sufficient # of stars.'
	
	if (values.bounds != None) and (np.sum(N_pix) != 0):
		print ''
		print 'Bounds of included pixel centers:'
		print '\t(l_min, l_max) = (%.3f, %.3f)' % (l_min, l_max)
		print '\t(b_min, b_max) = (%.3f, %.3f)' % (b_min, b_max)
	
	# Show footprint of stored pixels on sky
	if values.visualize:
		hp.visufunc.mollview(map=np.log(pix_map), nest=nest,
		                     title=r'# of stars', coord='G', xsize=5000)
		plt.show()
	
	return 0

if __name__ == '__main__':
	main()

