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


def lb2pix(nside, l, b, nest=True):
	theta = np.pi/180. * (90. - b)
	phi = np.pi/180. * l
	
	return hp.pixelfunc.ang2pix(nside, theta, phi, nest=nest)

def pix2lb(nside, ipix, nest=True):
	theta, phi = hp.pixelfunc.pix2ang(nside, ipix, nest=True)
	
	l = 180./np.pi * phi
	b = 90. - 180./np.pi * theta
	
	return l, b


def adaptive_subdivide(pix_idx, nside, obj,
                    n_stars_max, n_stars_min=10, nside_max=2048):
	# Subdivide pixel
	if (len(obj) > n_stars_max[nside]) and (nside < nside_max):
		sub_pix_idx = lb2pix(nside*2, obj['l'], obj['b'], nest=True)
		
		# Check that all pixels have more than minimum # of pixels
		over_threshold = True
		
		for i in xrange(4 * pix_idx, 4 * pix_idx + 4):
			idx = (sub_pix_idx == i)
			
			if np.sum(idx) < n_stars_min:
				over_threshold = False
				break
		
		if not over_threshold:
			return [(nside, pix_idx, obj)]
		
		# Return subdivided pixel
		ret = []
		
		for i in xrange(4 * pix_idx, 4 * pix_idx + 4):
			idx = (sub_pix_idx == i)
			
			tmp = adaptive_subdivide(i, nside*2, obj[idx],
			                         n_stars_max, n_stars_min, nside_max)
			
			for pix in tmp:
				ret.append(pix)
		
		return ret
		
	else:
		return [(nside, pix_idx, obj)]


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
			#if bounds != None:
			#	theta_0, phi_0 = hp.pix2ang(nside, pix_index, nest=nest)
			#	l_0 = 180./np.pi * phi_0
			#	b_0 = 90. - 180./np.pi * theta_0
			#	if (l_0 < bounds[0]) or (l_0 > bounds[1]) or (b_0 < bounds[2]) or (b_0 > bounds[3]):
			#		continue
			
			yield (pix_index, obj[block_indices])


def reducer(keyvalue):
	pix_index, obj = keyvalue
	obj = lsd.colgroup.fromiter(obj, blocks=True)
	
	# Scale errors
	err_scale = 1.3
	err_floor = 0.02
	obj['err'] = np.sqrt((err_scale * obj['err'])**2. + err_floor**2.)
	
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
	#mask_detect = np.sum(obj['mean'], axis=1).astype(np.bool)
	#mask_informative = (np.sum(obj['err'] > 1.e10, axis=1) < 3).astype(np.bool)
	#mask_keep = np.logical_and(mask_detect, mask_informative)
	
	#yield (pix_index, obj[mask_keep])
	
	yield (pix_index, obj)


def subdivider(keyvalue, nside, n_stars_max, n_stars_min, nside_max):
	pix_index, obj = keyvalue
	obj = lsd.colgroup.fromiter(obj, blocks=True)
	
	# Adaptively subdivide pixel
	ret = adaptive_subdivide(pix_index, nside, obj,
                             n_stars_max, n_stars_min, nside_max)
	
	for subpixel in ret:
		sub_nside, sub_idx, sub_obj = subpixel
		
		yield ((sub_nside, sub_idx), sub_obj)


def start_file(base_fname, index):
	fout = open('%s_%d.in' % (base_fname, index), 'wb')
	f.write(np.array([0], dtype=np.uint32).tostring())
	return f


def to_file(f, pix_index, nside, nest, EBV, data):
	close_file = False
	if type(f) == str:
		f = h5py.File(fname, 'a')
		close_file = True
	
	ds_name = '/photometry/pixel %d-%d' % (nside, pix_index)
	ds = f.create_dataset(ds_name, data.shape, data.dtype, chunks=True,
	                      compression='gzip', compression_opts=9)
	ds[:] = data[:]
	
	N_stars = data.shape[0]
	t,p = hp.pixelfunc.pix2ang(nside, pix_index, nest=nest)
	t *= 180. / np.pi
	p *= 180. / np.pi
	gal_lb = np.array([p, 90. - t], dtype='f8')
	
	att_f8 = np.array([EBV], dtype='f8')
	att_u8 = np.array([pix_index], dtype='u8')
	att_u4 = np.array([nside, N_stars], dtype='u4')
	att_u1 = np.array([nest], dtype='u1')
	
	ds.attrs['healpix_index'] = att_u8[0]
	ds.attrs['nested'] = att_u1[0]
	ds.attrs['nside'] = att_u4[0]
	#ds.attrs['N_stars'] = N_stars
	ds.attrs['l'] = gal_lb[0]
	ds.attrs['b'] = gal_lb[1]
	ds.attrs['EBV'] = att_f8[0]
	
	if close_file:
		f.close()
	
	return gal_lb


def main():
	parser = argparse.ArgumentParser(
	           prog='query_lsd.py',
	           description='Generate bayestar input files from PanSTARRS data.',
	           add_help=True)
	parser.add_argument('out', type=str, help='Output filename.')
	parser.add_argument('-nmin', '--nside-min', type=int, default=512,
	                    help='Lowest resolution in healpix nside parameter (default: 512).')
	parser.add_argument('-nmax', '--nside-max', type=int, default=512,
	                    help='Lowest resolution in healpix nside parameter (default: 512).')
	parser.add_argument('-rt', '--res-thresh', type=int, nargs='+', default=None,
	                    help='Maximum # of pixels for each healpix resolution (from lowest to highest).')
	parser.add_argument('-b', '--bounds', type=float, nargs=4, default=None,
	                    help='Restrict pixels to region enclosed by: l_min, l_max, b_min, b_max.')
	parser.add_argument('-min', '--min-stars', type=int, default=1,
	                    help='Minimum # of stars in pixel (default: 1).')
	parser.add_argument('-max', '--max-stars', type=int, default=50000,
	                    help='Maximum # of stars in file')
	parser.add_argument('-sdss', '--sdss', action='store_true',
	                    help='Only select objects identified in the SDSS catalog as stars.')
	parser.add_argument('-ext', '--maxAr', type=float, default=None,
	                    help='Maximum allowed A_r.')
	parser.add_argument('--n-bands', type=int, default=4,
	                    help='Min. # of passbands with detection.')
	parser.add_argument('--n-det', type=int, default=4,
	                    help='Min. # of detections.')
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	values = parser.parse_args(sys.argv[offset:])
	
	nPointlike = values.n_bands - 1
	if nPointlike == 0:
		nPointlike = 1
	
	# Handle adaptive pixelization parameters
	base_2_choices = [2**n for n in xrange(15)]
	if values.nside_min not in base_2_choices:
		raise ValueError('--nside-min is not a small power of two.')
	elif values.nside_max not in base_2_choices:
		raise ValueError('--nside-max is not a small power of two.')
	elif values.nside_max < values.nside_min:
		raise ValueError('--nside-max is less than --nside-min.')
	
	n_stars_max = None
	n_pixels_at_res = None
	nside_options = None
	
	if values.nside_min == values.nside_max:
		n_stars_max = {values.nside_max: 1}
		n_pixels_at_res = {values.nside_max: 0}
		nside_options = [values.nside_options]
	else:
		n_stars_max = {}
		n_pixels_at_res = {}
		nside_options = []
		
		nside = values.nside_min
		k = 0
		
		while nside < values.nside_max:
			n_stars_max[nside] = values.res_thresh[k]
			n_pixels_at_res[nside] = 0
			nside_options.append(nside)
			
			nside *= 2
			k += 1
		
		n_stars_max[values.nside_max] = 1
		n_pixels_at_res[values.nside_max] = 0
		nside_options.append(values.nside_max)
	
	# Determine the query bounds
	query_bounds = None
	if values.bounds != None:
		pix_scale = hp.pixelfunc.nside2resol(values.nside_min) * 180. / np.pi
		query_bounds = []
		query_bounds.append(max([0., values.bounds[0] - 3.*pix_scale]))
		query_bounds.append(min([360., values.bounds[1] + 3.*pix_scale]))
		query_bounds.append(max([-90., values.bounds[2] - 3.*pix_scale]))
		query_bounds.append(min([90., values.bounds[3] + 3.*pix_scale]))
	#else:
	#	query_bounds = [0., 360., -90., 90.]
		query_bounds = lsd.bounds.rectangle(query_bounds[0], query_bounds[2],
		                                    query_bounds[1], query_bounds[3],
		                                    coordsys='gal')
	#query_bounds = (query_bounds, []) 
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
			         "where (numpy.sum(nmag_ok > 0, axis=1) >= 4) & "
			         "(nmag_ok[:,0] > 0) & "
			         "(numpy.sum(mean - mean_ap < 0.1, axis=1) >= 2) & "
			         "(type == 6) & (rExt <= %.4f)" % values.maxAr)
	else:
		query = ("select obj_id, equgal(ra, dec) as (l, b), mean, err, "
		         "mean_ap, nmag_ok, maglimit, SFD.EBV(l, b) as EBV "
		         "from ucal_magsqx_noref "
		         "where (numpy.sum(nmag_ok > 0, axis=1) >= %d) "
		         "& (nmag_ok[:,0] > 0) "
		         "& (numpy.sum(nmag_ok, axis=1) >= %d) "
		         "& (numpy.sum(mean - mean_ap < 0.1, axis=1) >= %d)"
		         % (values.n_bands, values.n_det, nPointlike))
	
	query = db.query(query)
	
	# Initialize stats on pixels, # of stars, etc.
	l_min = np.inf
	l_max = -np.inf
	b_min = np.inf
	b_max = -np.inf
	
	N_stars = 0
	N_pix = 0
	N_min = np.inf
	N_max = -np.inf
	
	N_in_pixel = []
	N_pix_too_sparse = 0
	N_pix_out_of_bounds = 0
	
	fnameBase = abspath(values.out)
	fnameSuffix = 'h5'
	if fnameBase.endswith('.h5'):
		fnameBase = fnameBase[:-3]
	elif fnameBase.endswith('.hdf5'):
		fnameBase = fnameBase[:-5]
		fnameSuffix = 'hdf5'
	f = None
	nFiles = 0
	nInFile = 0
	
	# Write each pixel to the same file
	nest = True
	for (pix_info, obj) in query.execute([(mapper, values.nside_min, nest, values.bounds),
	                                      reducer,
	                                      (subdivider, values.nside_min, n_stars_max, values.min_stars, values.nside_max)],
	                                      #group_by_static_cell=True,
	                                      bounds=query_bounds):
		# Filter out pixels that have too few stars
		if len(obj) < values.min_stars:
			N_pix_too_sparse += 1
			
			continue
		
		nside, pix_index = pix_info
		
		# Filter out pixels that are outside of bounds
		l_center, b_center = pix2lb(nside, pix_index, nest=nest)
		
		if values.bounds != None:
			if (     (l_center < values.bounds[0])
			      or (l_center > values.bounds[1]) 
			      or (b_center < values.bounds[2]) 
			      or (b_center > values.bounds[3]) ):
				N_pix_out_of_bounds += 1
				continue
		
		# Prepare output for pixel
		outarr = np.empty(len(obj), dtype=[('obj_id','u8'),
		                                   ('l','f8'), ('b','f8'), 
		                                   ('mag','f4',5), ('err','f4',5),
		                                   ('maglimit','f4',5),
		                                   ('nDet','u4',5),
		                                   ('EBV','f4')])
		outarr['obj_id'][:] = obj['obj_id'][:]
		outarr['l'][:] = obj['l'][:]
		outarr['b'][:] = obj['b'][:]
		outarr['mag'][:] = obj['mean'][:]
		outarr['err'][:] = obj['err'][:]
		outarr['maglimit'][:] = obj['maglimit'][:]
		outarr['nDet'][:] = obj['nmag_ok'][:]
		outarr['EBV'][:] = obj['EBV'][:]
		EBV = np.percentile(obj['EBV'][:], 95.)
		
		# Open output file
		if f == None:
			fname = '%s.%.5d.%s' % (fnameBase, nFiles, fnameSuffix)
			f = h5py.File(fname, 'w')
			nInFile = 0
			nFiles += 1
		
		# Write to file
		gal_lb = to_file(f, pix_index, nside, nest, EBV, outarr)
		
		# Update stats
		N_pix += 1
		stars_in_pix = len(obj)
		N_stars += stars_in_pix
		nInFile += stars_in_pix
		N_in_pixel.append(stars_in_pix)
		n_pixels_at_res[nside] += 1
		
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
		
		# Close file if size exceeds max_stars
		if nInFile >= values.max_stars:
			f.close()
			f = None
	
	if f != None:
		f.close()
	
	if N_pix != 0:
		N_in_pixel = np.array(N_in_pixel)
		print '# of stars in footprint: %d.' % N_stars
		print '# of pixels in footprint: %d.' % N_pix
		print 'Stars per pixel:'
		print '    min: %d' % N_min
		print '    5%%: %d' % (np.percentile(N_in_pixel, 5.))
		print '    50%%: %d' % (np.percentile(N_in_pixel, 50.))
		print '    mean: %d' % (float(N_stars) / float(N_pix))
		print '    95%%: %d' % (np.percentile(N_in_pixel, 95.))
		print '    max: %d' % N_max
		print '# of pixels at each nside resolution:'
		
		for nside in nside_options:
			area_per_pix = hp.pixelfunc.nside2pixarea(nside, degrees=True)
			#area_per_pix = 4.*np.pi * (180./np.pi)**2. / (12. * nside**2.)
			area = n_pixels_at_res[nside] * area_per_pix
			print '    %d: %d (%.2f deg^2)' % (nside, n_pixels_at_res[nside], area)
		
		pct_sparse = 100. * float(N_pix_too_sparse) / float(N_pix_too_sparse + N_pix)
		print '# of pixels too sparse: %d (%.3f %%)' % (N_pix_too_sparse, pct_sparse)
		
		pct_out_of_bounds = 100. * float(N_pix_out_of_bounds) / float(N_pix_out_of_bounds + N_pix)
		print '# of pixels out of bounds: %d (%.3f %%)' % (N_pix_out_of_bounds, pct_out_of_bounds)
		
		print '# of files: %d.' % nFiles
	else:
		print 'No pixels in specified bounds with sufficient # of stars.'
	
	if (values.bounds != None) and (np.sum(N_pix) != 0):
		print ''
		print 'Bounds of included pixel centers:'
		print '\t(l_min, l_max) = (%.3f, %.3f)' % (l_min, l_max)
		print '\t(b_min, b_max) = (%.3f, %.3f)' % (b_min, b_max)
	
	return 0

if __name__ == '__main__':
	main()

