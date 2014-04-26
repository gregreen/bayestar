#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       query_lsd_multiple_2MASS.py
#       
#       Copyright 2014 Greg <greg@greg-UX31a>
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
#import pyfits
import h5py

import lsd

import iterators

import matplotlib.pyplot as plt


def mapper(qresult, target_tp, target_radius):
	obj = lsd.colgroup.fromiter(qresult, blocks=True)
	
	if (obj != None) and (len(obj) > 0):
		# Find nearest target center to each star
		theta = np.pi/180. * (90. - obj['b'])
		phi = np.pi/180. * obj['l']
		tp_star = np.array([theta, phi]).T
		d = great_circle_dist(tp_star, target_tp) / target_radius
		min_idx = np.argmin(d, axis=1)
		
		# Group together stars belonging to the same target
		for target_idx, block_idx in iterators.index_by_key(min_idx):
			yield (target_idx, (d[block_idx,target_idx], obj[block_idx]))


def reducer(keyvalue, max_per_pixel, n_bands, n_PS1_bands):
	key, val = keyvalue
	
	# Reorganize distances and object records pairs
	d = []
	obj = []
	
	for dd,oo in val:
		d.append(dd)
		obj.append(oo)
	
	obj = lsd.colgroup.fromiter(obj, blocks=True)
	d = lsd.colgroup.fromiter(d, blocks=True)
	
	# Determine which bands have good detections
	good_det = np.empty((len(obj), 8), dtype=np.bool)
	good_det[:,:5] = (obj['nmag_ok'] > 0) & ~PS1_saturated(obj) & (obj['err'] < 0.20)
	hq_idx = tmass_hq_phot(obj)
	good_det[:,5:] = hq_idx
	
	# Filter out stars that have detections in <N bands
	idx_good = (np.sum(good_det, axis=1) >= n_bands)
	
	# Filter out stars that have detections in <N PS1 bands
	idx_good &= (np.sum(good_det[:, :5], axis=1) >= n_PS1_bands)
	
	# Filter out stars that are extended in at least one 2MASS band
	idx_good &= ~tmass_ext(obj)
	
	
	# Copy in magnitudes and errors
	data = np.empty(len(obj), dtype=[('obj_id','u8'),
	                                 ('l','f8'), ('b','f8'), 
	                                 ('mag','8f4'), ('err','8f4'),
	                                 ('maglimit','8f4'),
	                                 ('nDet','8u4'),
	                                 ('EBV','f4')])
	                                 
	data['mag'][:,:5] = obj['mean'][:,:]
	data['err'][:,:5] = obj['err'][:,:]
	data['mag'][:,5] = obj['J'][:]
	data['err'][:,5] = obj['J_sig'][:]
	data['mag'][:,6] = obj['H'][:]
	data['err'][:,6] = obj['H_sig'][:]
	data['mag'][:,7] = obj['K'][:]
	data['err'][:,7] = obj['K_sig'][:]
	
	data['mag'][~good_det] = 0.
	data['err'][~good_det] = 1.e10
	
	data['maglimit'][:,:5] = obj['maglimit'][:]
	data['maglimit'][:,5:] = 18.	# TODO: Update this with map of 2MASS limits (possibly estimate on-the-fly)
	
	data['nDet'][:,:5] = obj['nmag_ok'][:]
	data['nDet'][:,5:] = 1			# TODO: Replace with real answer, if necessary
	
	data['EBV'][:] = obj['EBV'][:]
	
	data['obj_id'][:] = obj['obj_id'][:]
	data['l'][:] = obj['l'][:]
	data['b'][:] = obj['b'][:]
	
	data = data[idx_good]
	
	# Limit number of stars
	if max_per_pixel != None:
		if len(data) > max_per_pixel:
			d = d[idx_good]
			idx = np.argsort(d)
			data = data[idx[:max_per_pixel]]
	
	yield (key, data)
	
	'''
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
	
	#obj['EBV'][:] = d[:]
	
	# Combine and apply the masks
	mask_detect = np.sum(obj['mean'], axis=1).astype(np.bool)
	mask_informative = (np.sum(obj['err'] > 1.e10, axis=1) < 3).astype(np.bool)
	mask_keep = np.logical_and(mask_detect, mask_informative)
	obj = obj[mask_keep]
	
	# Limit number of stars
	if max_per_pixel != None:
		if len(obj) > max_per_pixel:
			d = d[mask_keep]
			idx = np.argsort(d)
			obj = obj[idx[:max_per_pixel]]
	
	yield (key, obj)
	'''


def start_file(base_fname, index):
	fout = open('%s_%d.in' % (base_fname, index), 'wb')
	f.write(np.array([0], dtype=np.uint32).tostring())
	return f


def to_file(f, target_idx, target_name, gal_lb, EBV, data):
	close_file = False
	if type(f) == str:
		f = h5py.File(fname, 'a')
		close_file = True
	
	ds_name = '/photometry/pixel 512-%d' % target_idx
	ds = f.create_dataset(ds_name, data.shape, data.dtype, chunks=True,
	                      compression='gzip', compression_opts=9)
	ds[:] = data[:]
	
	N_stars = data.shape[0]
	gal_lb = np.array(gal_lb, dtype='f8')
	pix_idx = target_idx
	nside = 512
	nest = True
	
	att_f8 = np.array([EBV], dtype='f8')
	att_u8 = np.array([pix_idx, target_idx], dtype='u8')
	att_u4 = np.array([nside, N_stars], dtype='u4')
	att_u1 = np.array([nest], dtype='u1')
	
	ds.attrs['healpix_index'] = att_u8[0]
	ds.attrs['nested'] = att_u1[0]
	ds.attrs['nside'] = att_u4[0]
	ds.attrs['l'] = gal_lb[0]
	ds.attrs['b'] = gal_lb[1]
	ds.attrs['EBV'] = att_f8[0]
	ds.attrs['target_index'] = att_u8[1]
	ds.attrs['target_name'] = target_name
	
	if close_file:
		f.close()
	
	return gal_lb


def great_circle_dist(tp0, tp1):
	'''
	Returns the great-circle distance bewteen two sets of coordinates,
	tp0 and tp1.
	
	Inputs:
	    tp0  (N, 2) numpy array. Each element is (theta, phi) in rad.
	    tp1  (M, 2) numpy array. Each element is (theta, phi) in rad.
	
	Output:
	    dist  (N, M) numpy array. dist[n,m] = dist(tp0[n], tp1[m]).
	'''
	
	N = tp0.shape[0]
	M = tp1.shape[0]
	out = np.empty((N,M), dtype=tp0.dtype)
	
	dist = lambda p0, t0, p1, t1: np.arccos(np.cos(t0)*np.cos(t1)
	                              + np.sin(t0)*np.sin(t1)*np.cos(p0-p1))
	
	if N <= M:
		for n in xrange(N):
			out[n,:] = dist(tp0[n,1], tp0[n,0], tp1[:,1], tp1[:,0])
	else:
		for m in xrange(M):
			out[:,m] = dist(tp0[:,1], tp0[:,0], tp1[m,1], tp1[m,0])
	
	return out


def get_bounds(infname):
	bounds = []
	name, l, b, radius = [], [], [], []
	
	f = open(infname, 'r')
	
	for line in f:
		line_stripped = line.lstrip().rstrip()
		if len(line_stripped) == 0:
			continue
		elif line_stripped[0] == '#':
			continue
		
		tmp = line_stripped.split()
		name.append(tmp[0])
		l.append(float(tmp[1]))
		b.append(float(tmp[2]))
		radius.append(float(tmp[3]))
		bounds.append( lsd.bounds.beam(l[-1], b[-1], radius[-1], coordsys='gal') )
	
	return bounds, name, np.array(l), np.array(b), np.array(radius)


def PS1_saturated(obj):
	'''
	Return indices of saturated detections in bands.
	'''
	
	sat = np.zeros((len(obj), 5), dtype=np.bool)
	PS1_sat_limit = [14.5, 14.5, 14.5, 14., 13.]
	
	for i,m in enumerate(PS1_sat_limit):
		idx = (obj['mean'][:,i] < m)
		sat[idx, i] = 1
	
	return sat


def tmass_ext(obj):
	'''
	Returns indices of extended objects.
	'''
	
	return (obj['ext_key'] > 0)


def tmass_hq_phot(obj):
	'''
	Return index of which detections in which bands
	are of high quality, as per the 2MASS recommendations:
	<http://www.ipac.caltech.edu/2mass/releases/allsky/doc/sec1_6b.html#composite>
	'''
	
	# Photometric quality in each passband
	idx = (obj['ph_qual'] == '0')
	obj['ph_qual'][idx] = '000'
	ph_qual = np.array(map(list, obj['ph_qual']))
	
	# Read flag in each passband
	idx = (obj['rd_flg'] == '0')
	obj['rd_flg'][idx] = '000'
	rd_flg = np.array(map(list, obj['rd_flg']))
	#rd_flg = (rd_flg == '1') | (rd_flg == '3')
	
	# Contamination flag in each passband
	idx = (obj['cc_flg'] == '0')
	obj['cc_flg'][idx] = '000'
	cc_flg = np.array(map(list, obj['cc_flg']))
	
	# Combine passband flags
	cond_1 = (ph_qual == 'A') | (rd_flg == '1') | (rd_flg == '3')
	cond_1 &= (cc_flg == '0')
	
	# Source quality flags
	cond_2 = (obj['use_src'] == 1) & (obj['gal_contam'] == 0)# & (obj['ext_key'] <= 0)
	
	# Combine all flags for each object
	hq = np.empty((len(obj), 3), dtype=np.bool)
	
	for i in range(3):
		hq[:,i] = cond_1[:,i] & cond_2
	
	return hq


def main():
	parser = argparse.ArgumentParser(
	           prog='query_lsd_multiple_2MASS.py',
	           description='Generate bayestar input files from PS1 and 2MASS photometry, given list of targets.',
	           add_help=True)
	parser.add_argument('targets', type=str, help='Input target list.\n'
	                                              'Each line should be of the form "name l b radius (deg)".')
	parser.add_argument('out', type=str, help='Output filename.')
	parser.add_argument('-min', '--min-stars', type=int, default=1,
	                    help='Minimum # of stars in pixel (default: 1).')
	parser.add_argument('-max', '--max-stars', type=int, default=50000,
	                    help='Maximum # of stars in file')
	parser.add_argument('-ppix', '--max-per-pix', type=int, default=None,
	                    help='Take at most N nearest stars to center of pixel.')
	parser.add_argument('--n-bands', type=int, default=4,
	                    help='Min. # of1 passbands with detection.')
	parser.add_argument('--n-PS1-bands', type=int, default=2,
	                    help='Min. # of PS1 passbands with detection.')
	parser.add_argument('--n-det', type=int, default=4,
	                    help='Min. # of PS1 detections.')
	parser.add_argument('--n-pointlike', type=int, default=2,
	                    help='Min. # of PS1 bands that pass\n'
	                         'Aperture-PSF magnitude cut.')
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	values = parser.parse_args(sys.argv[offset:])
	
	nPointlike = values.n_bands - 1
	if nPointlike == 0:
		nPointlike = 1
	
	# Determine the query bounds
	query_bounds, target_name, l, b, target_radius = get_bounds(values.targets)
	query_bounds = lsd.bounds.make_canonical(query_bounds)
	
	# Convert target positions to useful forms
	target_lb = np.empty((len(l), 2), dtype='f8')
	target_lb[:,0] = l
	target_lb[:,1] = b
	target_tp = np.empty((len(l), 2), dtype='f8')
	target_tp[:,0] = np.pi/180. * (90. - b)
	target_tp[:,1] = np.pi/180. * l
	
	# Set up the query
	db = lsd.DB(os.environ['LSD_DB'])
	query = None
	query = ("select equgal(ra, dec) as (l, b), "
	         "SFD.EBV(l, b) as EBV, "
	         "obj_id, maglimit, "
	         "mean, err, mean_ap, nmag_ok, "
	         "tmass.ph_qual as ph_qual, "
	         "tmass.use_src as use_src, "
	         "tmass.rd_flg as rd_flg, "
	         "tmass.ext_key as ext_key, "
	         "tmass.gal_contam as gal_contam, "
	         "tmass.cc_flg as cc_flg, "
	         "tmass.j_m as J, tmass.j_msigcom as J_sig, "
	         "tmass.h_m as H, tmass.h_msigcom as H_sig, "
	         "tmass.k_m as K, tmass.k_msigcom as K_sig "
	         "from ucal_magsqx_noref, "
	         "tmass(outer, matchedto=ucal_magsqx_noref, dmax=2.0, nmax=1)"
	         "where (numpy.sum(nmag_ok > 0, axis=1) >= %d) "
	         "& (numpy.sum(nmag_ok, axis=1) >= %d) "
	         "& (numpy.sum(mean - mean_ap < 0.1, axis=1) >= %d)"
	         % (values.n_PS1_bands, values.n_det, values.n_pointlike))
	
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
	
	n_in_band = np.zeros(8, dtype='u8')
	
	# Write each pixel to the same file
	for (t_idx, obj) in query.execute([(mapper, target_tp, target_radius),
	                                   (reducer, values.max_per_pix, values.n_bands, values.n_PS1_bands)],
	                                  bounds=query_bounds):
		if len(obj) < values.min_stars:
			continue
		
		'''
		# Prepare output for pixel
		outarr = np.zeros(len(obj), dtype=[('obj_id','u8'),
		                                   ('l','f8'), ('b','f8'), 
		                                   ('mag','f4',8), ('err','f4',8),
		                                   ('maglimit','f4',8),
		                                   ('nDet','u4',8),
		                                   ('EBV','f4')])
		outarr['obj_id'][:] = obj['obj_id'][:]
		outarr['l'][:] = obj['l'][:]
		outarr['b'][:] = obj['b'][:]
		
		outarr['mag'][:,:5] = obj['mean'][:]
		outarr['err'][:,:5] = obj['err'][:]
		outarr['maglimit'][:,:5] = obj['maglimit'][:]
		outarr['nDet'][:,:5] = obj['nmag_ok'][:]
		
		tmass_good = tmass_hq_phot(obj)
		
		
		
		idx = tmass_good[0]
		outarr['mag'][idx,5] = obj['J'][idx]
		outarr['err'][idx,5] = obj['J_sig'][idx]
		outarr['err'][~idx,5] = 1.e10
		
		idx = tmass_good[1]
		outarr['mag'][idx,6] = obj['H'][idx]
		outarr['err'][idx,6] = obj['H_sig'][idx]
		outarr['err'][~idx,6] = 1.e10
		
		idx = tmass_good[2]
		outarr['mag'][idx,7] = obj['K'][idx]
		outarr['err'][idx,7] = obj['K_sig'][idx]
		outarr['err'][~idx,7] = 1.e10
		
		outarr['EBV'][:] = obj['EBV'][:]
		'''
		
		EBV = np.percentile(obj['EBV'][:], 95.)
		
		n_in_band += np.sum(obj['err'] < 1.e9, axis=0)
		
		# Open output file
		if f == None:
			fname = '%s.%.5d.%s' % (fnameBase, nFiles, fnameSuffix)
			f = h5py.File(fname, 'w')
			nInFile = 0
			nFiles += 1
		
		# Write to file
		gal_lb = to_file(f, t_idx, target_name[t_idx],
		                    target_lb[t_idx], EBV, obj)
		
		# Update stats
		N_pix += 1
		stars_in_pix = len(obj)
		N_stars += stars_in_pix
		nInFile += stars_in_pix
		
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
		print '# of stars: %d.' % N_stars
		
		for k,n in enumerate(['g', 'r', 'i', 'z', 'y', 'J', 'H', 'K']):
			print '    %s: %d' % (n, n_in_band[k])
		
		print '# of targets: %d.' % N_pix
		print 'Stars per target:'
		print '    min: %d' % N_min
		print '    mean: %d' % (N_stars / N_pix)
		print '    max: %d' % N_max
		print '# of files: %d.' % nFiles
		
		if np.sum(N_pix) != 0:
			print ''
			print 'Bounds of included pixel centers:'
			print '\t(l_min, l_max) = (%.3f, %.3f)' % (l_min, l_max)
			print '\t(b_min, b_max) = (%.3f, %.3f)' % (b_min, b_max)
	else:
		print 'No pixels in specified bounds with sufficient # of stars.'
	
	return 0

if __name__ == '__main__':
	main()

