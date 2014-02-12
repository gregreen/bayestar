#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       PS1_err_curve.py
#       
#       Copyright 2013 Greg Green <greg@greg-G53JW>
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

import matplotlib as mplib
#mplib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import healpy as hp
import numpy as np
import scipy
import scipy.special
import pyfits
import h5py

import lsd

from ps import pssdsstransformall
import iterators


def flux2luptitudes(x, b):
	'''
	Convert flux to Luptitudes (asinh magnitudes).
	
	Inputs:
	    x  Flux in units of the flux at zeroeth magnitude
	    b  Dimensionless softening parameter
	'''
	
	return -2.5 / np.log(10.) * (np.arcsinh(x / (2. * b)) + np.log(b))


def luptitudes2flux(mu, b):
	'''
	Convert Luptitudes (asinh magnitudes) to flux (in
	units of the flux at zeroeth magnitude).
	
	Inputs:
	    mu  Luptitudes
	    b   Dimensionless softening parameter
	'''
	
	return -2. * b * np.sinh(np.log(10.) / 2.5 * mu + np.log(b))


def flux2mags(x):
	'''
	Convert flux to magnitudes.
	
	Input:
	    x  Flux in units of the flux at zeroeth magnitude
	'''
	
	return -2.5 / np.log(10.) * np.log(x)


def luptitudes2mags(mu, b):
	'''
	Convert Luptitudes (asinh magnitudes) to standard magnitudes.
	
	Inputs:
	    mu  Luptitudes
	    b   Dimensionless softening parameter
	'''
	
	x = luptitudes2flux(mu, b)
	return flux2mags(x)


def gc_dist(l_0, b_0, l_1, b_1):
	p_0 = np.pi / 180. * l_0
	t_0 = np.pi / 180. * b_0
	p_1 = np.pi / 180. * l_1
	t_1 = np.pi / 180. * b_1
	
	return np.arcsin(np.sqrt(np.sin(0.5*(t_1-t_0))**2 + np.cos(t_0) * np.cos(t_1) * np.sin(0.5*(p_1-p_0))**2))


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
		'''
		over_threshold = True
		
		for i in xrange(4 * pix_idx, 4 * pix_idx + 4):
			idx = (sub_pix_idx == i)
			
			if np.sum(idx) < n_stars_min:
				over_threshold = False
				break
		
		if not over_threshold:
			return [(nside, pix_idx, obj)]
		'''
		
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


def mapper(qresult, bounds):
	obj = lsd.colgroup.fromiter(qresult, blocks=True)
	
	if (obj != None) and (len(obj) > 0):
		#
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


def step_better(ax, x, y, *args, **kwargs):
	kwargs['where'] = 'post'
	
	y_tmp = np.append(y, y[-1])
	
	ax.step(x, y_tmp, *args, **kwargs)


def main():
	parser = argparse.ArgumentParser(
	           prog='PS1_err_curve.py',
	           description='Estimate PS1 error curve in each band.',
	           add_help=True)
	parser.add_argument('-b', '--bounds', type=float, nargs=4, default=None,
	                    help='Restrict pixels to region enclosed by: RA_min, RA_max, Dec_min, Dec_max.')
	parser.add_argument('--n-bands', type=int, default=1,
	                    help='Min. # of PS1 passbands with detection.')
	parser.add_argument('--n-det', type=int, default=4,
	                    help='Min. # of PS1 detections.')
	parser.add_argument('-w', '--n-workers', type=int, default=5,
	                    help='# of workers for LSD to use.')
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	args = parser.parse_args(sys.argv[offset:])
	
	mplib.rc('text', usetex=True)
	
	n_pointlike = args.n_bands - 1
	if n_pointlike == 0:
		n_pointlike = 1
	
	
	# Determine the query bounds
	query_bounds = None
	
	if args.bounds != None:
		query_bounds = lsd.bounds.rectangle(args.bounds[0], args.bounds[2],
		                                    args.bounds[1], args.bounds[3],
		                                    coordsys='equ')
	
	query_bounds = lsd.bounds.make_canonical(query_bounds)
	
	
	# Set up the query
	db = lsd.DB(os.environ['LSD_DB'])
	
	query = ("select mean, err, "
	         "mean_ap, nmag_ok, maglimit "
	         "from ucal_magsqx_noref "
	         "where (numpy.sum(nmag_ok > 0, axis=1) >= %d) "
	         "& (nmag_ok[:,0] > 0) "
	         "& (numpy.sum(nmag_ok, axis=1) >= %d) "
	         "& (numpy.sum(mean - mean_ap < 0.1, axis=1) >= %d)"
	         % (args.n_bands, args.n_det, n_pointlike))
	
	query = db.query(query)
	
	
	# Execute query
	rows = query.fetch(bounds=query_bounds, nworkers=args.n_workers)
	
	print 'Query returned %d objects.' % (len(rows))
	
	
	print 'maglimit:', np.median(rows['maglimit'], axis=0)
	
	# Difference from magnitude limit
	Delta_m = rows['mean'] - rows['maglimit']
	
	
	# Calculate error curve
	n_bins = 100
	bin_edges = np.linspace(-9., 1., n_bins+1)
	binned_err_curve = np.empty((5, n_bins, 3), dtype='f8')
	
	for band in xrange(5):
		idx = np.isfinite(Delta_m[:, band]) & (rows['mean'][:, band] > 0.)
		Dm = Delta_m[idx, band]
		err = rows['err'][idx, band]
		
		for n, (D0, D1) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
			idx = (Dm > D0) & (Dm <= D1)
			
			try:
				binned_err_curve[band, n, :] = np.percentile(err[idx], [5., 50., 95.])
			except:
				binned_err_curve[band, n, :] = np.nan
	
	
	# Fit the error curve
	Dm = 0.5 * (bin_edges[:-1] + bin_edges[1:])
	
	idx_0 = np.argmin(np.abs(Dm + 0.25))
	idx_1 = np.argmin(np.abs(Dm + 1.0))
	
	Dm_0 = Dm[idx_0]
	Dm_1 = Dm[idx_1]
	
	f_0 = binned_err_curve[:, idx_0, 1]
	f_1 = binned_err_curve[:, idx_1, 1]
	
	c = (Dm_1 - Dm_0) / (np.log(f_1) - np.log(f_0))
	a = f_0 * np.exp(-(Dm_0 - 0.16) / c)
	
	print 'a:', a
	print 'c:', c
	
	
	# Plot error curve
	fig = plt.figure()
	
	Dm = np.linspace(-8., 1., 2000)
	
	for band,color in enumerate(['cyan', 'b', 'g', 'r', 'k']):
		ax = fig.add_subplot(5,1,1+band)
		
		step_better(ax, bin_edges, binned_err_curve[band, :, 0], c=color)
		step_better(ax, bin_edges, binned_err_curve[band, :, 1], c=color)
		step_better(ax, bin_edges, binned_err_curve[band, :, 2], c=color)
		
		fit = a[band] * np.exp((Dm - 0.16) / c[band])
		fit *= 1. + np.random.normal(loc=0., scale=0.1, size=fit.shape)
		
		ax.plot(Dm, fit, c='k', alpha=0.5, lw=1.5)
		
		ax.set_xlim(-4., 1.)
	
	plt.show()
	
	
	return 0

if __name__ == '__main__':
	main()

