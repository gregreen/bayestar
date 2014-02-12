#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       s82_completeness.py
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


def main():
	parser = argparse.ArgumentParser(
	           prog='s82_completeness.py',
	           description='Estimate PS1 completeness by comparison with deep SDSS Stripe 82 photometry.',
	           add_help=True)
	#parser.add_argument('out', type=str, help='Filename for query output.')
	parser.add_argument('-b', '--bounds', type=float, nargs=4, default=None,
	                    help='Restrict pixels to region enclosed by: RA_min, RA_max, Dec_min, Dec_max.')
	parser.add_argument('--n-bands', type=int, default=4,
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
	
	query = ("select s82coadd.l as l, s82coadd.b as b, "
	                "s82coadd.ra as ra_s82, "
	                "s82coadd.dec as dec_s82, "
	                "s82coadd.psfcounts as s82_counts, "
	                "s82coadd.psfcountserr as s82_counts_err, "
	                "ucal_magsqx_noref.ra as ra_ps1, "
	                "ucal_magsqx_noref.dec as dec_ps1, "
	                "ucal_magsqx_noref.mean as ps1_mean, "
	                "ucal_magsqx_noref.err as ps1_err, "
	                "ucal_magsqx_noref.mean_ap as ps1_mean_ap, "
	                "ucal_magsqx_noref.maglimit as ps1_maglim, "
	                "ucal_magsqx_noref.nmag_ok as ps1_nmag_ok "
	         "from s82coadd, ucal_magsqx_noref(outer, matchedto=s82coadd, nmax=1, dmax=30) "
	         "where (s82coadd.objc_type == 6)")
	
	query = db.query(query)
	
	# Execute query
	rows = query.fetch(bounds=query_bounds, nworkers=args.n_workers)
	
	
	# Transform from luptitudes to AB magnitudes
	b_s82 = [1.0e-11, 0.43e-11, 0.81e-11, 1.4e-11, 3.7e-11]	# Stripe-82 ugriz softening parameters
	sdss_bands = 'ugriz'
	
	dtype = [(b, 'f8') for b in sdss_bands]
	s82_mags = np.empty(rows['s82_counts'].shape[0], dtype=dtype)
	
	for band, (name, b) in enumerate(zip(sdss_bands, b_s82)):
		s82_mags[name][:] = luptitudes2mags(rows['s82_counts'][:, band], b)
	
	
	# Transform SDSS magnitudes to synthetic PS1 magnitudes
	s82_ps1_mags = pssdsstransformall(s82_mags)
	
	
	# Filter objects which do not have 5-band Stripe-82 detections
	idx = np.isfinite(s82_ps1_mags) & (s82_ps1_mags > 0.) & (s82_ps1_mags < 30.)
	idx = np.all(idx, axis=1)
	
	print 'Stripe-82 objects filtered: %d of %d' % (np.sum(~idx), len(idx))
	
	rows = rows[idx]
	s82_mags = s82_mags[idx]
	s82_ps1_mags = s82_ps1_mags[idx]
	
	
	# Which Stripe-82 objects have PS1 matches
	ps1_mags = rows['ps1_mean']
	ps1_mask = (ps1_mags > 0.)
	
	match_dist = gc_dist(rows['ra_s82'], rows['dec_s82'], rows['ra_ps1'], rows['dec_ps1'])
	max_dist = 1. * np.pi / 180. / 3600. # One arcsecond
	idx = (match_dist < max_dist)
	
	for band in xrange(5):
		ps1_mask[:, band] = ps1_mask[:, band] & idx
	
	'''
	ps1_row_mask = (  (rows['ps1_nmag_ok'][:,0] > 0)
	                & (np.sum(rows['ps1_nmag_ok'] > 0, axis=1) >= args.n_bands)
	                & (np.sum(rows['ps1_nmag_ok'], axis=1) >= args.n_det)
	                & (np.sum(rows['ps1_mean'] - rows['ps1_mean_ap'] < 0.1, axis=1) >= n_pointlike)
	               )
	
	for band in xrange(ps1_mask.shape[1]):
		ps1_mask[:, band] = ps1_mask[:, band] & ps1_row_mask
	'''
	
	print 'Stripe-82 detections:', len(rows)
	print 'PS1 griz matches:', np.sum(ps1_mask, axis=0)
	
	
	# PS1 completeness in each magnitude bin
	mag_bins = np.linspace(-2., 4., 60)
	bin_min = mag_bins[:-1]
	bin_max = mag_bins[1:]
	mag_bin_center = 0.5 * (bin_min + bin_max)
	
	'''
	ps1_pct = np.empty((mag_bins.size-1, 5), dtype='f8')
	
	for band in xrange(5):
		for i, (mag_min, mag_max) in enumerate(zip(bin_min, bin_max)):
			idx = (s82_ps1_mags[:, band] >= mag_min) & (s82_ps1_mags[:, band] < mag_max)
			
			ps1_pct[i, band] = np.sum(ps1_mask[idx, band], axis=0) / float(np.sum(idx))
	'''
	
	# PS1 completeness in different spatial pixels
	pix_idx = lb2pix(128, rows['ra_s82'], rows['dec_s82'])
	
	pix_idx_unique = np.unique(pix_idx)
	n_pix = pix_idx_unique.size
	
	ps1_pct_area = np.empty((n_pix, mag_bins.size-1, 5), dtype='f8')
	maglim_area = np.empty((n_pix, 5), dtype='f8')
	n_stars = np.empty(n_pix)
	
	for k, p_idx in enumerate(pix_idx_unique):
		in_pix = (pix_idx == p_idx)
		n_stars[k] = np.sum(in_pix)
		
		print '%d stars in pixel %d.' % (n_stars[k], k + 1)
		
		s82_ps1_tmp = s82_ps1_mags[in_pix]
		ps1_mask_tmp = ps1_mask[in_pix]
		
		for band in xrange(5):
			maglim = rows['ps1_maglim'][:, band]
			idx = (maglim > 0.)
			maglim_area[k, band] = np.median(maglim[idx])
			maglim = maglim_area[k, band]
			
			for i, (mag_min, mag_max) in enumerate(zip(bin_min, bin_max)):
				idx = (s82_ps1_tmp[:, band] - maglim >= mag_min) & (s82_ps1_tmp[:, band] - maglim < mag_max)
				
				ps1_pct_area[k, i, band] = np.sum(ps1_mask_tmp[idx, band], axis=0) / float(np.sum(idx))
	
	idx = (n_stars > 1000)
	ps1_pct_area = np.percentile(ps1_pct_area[idx], [15.87, 50., 84.13], axis=0)
	ps1_pct_area = np.array(ps1_pct_area)
	
	
	idx = (ps1_pct_area[0, :, :] < 1.e-10)
	ps1_pct_area[0, idx] = 1.e-10 * (0.5 + np.random.random(ps1_pct_area[0, idx].shape))
	
	idx = (ps1_pct_area[1, :, :] < 1.e-9)
	ps1_pct_area[1, idx] = 1.e-9 * (0.5 + np.random.random(ps1_pct_area[0, idx].shape))
	
	idx = (ps1_pct_area[2, :, :] < 1.e-8)
	ps1_pct_area[2, idx] = 1.e-8 * (0.5 + np.random.random(ps1_pct_area[0, idx].shape))
	
	
	# PS1 completeness as a function of mag - maglimit
	mag_diff_bins = np.linspace(-6., 4., 100)
	bin_min = mag_diff_bins[:-1]
	bin_max = mag_diff_bins[1:]
	diff_bin_center = 0.5 * (bin_min + bin_max)
	
	ps1_pct_diff = np.empty((mag_diff_bins.size-1, 5), dtype='f8')
	s82_ps1_diff = s82_ps1_mags - rows['ps1_maglim']
	
	for band in xrange(5):
		idx_maglim = (rows['ps1_maglim'][:, band] > 0.)
		print band, np.median(rows['ps1_maglim'][idx_maglim, band])
		
		for i, (mag_min, mag_max) in enumerate(zip(bin_min, bin_max)):
			idx_bin = (s82_ps1_diff[:, band] >= mag_min) & (s82_ps1_diff[:, band] < mag_max)
			idx = idx_maglim & idx_bin
			
			ps1_pct_diff[i, band] = np.sum(ps1_mask[idx, band], axis=0) / float(np.sum(idx))
	
	
	# Completeness parameterization
	tmp_pct = ps1_pct_area[1, -20:, :]
	idx = (tmp_pct > 1.e-5) & np.isfinite(tmp_pct)
	comp_floor = np.median(tmp_pct[idx])
	print 'Completeness floor: %.2g' % comp_floor
	
	#dm_0 = [0.15, 0.23, 0.17, 0.12, 0.15]
	dm_0 = [0.16, 0.16, 0.16, 0.16, 0.16]
	dm_1 = 0.20
	
	'''
	comp_fit = 0.5 * (1. - scipy.special.erf((dm - 0.15) / 0.47))
	ax.plot(dm, comp_fit, lw=2., alpha=0.3, c='orange')
	comp_fit = (1. - comp_floor) * comp_fit + comp_floor
	'''
	
	#comp_fit = 1. / (1. + np.exp((dm - 0.13) / 0.20))
	#comp_fit_floor = (1. - comp_floor) * comp_fit + comp_floor
	
	
	# Plot completeness
	fig = plt.figure(figsize=(9,6), dpi=150)
	
	band_names = ['g', 'r', 'i', 'z', 'y']
	plt_colors = ['c', 'b', 'g', 'r', 'gray']
	
	for band, (name, color) in enumerate(zip(band_names, plt_colors)):
		maglim = rows['ps1_maglim'][:, band]
		idx = (maglim > 0.)
		maglim = np.percentile(maglim[idx], [15.87, 50., 84.13])
		
		print 'maglim_%d: %.2f + %.2f - %.2f' % (band,
		                                         maglim[1],
		                                         maglim[1] - maglim[0],
		                                         maglim[2] - maglim[1])
		
		ax = fig.add_subplot(2, 3, band+1)
		
		ax.axvline(x=0., c='k', ls=':', lw=1., alpha=0.2)
		ax.axhline(y=1., c='k', ls=':', lw=1., alpha=0.2)
		
		pos = np.all(ps1_pct_area[:, :, band] > 0, axis=0)
		
		ax.fill_between(mag_bin_center,
		                ps1_pct_area[0, :, band],
		                ps1_pct_area[2, :, band],
		                where=pos,
		                color=color,
		                edgecolor=color,
		                alpha=0.5,
		                label=r'$%s_{\mathrm{P1}}$' % name)
		
		ax.semilogy(mag_bin_center, ps1_pct_area[1, :, band],
		            c=color, alpha=0.5, label=r'$%s_{\mathrm{P1}}$' % name)
		
		tmp_pct = ps1_pct_area[1, -25:, band]
		idx = (tmp_pct > 1.e-5) & np.isfinite(tmp_pct)
		comp_floor = np.median(tmp_pct[idx])
		print 'Completeness floor %d: %.2g' % (band, comp_floor)
		comp_floor = 0.01
		
		dm = np.linspace(-2., 4., 1000)
		comp_fit = 1. / (1. + np.exp((dm - dm_0[band]) / dm_1))
		comp_fit_floor = (1. - comp_floor) * comp_fit + comp_floor
		
		ax.semilogy(dm, comp_fit, lw=2., alpha=0.5,
	                              c='k', ls='-',
	                              label=r'$\mathrm{Fit}$')
		
		ax.semilogy(dm, comp_fit_floor, lw=1., alpha=0.25,
		                                c='k', ls='--')
		
		ax.set_yscale('log')
		ax.set_xlim(-1., 2.)
		ax.set_ylim(0.005, 1.5)
		
		if band < 3:
			ax.xaxis.set_label_position('top')
			ax.xaxis.tick_top()
		
		ax.set_xticks([-0.5, 0.0, 0.5, 1.0, 1.5])
		ax.set_xlabel(r'$\Delta %s_{\mathrm{P1}} \ (\mathrm{mag})$' % name, fontsize=16)
		
		ax.yaxis.set_major_formatter(FormatStrFormatter(r'$%.2f$'))
		
		if band not in [0, 3]:
			ax.set_yticklabels([])
	
	fig.text(0.06, 0.5, r'$\mathrm{PS1 \ Completeness}$',
	         fontsize=16, va='center', ha='center', rotation='vertical')
	
	# Legend
	ax = fig.add_subplot(2, 3, 6)
	
	ax.fill_between([0.1, 0.35], [0.7, 0.7], [0.8, 0.8],
	                color='orange', edgecolor='orange', alpha=0.5)
	ax.plot([0.1, 0.35], [0.75, 0.75], color='orange', alpha=0.5)
	ax.text(0.45, 0.75, r'$1 \sigma \ \mathrm{Region}$',
	        ha='left', va='center', fontsize=16)
	
	ax.plot([0.1, 0.35], [0.5, 0.5], lw=2., alpha=0.5,
	                                c='k', ls='-')
	ax.text(0.45, 0.5, r'$\mathrm{Fit}$',
	        ha='left', va='center', fontsize=16)
	
	ax.plot([0.1, 0.35], [0.3, 0.3], lw=1., alpha=0.25,
	                                c='k', ls='--')
	ax.text(0.45, 0.3, r'$\mathrm{Fit \ with \ floor}$',
	        ha='left', va='center', fontsize=16)
	
	ax.set_xlim([0., 1.])
	ax.set_ylim([0.05, 1.05])
	
	ax.axis('off')
	
	
	fig.subplots_adjust(top=0.85, bottom=0.13,
	                    left=0.13, right=0.95,
	                    hspace=0., wspace=0.)
	
	fig.savefig('tmp.png', dpi=300)
	
	#ax.set_title(r'$\mathrm{Absolute}$', fontsize=16)
	#ax.set_xlabel(r'$\mathrm{\Delta m_{P1} \ (mag)}$', fontsize=16)
	#ax.set_xlabel(r'$\mathrm{Stripe-82 \ asinh \ mag}$', fontsize=14)
	#ax.set_ylabel(r'$\mathrm{PS1 \ Completeness}$', fontsize=16)
	
	'''
	ax = fig.add_subplot(2,1,2)
	
	band_names = ['g', 'r', 'i', 'z', 'y']
	plt_colors = ['c', 'b', 'g', 'r', 'gray']
	
	for band, (name, color) in enumerate(zip(band_names, plt_colors)):
		ax.semilogy(diff_bin_center, ps1_pct_diff[:, band], c=color, label=r'$%s$' % name)
	
	ax.set_title(r'$\mathrm{Relative}$', fontsize=16)
	ax.set_xlabel(r'$\mathrm{mag \ - \ maglim}$', fontsize=14)
	#ax.set_xlabel(r'$\mathrm{Stripe-82 \ asinh \ mag}$', fontsize=14)
	ax.set_ylabel(r'$\mathrm{PS1 \ Completeness}$', fontsize=14)
	ax.legend()
	
	ax.set_xlim(-1., 2.)
	ax.set_ylim(0.005, 1.5)
	
	
	fig.savefig('tmp.png', dpi=200)
	'''
	
	# Histograms of synthetic PS1 vs real PS1 magnitudes
	fig = plt.figure()
	
	diff = ps1_mags - s82_ps1_mags
	
	for band, name in enumerate(band_names):
		ax = fig.add_subplot(1, 5, band+1)
		
		ax.hist(diff[ps1_mask[:, band], band], bins=50)
		
		ax.set_title(r'$\Delta %s$' % name, fontsize=16)
		ax.set_xlabel(r'$\Delta %s \ (\mathrm{mag})$' % name, fontsize=14)
	
	
	# Color-color diagrams of synthetic PS1 photometry
	fig = plt.figure()
	
	for c2 in xrange(4):
		for c1 in xrange(c2):
			ax = fig.add_subplot(3, 4, 1 + 3*c1 + c2)
			
			diff_1 = s82_ps1_mags[:, c1] - s82_ps1_mags[:, c1+1]
			diff_2 = s82_ps1_mags[:, c2] - s82_ps1_mags[:, c2+1]
			
			ax.scatter(diff_1, diff_2, s=1.5, alpha=0.05, edgecolor='none')
			
			xlim = ax.get_xlim()
			ylim = ax.get_ylim()
			
			ax.set_xlim(xlim[1], xlim[0])
			ax.set_ylim(ylim[1], ylim[0])
	
	plt.show()
		
	
	return 0

if __name__ == '__main__':
	main()

