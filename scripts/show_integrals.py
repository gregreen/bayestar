#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#
#       show_integrals.py
#       
#       Copyright 2014 Greg <greg@greg-G53JW>
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

import sys, argparse
from os.path import abspath, expanduser

import numpy as np
import scipy.optimize as opt
from scipy.ndimage.filters import gaussian_filter

import matplotlib as mplib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

import h5py

import hdf5io


def los_integral(surfs, dEBV, EBV_lim=(0., 5.), subsampling=1):
	n_stars, n_DM, n_EBV = surfs.shape
	n_regions = dEBV.size - 1
	
	assert (n_DM % n_regions == 0)
	
	n_pix_per_bin = n_DM / n_regions
	n_samples = subsampling * n_pix_per_bin
	
	EBV_per_pix = (EBV_lim[1] - EBV_lim[0]) / float(n_EBV)
	
	EBV = np.hstack([np.cumsum(dEBV), 0.]) - EBV_lim[0]
	
	x = np.arange(subsampling * n_DM).astype('f4') / float(subsampling * n_pix_per_bin)
	x_floor = x.astype('i4')
	a = (x - x_floor.astype('f4'))
	
	y = ((1. - a) * EBV[x_floor] + a * EBV[x_floor+1]) / EBV_per_pix
	y_floor = y.astype('i4')
	a = (y - y_floor.astype('f4'))
	
	x_pix = np.arange(subsampling * n_DM).astype('f4') / subsampling
	x_pix_floor = x_pix.astype('i4')
	
	p = ( (1. - a) * surfs[:, x_pix_floor, y_floor]
	           + a * surfs[:, x_pix_floor, y_floor+1] )
	
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.imshow(np.sum(surfs, axis=0).T, origin='lower',
	          aspect='auto', interpolation='nearest')
	ax.plot(x_pix, y, 'g-', marker='.')
	ax.set_xlim(0, n_DM)
	ax.set_ylim(0, n_EBV)
	plt.show()
	
	return np.sum(p, axis=1) / float(subsampling)


def los2ax(ax, fname, group, DM_lim, *args, **kwargs):
	chain = hdf5io.TChain(fname, '%s/los' % group)
	
	mu = np.linspace(DM_lim[0], DM_lim[1], chain.get_nDim())
	if 'alpha' not in kwargs:
		kwargs['alpha'] = 1. / np.power(chain.get_nSamples(), 0.55)
	
	# Plot all paths
	EBV_all = np.cumsum(np.exp(chain.get_samples(0)), axis=1)
	lnp = chain.get_lnp()[0, 1:]
	#lnp = np.exp(lnp - np.max(lnp))
	lnp_min, lnp_max = np.percentile(lnp, [10., 90.])
	lnp = (lnp - lnp_min) / (lnp_max - lnp_min)
	lnp[lnp > 1.] = 1.
	lnp[lnp < 0.] = 0.
	
	for i,EBV in enumerate(EBV_all[1:]):
		c = (1.-lnp[i], 0., lnp[i])
		kwargs['c'] = c
		ax.plot(mu, EBV, *args, **kwargs)
	
	kwargs['c'] = 'g'
	kwargs['lw'] = 1.5
	kwargs['alpha'] = 0.5
	ax.plot(mu, EBV_all[0], *args, **kwargs)
	
	# Plot mean path
	#y = np.mean(EBV_all, axis=0)
	#y_err = np.std(EBV_all, axis=0)
	#ax.errorbar(mu, y, yerr=y_err, c='g', ecolor=(0., 1., 0., 0.5), alpha=0.3)
	
	# Plot best path
	#i = np.argsort(chain.get_lnp(0))[::-1]
	#alpha = 1.
	#for ii in i[:3]:
	#	ax.plot(mu, EBV_all[ii], 'r-', alpha=alpha)
	#	alpha *= 0.5
	#alpha = 1.
	#for ii in i[-10:]:
	#	ax.plot(mu, EBV_all[ii], 'k-', alpha=alpha)
	#	alpha *= 0.85
	
	ax.set_xlim(DM_lim[0], DM_lim[1]) 

def clouds2ax(ax, fname, group, DM_lim, *args, **kwargs):
	chain = hdf5io.TChain(fname, '%s/clouds' % group)
	mu_range = np.linspace(DM_lim[0], DM_lim[1], chain.get_nDim())
	if 'alpha' not in kwargs:
		kwargs['alpha'] = 1. / np.power(chain.get_nSamples(), 0.55)
	
	# Plot all paths
	N_clouds = chain.get_nDim() / 2
	N_paths = chain.get_nSamples()
	mu_tmp = np.cumsum(chain.get_samples(0)[:,:N_clouds], axis=1)
	EBV_tmp = np.cumsum(np.exp(chain.get_samples(0)[:,N_clouds:]), axis=1)
	
	mu_all = np.zeros((N_paths, 2*(N_clouds+1)), dtype='f8')
	EBV_all = np.zeros((N_paths, 2*(N_clouds+1)), dtype='f8')
	mu_all[:,0] = mu_range[0]
	mu_all[:,1:-1:2] = mu_tmp
	mu_all[:,2:-1:2] = mu_tmp
	mu_all[:,-1] = mu_range[-1]
	EBV_all[:,2:-1:2] = EBV_tmp
	EBV_all[:,3::2] = EBV_tmp
	#EBV_all[:,-1] = EBV_tmp[:,-1]
	
	lnp = chain.get_lnp()[0, 1:]
	lnp_min, lnp_max = np.percentile(lnp, [10., 90.])
	lnp = (lnp - lnp_min) / (lnp_max - lnp_min)
	lnp[lnp > 1.] = 1.
	lnp[lnp < 0.] = 0.
	
	for i,(mu,EBV) in enumerate(zip(mu_all[1:], EBV_all[1:])):
		c = (1.-lnp[i], 0., lnp[i])
		kwargs['c'] = c
		ax.plot(mu, EBV, *args, **kwargs)
	
	kwargs['c'] = 'g'
	kwargs['alpha'] = 0.5
	ax.plot(mu_all[0], EBV_all[0], *args, **kwargs)
	
	# Plot mean path
	#y = np.mean(EBV_all, axis=0)
	#y_err = np.std(EBV_all, axis=0)
	#ax.errorbar(mu, y, yerr=y_err, c='g', ecolor=(0., 1., 0., 0.5), alpha=0.3)
	
	# Plot best path
	#i = np.argsort(chain.get_lnp(0))[::-1]
	#alpha = 1.
	#for ii in i[:3]:
	#	ax.plot(mu, EBV_all[ii], 'r-', alpha=alpha)
	#	alpha *= 0.5
	#alpha = 1.
	#for ii in i[-10:]:
	#	ax.plot(mu, EBV_all[ii], 'k-', alpha=alpha)
	#	alpha *= 0.85
	
	ax.set_xlim(DM_lim[0], DM_lim[1]) 

def find_contour_levels(pdf, pctiles):
	norm = np.sum(pdf)
	pctile_diff = lambda pixval, target: np.sum(pdf[pdf > pixval]) / norm - target
	
	levels = []
	
	for P in pctiles:
		l = opt.brentq(pctile_diff, np.min(pdf), np.max(pdf),
		               args=P/100., xtol=1.e-5, maxiter=25)
		levels.append(l)
	
	return np.array(levels)

def read_photometry(fname, loc):
	f = h5py.File(fname, 'r')
	group = '/photometry/pixel %d-%d' % (loc[0], loc[1])
	data = f[group][:]
	f.close()
	
	return data


def main():
	# Parse commandline arguments
	parser = argparse.ArgumentParser(prog='show_integrals',
	              description='Shows information on line-of-sight integrals from Bayestar.',
	              add_help=True)
	parser.add_argument('input', type=str, help='Bayestar input file.')
	parser.add_argument('output', type=str, help='Bayestar output file.')
	parser.add_argument('loc', type=int, nargs=2, help='HEALPix nside and index.')
	parser.add_argument('-o', '--output', type=str, help='Filename for plot.')
	parser.add_argument('-s', '--show', action='store_true', help='Show plot.')
	
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	args = parser.parse_args(sys.argv[offset:])
	
	if (args.output == None) and not args.show:
		print 'Either --output or --show must be given.'
		return 0
	
	outfname = abspath(expanduser(args.output))
	group = 'pixel %d-%d' % (args.loc[0], args.loc[1])
	
	# Load in evidences
	dset = '%s/stellar chains' % group
	chain = hdf5io.TChain(outfname, dset)
	lnZ = chain.get_lnZ()[:]
	lnZ_max = np.percentile(lnZ[np.isfinite(lnZ)], 95.)
	lnZ_idx = (lnZ > lnZ_max - 5.)
	
	EBV_star = chain.get_samples()[:, 0, 0]
	print np.median(EBV_star)
	
	# Load in pdfs
	dset = '%s/stellar pdfs' % group
	pdf = hdf5io.TProbSurf(outfname, dset)
	x_min, x_max = pdf.x_min, pdf.x_max
	surfs = pdf.get_p()
	
	# Load in piecewise-linear fit
	chain = hdf5io.TChain(outfname, '%s/los' % group)
	
	dEBV = np.exp(chain.get_samples()[0,:,:])
	
	p_best = los_integral(surfs, dEBV[0,:],
	                      EBV_lim=(x_min[1], x_max[1]),
	                      subsampling=10)
	
	p_prime = los_integral(surfs, 1.05*dEBV[0,:],
	                       EBV_lim=(x_min[1], x_max[1]),
	                       subsampling=10)
	
	print np.min(p_best), np.mean(p_best), np.median(p_best), np.max(p_best)
	
	# Load in photometry
	infname = abspath(expanduser(args.input))
	phot = read_photometry(infname, args.loc)
	
	print np.percentile(phot['EBV'], [5., 50., 95.])
	
	
	# Color-color diagram
	
	gr = phot['mag'][:,0] - phot['mag'][:,1]
	ri = phot['mag'][:,1] - phot['mag'][:,2]
	
	idx = (gr < 3.) & (gr > -0.5) & (ri < 2.5) & (ri > -0.5)
	idx &= (phot['err'][:,0] < 1.e9) & (phot['err'][:,1] < 1.e9) & (phot['err'][:,2] < 1.e9)
	idx &= lnZ_idx
	
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	
	c = np.log(p_best)
	cidx = np.isfinite(c)
	c[~cidx] = np.min(c[cidx])
	
	ax.scatter(gr[idx], ri[idx], c=-c[idx], s=6, edgecolor='none', cmap='YlOrRd')
	
	
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	
	c = np.log(p_best) - np.log(p_prime)
	cidx = np.isfinite(c)
	c[~cidx] = np.min(c[cidx])
	vmax = np.max(np.abs(c))
	
	ax.scatter(gr[idx], ri[idx], c=c[idx],
	           s=6, edgecolor='none', cmap='bwr',
	           vmin=-vmax, vmax=vmax)
	
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	
	c = lnZ
	
	ax.scatter(gr[idx], ri[idx], c=-c[idx],
	           s=6, edgecolor='none', cmap='YlOrRd')
	
	plt.show()
	
	return 0

if __name__ == '__main__':
	main()

