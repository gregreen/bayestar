#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  pctile-test.py
#  
#  Copyright 2012 Greg Green <greg@greg-UX31A>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

import numpy as np

from scipy.interpolate import interp2d, RectBivariateSpline
import scipy.ndimage.interpolation as interp
import scipy.stats

import matplotlib.pyplot as plt
import matplotlib as mplib

import argparse, sys
from os.path import abspath

import h5py

import hdf5io


def stack_shifted(bounds, p, shift, norm):
	dx = shift[0] * p.shape[1] / (bounds[1] - bounds[0])
	dy = shift[1] * p.shape[2] / (bounds[3] - bounds[2])
	dxy = np.vstack([dx,dy]).T
	p_stacked = np.zeros(p.shape[1:], dtype='f8')
	for surf,D,Z in zip(p,dxy,norm):
		tmp = interp.shift(surf, D) / Z
		p_stacked += tmp #*= tmp + 1.e-5*np.exp(-tmp/1.e-2)
	return p_stacked


def P_star(bounds, p, truth):
	idx_DM = ( (truth['DM'] - bounds[0]) / (bounds[1] - bounds[0])
	                                         * p.shape[1] ).astype('i8')
	idx_Ar = ( (truth['EBV'] - bounds[2]) / (bounds[3] - bounds[2])
	                                         * p.shape[2] ).astype('i8')
	
	idx = (idx_DM > p.shape[1])
	idx_DM[idx] = p.shape[1] - 1
	idx = (idx_DM < 0)
	idx_DM[idx] = 0
	
	idx = (idx_EBV > p.shape[2])
	idx_EBV[idx] = p.shape[2] - 1
	idx = (idx_EBV < 0)
	idx_EBV[idx] = 0
	
	idx = [np.arange(p.shape[0]), idx_DM, idx_Ar]
	
	threshold = p[idx]
	
	P_ret = np.empty(p.shape[0], dtype='f8')
	for i,pp in enumerate(p):
		idx = pp > threshold[i]
		gtr, less = np.sum(pp[idx]), np.sum(pp[~idx])
		P_ret[i] = less / (gtr + less)
	
	return P_ret

def P_star_2(lnp, bounds, p, truth):
	idx_DM = ( (truth['DM'] - bounds[0]) / (bounds[1] - bounds[0])
	                                         * p.shape[1] ).astype('i8')
	idx_Ar = ( (truth['EBV'] - bounds[2]) / (bounds[3] - bounds[2])
	                                         * p.shape[2] ).astype('i8')
	
	idx = [np.arange(p.shape[0]), idx_DM, idx_Ar]
	
	threshold = p[idx]
	
	P_ret = np.empty(p.shape[0], dtype='f8')
	for k,(lnp_samples, p_threshold) in enumerate(zip(lnp, threshold)):
		idx = (lnp_samples < np.log(p_threshold))
		P_ret[k] = np.sum(idx)
	
	return P_ret / float(lnp.shape[0])


def binom_confidence(nbins, ntrials, confidence):
	q = 0.5 * (1. - confidence)
	qprime = (1. - q)**(1./nbins)
	
	rv = scipy.stats.binom(ntrials, 1./float(nbins))
	P = rv.cdf(np.arange(ntrials+1))
	
	lower = np.where((1. - P) >= qprime)[0][-1]
	upper = np.where(P < qprime)[0][-1] + 1
	
	return lower, upper


def main():
	parser = argparse.ArgumentParser(
	              prog='mock-comparison.py',
	              description='Compares results from Bayestar for mock data '
	                          'with true stellar parameters.',
	              add_help=True)
	parser.add_argument('input', type=str, help='Bayestar input file with true parameters.')
	parser.add_argument('output', type=str, help='Bayestar output file with surfaces.')
	parser.add_argument('index', type=int, help='HEALPix index of pixel.')
	parser.add_argument('--stack-out', '-so', type=str, default=None,
	                       help='Output filename for stacked pdf plot.')
	parser.add_argument('--pct-out', '-po', type=str, default=None,
	                        help='Output filename for percentile plot.')
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	args = parser.parse_args(sys.argv[offset:])
	
	if (args.stack_out == None) and (args.pct_out == None):
		print "Either '--stack-out' or '--pct-out' (or both) must be specified."
		return 0
	
	# Read in pdfs
	group = 'pixel %d' % (args.index)
	dset = '%s/stellar pdfs' % group
	pdf = hdf5io.TProbSurf(args.output, dset)
	x_min, x_max = pdf.x_min, pdf.x_max
	p = pdf.get_p()[:,:]
	
	# Read in convergence information
	dset = '%s/stellar chains' % group
	chain = hdf5io.TChain(args.output, dset)
	lnp = chain.get_lnp()[:]
	lnZ = chain.get_lnZ()[:]
	conv = chain.get_convergence()[:]
	tmp_samples = chain.get_samples()[:]
	samples = np.empty(tmp_samples.shape, dtype='f8')
	samples[:,:,0] = tmp_samples[:,:,1]
	samples[:,:,1] = tmp_samples[:,:,0]
	samples[:,:,2] = tmp_samples[:,:,2]
	samples[:,:,3] = tmp_samples[:,:,3]
	
	lnp_norm = np.empty(lnp.shape, dtype='f8')
	lnp_norm[:] = lnp[:]
	lnZ.shape = (lnZ.size, 1)
	lnp_norm -= np.repeat(lnZ, lnp.shape[1], axis=1)
	lnZ.shape = (lnZ.size)
	
	lnZ_max = np.max(lnZ[np.isfinite(lnZ)])
	lnZ_idx = (lnZ > lnZ_max - 10.)
	
	mean = np.mean(samples, axis=1)
	
	mean.shape = (mean.shape[0], 1, mean.shape[1])
	Delta = np.repeat(mean, samples.shape[1], axis=1)
	mean.shape = (mean.shape[0], mean.shape[2])
	Delta -= samples
	cov = np.einsum('ijk,ijl->ikl', Delta, Delta) / float(samples.shape[1])
	
	bounds = [x_min[0], x_max[0], x_min[1], x_max[1]]
	
	# Read in true parameter values
	f = h5py.File(args.input, 'r')
	dset = f['/parameters/pixel %d' % (args.index)]
	
	fields = ['DM', 'EBV', 'Mr', 'FeH']
	dtype = [(field, 'f8') for field in fields]
	truth = np.empty(len(dset), dtype=dtype)
	
	for field in fields:
		truth[field][:] = dset[field][:]
	
	f.close()
	
	norm = np.sum(np.sum(p, axis=1), axis=1)
	
	# Set matplotlib style attributes
	mplib.rc('text', usetex=True)
	mplib.rc('xtick.major', size=6)
	mplib.rc('xtick.minor', size=4)
	mplib.rc('ytick.major', size=6)
	mplib.rc('ytick.minor', size=4)
	mplib.rc('xtick', direction='out')
	mplib.rc('ytick', direction='out')
	mplib.rc('axes', grid=False)
	
	# Percentile statistics
	if args.pct_out != None:
		pct_fname = abspath(args.pct_out)
		
		P_indiv = P_star(bounds, p, truth)
		#print P_indiv
		
		#P_indiv = P_star_2(lnp_norm, bounds, p, truth)
		#print P_indiv_2
		
		fig = plt.figure(figsize=(4,3), dpi=200)
		ax = fig.add_subplot(1,1,1)
		
		ax.hist(P_indiv, alpha=0.6)
		
		lower, upper = binom_confidence(10, p.shape[0], 0.95)
		ax.fill_between([0., 1.], [lower, lower], [upper, upper], facecolor='g', alpha=0.2)
		
		lower, upper = binom_confidence(10, p.shape[0], 0.50)
		ax.fill_between([0., 1.], [lower, lower], [upper, upper], facecolor='g', alpha=0.2)
		
		ax.set_xlim(0., 1.)
		ax.set_xlabel(r'$\% \mathrm{ile}$', fontsize=16)
		ax.set_ylabel(r'$\mathrm{\# \ of \ stars}$', fontsize=16)
		fig.subplots_adjust(left=0.18, bottom=0.18)
		
		fig.savefig(pct_fname, dpi=300)
	
	# Shifted and stacked pdfs
	if args.stack_out != None:
		stack_fname = abspath(args.stack_out)
		
		# Simple statistics
		Delta_DM = (truth['DM']-mean[:,0]) / np.sqrt(cov[:,0,0])
		Delta_Ar = (truth['EBV']-mean[:,1]) / np.sqrt(cov[:,1,1])
		Delta_Mr = (truth['Mr']-mean[:,2]) / np.sqrt(cov[:,2,2])
		Delta_FeH = (truth['FeH']-mean[:,3]) / np.sqrt(cov[:,3,3])
		
		w_x = x_max[0] - x_min[0]
		w_y = x_max[1] - x_min[1]
		dx = x_min[0] + 0.5*w_x - truth['DM']
		dy = x_min[1] + 0.5*w_y - truth['EBV']
		bounds_new = [-0.5*w_x, 0.5*w_x, -0.5*w_y, 0.5*w_y]
		stack = stack_shifted(bounds, p, [dx,dy], norm)
		
		DM_range = np.linspace(bounds_new[0], bounds_new[1], stack.shape[0])
		p_DM = np.sum(stack, axis=1)
		p_DM /= np.sum(p_DM)
		
		Ar_range = np.linspace(bounds_new[2], bounds_new[3], stack.shape[1])
		p_Ar = np.sum(stack, axis=0)
		p_Ar /= np.sum(p_Ar)
		
		# Determine geometry of density plot and histograms
		main_left, main_bottom = 0.18, 0.16
		main_width, main_height = 0.63, 0.65
		buffer_right, buffer_top = 0., 0.
		histx_height, histy_width = 0.12, 0.09
		rect_main = [main_left, main_bottom, main_width, main_height]
		rect_histx = [main_left, main_bottom+main_height+buffer_top, main_width, histx_height]
		rect_histy = [main_left+main_width+buffer_right, main_bottom, histy_width, main_height]
		
		# Set up the figure with a density plot and two histograms
		fig = plt.figure(figsize=(4,3), dpi=150)
		ax_density = fig.add_axes(rect_main)
		ax_histx = fig.add_axes(rect_histx)
		ax_histy = fig.add_axes(rect_histy)
		
		xlim = [-2.,2.]
		ylim = [-1.,1.]
		
		ax_density.imshow(stack.T, extent=bounds_new, origin='lower', vmin=0.,
						  aspect='auto', cmap='hot', interpolation='nearest')
		ax_density.plot([0., 0.], [ylim[0]-1.,ylim[1]+1.], 'c:', lw=0.5, alpha=0.35)
		ax_density.plot([xlim[0]-1., xlim[1]+1.], [0., 0.], 'c:', lw=0.5, alpha=0.35)
		ax_density.set_xlim(xlim)
		ax_density.set_ylim(ylim)
		
		ax_histx.fill_between(DM_range, y1=p_DM, alpha=0.4, facecolor='b')
		ax_histx.plot([0., 0.], [0., 1.1*np.max(p_DM)], 'g-', lw=0.5)
		ax_histx.set_ylim(0., 1.1*np.max(p_DM))
		ax_histx.set_xlim(xlim)
		ax_histx.set_xticklabels([])
		ax_histx.set_yticklabels([])
		
		ax_histy.fill_betweenx(Ar_range, x1=p_Ar, alpha=0.4, facecolor='b')
		ax_histy.plot([0., 1.1*np.max(p_Ar)], [0., 0.], 'g-', lw=0.5)
		ax_histy.set_xlim(0., 1.1*np.max(p_Ar))
		ax_histy.set_ylim(ylim)
		ax_histy.set_xticklabels([])
		ax_histy.set_yticklabels([])
		
		fig.savefig(stack_fname, dpi=300)
	
	plt.show()
	
	return 0

if __name__ == '__main__':
	main()

