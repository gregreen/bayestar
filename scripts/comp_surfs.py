#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#
#       comp_surfs.py
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

import hdf5io

from plotpdf import los2ax, clouds2ax


def main():
	# Parse commandline arguments
	parser = argparse.ArgumentParser(prog='comp_surfs',
	              description='Compare two different sets of stellar probability density surfaces.',
	              add_help=True)
	parser.add_argument('input1', type=str, help='1st Bayestar output file.')
	parser.add_argument('loc1', type=int, nargs=2, help='HEALPix nside and index.')
	parser.add_argument('input2', type=str, help='2nd Bayestar output file.')
	parser.add_argument('loc2', type=int, nargs=2, help='HEALPix nside and index.')
	parser.add_argument('-o', '--output', type=str, help='Filename for plot.')
	parser.add_argument('-s', '--show', action='store_true', help='Show plot.')
	parser.add_argument('-los', '--show-los', action='store_true',
	                            help='Show piecewise-linear l.o.s. reddening.')
	parser.add_argument('-cl', '--show-clouds', action='store_true',
	                            help='Show cloud model of reddening.')
	parser.add_argument('-ind', '--show-individual', action='store_true',
	                            help='Show individual stellar pdfs above main plot.')
	parser.add_argument('-ovplt', '--overplot-clouds', type=float, nargs='+', default=None,
	                            help='Overplot clouds at specified distance/depth pairs.')
	parser.add_argument('-fig', '--figsize', type=float, nargs=2, default=(7, 5.),
	                          help='Figure width and height in inches.')
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	args = parser.parse_args(sys.argv[offset:])
	
	if (args.output == None) and not args.show:
		print 'Either --output or --show must be given.'
		return 0
	
	input_params = [(args.input1, args.loc1),
	                (args.input2, args.loc2)]
	
	# Load in pdfs
	x_min, x_max = [4., 0.], [19., 5.]
	pdf_stack = []
	EBV_max = []
	
	for fname, loc in input_params:
		print 'Loading surfaces from %s (pixel %d - %d) ...' % (fname, loc[0], loc[1])
		
		fname = abspath(expanduser(fname))
		group = 'pixel %d-%d' % (loc[0], loc[1])
		
		# Load in pdfs
		pdf_indiv = None
		
		dset = '%s/stellar chains' % group
		chain = hdf5io.TChain(fname, dset)
		
		conv = chain.get_convergence()[:]
		
		lnZ = chain.get_lnZ()[:]
		lnZ_max = np.percentile(lnZ[np.isfinite(lnZ)], 98.)
		lnZ_idx = (lnZ > lnZ_max - 5.)
		
		print 'ln(Z_98) = %.2f' % (lnZ_max)
		
		for Delta_lnZ in [2.5, 5., 10., 15., 100.]:
			tmp = np.sum(lnZ < lnZ_max - Delta_lnZ) / float(lnZ.size)
			print '%.2f %% fail D = %.1f cut' % (100.*tmp, Delta_lnZ)
		
		stack_tmp = None
		
		try:
			dset = '%s/stellar pdfs' % group
			pdf = hdf5io.TProbSurf(fname, dset)
			x_min, x_max = pdf.x_min, pdf.x_max
			stack_tmp = np.sum(pdf.get_p()[lnZ_idx], axis=0)
			
		except:
			print 'Using chains to create image of stacked pdfs...'
			
			star_samples = chain.get_samples()[:, :, 0:2]
			
			idx = conv & lnZ_idx
			
			star_samples = star_samples[idx]
			
			n_stars_tmp, n_star_samples, n_star_dim = star_samples.shape
			star_samples.shape = (n_stars_tmp * n_star_samples, n_star_dim)
			
			res = (501, 121)
			
			E_range = np.linspace(x_min[1], x_max[1], res[0]*2+1)
			DM_range = np.linspace(x_min[0], x_max[0], res[1]*2+1)
			
			stack_tmp, tmp1, tmp2 = np.histogram2d(star_samples[:,0],
			                                       star_samples[:,1],
			                                       bins=[E_range, DM_range])
			
			stack_tmp = gaussian_filter(pdf_stack.astype('f8'),
			                            sigma=(4, 2), mode='reflect')
			stack_tmp = pdf_stack.reshape([res[0], 2, res[1], 2]).mean(3).mean(1)
			stack_tmp = pdf_stack.T
		
		tmp = stack_tmp / np.max(stack_tmp)
		
		norm = 1. / np.power(np.max(tmp, axis=1), 0.8)
		norm[np.isinf(norm)] = 0.
		tmp = np.einsum('ij,i->ij', tmp, norm)
		
		w_y = np.mean(tmp, axis=0)
		y_max = np.max(np.where(w_y > 1.e-2)[0])
		EBV_max.append(y_max * (5. / tmp.shape[1]))
		
		stack_tmp /= np.sum(stack_tmp)
		pdf_stack.append(stack_tmp)
	
	pdf_stack = pdf_stack[1] - pdf_stack[0]
	
	# Normalize peak to unity at each distance
	pdf_stack /= np.max(np.abs(pdf_stack))
	
	#norm = 1. / np.power(np.max(np.abs(pdf_stack), axis=1), 0.8)
	#norm[np.isinf(norm)] = 0.
	#pdf_stack = np.einsum('ij,i->ij', pdf_stack, norm)
	
	# Determine maximum E(B-V)
	EBV_max = np.max(EBV_max)
	
	
	# Set matplotlib style attributes
	mplib.rc('text', usetex=True)
	mplib.rc('xtick.major', size=5)
	mplib.rc('xtick.minor', size=2)
	mplib.rc('ytick.major', size=5)
	mplib.rc('ytick.minor', size=2)
	mplib.rc('xtick', direction='out')
	mplib.rc('ytick', direction='out')
	mplib.rc('axes', grid=False)
	
	# Set up figure
	fig = plt.figure(figsize=args.figsize, dpi=150)
	
	ax = fig.add_subplot(1,1,1)
	
	ax.set_xlabel(r'$\mu$', fontsize=16)
	ax.set_ylabel(r'$\mathrm{E} \left( B - V \right)$', fontsize=16)
	ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
	ax.xaxis.set_minor_locator(AutoMinorLocator())
	ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
	ax.yaxis.set_minor_locator(AutoMinorLocator())
	fig.subplots_adjust(bottom=0.12, left=0.12, right=0.98)
	
	bounds = [x_min[0], x_max[0], x_min[1], x_max[1]]
	vmax = np.max(np.abs(pdf_stack))
	
	ax.imshow(pdf_stack.T, extent=bounds, origin='lower',
	             aspect='auto', cmap='bwr', interpolation='nearest',
	             vmin=-vmax, vmax=vmax)
	
	ax.set_ylim(x_min[1], EBV_max)
	
	# Save/show plot
	if args.output != None:
		fig.savefig(args.output, dpi=300)
	if args.show:
		plt.show()
	
	return 0

if __name__ == '__main__':
	main()

