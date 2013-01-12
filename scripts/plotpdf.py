#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#
#       plotpdf.py
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

import sys, argparse
from os.path import abspath, expanduser

import numpy as np

import matplotlib as mplib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

import hdf5io

def los2ax(ax, fname, group, *args, **kwargs):
	chain = hdf5io.TChain(fname, group)
	mu = np.linspace(5., 20., chain.ndim)
	if 'alpha' not in kwargs:
		kwargs['alpha'] = 1. / np.power(chain.data.shape[0], 0.55)
	
	# Plot all paths
	EBV_all = np.cumsum(np.exp(chain.data['x']), axis=1)
	for EBV in EBV_all:
		ax.plot(mu, EBV, *args, **kwargs)
	
	# Plot mean path
	y = np.mean(EBV_all, axis=0)
	y_err = np.std(EBV_all, axis=0)
	ax.errorbar(mu, y, yerr=y_err, c='g', ecolor=(0., 1., 0., 0.5), alpha=0.3)
	
	# Plot best path
	i = np.argsort(chain.data['ln_p'])[::-1]
	alpha = 1.
	for ii in i[:3]:
		ax.plot(mu, EBV_all[ii], 'r-', alpha=alpha)
		alpha *= 0.5
	#alpha = 1.
	#for ii in i[-10:]:
	#	ax.plot(mu, EBV_all[ii], 'k-', alpha=alpha)
	#	alpha *= 0.85
	
	ax.set_xlim(5., 20.) 

def main():
	# Parse commandline arguments
	parser = argparse.ArgumentParser(prog='plotpdf',
	              description='Plots posterior distributions produced by galstar',
	              add_help=True)
	parser.add_argument('input', type=str, help='Bayestar output file.')
	parser.add_argument('index', type=int, help='Healpix index.')
	parser.add_argument('-o', '--output', type=str, help='Filename for plot.')
	parser.add_argument('-s', '--show', action='store_true', help='Show plot.')
	parser.add_argument('-pdfs', '--show-pdfs', action='store_true',
	                            help = 'Show stellar pdfs in background.')
	#parser.add_argument('--testfn', nargs='+', type=str, default=None,
	#                    help='ASCII file with true stellar parameters (same as used for galstar input).')
	#parser.add_argument('-cnv', '--converged', action='store_true',
	#                    help='Show only converged stars.')
	parser.add_argument('-fig', '--figsize', type=float, nargs=2, default=(7, 4.5),
	                          help='Figure width and height in inches.')
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	args = parser.parse_args(sys.argv[offset:])
	
	if (args.output == None) and not args.show:
		print 'Either --output or --show must be given.'
		return 0
	
	# Load in pdfs
	fname = abspath(expanduser(args.input))
	pdf = hdf5io.get_stellar_pdfs(fname, args.index)
	pdf_stack = np.zeros(pdf[0].shape, pdf[0].dtype)
	for p in pdf:
		pdf_stack += p
	del pdf
	
	# Normalize peak to unity at each distance
	norm = 1. / np.max(pdf_stack, axis=1)
	pdf_stack = np.einsum('ij,i->ij', pdf_stack, norm)
	
	# Determine maximum E(B-V)
	w_y = np.mean(pdf_stack, axis=0)
	y_max = np.max(np.where(w_y > 1.e-2)[0])
	EBV_max = y_max * (5. / pdf_stack.shape[1])
	#pdf_stack = pdf_stack[:,:y_max]
	
	# Set matplotlib style attributes
	mplib.rc('text', usetex=True)
	mplib.rc('xtick.major', size=6)
	mplib.rc('xtick.minor', size=2)
	mplib.rc('ytick.major', size=6)
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
	fig.subplots_adjust(bottom=0.12)
	
	bounds = [5., 20., 0., 5.]
	if args.show_pdfs:
		ax.imshow(pdf_stack.T, extent=bounds, origin='lower',
		             aspect='auto', cmap='hot', interpolation='nearest')
	
	# Plot l.o.s. extinction to figure
	group = 'pixel %d/los extinction/' % (args.index)
	los2ax(ax, fname, group, 'c')
	
	ax.set_ylim(0., EBV_max)
	
	# Save/show plot
	if args.output != None:
		fig.savefig(args.output, dpi=300)
	if args.show:
		plt.show()
	
	return 0

if __name__ == '__main__':
	main()

