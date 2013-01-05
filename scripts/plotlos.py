#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  plotlos.py
#  
#  Copyright 2013 Greg Green <greg@greg-UX31A>
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

import sys, argparse
from os.path import abspath, expanduser

import matplotlib.pyplot as plt
import matplotlib as mplib
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

import hdf5io

def los2ax(ax, fname, group, *args, **kwargs):
	chain = hdf5io.TChain(fname, group)
	mu = np.linspace(5., 20., chain.ndim)
	if 'alpha' not in kwargs:
		kwargs['alpha'] = 1. / np.power(chain.data.shape[0], 0.65)
	# Plot all paths
	for EBV in chain.data['x']:
		ax.plot(mu, EBV, *args, **kwargs)
	# Plot mean path
	y = np.mean(chain.data['x'], axis=0)
	y_err = np.std(chain.data['x'], axis=0)
	ax.errorbar(mu, y, yerr=y_err, c='r', ecolor=(1., 0., 0., 0.3), alpha=0.3)
	x = [5., 8., 8., 12., 12., 20.]
	y = [0., 0., 0.5, 0.5, 0.7, 0.7]
	ax.plot(x, y, 'b-', alpha=0.2)
	ax.set_xlim(5., 20.)
	ax.set_ylim(0., np.mean(chain.data['x'][:,-4]))


def main():
	# Parse commandline arguments
	parser = argparse.ArgumentParser(prog='plotlos.py',
	                      description='Plot l.o.s. extinction profile.',
	                      add_help=True)
	parser.add_argument('fname', type=str, help='Bayestar output file.')
	parser.add_argument('index', type=int, help='Healpix index.')
	parser.add_argument('--output', '-o', type=str, help='Output filename')
	parser.add_argument('--show', '-s', action='store_true', help='Show plot.')
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	args = parser.parse_args(sys.argv[offset:])
	
	if (args.output == None) and not args.show:
		print 'Either --output or --show must be given.'
		return 0
	
	# Set matplotlib defaults
	mplib.rc('text', usetex=True)
	mplib.rc('xtick.major', size=6)
	mplib.rc('xtick.minor', size=4)
	mplib.rc('axes', grid=True)
	
	# Set up figure
	fig = plt.figure(figsize=(7,5), dpi=150)
	ax = fig.add_subplot(1,1,1)
	ax.set_xlabel(r'$\mu$', fontsize=16)
	ax.set_ylabel(r'$\mathrm{E} \left( B - V \right)$', fontsize=16)
	ax.grid(which='major', alpha=0.3)
	ax.grid(which='minor', alpha=0.1)
	ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
	ax.xaxis.set_minor_locator(AutoMinorLocator())
	ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
	ax.yaxis.set_minor_locator(AutoMinorLocator())
	
	# Plot l.o.s. extinction to figure
	group = 'pixel %d/los extinction/' % (args.index)
	fname = abspath(expanduser(args.fname))
	los2ax(ax, fname, group, 'k')
	
	# Save/show plot
	if args.output != None:
		fig.savefig(args.output, dpi=300)
	if args.show:
		plt.show()
	
	return 0

if __name__ == '__main__':
	main()

