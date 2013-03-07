#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  distance.py
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
import h5py
import healpy

import sys, argparse
from os.path import abspath, expanduser

import matplotlib.pyplot as plt
import matplotlib as mplib
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

class TProbDist:
	def __init__(self, fname, healpixIdx, muEBV, sigmaEBV):
		f = h5py.File(fname, 'r')
		dset = '/pixel %d/los' % healpixIdx
		self.DeltaEBV = np.exp(f[dset][0,:,1:])	# (nChains, nDists)
		minDM, maxDM, nDM = 5., 20., self.DeltaEBV.shape[1]
		self.DM = np.linspace(minDM, maxDM, nDM)
		self.dDM = self.DM[1] - self.DM[0]
		self.EBV = np.cumsum(self.DeltaEBV, axis=1)
		
		self.muEBV = muEBV
		self.sigmaEBV = sigmaEBV
	
	def __call__(self, DM, chainIdx=0):
		Delta = (self.muEBV - self.getEBV(DM, chainIdx)) / self.sigmaEBV
		lnp = self.lnPriorDM(DM) - 0.5 * Delta * Delta
		
		# Out-of-bounds DM --> ln(p) = -inf
		idx = np.isnan(Delta)
		lnp[idx] = -np.inf
		
		return lnp
	
	def getEBV(self, DM, chainIdx=0):
		# Handle out-of-bounds DM
		idxOut = (DM < self.DM[0]) | (DM >= self.DM[-1])
		DMcopy = DM.copy()
		DMcopy[idxOut] = DM[0]
		
		# Get E(B-V) for each DM
		binIdx = np.digitize(DMcopy, self.DM) - 1
		a = (DMcopy - self.DM[binIdx]) / self.dDM
		EBVleft = self.EBV[chainIdx, binIdx]
		EBVright = self.EBV[chainIdx, binIdx+1]
		EBV = a * EBVright + (1. - a) * EBVleft
		
		# Out-of-bounds --> NaN
		EBV[idxOut] = np.nan
		
		return EBV
	
	def lnPriorDM(self, DM):
		return 1.
		#Delta = (DM - 12.) / 10.
		#return -0.5 * Delta * Delta
	
	def get_nChains(self):
		return self.DeltaEBV.shape[0]


def idxGtrThan(x, x0):
	idx = np.arange(len(x))
	tmp = np.where(x > x0, idx, len(x))
	return int(np.min(tmp))


def main():
	# Parse commandline arguments
	parser = argparse.ArgumentParser(prog='distance.py',
	                      description='Produce distance estimate for object with reddening estimate.',
	                      add_help=True)
	parser.add_argument('fname', type=str, help='Bayestar output file')
	parser.add_argument('index', type=int, help='Healpix index')
	parser.add_argument('muEBV', type=float, help='Mean E(B-V)')
	parser.add_argument('sigmaEBV', type=float, help='Std. dev. in E(B-V)')
	parser.add_argument('--output', '-o', type=str, help='Output filename')
	parser.add_argument('--show', '-s', action='store_true', help='Show plot')
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	args = parser.parse_args(sys.argv[offset:])
	
	# Calculate p(DM)
	lnp = TProbDist(args.fname, args.index, args.muEBV, args.sigmaEBV)
	DM = np.linspace(5., 20., 500)
	p = np.zeros(len(DM), dtype='f8')
	for i in xrange(lnp.get_nChains()):
		p_tmp = np.exp(lnp(DM, chainIdx=i))
		p += p_tmp / np.sum(p_tmp)
	p *= (DM[-1] - DM[0]) / np.sum(p)
	
	# Calculate median, 1-sigma-equivalent percentiles
	P = np.cumsum(p) / (DM[-1] - DM[0])
	DeltaDM = DM[1] - DM[0]
	lowIdx = idxGtrThan(P, 0.1587)
	medIdx = idxGtrThan(P, 0.5)
	highIdx = idxGtrThan(P, 0.8413)
	lowDM = lowIdx * DeltaDM + DM[0]
	medDM = medIdx * DeltaDM + DM[0]
	highDM = highIdx * DeltaDM + DM[0]
	print 'DM = %.2f + %.2f - %.2f' % (lowDM, medDM-lowDM, highDM-medDM)
	
	# Set matplotlib style attributes
	mplib.rc('text', usetex=True)
	mplib.rc('xtick.major', size=6)
	mplib.rc('xtick.minor', size=4)
	mplib.rc('ytick.major', size=6)
	mplib.rc('ytick.minor', size=4)
	mplib.rc('xtick', direction='out')
	mplib.rc('ytick', direction='out')
	mplib.rc('axes', grid=False)
	
	fig = plt.figure(figsize=(7,5), dpi=150)
	ax = fig.add_subplot(1,1,1)
	
	# Plot E(B-V) vs. DM curves
	nChains = lnp.EBV.shape[0]
	for i in xrange(nChains):
		ax.plot(lnp.DM, lnp.EBV[i,:], 'k-', alpha=0.01)
	
	# Plot E(B-V) data
	for i in xrange(1,3):
		lowEBV = args.muEBV - i*args.sigmaEBV
		highEBV = args.muEBV + i*args.sigmaEBV
		ax.fill_between(DM, lowEBV, highEBV, facecolor='red', alpha=0.3)
	
	# Plot p(DM)
	p *= np.max(lnp.EBV) / np.max(p)
	ax.plot(DM, p)
	ax.fill_between(DM[:lowIdx+1], 0, p[:lowIdx+1], facecolor='blue', alpha=0.25)
	ax.fill_between(DM[lowIdx:highIdx+1], 0, p[lowIdx:highIdx+1], facecolor='blue', alpha=0.5)
	ax.fill_between(DM[highIdx:], 0, p[highIdx:], facecolor='blue', alpha=0.25)
	
	# Format p(DM) axis
	ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
	ax.xaxis.set_minor_locator(AutoMinorLocator())
	ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
	ax.yaxis.set_minor_locator(AutoMinorLocator())
	ax.set_xlim(DM[0], DM[-1])
	ylim = ax.get_ylim()
	ax.set_ylim(0, ylim[1])
	ax.set_xlabel(r'$\mu$', fontsize=16)
	ax.set_ylabel(r'$p \left( \mu \right)$', fontsize=16)
	
	# Format figure
	fig.subplots_adjust(left=0.15, bottom=0.15)
	
	# Show / save plot
	if args.output != None:
		fig.savefig(args.output, dpi=300)
	if args.show:
		plt.show()
	
	return 0

if __name__ == '__main__':
	main()

