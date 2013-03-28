#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  plotmap.py
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

import matplotlib as mplib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from mpl_toolkits.axes_grid1 import ImageGrid

import argparse, sys

import healpy as hp
import h5py


def getLOS(fname):
	f = h5py.File(fname, 'r')
	
	# Get a list of pixels in the file
	pixels = []
	for name,item in f.iteritems():
		try:
			pixels.append(int( name.split()[1] ))
		except:
			pass
	nPixels = len(pixels)
	
	# Get number of clouds and samples
	nCloudSamples, nClouds = 1, 1
	try:
		tmp, nCloudSamples, nClouds = f['pixel %d/clouds' % pixels[0]].shape
		nClouds = (nClouds - 1) / 2
	except:
		pass
	shape = (nPixels, nCloudSamples, nClouds)
	CloudDeltaMu = np.zeros(shape, dtype='f8')
	CloudDeltaLnEBV = np.zeros(shape, dtype='f8')
	
	# Get number of regions and samples for piecewise model
	nPiecewiseSamples, nSlices = 1, 1
	try:
		tmp, nPiecewiseSamples, nSlices = f['pixel %d/los' % pixels[0]].shape
		nSlices -= 1
	except:
		pass
	shape = (nPixels, nPiecewiseSamples, nSlices)
	PiecewiseDeltaLnEBV = np.zeros(shape, dtype='f8')
	
	# Extract l.o.s. fits from each pixel
	for i,pixIdx in enumerate(pixels):
		# Cloud model
		try:
			group = f['pixel %d/clouds' % pixIdx]
		except:
			pass
		
		try:
			CloudDeltaMu[i] = group[0, :, 1:nClouds+1]
			CloudDeltaLnEBV[i] = group[0, :, nClouds+1:]
		except:
			CloudDeltaMu[i] = 0
			CloudDeltaLnEBV[i] = np.nan
		
		# Piecewise model
		try:
			group = f['pixel %d/los' % pixIdx]
		except:
			pass
		
		try:
			PiecewiseDeltaLnEBV[i] = group[0,:,1:]
		except:
			PiecewiseDeltaLnEBV[i] = np.nan
	
	CloudMuAnchor = np.cumsum(CloudDeltaMu, axis=2)
	CloudDeltaEBV = np.exp(CloudDeltaLnEBV)
	PiecewiseDeltaEBV = np.exp(PiecewiseDeltaLnEBV)
	
	return pixels, CloudMuAnchor, CloudDeltaEBV, PiecewiseDeltaEBV

def getLOSFromMultiple(fnames):
	partial = []
	nPixels = 0
	for fname in fnames:
		try:
			partial.append(getLOS(fname))
			nPixels += len(partial[-1][0])
		except:
			pass
	pixels = np.empty(nPixels, dtype='u4')
	
	tmp, nCloudSamples, nClouds = partial[0][1].shape
	shape = (nPixels, nCloudSamples, nClouds)
	CloudMuAnchor = np.empty(shape, dtype='f8')
	CloudDeltaEBV = np.empty(shape, dtype='f8')
	
	tmp, nPiecewiseSamples, nSlices = partial[0][3].shape
	shape = (nPixels, nPiecewiseSamples, nSlices)
	PiecewiseDeltaEBV = np.empty(shape, dtype='f8')
	
	startIdx = 0
	for i,part in enumerate(partial):
		endIdx = startIdx + len(part[1])
		pixels[startIdx:endIdx] = part[0][:]
		CloudMuAnchor[startIdx:endIdx] = part[1][:]
		CloudDeltaEBV[startIdx:endIdx] = part[2][:]
		PiecewiseDeltaEBV[startIdx:endIdx] = part[3][:]
		startIdx += len(part[1])
	
	return pixels, CloudMuAnchor, CloudDeltaEBV, PiecewiseDeltaEBV

def calcCloudEBV(muAnchor, DeltaEBV, mu):
	foreground = (muAnchor < mu)
	return np.cumsum(foreground * DeltaEBV, axis=2)[:,:,-1]

def calcPiecewiseEBV(muAnchor, DeltaEBV, mu):
	nPixels, nSamples, nSlices = DeltaEBV.shape
	
	idx = np.where(muAnchor >= mu, np.arange(nSlices), nSlices+1)
	lowIdx = np.min(idx)
	
	EBVslice = np.cumsum(DeltaEBV, axis=2)
	
	if lowIdx == nSlices - 1:
		return EBVslice[:,:,-1]
	
	lowMu = muAnchor[lowIdx]
	highMu = muAnchor[lowIdx+1]
	
	a = (mu - lowMu) / (highMu - lowMu)
	EBVinterp = (1. - a) * EBVslice[:,:,lowIdx]
	EBVinterp += a * EBVslice[:,:,lowIdx+1]
	
	return EBVinterp

def calcPiecewiseEBV2(muAnchor, DeltaEBV, mu):
	'''
	Evaluates E(B-V) at a set of distances for the piecewise-linear model.
	
	Inputs:
	    muAnchor  (nSlices)                     Distances of slices returned by bayestar.
	    DeltaEBV  (nPixels, nSamples, nSlices)  DeltaEBV between slices.
	    mu        (nMu)                         Distances at which to evaluate E(B-V).
	
	Outputs:
	'''
	
	nPixels, nSamples, nSlices = DeltaEBV.shape
	nMu = mu.shape
	
	print type(mu)
	print mu.dtype
	print mu.shape
	print nMu
	
	muTmp = mu.copy()
	muAnchorTmp = muAnchor.copy()
	muTmp.shape = (nMu, 1)
	muAnchorTmp.shape = (1, nSlices)
	muTmp = np.repeat(muTmp, nSlices, axis=1)
	muAnchorTmp = np.repeat(muAnchorTmp, nMu, axis=0)
	
	muDiff = muAnchorTmp - muTmp
	n = np.arange(nSlices)
	n.shape = (1, nSlices)
	n = np.repeat(n, nMu, axis=0)
	
	idx = np.where(muDiff >= 0., n, nSlices+1)
	lowIdx = np.min(idx, axis=1)
	
	lowMu = muAnchor[lowIdx]
	highMu = muAnchor[lowIdx+1]	# TODO: deal with boundary cases
	
	EBVinterp = np.zeros(DeltaEBV.shape, dtype='f8')
	EBVslice = np.cumsum(DeltaEBV, axis=2)
	
	a = (mu - lowMu) / (highMu - lowMu)
	for i in xrange(nMu):
		EBVinterp[:,:,i] += (1. - a[i]) * EBVslice[:,:,lowIdx[i]]
		EBVinterp[:,:,i] += a[i] * EBVslice[:,:,highIdx[i]]
	
	return EBVinterp
	

def rasterizeMap(pixels, EBV, nside=512, nest=True, oversample=4):
	# Determine pixel centers and bounds
	pixels = np.array(pixels)
	theta, phi = hp.pix2ang(nside, pixels, nest=nest)
	lCenter, bCenter = 180./np.pi * phi, 90. - 180./np.pi * theta
	
	pixLength = np.sqrt( hp.nside2pixarea(nside, degrees=True) )
	lMin, lMax = np.min(lCenter)-pixLength/2., np.max(lCenter)+pixLength/2.
	bMin, bMax = np.min(bCenter)-pixLength/2., np.max(bCenter)+pixLength/2.
	
	# Set resolution of image
	xSize = int( oversample * (lMax - lMin) / pixLength )
	ySize = int( oversample * (bMax - bMin) / pixLength )
	
	# Make grid of pixels to plot
	l, b = np.mgrid[0:xSize, 0:ySize].astype(np.float32) + 0.5
	l = lMax - (lMax - lMin) * l / float(xSize)
	b = bMin + (bMax - bMin) * b / float(ySize)
	theta, phi = np.pi/180. * (90. - b), np.pi/180. * l
	del l, b
	
	pixIdx = hp.ang2pix(nside, theta, phi, nest=nest)
	idxMap = np.empty(12*nside*nside, dtype='i8')
	idxMap[:] = -1
	idxMap[pixels] = np.arange(len(pixels))
	idxEBV = idxMap[pixIdx]
	mask = (idxEBV == -1)
	del idxMap
	
	# Grab pixels from map
	img = None
	if len(EBV.shape) == 1:
		img = EBV[idxEBV]
		img[mask] = np.nan
		img.shape = (xSize, ySize)
	elif len(EBV.shape) == 2:
		img = EBV[:,idxEBV]
		img[mask] = np.nan
		img.shape = (img.shape[0], xSize, ySize)
	else:
		raise Exception('EBV must be either 1- or 2-dimensional.')
	
	bounds = (lMin, lMax, bMin, bMax)
	return img, bounds

def calcEBV(muAnchor, DeltaEBV, mu, model='piecewise', maxSpread=None, calcSpread=False):
	EBV = None
	if model == 'piecewise':
		EBV = calcPiecewiseEBV(muAnchor, DeltaEBV, mu)
	elif model == 'clouds':
		EBV = calcCloudEBV(muAnchor, DeltaEBV, mu)
	else:
		raise ValueError("Unrecognized extinction model: '%s'" % model)
	
	EBVcenter = None
	if calcSpread:
		EBVcenter = np.percentile(EBV, 95., axis=1) - np.percentile(EBV, 5., axis=1)
	else:
		EBVcenter = np.median(EBV, axis=1)
		if maxSpread != None:
			EBVspread = np.percentile(EBV, 95., axis=1) - np.percentile(EBV, 5., axis=1)
			idx = EBVspread > maxSpread
			EBVcenter[idx] = np.nan
	
	return EBVcenter

def plotEBV(ax, pixels, muAnchor, DeltaEBV, mu,
                nside=512, nest=True, model='piecewise',
                maxSpread=None, plotSpread=False, **kwargs):
	# Generate rasterized image of E(B-V)
	calcEBV(muAnchor, DeltaEBV, model, maxSpread, plotSpread)
	img, bounds = rasterizeMap(pixels, EBVcenter, nside, nest)
	
	# Configure plotting options
	if 'vmin' not in kwargs:
		kwargs['vmin'] = np.min(img[np.isfinite(img)])
	if 'vmax' not in kwargs:
		kwargs['vmax'] = np.max(img[np.isfinite(img)])
	if 'aspect' not in kwargs:
		kwargs['aspect'] = 'auto'
	if 'interpolation' not in kwargs:
		kwargs['interpolation'] = 'nearest'
	if 'origin' in kwargs:
		print "Ignoring option 'origin'."
	if 'extent' in kwargs:
		print "Ignoring option 'extent'."
	kwargs['origin'] = 'lower'
	kwargs['extent'] = [bounds[1], bounds[0], bounds[2], bounds[3]]
	kwargs['cmap'] = 'binary'
	
	# Plot
	ax.imshow(img.T, **kwargs)
	
	kwargs['vmin'] = 0.
	kwargs['vmax'] = 1.
	mask = np.isnan(img.T)
	shape = (img.shape[1], img.shape[0], 4)
	maskImg = np.zeros(shape, dtype='f8')
	maskImg[:,:,1] = 0.4
	maskImg[:,:,2] = 1.
	maskImg[:,:,3] = 0.65 * mask.astype('f8')
	ax.imshow(maskImg, **kwargs)


def main():
	parser = argparse.ArgumentParser(prog='plotmap.py',
	                                 description='Generate a map of E(B-V) from bayestar output.',
	                                 add_help=True)
	parser.add_argument('input', type=str, nargs='+', help='Bayestar output files.')
	parser.add_argument('--output', '-o', type=str, help='Output filename for plot.')
	parser.add_argument('--show', '-sh', action='store_true', help='Show plot.')
	parser.add_argument('--dists', '-d', type=float, nargs=3,
	                                     default=(5., 20., 21),
	                                     help='DM min, DM max, # of distance slices.')
	parser.add_argument('--nside', '-n', type=int, default=512,
	                                     help='HealPIX nside parameter.')
	parser.add_argument('--model', '-m', type=str, default='piecewise',
	                                     choices=('piecewise', 'clouds'),
	                                     help="Extinction model: 'piecewise' or 'clouds'")
	parser.add_argument('--mask', '-msk', type=float, default=None,
	                                      help='Hide parts of map where 95\% - 5\% of E(B-V) is greater than given value')
	parser.add_argument('--spread', '-sp', action='store_true',
	                                       help='Plot 95\% - 5\% of E(B-V)')
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	args = parser.parse_args(sys.argv[offset:])
	
	fnames = args.input
	pixels, CloudMuAnchor, CloudDeltaEBV, PiecewiseDeltaEBV = getLOSFromMultiple(fnames)
	
	tmp1, tmp2, nSlices = PiecewiseDeltaEBV.shape
	PiecewiseMuAnchor = np.linspace(5., 20., nSlices)
	
	mplib.rc('text', usetex=True)
	mplib.rc('xtick.major', size=6)
	mplib.rc('xtick.minor', size=2)
	mplib.rc('ytick.major', size=6)
	mplib.rc('ytick.minor', size=2)
	mplib.rc('xtick', direction='out')
	mplib.rc('ytick', direction='out')
	mplib.rc('axes', grid=False)
	
	muMin, muMax = args.dists[:2]
	muN = int(args.dists[2])
	mu = np.linspace(muMin, muMax, muN)
	
	pixels, muAnchor, DeltaEBV, mu,
            nside=512, nest=True, model='piecewise',
            maxSpread=None, calcSpread=False
            
	# Get upper limit on E(B-V)
	EBVs = None
	if args.model == 'piecewise':
		EBVs = calcEBV(PiecewiseMuAnchor, PiecewiseDeltaEBV, mu[-1],
		               args.model, args.mask, args.spread)
	elif args.model == 'clouds':
		EBVs = calcEBV(CloudMuAnchor, CloudDeltaEBV, mu[-1],
		               args.model, args.mask, args.spread)
	EBVmax = np.percentile(EBVs, 98.)
	del EBVs
	
	# Determine output filename
	fname = args.output
	if fname != None:
		if fname.endswith('.png'):
			fname = fname[:-4]
	
	# Plot at each distance
	for i in xrange(muN):
		print 'mu = %.2f (%d of %d)' % (mu[i], i+1, muN)
		
		fig = plt.figure(dpi=150)
		ax = fig.add_subplot(1,1,1)
		
		if args.model == 'piecewise':
			plotEBV(ax, pixels, PiecewiseMuAnchor, PiecewiseDeltaEBV, mu[i],
			        nside=args.nside, nest=True, model=args.model,
			        maxSpread=args.mask, plotSpread=args.spread,
			        vmin=0., vmax=EBVmax)
		elif args.model == 'clouds':
			plotEBV(ax, pixels, CloudMuAnchor, CloudDeltaEBV, mu[i],
			        nside=args.nside, nest=True, model=args.model,
			        maxSpread=args.mask, plotSpread=args.spread,
			        vmin=0., vmax=EBVmax)
		
		ax.set_xlabel(r'$\ell$', fontsize=16)
		ax.set_ylabel(r'$b$', fontsize=16)
		
		ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
		ax.xaxis.set_minor_locator(AutoMinorLocator())
		ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
		ax.yaxis.set_minor_locator(AutoMinorLocator())
		
		d = 10.**(mu[i]/5. - 2.)
		ax.set_title(r'$\mu = %.2f \ \ \ d = %.2f \, \mathrm{kpc}$' % (mu[i], d), fontsize=16)
		
		if fname != None:
			full_fname = '%s.%s.%.2d.png' % (fname, args.model, i)
			fig.savefig(full_fname, dpi=150)
	
	if args.show:
		plt.show()
	
	return 0

if __name__ == '__main__':
	main()

