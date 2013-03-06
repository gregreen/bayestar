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

import healpy as hp
import h5py

def getClouds(fname):
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
	tmp, nSamples, nClouds = f['pixel %d/clouds' % pixels[0]].shape
	nClouds = (nClouds - 1) / 2
	
	# Extract cloud fit from each pixel
	shape = (nPixels, nSamples, nClouds)
	DeltaMu = np.zeros(shape, dtype='f8')
	DeltaLnEBV = np.zeros(shape, dtype='f8')
	for i,pixIdx in enumerate(pixels):
		try:
			group = f['pixel %d/clouds' % pixIdx]
		except:
			continue
		
		try:
			DeltaMu[i] = group[0, :, 1:nClouds+1]
			DeltaLnEBV[i] = group[0, :, nClouds+1:]
		except:
			DeltaMu[i] = 0
			DeltaLnEBV[i] = np.nan
	
	muAnchor = np.cumsum(DeltaMu, axis=2)
	DeltaEBV = np.exp(DeltaLnEBV)
	
	return pixels, muAnchor, DeltaEBV

def getCloudsFromMultiple(fnames):
	partial = []
	nPixels = 0
	for fname in fnames:
		try:
			partial.append( getClouds(fname) )
			nPixels += len(partial[-1][1])
		except:
			pass
	tmp, nSamples, nClouds = partial[0][1].shape
	
	pixels = np.empty(nPixels, dtype='u4')
	shape = (nPixels, nSamples, nClouds)
	muAnchor = np.empty(shape, dtype='f8')
	DeltaEBV = np.empty(shape, dtype='f8')
	
	startIdx = 0
	for i,part in enumerate(partial):
		endIdx = startIdx + len(part[1])
		pixels[startIdx:endIdx] = part[0][:]
		muAnchor[startIdx:endIdx] = part[1][:]
		DeltaEBV[startIdx:endIdx] = part[2][:]
		startIdx += len(part[1])
	
	return pixels, muAnchor, DeltaEBV

def calcEBV(muAnchor, DeltaEBV, mu):
	foreground = (muAnchor < mu)
	return np.cumsum(foreground * DeltaEBV, axis=2)[:,:,-1]

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

def plotEBV(ax, pixels, muAnchor, DeltaEBV, mu, nside=512, nest=True, **kwargs):
	# Generate rasterized image of E(B-V)
	EBV = calcEBV(muAnchor, DeltaEBV, mu)
	#idx1 = np.arange(EBV.shape[0])
	#idx2 = np.random.randint(EBV.shape[1], size=EBV.shape[1])
	#print EBV.shape
	#print np.median(EBV, axis=1).shape
	#EBV = EBV[idx1,idx2]
	print EBV.shape
	EBV = np.median(EBV, axis=1) #np.percentile(EBV, 95., axis=1) - np.percentile(EBV, 5., axis=1) #np.mean(EBV, axis=1)
	img, bounds = rasterizeMap(pixels, EBV, nside, nest)
	
	# Configure plotting options
	if 'vmin' not in kwargs:
		kwargs['vmin'] = np.min(img[np.isfinite(img)])
	if 'vmax' not in kwargs:
		kwargs['vmax'] = np.max(img[np.isfinite(img)])
	if 'aspect' not in kwargs:
		kwargs['aspect'] = 'auto'
	if 'origin' in kwargs:
		print "Ignoring option 'origin'."
	if 'extent' in kwargs:
		print "Ignoring option 'extent'."
	kwargs['origin'] = 'lower'
	kwargs['extent'] = bounds
	
	# Plot
	ax.imshow(img.T, **kwargs)


def main():
	fnames = ['/n/wise/ggreen/bayestar/output/CMa.%.5d.h5' % i for i in xrange(18)]
	pixels, muAnchor, DeltaEBV = getCloudsFromMultiple(fnames)
	
	'''
	nMu = 100
	xRange = np.linspace(5., 15., nMu)
	yRange = np.empty(nMu, dtype='f8')
	yPctile = np.empty((4, nMu), dtype='f8')
	pctile = [5., 25., 75., 95.]
	for i,x in enumerate(xRange):
		tmpEBV = calcEBV(muAnchor, DeltaEBV, x)[1]
		yRange[i] = np.mean(tmpEBV)
		yPctile[:,i] = np.percentile(tmpEBV, pctile)
	
	fig = plt.figure(figsize=(7,5), dpi=150)
	ax = fig.add_subplot(1,1,1)
	ax.fill_between(xRange, yPctile[0], yPctile[-1], facecolor='b', alpha=0.5)
	ax.fill_between(xRange, yPctile[1], yPctile[-2], facecolor='b', alpha=0.5)
	#ax.plot(xRange, yRange, 'b')
	plt.show()
	'''
	
	mplib.rc('text', usetex=True)
	mplib.rc('xtick.major', size=6)
	mplib.rc('xtick.minor', size=2)
	mplib.rc('ytick.major', size=6)
	mplib.rc('ytick.minor', size=2)
	mplib.rc('xtick', direction='out')
	mplib.rc('ytick', direction='out')
	mplib.rc('axes', grid=False)
	
	fig = plt.figure(figsize=(7., 7.), dpi=150)
	grid = ImageGrid(fig, 111, nrows_ncols=(3, 3), axes_pad=0.05)
	
	mu = np.linspace(5., 11., 9)
	print mu
	for i in xrange(9):
		ax = grid[i]
		plotEBV(ax, pixels, muAnchor, DeltaEBV, mu[i], vmin=0., vmax=1.)
	
	#ax.set_xlabel(r'$\ell$', fontsize=16)
	#ax.set_ylabel(r'$b$', fontsize=16)
	
	#ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
	#ax.xaxis.set_minor_locator(AutoMinorLocator())
	#ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
	#ax.yaxis.set_minor_locator(AutoMinorLocator())
	
	plt.show()
	
	return 0

if __name__ == '__main__':
	main()

