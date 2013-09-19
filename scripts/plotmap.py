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

# TO DO: Get nside from pixel directory

import numpy as np

import matplotlib as mplib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from mpl_toolkits.axes_grid1 import ImageGrid

import argparse, sys

import healpy as hp
import h5py


class los_collection:
	def __init__(self, fnames):
		# Pixel locations
		self.pix_idx = []
		self.nside = []
		
		# Cloud fit data
		self.cloud_delta_mu = []
		self.cloud_delta_lnEBV = []
		self.cloud_mask = []
		
		self.n_clouds = None
		self.n_cloud_samples = None
		
		# Piecewise-linear fit data
		self.los_delta_lnEBV = []
		self.los_mask = []
		
		self.n_slices = None
		self.n_los_samples = None
		self.DM_min = None
		self.DM_max = None
		
		# Load files
		self.load_files(fnames)
	
	def load_file_indiv(self, fname):
		print 'Loading %s ...' % fname
		
		f = h5py.File(fname, 'r')
		
		# Load each pixel
		
		for name,item in f.iteritems():
			# Load pixel position
			try:
				pix_idx_tmp = item.attrs['healpix_index']
				nside_tmp = item.attrs['nside']
			except:
				continue
			
			self.pix_idx.append(pix_idx_tmp)
			self.nside.append(nside)
			
			# Load cloud fit
			try:
				cloud_samples_tmp = item['clouds'][:, 1:, 1:]
				tmp, n_cloud_samples, n_clouds = cloud_samples_tmp.shape
				
				self.cloud_delta_mu.append(cloud_samples_tmp[:n_clouds])
				self.cloud_delta_lnEBV.append(cloud_samples_tmp[n_clouds:])
				
				if self.n_cloud_samples != None:
					if n_cloud_samples != self.n_cloud_samples:
						raise ValueError('# of cloud fit samples in "%s" different from other pixels') % name
					if n_clouds != self.n_clouds:
						raise ValueError('# of cloud fit clouds in "%s" different from other pixels') % name
				else:
					self.n_cloud_samples = n_cloud_samples
					self.n_clouds = n_clouds
				
				self.cloud_mask.append(True)
				
			except:
				self.cloud_mask.append(False)
			
			# Load piecewise-linear fit
			try:
				los_samples_tmp = item['los'][:, 1:, 1:]
				tmp, n_los_samples, n_slices = los_samples_tmp.shape
				
				DM_min = item['los'].attrs['DM_min']
				DM_max = item['los'].attrs['DM_max']
				
				self.los_lnEBV.append(los_samples_tmp)
				
				if self.n_los_samples != None:
					if n_los_samples != self.n_los_samples:
						raise ValueError('# of l.o.s. fit samples in "%s" different from other pixels') % name
					if n_slices != self.n_slices:
						raise ValueError('# of l.o.s. regions in "%s" different from other pixels') % name
					if DM_min != self.DM_min:
						raise ValueError('DM_min in "%s" different from other pixels') % name
					if DM_max != self.DM_max:
						raise ValueError('DM_min in "%s" different from other pixels') % name
				else:
					self.n_los_samples = n_los_samples
					self.n_slices = n_slices
					self.DM_min = DM_min
					self.DM_max = DM_max
				
				self.los_mask.append(True)
				
			except:
				self.los_mask.append(False)
		
		f.close()
	
	def load_files(self, fnames):
		# Create a giant lists of info from all pixels
		for fname in fnames:
			self.load_file_indiv(fname)
		
		# Combine pixel information
		self.pix_idx = np.array(self.pix_idx)
		self.nside = np.array(self.nside)
		
		# Combine cloud fits
		self.cloud_delta_mu = np.concatenate(self.cloud_delta_mu, axis=0)
		self.cloud_delta_lnEBV = np.concatenate(self.cloud_delta_lnEBV, axis=0)
		
		# Combine piecewise-linear fits
		self.los_delta_lnEBV = np.concatenate(self.los_delta_lnEBV, axis=0)
		
		# Calculate derived information
		self.cloud_mu_anchor = np.cumsum(self.cloud_delta_mu, axis=2)
		self.cloud_delta_EBV = np.exp(self.cloud_delta_lnEBV)
		
		self.los_delta_EBV = np.exp(self.los_delta_lnEBV)
		self.DM_anchor = np.linspace(self.DM_min, self.DM_max, self.n_slices)
		
		


def getLOS(fname):
	print 'Loading %s ...' % fname
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
		nCloudSamples -= 1
	except:
		pass
	shape = (nPixels, nCloudSamples, nClouds)
	CloudDeltaMu = np.zeros(shape, dtype='f8')
	CloudDeltaLnEBV = np.zeros(shape, dtype='f8')
	
	# Get number of regions and samples for piecewise model
	nPiecewiseSamples, nSlices, DM_min, DM_max = 1, 1, 1, 1
	try:
		dset = f['pixel %d/los' % pixels[0]]
		tmp, nPiecewiseSamples, nSlices = dset.shape
		DM_min = dset.attrs['DM_min']
		DM_max = dset.attrs['DM_max']
		nSlices -= 1
		nPiecewiseSamples -= 1
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
			CloudDeltaMu[i] = group[0, 1:, 1:nClouds+1]
			CloudDeltaLnEBV[i] = group[0, 1:, nClouds+1:]
		except:
			CloudDeltaMu[i] = 0
			CloudDeltaLnEBV[i] = np.nan
		
		# Piecewise model
		try:
			group = f['pixel %d/los' % pixIdx]
		except:
			pass
		
		try:
			PiecewiseDeltaLnEBV[i] = group[0,1:,1:]
		except:
			PiecewiseDeltaLnEBV[i] = np.nan
	
	f.close()
	
	CloudMuAnchor = np.cumsum(CloudDeltaMu, axis=2)
	CloudDeltaEBV = np.exp(CloudDeltaLnEBV)
	PiecewiseDeltaEBV = np.exp(PiecewiseDeltaLnEBV)
	
	return (pixels, CloudMuAnchor, CloudDeltaEBV,
	                DM_min, DM_max, PiecewiseDeltaEBV)

def getLOSFromMultiple(fnames):
	partial = []
	nPixels = 0
	for fname in fnames:
		try:
			partial.append(getLOS(fname))
			nPixels += len(partial[-1][0])
		except:
			print 'Loading Failed.'
	pixels = np.empty(nPixels, dtype='u4')
	
	tmp, nCloudSamples, nClouds = partial[0][1].shape
	shape = (nPixels, nCloudSamples, nClouds)
	CloudMuAnchor = np.empty(shape, dtype='f8')
	CloudDeltaEBV = np.empty(shape, dtype='f8')
	
	DM_min, DM_max = partial[0][3:5]
	tmp, nPiecewiseSamples, nSlices = partial[0][5].shape
	shape = (nPixels, nPiecewiseSamples, nSlices)
	PiecewiseDeltaEBV = np.empty(shape, dtype='f8')
	
	startIdx = 0
	for i,part in enumerate(partial):
		endIdx = startIdx + len(part[1])
		pixels[startIdx:endIdx] = part[0][:]
		CloudMuAnchor[startIdx:endIdx] = part[1][:]
		CloudDeltaEBV[startIdx:endIdx] = part[2][:]
		PiecewiseDeltaEBV[startIdx:endIdx] = part[5][:]
		startIdx += len(part[1])
	
	return (pixels, CloudMuAnchor, CloudDeltaEBV,
	                DM_min, DM_max, PiecewiseDeltaEBV)

def calcCloudEBV(muAnchor, DeltaEBV, mu):
	foreground = (muAnchor < mu)
	return np.cumsum(foreground * DeltaEBV, axis=2)[:,:,-1]

def calcPiecewiseEBV(muAnchor, DeltaEBV, mu):
	nPixels, nSamples, nSlices = DeltaEBV.shape
	
	idx = np.where(muAnchor >= mu, -1, np.arange(nSlices))
	lowIdx = np.max(idx)
	
	EBVslice = np.cumsum(DeltaEBV, axis=2)
	
	if lowIdx == nSlices - 1:
		return EBVslice[:,:,-1]
	
	lowMu = muAnchor[lowIdx]
	highMu = muAnchor[lowIdx+1]
	
	#print lowMu, mu, highMu
	#print EBVslice[:,0,lowIdx], EBVslice[:,0,lowIdx+1]
	
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

def calcEBV(muAnchor, DeltaEBV, mu, model='piecewise',
            maxSpread=None, method='median'):
	EBV = None
	if model == 'piecewise':
		EBV = calcPiecewiseEBV(muAnchor, DeltaEBV, mu)
	elif model == 'clouds':
		EBV = calcCloudEBV(muAnchor, DeltaEBV, mu)
	else:
		raise ValueError("Unrecognized extinction model: '%s'" % model)
	
	EBVcenter = None
	if type(method) in [float, int]:
		EBVcenter = np.percentile(EBV[:,1:], float(method), axis=1)
		if maxSpread != None:
			EBVspread = np.percentile(EBV[:,1:], 95., axis=1) - np.percentile(EBV[:,1:], 5., axis=1)
			idx = EBVspread > maxSpread
			EBVcenter[idx] = np.nan
	elif method == 'spread':
		EBVcenter = np.percentile(EBV[:,1:], 95., axis=1) - np.percentile(EBV[:,1:], 5., axis=1)
	elif method == 'median':
		EBVcenter = np.median(EBV[:,1:], axis=1)
		if maxSpread != None:
			EBVspread = np.percentile(EBV[:,1:], 95., axis=1) - np.percentile(EBV[:,1:], 5., axis=1)
			idx = EBVspread > maxSpread
			EBVcenter[idx] = np.nan
	elif method == 'mean':
		EBVcenter = np.mean(EBV[:,1:], axis=1)
		if maxSpread != None:
			EBVspread = np.percentile(EBV[:,1:], 95., axis=1) - np.percentile(EBV[:,1:], 5., axis=1)
			idx = EBVspread > maxSpread
			EBVcenter[idx] = np.nan
	elif method == 'best':
		EBVcenter = EBV[:,0]
		if maxSpread != None:
			EBVspread = np.percentile(EBV[:,1:], 95., axis=1) - np.percentile(EBV[:,1:], 5., axis=1)
			idx = EBVspread > maxSpread
			EBVcenter[idx] = np.nan
	else:
		raise ValueError("Unknown option: method='%s'" % method)
	
	return EBVcenter

def plotEBV(ax, pixels, muAnchor, DeltaEBV, mu,
                nside=512, nest=True, model='piecewise',
                maxSpread=None, method='median', **kwargs):
	# Generate rasterized image of E(B-V)
	EBVcenter = calcEBV(muAnchor, DeltaEBV, mu, model, maxSpread, method)
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
	imgRes = ax.imshow(img.T, **kwargs)
	
	kwargs['vmin'] = 0.
	kwargs['vmax'] = 1.
	mask = np.isnan(img.T)
	shape = (img.shape[1], img.shape[0], 4)
	maskImg = np.zeros(shape, dtype='f8')
	maskImg[:,:,1] = 0.4
	maskImg[:,:,2] = 1.
	maskImg[:,:,3] = 0.65 * mask.astype('f8')
	ax.imshow(maskImg, **kwargs)
	
	return imgRes

class PixelIdentifier:
	def __init__(self, ax, nside, nest=True):
		self.ax = ax
		self.cid = ax.figure.canvas.mpl_connect('button_press_event', self)
		
		self.nside = nside
		self.nest = nest
	
	def __call__(self, event):
		if event.inaxes != self.ax:
			return
		
		# Determine healpix index of point
		l, b = event.xdata, event.ydata
		theta = np.pi/180. * (90. - b)
		phi = np.pi/180. * l
		pix_idx = hp.ang2pix(self.nside, theta, phi, nest=self.nest)
		
		print '(%.2f, %.2f) -> %d' % (l, b, pix_idx)

def main():
	parser = argparse.ArgumentParser(prog='plotmap.py',
	                                 description='Generate a map of E(B-V) from bayestar output.',
	                                 add_help=True)
	parser.add_argument('input', type=str, nargs='+', help='Bayestar output files.')
	parser.add_argument('--output', '-o', type=str, help='Output filename for plot.')
	parser.add_argument('--show', '-sh', action='store_true', help='Show plot.')
	parser.add_argument('--dists', '-d', type=float, nargs=3,
	                                     default=(4., 19., 21),
	                                     help='DM min, DM max, # of distance slices.')
	parser.add_argument('--nside', '-n', type=int, default=512,
	                                     help='HealPIX nside parameter.')
	parser.add_argument('--model', '-m', type=str, default='piecewise',
	                                     choices=('piecewise', 'clouds'),
	                                     help="Extinction model: 'piecewise' or 'clouds'")
	parser.add_argument('--mask', '-msk', type=float, default=None,
	                                      help=r'Hide parts of map where 95%% - 5%% of E(B-V) is greater than given value')
	parser.add_argument('--method', '-mtd', type=str, default='median',
	                                        choices=('median', 'mean', 'best', '5th', '95th', 'spread'),
	                                        help='Measure of E(B-V) to plot.')
	parser.add_argument('--spread', '-sp', action='store_true',
	                                       help='Plot 95%% - 5%% of E(B-V)')
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	args = parser.parse_args(sys.argv[offset:])
	
	fnames = args.input
	pixels, CloudMuAnchor, CloudDeltaEBV, DM_min, DM_max, PiecewiseDeltaEBV = getLOSFromMultiple(fnames)
	
	tmp1, tmp2, nSlices = PiecewiseDeltaEBV.shape
	PiecewiseMuAnchor = np.linspace(DM_min, DM_max, nSlices)
	
	method = args.method
	if method == '5th':
		method = 5.
	elif method == '95th':
		method = 95.
	
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
	
	# Get upper limit on E(B-V)
	EBVs = None
	if args.model == 'clouds':
		if CloudDeltaEBV.shape[2] == 1:
			dist = np.power(10., CloudMuAnchor/5. + 1.)
			print 'Cloud distance:'
			print '  d = %.3f +- %.3f pc' % (np.mean(dist), np.std(dist))
			print '  mu = %.3f +- %.3f' % (np.mean(CloudMuAnchor), np.std(CloudMuAnchor))
			print 'distance percentiles:'
			print '  15.84%%: %.3f pc' % (np.percentile(dist, 15.84))
			print '  50.00%%: %.3f pc' % (np.percentile(dist, 50.))
			print '  84.16%%: %.3f pc' % (np.percentile(dist, 84.16))
			print 'E(B-V) = %.3f +- %.3f pc' % (np.mean(CloudDeltaEBV), np.std(CloudDeltaEBV))
			print 'E(B-V) percentiles:'
			print '  15.84%%: %.3f mag' % (np.percentile(CloudDeltaEBV, 15.84))
			print '  50.00%%: %.3f mag' % (np.percentile(CloudDeltaEBV, 50.))
			print '  84.16%%: %.3f mag' % (np.percentile(CloudDeltaEBV, 84.16))
	
	EBVmax = None
	for m in mu[::-1]:
		if args.model == 'piecewise':
			EBVs = calcEBV(PiecewiseMuAnchor, PiecewiseDeltaEBV, m,
						   args.model, args.mask, method)
		elif args.model == 'clouds':
			EBVs = calcEBV(CloudMuAnchor, CloudDeltaEBV, m,
						   args.model, args.mask, method)
		idx = ~np.isnan(EBVs)
		try:
			EBVmax = np.percentile(EBVs[idx], 98.)
		except:
			pass
		if EBVmax != None:
			print 'max EBV = %.3f' % EBVmax
			print 'EBV(mu=%.2f) = %.3f +- %.3f' % (mu[-1], np.mean(EBVs[idx]), np.std(EBVs[idx]))
			del EBVs
			break
		del EBVs
	
	# Determine output filename
	fname = args.output
	if fname != None:
		if fname.endswith('.png'):
			fname = fname[:-4]
	
	pix_identifiers = []
	
	# Plot at each distance
	for i in xrange(muN):
		print 'mu = %.2f (%d of %d)' % (mu[i], i+1, muN)
		
		fig = plt.figure(dpi=150)
		ax = fig.add_subplot(1,1,1)
		
		if args.model == 'piecewise':
			img = plotEBV(ax, pixels, PiecewiseMuAnchor, PiecewiseDeltaEBV, mu[i],
			              nside=args.nside, nest=True, model=args.model,
			              maxSpread=args.mask, method=method,
			              vmin=0., vmax=EBVmax)
		elif args.model == 'clouds':
			img = plotEBV(ax, pixels, CloudMuAnchor, CloudDeltaEBV, mu[i],
			              nside=args.nside, nest=True, model=args.model,
			              maxSpread=args.mask, method=method,
			              vmin=0., vmax=EBVmax)
		
		fig.subplots_adjust(bottom=0.12, left=0.12, right=0.89, top=0.9)
		cax = fig.add_axes([0.9, 0.12, 0.03, 0.78])
		cb = fig.colorbar(img, cax=cax)
		
		ax.set_xlabel(r'$\ell$', fontsize=16)
		ax.set_ylabel(r'$b$', fontsize=16)
		
		ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
		ax.xaxis.set_minor_locator(AutoMinorLocator())
		ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
		ax.yaxis.set_minor_locator(AutoMinorLocator())
		
		d = 10.**(mu[i]/5. - 2.)
		ax.set_title(r'$\mu = %.2f \ \ \ d = %.2f \, \mathrm{kpc}$' % (mu[i], d), fontsize=16)
		
		pix_identifiers.append(PixelIdentifier(ax, args.nside))
		
		if fname != None:
			full_fname = '%s.%s.%s.%.5d.png' % (fname, args.model, args.method, i)
			fig.savefig(full_fname, dpi=150)
	
	if args.show:
		plt.show()
	
	return 0

if __name__ == '__main__':
	main()

