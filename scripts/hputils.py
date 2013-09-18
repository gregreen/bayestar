#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  hputils.py
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

print 'import np'
import numpy as np
print 'import hp'
import healpy as hp

print 'import plt'
import matplotlib.pyplot as plt

print 'done importing'

def lb2pix(nside, l, b, nest=True):
	'''
	Convert (l, b) to pixel index.
	'''
	
	theta = np.pi/180. * (90. - b)
	phi = np.pi/180. * l
	
	return hp.pixelfunc.ang2pix(nside, theta, phi, nest=nest)


def pix2lb(nside, ipix, nest=True):
	'''
	Convert pixel index to (l, b).
	'''
	
	theta, phi = hp.pixelfunc.pix2ang(nside, ipix, nest=True)
	
	l = 180./np.pi * phi
	b = 90. - 180./np.pi * theta
	
	return l, b


def Mollweide_inv(x, y, lam_0=0.):
	'''
	Inverse Mollweide transformation.
	
	Returns (phi, lam), given (x, y), where
	x and y can each range from -1 to 1.
	
	x and y can be floats or numpy float arrays.
	
	lam_0 is the central longitude of the map.
	'''
	
	theta = np.arcsin(y / np.sqrt(2.))
	
	phi = np.arcsin((2. * theta + np.sin(2. * theta)) / np.pi)
	
	lam = lam_0 + np.pi / (2. * np.sqrt(2.) * np.cos(theta))


def Mollweide_theta(phi, iterations=10):
	theta = phi[:]
	sin_phi = np.sin(phi)
	
	for i in xrange(iterations):
		theta -= (theta + np.sin(theta) - np.pi * sin_phi) / (1. + np.cos(theta))
	
	return theta

def Mollweide(phi, lam, y, lam_0=0.):
	theta = Mollweide_theta(phi)
	
	x = 2. * np.sqrt(2.) * (lam - lam_0) * np.cos(theta)
	y = np.sqrt(2.) * np.sin(theta)


def rasterize_map(pix_idx, pix_val, size, nside, nest=True):
	pix_scale = 180./np.pi * hp.nside2resol(nside)
	
	# Determine pixel centers and bounds
	l_0, b_0 = pix2lb(nside, pix_idx, nest=nest)
	x_0, y_0 = Mollweide(b_0, l_0)
	
	x_min = np.min(x_0)
	x_max = np.max(x_0)
	y_min = np.min(y_0)
	y_max = np.max(y_0)
	
	# Make grid of display-space pixels
	x_size, y_size = size
	
	x, y = np.mgrid[0:x_size, 0:y_size].astype(np.float32) + 0.5
	x = x_min + (x_max - x_min) * x / float(x_size)
	y = y_min + (y_max - y_min) * y / float(y_size)
	
	# Convert display-space pixels to (l, b)
	l, b = Mollweide_inv(x, y)
	
	# Convert (l, b) to healpix indices
	disp_idx = lb2pix(nside, l, b, nest=nest)
	#mask = 
	
	# Grab pixel values
	img = None
	if len(pix_val.shape) == 1:
		img = pix_val[disp_idx]
		#img[mask] = np.nan
		img.shape = (x_size, y_size)
	elif len(pix_val.shape) == 2:
		img = pix_val[:,disp_idx]
		#img[mask] = np.nan
		img.shape = (img.shape[0], x_size, y_size)
	else:
		raise Exception('pix_val must be either 1- or 2-dimensional.')
	
	bounds = (x_min, x_max, y_min, y_min)
	return img, bounds


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


def main():
	nside = 4
	nest = True
	size = (500, 500)
	
	print 'hello'
	n_pix = hp.pixelfunc.nside2npix(nside)
	pix_idx = np.arange(n_pix)
	pix_val = pix_idx[:]
	
	print 'world'
	# Rasterize map
	img, bounds = rasterize_map(pix_idx, pix_val,
	                            size, nside, nest=nest)
	
	print 'plotting'
	# Plot map
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	
	ax.imshow(img, extent=bounds, interpolation='nearest')
	
	plt.show()
	
	return 0


if __name__ == '__main__':
	main()
