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

import numpy as np
import healpy as hp

import matplotlib.pyplot as plt


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
	
	theta, phi = hp.pixelfunc.pix2ang(nside, ipix, nest=nest)
	
	l = 180./np.pi * phi
	b = 90. - 180./np.pi * theta
	
	return l, b


def wrap_longitude(lon, delta_lon, degrees=True):
	'''
	Shift longitudes by delta_lon, and wrap
	back to range [0, 360].
	
	If degrees=False, then radians are assumed.
	'''
	
	lon_shifted = lon + delta_lon
	
	if degrees:
		return np.mod(lon_shifted, 360.)
	else:
		return np.mod(lon_shifted, 2. * np.pi)


def shift_lon_lat(lon, lat, delta_lon, delta_lat,
                  degrees=True, clip=False):
	'''
	Shift latitudes and longitudes, but do not
	move them off edges map, and do not wrap
	longitude.
	'''
	
	lon_shifted = lon + delta_lon
	lat_shifted = lat + delta_lat
	
	if clip:
		idx = (lon_shifted > 360.)
		lon_shifted[idx] = 360.
		
		idx = (lon_shifted < 0.)
		lon_shifted[idx] = 0.
		
		idx = (lat_shifted > 90.)
		lat_shifted[idx] = 90.
		
		idx = (lat_shifted < -90.)
		lat_shifted[idx] = -90.
	
	return lon_shifted, lat_shifted


class Mollweide_projection:
	def __init__(self, lam_0=180.):
		'''
		lam_0 is the central longitude of the map.
		'''
		
		self.lam_0 = np.pi/180. * lam_0
	
	def proj(self, phi, lam, iterations=10):
		'''
		Mollweide projection.
		
		phi = latitude
		lam = longitude
		'''
		
		theta = self.Mollweide_theta(phi, iterations)
		
		x = 2. * np.sqrt(2.) * (lam - self.lam_0) * np.cos(theta) / np.pi
		y = np.sqrt(2.) * np.sin(theta)
		
		return x, y
	
	def inv(self, x, y):
		'''
		Inverse Mollweide projection.
		
		Returns (phi, lam), given (x, y), where
		x and y can each range from -1 to 1.
		
		phi = latitude
		lam = longitude
		
		x and y can be floats or numpy float arrays.
		'''
		
		theta = np.arcsin(y / np.sqrt(2.))
		
		phi = np.arcsin((2. * theta + np.sin(2. * theta)) / np.pi)
		
		lam = self.lam_0 + np.pi * x / (2. * np.sqrt(2.) * np.cos(theta))
		
		return phi, lam
	
	def Mollweide_theta(self, phi, iterations):
		theta = np.arcsin(2. * phi / np.pi)
		sin_phi = np.sin(phi)
		
		for i in xrange(iterations):
			theta -= 0.5 * (2. * theta + np.sin(2. * theta) - np.pi * sin_phi) / (1. + np.cos(2. * theta))
		
		return theta


class Cartesian_projection:
	def __init__(self):
		pass
	
	def proj(self, phi, lam):
		x = 180./np.pi * lam
		y = 180./np.pi * phi
		
		return x, y
	
	def inv(self, x, y):
		lam = np.pi/180. * x
		phi = np.pi/180. * y
		
		return phi, lam


def rasterize_map(pix_idx, pix_val,
                  nside, size,
                  nest=True, clip=True,
                  proj=Cartesian_projection()):
	'''
	Rasterize a healpix map.
	'''
	
	pix_scale = 180./np.pi * hp.nside2resol(nside)
	
	# Determine pixel centers
	l_0, b_0 = pix2lb(nside, pix_idx, nest=nest)
	l_0 = 360. - wrap_longitude(l_0, 180.)
	
	# Determine display-space bounds
	shift = [(0., 0.), (1., 0.), (0., 1.), (-1., 0.), (0., -1.)]
	x_min, x_max, y_min, y_max = [], [], [], []
	
	#for (s_x, s_y) in shift:
	for s_x in np.linspace(-pix_scale, pix_scale, 3):
		for s_y in np.linspace(-pix_scale, pix_scale, 3):
			l, b = shift_lon_lat(l_0, b_0, 0.75*s_x, 0.75*s_y, clip=True)
			
			#print l
			#print b
			
			x_0, y_0 = proj.proj(np.pi/180. * b, np.pi/180. * l)
			
			x_min.append(np.min(x_0))
			x_max.append(np.max(x_0))
			y_min.append(np.min(y_0))
			y_max.append(np.max(y_0))
	
	x_min = np.min(x_min)
	x_max = np.max(x_max)
	y_min = np.min(y_min)
	y_max = np.max(y_max)
	
	# Make grid of display-space pixels
	x_size, y_size = size
	
	x, y = np.mgrid[0:x_size, 0:y_size].astype(np.float32) + 0.5
	x = x_min + (x_max - x_min) * x / float(x_size)
	y = y_min + (y_max - y_min) * y / float(y_size)
	
	# Convert display-space pixels to (l, b)
	b, l = proj.inv(x, y)
	l *= 180./np.pi
	b *= 180./np.pi
	
	# Generate clip mask
	mask = None
	
	if clip:
		mask = (l < 0.) | (l > 360.) | (b < -90.) | (b > 90.)
	
	# Convert (l, b) to healpix indices
	l = 360. - wrap_longitude(l, 180.)
	disp_idx = lb2pix(nside, l, b, nest=nest)
	#mask = 
	
	# Generate full map
	n_pix = hp.pixelfunc.nside2npix(nside)
	pix_idx_full = np.arange(n_pix)
	pix_val_full = np.empty(n_pix, dtype='f8')
	pix_val_full[:] = np.nan
	pix_val_full[pix_idx] = pix_val[:]
	
	# Grab pixel values
	img = None
	if len(pix_val.shape) == 1:
		img = pix_val_full[disp_idx]
		
		if clip:
			img[mask] = np.nan
			
		img.shape = (x_size, y_size)
		
	elif len(pix_val.shape) == 2:
		img = pix_val[:,disp_idx]
		
		if clip:
			img[:,mask] = np.nan
		
		img.shape = (img.shape[0], x_size, y_size)
		
	else:
		raise Exception('pix_val must be either 1- or 2-dimensional.')
	
	bounds = (x_min, x_max, y_min, y_max)
	
	return img, bounds


def test_Mollweide():
	proj = Mollweide_projection()
	
	phi = np.pi * (np.random.random(10) - 0.5)
	lam = 2. * np.pi * (np.random.random(10) - 0.5)
	
	x, y = proj.proj(phi, lam)
	phi_1, lam_1 = proj.inv(x, y)
	
	print 'lat  lon  x    y'
	
	for i in xrange(len(phi)):
		print '%.2f %.2f %.2f %.2f' % (phi[i]*180./np.pi, lam[i]*180./np.pi, x[i], y[i])
	
	print ''
	print "phi  phi'  lam  lam'"
	for i in xrange(len(phi)):
		print '%.2f %.2f %.2f %.2f' % (phi[i], phi_1[i], lam[i], lam_1[i])


def test_proj():
	nside = 16
	nest = True
	clip = True
	size = (1000, 500)
	
	n_pix = hp.pixelfunc.nside2npix(nside)
	pix_idx = np.arange(n_pix)#[:256]
	l, b = pix2lb(nside, pix_idx, nest=nest)
	pix_val = pix_idx[:]
	
	# Plot map
	
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	
	# Rasterize map
	img, bounds = rasterize_map(pix_idx[:64], pix_val[:64],
	                            nside, size,
	                            nest=nest, clip=clip,
	                            proj=Mollweide_projection(lam_0=180.))
	
	cimg = ax.imshow(img.T, extent=bounds,
	                 origin='lower', interpolation='nearest',
	                 aspect='auto')
	
	# Rasterize map
	img, bounds = rasterize_map(pix_idx[64:], pix_val[64:],
	                            nside, size,
	                            nest=nest, clip=clip,
	                            proj=Mollweide_projection(lam_0=180.))
	
	cimg = ax.imshow(img.T, extent=bounds,
	                 origin='lower', interpolation='nearest',
	                 aspect='auto')
	
	cimg = ax.imshow(img.T, extent=bounds,
	                 origin='lower', interpolation='nearest',
	                 aspect='auto')
	
	# Color bar
	fig.subplots_adjust(left=0.10, right=0.90, bottom=0.20, top=0.90)
	cax = fig.add_axes([0.10, 0.10, 0.80, 0.05])
	fig.colorbar(cimg, cax=cax, orientation='horizontal')
	
	
	plt.show()


def main():
	#test_Mollweide()
	test_proj()


if __name__ == '__main__':
	main()
