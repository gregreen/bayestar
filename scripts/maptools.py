#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  maptools.py
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

import healpy as hp
import h5py

import hputils


def reduce_to_single_res(pix_idx, nside, pix_val):
	nside_unique = np.unique(nside)
	nside_max = np.max(nside_unique)
	
	idx = (nside == nside_max)
	pix_idx_exp = [pix_idx[idx]]
	pix_val_exp = [pix_val[idx]]
	
	for n in nside_unique[:-1]:
		n_rep = (nside_max / n)**2
		
		idx = (nside == n)
		n_pix = np.sum(idx)
		
		pix_idx_n = np.repeat(n_rep * pix_idx[idx], n_rep, axis=0)
		
		pix_adv = np.mod(np.arange(n_rep * n_pix), n_rep)
		pix_idx_n += pix_adv
		
		#for k in xrange(1, n_rep):
		#	pix_idx_n[k*n_pix:(k+1)*n_pix] += k
		
		pix_val_n = np.repeat(pix_val[idx], n_rep, axis=0)
		
		pix_idx_exp.append(pix_idx_n)
		pix_val_exp.append(pix_val_n)
	
	pix_idx_exp = np.concatenate(pix_idx_exp, axis=0)
	pix_val_exp = np.concatenate(pix_val_exp, axis=0)
	
	return nside_max, pix_idx_exp, pix_val_exp


class los_collection:
	'''
	Loads line-of-sight fits from Bayestar
	output files, and generates maps at
	requested distances.
	'''
	
	def __init__(self, fnames, bounds=None):
		'''
		fnames is a list of Bayestar output files
		containing line-of-sight fit information.
		
		
		Do not load pixels whose centers are outside of the range set
		by <bounds>, where
		
		    bounds = [l_min, l_max, b_min, b_max]
		
		If <bounds> is None, then all pixels are loaded.
		'''
		
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
		self.load_files(fnames, bounds=bounds)
	
	def load_file_indiv(self, fname, bounds=None):
		'''
		Loads data on the line-of-sight fits from one
		Bayestar output file.
		
		Do not load pixels whose centers are outside of the range set
		by <bounds>, where
		
		    bounds = [l_min, l_max, b_min, b_max]
		
		If <bounds> is None, then all pixels are loaded.
		'''
		
		print 'Loading %s ...' % fname
		
		f = None
		
		try:
			f = h5py.File(fname, 'r')
		except:
			raise IOError('Unable to open %s.' % fname)
		
		# Load each pixel
		
		for name,item in f.iteritems():
			# Load pixel position
			try:
				pix_idx_tmp = int(item.attrs['healpix_index'][0])
				nside_tmp = int(item.attrs['nside'][0])
			except:
				continue
			
			# Check if pixel is in bounds
			if bounds != None:
				l, b = hputils.pix2lb_scalar(nside_tmp, pix_idx_tmp, nest=True)
				
				if (     (l < bounds[0]) or (l > bounds[1])
				      or (b < bounds[2]) or (b > bounds[3])  ):
					continue
			
			self.pix_idx.append(pix_idx_tmp)
			self.nside.append(nside_tmp)
			
			# Load cloud fit
			try:
				cloud_samples_tmp = item['clouds'][:, 1:, 1:]
				tmp, n_cloud_samples, n_clouds = cloud_samples_tmp.shape
				n_clouds /= 2
				
				self.cloud_delta_mu.append(cloud_samples_tmp[:, :, :n_clouds])
				self.cloud_delta_lnEBV.append(cloud_samples_tmp[:, :, n_clouds:])
				
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
				
				self.los_delta_lnEBV.append(los_samples_tmp)
					
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
	
	def load_files(self, fnames, bounds=None):
		'''
		Loads data on the line-of-sight fits from a set
		of Bayestar output files.
		
		Do not load pixels whose centers are outside of the range set
		by <bounds>, where
		
		    bounds = [l_min, l_max, b_min, b_max]
		
		If <bounds> is None, then all pixels are loaded.
		'''
		
		# Create a giant lists of info from all pixels
		for fname in fnames:
			self.load_file_indiv(fname, bounds=bounds)
		
		# Combine pixel information
		self.pix_idx = np.array(self.pix_idx).astype('i8')
		self.nside = np.array(self.nside)
		
		# Combine cloud fits
		self.cloud_mask = np.array(self.cloud_mask).astype(np.bool)
		self.cloud_delta_mu = np.concatenate(self.cloud_delta_mu, axis=0)
		self.cloud_delta_lnEBV = np.concatenate(self.cloud_delta_lnEBV, axis=0)
		
		# Combine piecewise-linear fits
		self.los_mask = np.array(self.los_mask).astype(np.bool)
		self.los_delta_lnEBV = np.concatenate(self.los_delta_lnEBV, axis=0)
		
		# Calculate derived information
		self.cloud_mu_anchor = np.cumsum(self.cloud_delta_mu, axis=2)
		self.cloud_delta_EBV = np.exp(self.cloud_delta_lnEBV)
		
		self.los_delta_EBV = np.exp(self.los_delta_lnEBV)
		self.los_EBV = np.cumsum(self.los_delta_EBV, axis=2)
		self.los_mu_anchor = np.linspace(self.DM_min, self.DM_max, self.n_slices)
		self.los_dmu = np.diff(self.los_mu_anchor)[0]
	
	def get_nside_levels(self):
		'''
		Returns the unique nside values present in the
		map.
		'''
		
		return np.unique(self.nside)
	
	def get_los_mu_anchors(self):
		'''
		Returns the anchor distance moduli for the
		piecewise-linear fits.
		'''
		
		return self.los_mu_anchor
	
	def get_los(self, nside=None):
		'''
		Returns data on the piecewise-linear fits.
		
		If no nside is specified, then returns
		
		    pix_idx, nside, delta_EBV
		
		for each pixel. If an nside is specified, then only
		pixels at the given nside are returned, and the output
		omits nside. The shape of the last array in the
		output is
		
		    shape = (n_pixels, n_samples, n_slices)
		
		where the first sample is the best fit, and the
		remaining samples are drawn from the posterior.
		'''
		
		if nside == None:
			idx = self.los_mask
			
			return self.pix_idx[idx], self.nside[idx], self.los_delta_EBV
		else:
			idx_0 = self.los_mask & (self.nside == nside)
			
			nside_tmp = self.nside[self.los_mask]
			idx_1 = nside_tmp == nside
			
			return self.pix_idx[idx_0], self.los_delta_EBV[idx_1]
	
	def get_clouds(self, nside=None):
		'''
		Returns data on the cloud fits.
		
		If no nside is specified, then returns
		
		    pix_idx, nside, cloud_delta_mu, cloud_delta_EBV
		
		for each pixel. If an nside is specified, then only
		pixels at the given nside are returned, and the output
		omits nside. The shape of the last two arrays in the
		output is
		
		    shape = (n_pixels, n_samples, n_clouds)
		
		where the first sample is the best fit, and the
		remaining samples are drawn from the posterior.
		'''
		
		if nside == None:
			idx = self.cloud_mask
			
			return (self.pix_idx[idx], self.nside[idx],
			        self.cloud_delta_mu, self.cloud_delta_EBV)
		else:
			idx_0 = self.cloud_mask & (self.nside == nside)
			
			nside_tmp = self.nside[self.cloud_mask]
			idx_1 = nside_tmp == nside
			
			return (self.pix_idx[idx_0], self.cloud_delta_mu[idx_1],
			                             self.cloud_delta_EBV[idx_1])
	
	def calc_cloud_EBV(self, mu):
		'''
		Returns an array of E(B-V) evaluated at
		distance modulus mu, with
		
		    shape = (n_pixels, n_samples)
		
		The first sample is the best fit sample.
		The rest are drawn from the posterior.
		
		Uses the cloud model.
		'''
		
		foreground = (self.cloud_mu_anchor < mu)
		
		return np.sum(foreground * self.cloud_delta_EBV, axis=2)
	
	def calc_piecewise_EBV(self, mu):
		'''
		Returns an array of E(B-V) evaluated at
		distance modulus mu, with
		
		    shape = (n_pixels, n_samples)
		
		The first sample is the best fit sample.
		The rest are drawn from the posterior.
		
		Uses the piecewise-linear model.
		'''
		
		idx = np.where(self.los_mu_anchor >= mu, -1, np.arange(self.n_slices))
		low_idx = np.max(idx)
		
		if low_idx == self.n_slices - 1:
			return self.los_EBV[:,:,-1]
		
		low_mu = self.los_mu_anchor[low_idx]
		high_mu = self.los_mu_anchor[low_idx+1]
		
		a = (mu - low_mu) / (high_mu - low_mu)
		EBV_interp = (1. - a) * self.los_EBV[:,:,low_idx]
		EBV_interp += a * self.los_EBV[:,:,low_idx+1]
		
		return EBV_interp
	
	def est_dEBV_pctile(self, pctile, delta_mu=0.1,
	                          fit='piecewise'):
		'''
		Estimate the requested percentile of
		
		    dE(B-V) / dDM
		
		over the whole map.
		
		Use the distance modulus step <delta_mu>
		to estimate the derivative.
		'''
		
		if fit == 'piecewise':
			return np.percentile(self.los_delta_EBV, pctile) / self.los_dmu
		elif fit == 'cloud':
			return np.percentile(self.cloud_delta_EBV, pctile) / delta_mu
		else:
			raise ValueError('Unrecognized fit type: "%s"' % fit)
	
	def gen_EBV_map(self, mu, fit='piecewise',
	                          method='median',
	                          mask_sigma=None,
	                          delta_mu=None):
		'''
		Returns an array of E(B-V) evaluated at
		distance modulus mu, with
		
		    shape = (n_pixels,)
		
		Also returns an array of HEALPix pixel indices,
		and a single nside parameter, equal to the
		highest nside resolution in the map.
		
		The order of the output is
		
		    nside, pix_idx, EBV
		
		The fit option can be either 'piecewise' or 'cloud',
		depending on which type of fit the map should use.
		
		The method option determines which measure of E(B-V)
		is returned. The options are
		
		    'median', 'mean', 'best',
		    'sample', 'sigma', float (percentile)
		
		'sample' generates a random map, drawn from the
		posterior. 'sigma' returns the percentile-equivalent
		of the standard deviation (half the 84.13%% - 15.87%% range).
		If method is a float, then the corresponding
		percentile map is returned.
		
		If mask_sigma is a float, then pixels where sigma is
		greater than the provided threshold will be masked out.
		'''
		
		EBV = None
		
		if fit == 'piecewise':
			EBV = self.calc_piecewise_EBV(mu)
		elif fit == 'cloud':
			EBV = self.calc_cloud_EBV(mu)
		else:
			raise ValueError('Unrecognized fit type: "%s"' % fit)
		
		# Calculate rate of reddening (dEBV/dDM), if requested
		if delta_mu != None:
			if fit == 'piecewise':
				EBV -= self.calc_piecewise_EBV(mu-delta_mu)
			elif fit == 'cloud':
				EBV -= self.calc_cloud_EBV(mu-delta_mu)
			
			EBV /= delta_mu
		
		# Mask regions with high uncertainty
		if mask_sigma != None:
			sigma = self.take_measure(EBV, 'sigma')
			mask_idx = (sigma > mask_sigma)
		
		# Reduce EBV in each pixel to one value
		EBV = self.take_measure(EBV, method)
		
		if mask_sigma != None:
			EBV[mask_idx] = np.nan
		
		# Reduce to one HEALPix nside resolution
		mask = self.los_mask
		pix_idx = self.pix_idx[mask]
		nside = self.nside[mask]
		
		nside, pix_idx, EBV = reduce_to_single_res(pix_idx, nside, EBV)
		
		return nside, pix_idx, EBV
	
	def take_measure(self, EBV, method):
		if method == 'median':
			return np.median(EBV[:,1:], axis=1)
		elif method == 'mean':
			return np.mean(EBV[:,1:], axis=1)
		elif method == 'best':
			return EBV[:,1]
		elif method == 'sample':
			n_pix, n_samples = EBV.shape
			j = np.random.randint(1, high=n_samples, size=n_pix)
			i = np.arange(n_pix)
			return EBV[i,j]
		elif method == 'sigma':
			high = np.percentile(EBV[:,1:], 84.13, axis=1)
			low = np.percentile(EBV[:,1:], 15.87, axis=1)
			return 0.5 * (high - low)
		elif type(method) == float:
			return np.percentile(EBV[:,1:], method, axis=1)
		else:
			raise ValueError('method not implemented: "%s"' % (str(method)))
	
	def rasterize(self, mu, size,
	                    method='median', fit='piecewise',
	                    mask_sigma=None, delta_mu=None,
	                    clip=True,
	                    proj=hputils.Cartesian_projection(),
	                    l_cent=0., b_cent=0.):
		'''
		Rasterize the map, returning an image of the specified size.
		
		The <fit> argument can be either 'piecewise' or 'cloud',
		depending on which type of fit the map should use.
		
		The <method> argument determines which measure of E(B-V)
		is returned. The options are
		
			'median', 'mean', 'best',
			'sample', 'sigma', float (percentile)
		
		'sample' generates a random map, drawn from the
		posterior. 'sigma' returns the percentile-equivalent
		of the standard deviation (half the 84.13%% - 15.87%% range).
		If method is a float, then the corresponding
		percentile map is returned.
		
		If <mask_sigma> is a float, then pixels where sigma is
		greater than the provided threshold will be masked out.
		
		If <clip> is true, then map does not wrap around at
		l = 180 deg. If
		
		The variable <proj> is a class representing a projection.
		The module hputils.py has two built-in projections,
		Cartesian_projection() and Mollweide_projection(). The user
		can supply their own custom projection class, if desired.
		The projection class must have two functions,
		
		    proj(lat, lon) --> (x, y)
		    inv(x, y) -> (lat, lon, out_of_bounds)
		
		The optional argument <l_cent> and <b_cent> determine the
		Galactic longitude and latitude on which the map is centered,
		respectively.
		'''
		
		nside, pix_idx, EBV = self.gen_EBV_map(mu, fit=fit,
		                                       method=method,
		                                       mask_sigma=mask_sigma,
		                                       delta_mu=delta_mu)
		
		img, bounds = hputils.rasterize_map(pix_idx, EBV, nside, size,
		                                    nest=True, clip=clip, proj=proj,
		                                    l_cent=l_cent, b_cent=b_cent)
		
		return img, bounds


class job_completion_counter:
	'''
	Checks the status of a set of Bayestar jobs. Has the
	ability to generate a rasterized map of the job
	completion.
	'''
	
	def __init__(self, infiles, outfiles):
		'''
		infiles is a list of Bayestar input files, while
		outfiles is a list of Bayestar output files.
		'''
		
		# Load files
		self.load_completion(infiles, outfiles)
	
	def load_output_indiv(self, outfname):
		'''
		Looks to see which pixels are finished in a job
		'''
		
		#print 'Loading %s ...' % outfname
		
		f = None
		
		try:
			f = h5py.File(outfname, 'r')
		except:
			raise IOError('Unable to open %s.' % outfname)
		
		# Load each pixel
		
		for name,item in f.iteritems():
			# Load pixel position
			try:
				pix_idx_tmp = item.attrs['healpix_index'][0]
				nside_tmp = item.attrs['nside'][0]
			except:
				continue
			
			# Check which elements of output are present in pixel
			keys = item.keys()
			
			star_tmp = ('stellar chains' in keys)
			cloud_tmp = ('clouds' in keys)
			los_tmp = ('los' in keys)
			
			self.completion_dict[(nside_tmp, pix_idx_tmp)] = (star_tmp, cloud_tmp, los_tmp)
			
			# Update number of stars completed
			if los_tmp:
				nstars_tmp, infname_tmp = self.att_dict[(nside_tmp, pix_idx_tmp)]
				self.nstars_complete += nstars_tmp
		
		f.close()
	
	def load_input_indiv(self, infname):
		'''
		Looks to see which pixels are in an input file.
		'''
		
		#print 'Loading %s ...' % infname
		
		f = None
		
		try:
			f = h5py.File(infname, 'r')
		except:
			raise IOError('Unable to open %s.' % infname)
		
		# Load each pixel
		
		for name,item in f['/photometry'].iteritems():
			# Load pixel position
			try:
				pix_idx_tmp = item.attrs['healpix_index']
				nside_tmp = item.attrs['nside']
				nstars_tmp = item.size
			except:
				continue
			
			self.completion_dict[(nside_tmp, pix_idx_tmp)] = (0, 0, 0)
			self.att_dict[(nside, pix_idx_tmp)] = (nstars_tmp, infname)
			self.nstars_input += nstars_tmp
		
		f.close()
	
	def load_completion(self, infnames, outfnames):
		# Information of completeness of jobs
		self.completion_dict = {}
		self.att_dict = {}
		self.nstars_input = 0
		self.nstars_complete = 0
		
		# Load information from input and output files
		for fname in infnames:
			try:
				self.load_input_indiv(fname)
			except:
				print 'Could not open %s !' % fname
		
		for fname in outfnames:
			try:
				self.load_output_indiv(fname)
			except:
				print 'Could not open %s !' % fname
		
		# Generate map of completion
		locs = np.array(self.completion_dict.keys(), dtype='i8')
		
		self.nside = locs[:,0]
		self.pix_idx = locs[:,1]
		
		completion = np.array(self.completion_dict.values(), dtype='i4')
		
		self.star = completion[:,0]
		self.cloud = completion[:,1]
		self.los = completion[:,2]
	
	def get_incomplete_inputs(self, method='both'):
		if method == 'star':
			comp_map = self.star[:]
		elif method == 'cloud':
			comp_map = self.star & self.cloud
		elif method == 'piecewise':
			comp_map = self.star & self.los
		elif method == 'both':
			comp_map = self.star & self.cloud & self.los
		else:
			raise ValueError("Unrecognized completion method: '%s'" % method)
		
		# Determine which input files have been completely processed
		incomplete = []
		
		for k,(nside,idx) in enumerate(zip(self.nside, self.pix_idx)):
			if not comp_map[k]:
				nstars_tmp, infname_tmp = self.att_dict[(nside, idx)]
				incomplete.append(infname_tmp)
		
		incomplete = np.array(incomplete)
		
		return np.unique(incomplete)
	
	def get_pct_complete(self):
		return 100. * float(self.nstars_complete) / float(self.nstars_input)
	
	def rasterize(self, size, method='both',
	                          proj=hputils.Cartesian_projection(),
	                          l_cent=0., b_cent=0.):
		'''
		Rasterize the completion map, returning an image of the specified size.
		
		The argument <method> indicates how completion should be calculated.
		The options are:
		
		    "cloud"      mark as complete if cloud fit present
		    "piecewise"  mark as complete if piecewise-linear fit present
		    "both"       mark as complete if both l.o.s. fits present
		
		The argument <proj> is a class representing a projection.
		The module hputils.py has two built-in projections,
		Cartesian_projection() and Mollweide_projection(). The user
		can supply their own custom projection class, if desired.
		The projection class must have two methods,
		
		    proj(lat, lon) --> (x, y)
		    inv(x, y) -> (lat, lon, out_of_bounds)
		
		The optional argument <l_cent> and <b_cent> determine the
		Galactic longitude and latitude on which the map is centered,
		respectively.
		'''
		
		comp_map = 1 + self.star
		
		if method == 'cloud':
			comp_map += self.cloud
		elif method == 'piecewise':
			comp_map += self.los
		elif method == 'both':
			comp_map += (self.cloud & self.los).astype('i4')
		else:
			raise ValueError("Unrecognized method: '%s'" % method)
		
		nside, pix_idx, val = reduce_to_single_res(self.pix_idx, self.nside, comp_map)
		
		img, bounds = hputils.rasterize_map(pix_idx, val, nside, size,
		                                    nest=True, clip=True, proj=proj,
		                                    l_cent=l_cent, b_cent=b_cent)
		
		return img, bounds


def test_load():
	dirname = r'/n/fink1/ggreen/bayestar/output/AquilaSouth'
	fnames = ['%s//AquilaSouth.%.5d.h5' % (dirname, n) for n in xrange(1)]
	
	los_coll = los_collection(fnames)
	
	nsides = los_coll.get_nside_levels()
	
	for n in nsides:
		pix_idx, cloud_delta_mu, cloud_delta_EBV = los_coll.get_clouds(nside=n)
		
		for idx, dm, dA in zip(pix_idx, cloud_delta_mu, cloud_delta_EBV):
			print n, idx, np.median(dm), np.median(dA)
	
	mu = 10.
	
	los_EBV = los_coll.calc_piecewise_EBV(mu)
	cloud_EBV = los_coll.calc_cloud_EBV(mu)
	
	print np.median(los_EBV, axis=1)
	print np.median(cloud_EBV, axis=1)
	
	nside, pix_idx, EBV = los_coll.gen_EBV_map(mu, fit='cloud', method='median')
	
	for i, A in zip(pix_idx, EBV):
		print i, A
	
	# Rasterize map
	size = (2000, 1000)
	
	img, bounds = los_coll.rasterize(mu, size,
	                                 method='best', fit='cloud',
	                                 proj=hputils.Mollweide_projection())
	
	# Plot map
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	
	cimg = ax.imshow(img.T, extent=bounds,
	                 vmin=0., cmap='binary',
	                 origin='lower', interpolation='nearest',
	                 aspect='auto')
	
	# Color bar
	fig.subplots_adjust(left=0.10, right=0.90, bottom=0.20, top=0.90)
	cax = fig.add_axes([0.10, 0.10, 0.80, 0.05])
	fig.colorbar(cimg, cax=cax, orientation='horizontal')
	
	plt.show()


def main():
	test_load()
	
	return 0


if __name__ == '__main__':
	main()
