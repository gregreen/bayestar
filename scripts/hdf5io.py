#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  hdf5io.py
#  
#  I/O for files used by Bayestar, including Markov Chains and stellar
#  template libraries.
#  
#  This file is part of Bayestar.
#  Copyright 2012 Greg Green <greg@greg-UX31A>
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
import scipy.stats
import h5py

import kerneldensity


########################################################################
# Gridded Synthetic Photometry
########################################################################
class TSyntheticPhotometry:
	
	def __init__(self, fname, **kwargs):
		self.load(fname, **kwargs)
	
	def load(self, fname, dset='PARSEC PS1 Templates', dim='Dimensions'):
		f = h5py.File(fname, 'r')
		
		data = f[dset]
		self.data = np.empty(data.shape, dtype=data.dtype)
		self.data[:] = data[:]
		
		dim = f[dim]
		self.dim = np.empty(dim.shape, dtype=dim.dtype)
		self.dim[:] = dim[:]
		
		f.close()
		
		self.unique = {}
		self.unique['logMass_init'] = np.unique(self.data['logMass_init'])
		self.unique['logtau'] = np.unique(self.data['logtau'])
		self.unique['Z'] = np.unique(self.data['Z'])
		self.logZ_Sun = np.log10(0.0122)
		self.unique['logZ'] = np.log10(np.unique(self.data['Z'])) - self.logZ_Sun
	
	def save(self, fname):
		f = h5py.File(fname, 'w')
		ds = f.create_dataset('PARSEC PS1 Templates',
		                      self.data.shape,
		                      self.data.dtype)
		ds[:] = self.data[:]
		
		dtype = [('N_Z', 'i4'),
				 ('N_logtau', 'i4'),
				 ('N_logMass_init', 'i4'),
				 ('Z_min', 'f4'),
				 ('Z_max', 'f4'),
				 ('logtau_min', 'f4'),
				 ('logtau_max', 'f4'),
				 ('logMass_init_min', 'f4'),
				 ('logMass_init_max', 'f4')]
		dim = np.empty(1, dtype=dtype)
		for name in ['Z', 'logtau', 'logMass_init']:
			x = np.unique(self.data[name])
			dim['N_%s' % name] = x.size
			dim['%s_min' % name] = np.min(x)
			dim['%s_max' % name] = np.max(x)
		ds = f.create_dataset('Dimensions', dim.shape, dim.dtype)
		ds[:] = dim[:]
	
	def sort(self, order=['logMass_init', 'logtau', 'Z']):
		self.data.sort(order=order)
	
	def get_unique(self, field):
		return self.unique(field)
	
	def isosurface(self, **kwargs):
		field, value = [], []
		for key in kwargs:
			if key in ['logtau', 'logMass_init', 'Z']:
				field.append(key)
				value.append(kwargs[key])
			else:
				raise ValueError('Unrecognized field: %s' % key)
		if len(field) == 0:
			raise ValueError('No fields specified.')
		
		idx = None
		for f,v in zip(field, value):
			nearest = self.unique[f][np.argmin(np.abs(self.unique[f] - v))]
			#print '%s = %.4f' % (f, nearest)
			if idx == None:
				idx = self.data[f] == nearest
			else:
				idx = idx & (self.data[f] == nearest)
		
		return self.data[idx]


########################################################################
# Markov Chain
########################################################################
class TChain:
	def __init__(self, fname=None, dset=None):
		f = None
		
		if fname != None:
			if type(dset) != str:
				raise TypeError("If 'fname' is provided, 'dset' must be "
				                "a string containing the name of the dataset.")
			f = h5py.File(fname, 'r')
			dset = f[dset]
		
		if dset == None:
			raise ValueError('A dataset name or object must be provided.')
		
		self.load(dset)
		
		if f != None:
			f.close()
	
	def load(self, dset):
		self.nChains, self.nSamples, self.nDim = dset.shape
		self.nDim -= 1
		self.nSamples -= 1
		
		self.converged = dset.attrs['converged'][:]
		self.lnZ = dset.attrs['ln(Z)'][:]
		
		self.lnp = dset[:,1:,0]
		self.coords = dset[:,1:,1:]
		
		self.lnp_best = dset[0,1:,0]
		self.coords_best = dset[0,1:,1:]
		
		self.GR = dset[:,0,1:]
		
		self.lnp_max = np.max(self.lnp)
		self.x_min = np.min(self.coords, axis=1)
		self.x_max = np.max(self.coords, axis=1)
	
	def get_samples(self, chainIdx=None):
		if chainIdx == None:
			return self.coords
		else:
			return self.coords[chainIdx]
	
	def get_lnp(self, chainIdx=None):
		if chainIdx == None:
			return self.lnp
		else:
			return self.lnp[chainIdx]
	
	def get_convergence(self, chainIdx=None):
		if chainIdx == None:
			return self.converged
		else:
			return self.converged[chainIdx]
	
	def get_lnZ(self, chainIdx=None):
		if chainIdx == None:
			return self.lnZ
		else:
			return self.lnZ[chainIdx]
	
	def get_nChains(self):
		return self.nChains
	
	def get_nSamples(self):
		return self.nSamples
	
	def get_nDim(self):
		return self.nDim



########################################################################
# Probability surfaces
########################################################################
class TProbSurf:
	def __init__(self, fname=None, dset=None):
		f = None
		close = False
		
		if fname != None:
			if type(dset) != str:
				raise TypeError("If 'fname' is provided, 'dset' must be "
				                "a string containing the name of the dataset.")
			if type(fname) == h5py._hl.files.File:
				f = fname
			else:
				f = h5py.File(fname, 'r')
				close = True
			dset = f[dset]
		
		if dset == None:
			raise ValueError('A dataset name or object must be provided.')
		
		self.load(dset)
		
		if close:
			f.close()
	
	def load(self, dset):
		self.nImages = dset.shape[0]
		self.nPix = dset.shape[1:]
		
		self.x_min = dset.attrs['min'][::-1]
		self.x_max = dset.attrs['max'][::-1]
		self.p = np.swapaxes(dset[:,:,:], 1, 2)
		
		self.p_max = np.max(np.max(self.p, axis=2), axis=1)
	
	def get_p(self, imgIdx=None):
		if imgIdx == None:
			return self.p
		else:
			return self.p[imgIdx]
	
	def get_n_stars(self):
		return self.nImages



########################################################################
# Batch processing of Markov chains
########################################################################

def get_stellar_chains(fname):
	'''
	Load all the stellar Markov chains from an hdf5 file.
	'''
	
	chain = []
	index = []
	tmp_index = None
	
	f = h5py.File(fname, 'r')
	
	for name,item in f.iteritems():
		# Check if group name is of type 'star #'
		name_spl = name.split()
		if name_spl[0] == 'star':
			try:
				tmp_index = int(name_spl[1])
			except:
				continue
			# Look for 'chain' group
			for subname,subitem in item.iteritems():
				if subname == 'chain':
					chain.append(TChain(group=subitem))
					index.append(tmp_index)
					break
	
	f.close()
	
	ret = zip(index,chain)
	ret.sort()
	
	return ret

def get_los_chains(fname):
	'''
	
	'''
	
	pass



def grid_pdfs(chain, bounds, samples, axes=(0,1), subsample=None):
	'''
	Return a stack of probability density grids, evaluated at regular
	intervals in the given bounds.
	
	Inputs:
	    bounds   [(min, max), ...] - min/max in each dimension.
	    
	    samples  [n1, n2, ...] - number of samples to take along each
	                             axis.
	    
	    axes     List specifying which dimensions to use.
	
	Output:
	    pdf[nchain, dim1, dim2, ...]
	'''
	
	if (len(bounds) != 2) or (len(samples) != 2) or (len(axes) != 2):
		raise ValueError("'bounds', 'samples' and 'axes' must all be"
		                 "of length 2.")
	
	coords = []
	for (xmin,xmax),n in zip(bounds,samples):
		coords.append(np.linspace(xmin, xmax, n))
	X,Y = np.meshgrid(coords[0], coords[1])
	x = np.vstack([X.flatten(), Y.flatten()]).T
	#print x
	
	pdf = np.empty((len(chain), samples[0]*samples[1]), 'f8')
	
	for i,c in enumerate(chain):
		sample = c.get_element()['x'][:,axes]
		if subsample != None:
			sample = sample[:subsample]
		kde = kerneldensity.TKernelDensity(sample)
		pdf[i,:] = kde(x)
		del kde
	
	pdf.shape = (len(chain), samples[1], samples[0])
	
	return pdf.transpose([0,2,1])

def get_stellar_pdfs(fname, pixel):
	'''
	Load all the stellar pdfs from a pixel.
	'''
	
	pdf = []
	tmp_index = None
	
	group = 'pixel %d' % pixel
	
	f = h5py.File(fname, 'r')
	g = f[group]
	
	for name,item in g.iteritems():
		# Check if group name is of type 'star #'
		name_spl = name.split()
		if name_spl[0] == 'star':
			try:
				tmp_index = int(name_spl[1])
			except:
				continue
			# Look for 'DM_EBV' group
			for subname,subitem in item.iteritems():
				if subname == 'DM_EBV':
					pdf.append(subitem[:,:])
					break
	
	f.close()
	
	return pdf

########################################################################
# Gridded Empirical Photometry
########################################################################
class TEmpiricalPhotometry:
	
	def __init__(self, fname, **kwargs):
		self.load(fname, **kwargs)
	
	def load(self, fname, r_max=None):
		# Load in templates
		f = open(fname, 'r')
		row = []
		for l in f:
			line = l.rstrip().lstrip()
			if len(line) == 0:	# Empty line
				continue
			if line[0] == '#':	# Comment
				continue
			data = line.split()
			if len(data) < 6:
				print 'Error reading in stellar templates.'
				print 'The following line does not have the correct number of entries (6 expected):'
				print line
				return 0
			row.append([float(s) for s in data])
		f.close()
		
		template = np.array(row, dtype=np.float64)
		if r_max != None:
			idx = template[:,0] <= r_max
			template = template[idx]
		
		self.data = np.empty(len(template), dtype=[('FeH', 'f8'),
		                                           ('M_g', 'f8'),
		                                           ('M_r', 'f8'),
		                                           ('M_i', 'f8'),
		                                           ('M_z', 'f8'),
		                                           ('M_y', 'f8')])
		self.data['FeH'][:] = template[:,1]
		self.data['M_r'][:] = template[:,0]
		self.data['M_g'][:] = template[:,2] + self.data['M_r'] # (g-r) + r
		self.data['M_i'][:] = self.data['M_r'] - template[:,3] # r - (r-i)
		self.data['M_z'][:] = self.data['M_i'] - template[:,4] # i - (i-z)
		self.data['M_y'][:] = self.data['M_z'] - template[:,5] # z - (z-y)
		
		self.unique = {}
		self.unique['M_r'] = np.unique(self.data['M_r'])
		self.unique['FeH'] = np.unique(self.data['FeH'])
	
	def get_unique(self, field):
		return self.unique(field)
	
	def isosurface(self, **kwargs):
		field, value = [], []
		for key in kwargs:
			if key in ['M_r', 'FeH']:
				field.append(key)
				value.append(kwargs[key])
			else:
				raise ValueError('Unrecognized field: %s' % key)
		if len(field) == 0:
			raise ValueError('No fields specified.')
		
		idx = None
		for f,v in zip(field, value):
			nearest = self.unique[f][np.argmin(np.abs(self.unique[f] - v))]
			#print '%s = %.4f' % (f, nearest)
			if idx == None:
				idx = self.data[f] == nearest
			else:
				idx = idx & (self.data[f] == nearest)
		
		return self.data[idx]


def main():
	synth_fname = '../data/PS1templates_sorted.h5'
	
	emp_fname = ['../data/PScolors_old.dat',
	             '../data/PScolors_Doug.dat']#,
	             #'../data/PScolors.dat']
	emp_name = ['Tonry', 'Finkbeiner']#, 'Greg']
	
	emp_track = []
	for fn in emp_fname:
		emplib = TEmpiricalPhotometry(fn, r_max=15.)
		emp_track.append(emplib.isosurface(FeH=0.))
	
	synthlib = TSyntheticPhotometry(synth_fname)
	logtau = [8., 9., 10.]
	solar_track = []
	for lt in logtau:
		tmp = synthlib.isosurface(logtau=lt, Z=0.0152)
		idx = tmp['logMass_init'] < 0.95 * np.max(tmp['logMass_init'])
		solar_track.append(tmp[idx])
	
	import matplotlib.pyplot as plt
	import matplotlib as mplib
	from matplotlib.ticker import MaxNLocator, AutoMinorLocator
	
	# Set matplotlib defaults
	mplib.rc('text', usetex=True)
	mplib.rc('xtick.major', size=6)
	mplib.rc('xtick.minor', size=4)
	mplib.rc('axes', grid=True)
	
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	
	for lt,st in zip(logtau, solar_track):
		ax.plot(st['M_g'] - st['M_r'], st['M_r'], '-', ms=1.0,
		        label=(r'$\log_{10} \tau = %.1f$' % lt))
	
	for lab,et in zip(emp_name, emp_track):
		ax.plot(et['M_g'] - et['M_r'], et['M_r'], '-', ms=1.0, 
		        label=r"$\mathrm{Mario's \ templates \ (%s)}$" % lab)
	
	# New tracks
	new_fname = '/home/greg/Downloads/output389285580561.dat'
	data = np.loadtxt(new_fname, usecols=(7,8,9,10,11), dtype=[('M_g', 'f8'),
	                                                            ('M_r', 'f8'),
	                                                            ('M_i', 'f8'),
	                                                            ('M_z', 'f8'),
	                                                            ('M_y', 'f8')])
	print data['M_g']
	print data['M_r']
	#ax.scatter(data['M_g'] - data['M_r'], data['M_r'], c='k', s=1.0)
	
	ylim = ax.get_ylim()
	ax.set_ylim([ylim[1], ylim[0]])
	
	ax.legend(loc=3)
	
	ax.set_xlabel(r'$g - r$', fontsize=16)
	ax.set_ylabel(r'$r$', fontsize=16)
	ax.tick_params(axis='both', which='major', labelsize=16)
	ax.tick_params(axis='both', which='minor', labelsize=16)
	
	ax.grid(which='major', alpha=0.3)
	ax.grid(which='minor', alpha=0.1)
	ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
	ax.xaxis.set_minor_locator(AutoMinorLocator())
	ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
	ax.yaxis.set_minor_locator(AutoMinorLocator())
	
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	st = solar_track[1]
	#ax.scatter(st['M_g'] - st['M_r'], st['M_r'] - st['M_i'], c=st['logMass_init'])
	#ax.scatter(data['M_g'] - data['M_r'], data['M_r'] - data['M_i'], c='k')
	for lab,et,c in zip(emp_name, emp_track, ['b.','r.','c.']):
		ax.plot(et['M_g'] - et['M_r'], et['M_r'] - et['M_i'], c, ms=1.0, 
		        label=r"$\mathrm{Mario's \ templates \ (%s)}$" % lab)
	for lt,st in zip(logtau, solar_track):
		ax.plot(st['M_g'] - st['M_r'], st['M_r'] - st['M_i'], '-', ms=1.0,
		        label=(r'$\log_{10} \tau = %.1f$' % lt))
	ax.set_xlabel(r'$g - r$', fontsize=16)
	ax.set_ylabel(r'$r - i$', fontsize=16)
	ax.legend(loc='upper left')
	
	plt.show()
	
	
	return 0

if __name__ == '__main__':
	main()

