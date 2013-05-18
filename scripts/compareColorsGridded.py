#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  compareColorsGridded.py
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
import scipy, scipy.stats, scipy.special
import h5py
import time

import argparse, sys, os

import matplotlib.pyplot as plt
import matplotlib as mplib
from mpl_toolkits.axes_grid1 import Grid
from matplotlib.ticker import MaxNLocator, AutoMinorLocator, FormatStrFormatter

import hdf5io


class TStellarModel:
	def __init__(self, fname):
		self.load(fname)
	
	def load(self, fname):
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
				txt = 'Error reading in stellar templates.\n'
				txt += 'The following line does not have the correct number of entries (6 expected):\n'
				txt += line
				raise Exception(txt)
			row.append([float(s) for s in data])
		f.close()
		template = np.array(row, dtype='f8')
		
		fields = ['Mr', 'FeH', 'gr', 'ri', 'iz', 'zy']
		dtype = [(field, 'f8') for field in fields]
		self.data = np.empty(len(template), dtype=dtype)
		for i,field in enumerate(fields):
			self.data[field][:] = template[:,i]
		
		self.FeH = np.unique(self.data['FeH'])
	
	def get_isochrone(self, FeH):
		if FeH >= np.max(self.FeH):
			FeH_eval = np.max(self.FeH)
			idx = (self.data['FeH'] == FeH_eval)
			return self.data[idx]
		elif FeH <= np.min(self.FeH):
			FeH_eval = np.min(self.FeH)
			idx = (self.data['FeH'] == FeH_eval)
			return self.data[idx]
		else:
			k = np.arange(self.FeH.size)
			#print np.where(FeH > self.FeH, k, -1)
			#print self.FeH
			idx = np.max(np.where(FeH > self.FeH, k, -1))
			FeH_eval = [self.FeH[idx], self.FeH[idx+1]]
			a = float(FeH_eval[1] - FeH) / float(FeH_eval[1] - FeH_eval[0])
			idx = (self.data['FeH'] == FeH_eval[0])
			d1 = self.data[idx]
			idx = (self.data['FeH'] == FeH_eval[1])
			d2 = self.data[idx]
			
			if np.any(d1['Mr'] != d2['Mr']):
				raise Exception('Expected Mr samples to be the same for each metallicity.')
			
			fields = ['Mr', 'FeH', 'gr', 'ri', 'iz', 'zy']
			dtype = [(field, 'f8') for field in fields]
			ret = np.empty(len(d1), dtype=dtype)
			
			#print FeH_eval
			#print a
			
			for field in fields:
				ret[field][:] = a * d1[field][:] + (1. - a) * d2[field][:]
			
			return ret

def read_photometry(fname, target_name):
	f = h5py.File(fname, 'r')
	
	# Hack to get the file to read properly
	#try:
	#	f.items()
	#except:
	#	pass
	
	phot = f['photometry']
	
	# Load in photometry from selected target
	for name,item in phot.iteritems():
		if 'pixel' in name:
			t_name = item.attrs['target_name']
			if t_name == target_name:
				mags = item['mag'][:]
				errs = item['err'][:]
				EBV_SFD = item['EBV'][:]
				pix_idx = int(name.split()[1])
				return mags, errs, EBV_SFD, t_name, pix_idx
	return None

def read_evidences(fname, pix_idx):
	f = h5py.File(fname, 'r')
	
	lnZ = None
	
	dset = '/pixel %d/stellar chains' % pix_idx
	
	try:
		lnZ = f[dset].attrs['ln(Z)'][:]
	except:
		print 'Dataset "%s" does not exist.' % dset
	
	return lnZ

def get_reddening_vector():
	return np.array([3.172, 2.271, 1.682, 1.322, 1.087])

def dereddened_mags(mags, EBV):
	R = get_reddening_vector()
	if type(EBV) == float:
		R.shape = (1, R.size)
		R = np.repeat(R, len(mags), axis=0)
		return mags - EBV * R
	elif type(EBV) == np.ndarray:
		return mags - np.einsum('i,j->ij', EBV, R)
	else:
		raise TypeError('EBV has unexpected type: %s' % type(EBV))

def plot_cluster(ax, template, mags):
	pass

def main():
	parser = argparse.ArgumentParser(prog='compareColorsGridded.py',
	                                 description='Compare photometric colors to model colors.',
	                                 add_help=True)
	parser.add_argument('--templates', '-t', type=str, required=True,
	                    help='Stellar templates (in ASCII format).')
	parser.add_argument('--photometry', '-ph', type=str, required=True,
	                    help='Bayestar input file with photometry.')
	parser.add_argument('--evidences', '-ev', type=str, default=None,
	                    help='Bayestar output file with evidences.')
	parser.add_argument('--name', '-n', type=str, required=True,
	                    help='Region name.')
	parser.add_argument('--output', '-o', type=str, default=None, help='Plot filename.')
	parser.add_argument('--show', '-sh', action='store_true', help='Show plot.')
	
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	args = parser.parse_args(sys.argv[offset:])
	
	templates = TStellarModel(args.templates)
	color_names = ['gr', 'ri', 'iz', 'zy']
	
	lnZ_max = 0.
	Delta_lnZ = 25.
	
	# Load photometry
	ret = read_photometry(args.photometry, args.name)
	if ret == None:
		print 'Target "%s" not found.' % args.name
		return 0
	mags, errs, EBV, t_name, pix_idx = ret
	mags = dereddened_mags(mags, EBV)
	colors = -np.diff(mags, axis=1)
	
	# Load evidences
	lnZ = np.zeros(len(mags), dtype='f8')
	if args.evidences != None:
		ret = read_evidences(args.evidences, pix_idx)
		if ret != None:
			lnZ = ret[:]
	
	print '  name: %s' % (args.name)
	print '  E(B-V): %.4f' % (np.percentile(EBV, 95.))
	print '  # of stars: %d' % (len(mags))
	
	# Compute mask for each color
	idx = []
	for i in xrange(4):
		idx.append( (mags[:,i] > 10.) & (mags[:,i] < 28.)
		          & (mags[:,i+1] > 10.) & (mags[:,i+1] < 28.) )
	
	# Compute display limits for each color
	lim = np.empty((4,2), dtype='f8')
	for i in xrange(4):
		lim[i,0], lim[i,1] = np.percentile(colors[idx[i],i], [2., 98.])
	w = np.reshape(np.diff(lim, axis=1), (4,))
	lim[:,0] -= 0.15 * w
	lim[:,1] += 0.15 * w
	
	lim_bounds = np.array([[-0.2, 1.6],
	                       [-0.3, 2.0],
	                       [-0.2, 1.1],
	                       [-0.15, 0.45]])
	for i in xrange(4):
		lim[i,0] = max(lim[i,0], lim_bounds[i,0])
		lim[i,1] = min(lim[i,1], lim_bounds[i,1])
	
	# Set matplotlib style attributes
	mplib.rc('text', usetex=True)
	mplib.rc('xtick.major', size=6)
	mplib.rc('xtick.minor', size=4)
	mplib.rc('ytick.major', size=6)
	mplib.rc('ytick.minor', size=4)
	mplib.rc('xtick', direction='in')
	mplib.rc('ytick', direction='in')
	mplib.rc('axes', grid=False)
	
	# Set up figure
	fig = plt.figure(figsize=(6,6), dpi=150)
	axgrid = Grid(fig, 111,
	              nrows_ncols=(3,3),
	              axes_pad=0.05,
	              add_all=True,
	              label_mode='L')
	
	# Grid of axes
	for row in xrange(3):
		color_y = colors[:,row+1]
		
		for col in xrange(row+1):
			color_x = colors[:,col]
			idx_xy = idx[col] & idx[row+1]
			
			ax = axgrid[3*row + col]
			
			# Empirical
			ax.scatter(color_x[idx_xy], color_y[idx_xy],
			           c=lnZ[idx_xy], cmap='Spectral',
			           vmin=lnZ_max-Delta_lnZ, vmax=lnZ_max,
			           s=1.5, alpha=0.1, edgecolor='none')
			
			# Model
			cx, cy = color_names[col], color_names[row+1]
			for FeH, style in zip([0., -1., -2.], ['c-', 'y-', 'r-']):
				isochrone = templates.get_isochrone(FeH)
				ax.plot(isochrone[cx], isochrone[cy], style, lw=1, alpha=0.25)
			
			ax.set_xlim(lim[col])
			ax.set_ylim(lim[row+1])
	
	# Format x-axes
	for i,c in enumerate(color_names[:-1]):
		color_label = r'$%s - %s$' % (c[0], c[1])
		ax = axgrid[6+i]
		ax.set_xlabel(color_label, fontsize=16)
		ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
		ax.xaxis.set_minor_locator(AutoMinorLocator())
	
	# Format y-axes
	for i,c in enumerate(color_names[1:]):
		color_label = r'$%s - %s$' % (c[0], c[1])
		ax = axgrid[3*i]
		ax.set_ylabel(color_label, fontsize=16)
		ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
		ax.yaxis.set_minor_locator(AutoMinorLocator())
	
	# Colorbar
	fig.subplots_adjust(left=0.12, right=0.85, top=0.98, bottom=0.10)
	
	cax = fig.add_axes([0.87, 0.10, 0.03, 0.88])
	norm = mplib.colors.Normalize(vmin=-25., vmax=0.)
	mappable = mplib.cm.ScalarMappable(cmap='Spectral', norm=norm)
	mappable.set_array(np.array([-25., 0.]))
	fig.colorbar(mappable, cax=cax, ticks=[0., -5., -10., -15., -20., -25.])
	
	cax.yaxis.set_label_position('right')
	cax.yaxis.tick_right()
	cax.set_ylabel(r'$\mathrm{ln} \left( Z \right)$', rotation='vertical', fontsize=16)
	
	if args.output != None:
		fig.savefig(args.output, dpi=300)
	
	if args.show:
		plt.show()
	
	return 0

if __name__ == '__main__':
	main()

