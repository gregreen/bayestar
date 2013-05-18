#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  compareColors.py
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
	parser = argparse.ArgumentParser(prog='compareColors.py',
	                                 description='Compare photometric colors to model colors.',
	                                 add_help=True)
	parser.add_argument('--templates', '-t', type=str, required=True,
	                    help='Stellar templates (in ASCII format).')
	parser.add_argument('--photometry', '-ph', type=str, nargs='+', required=True,
	                    help='Bayestar input file(s) with photometry.')
	parser.add_argument('--evidences', '-ev', type=str, nargs='+', default=None,
	                    help='Bayestar output file(s) with evidences.')
	parser.add_argument('--names', '-n', type=str, nargs='+', required=True,
	                    help='Region names.')
	parser.add_argument('--output', '-o', type=str, default=None, help='Plot filename.')
	parser.add_argument('--show', '-sh', action='store_true', help='Show plot.')
	
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	args = parser.parse_args(sys.argv[offset:])
	
	n_targets = len(args.names)
	
	templates = TStellarModel(args.templates)
	color_names = ['gr', 'ri', 'iz', 'zy']
	
	# Set matplotlib style attributes
	mplib.rc('text', usetex=True)
	mplib.rc('xtick.major', size=6)
	mplib.rc('xtick.minor', size=4)
	mplib.rc('ytick.major', size=6)
	mplib.rc('ytick.minor', size=4)
	mplib.rc('xtick', direction='in')
	mplib.rc('ytick', direction='in')
	mplib.rc('axes', grid=False)
	
	fig = plt.figure(figsize=(6., 2.*n_targets), dpi=150)
	axgrid = Grid(fig, 111,
	              nrows_ncols=(n_targets, 3),
	              axes_pad=0.05,
	              add_all=True,
	              label_mode='L')
	
	color_min = np.empty(len(color_names), dtype='f8')
	color_max = np.empty(len(color_names), dtype='f8')
	color_min[:] = np.inf
	color_max[:] = -np.inf
	
	for target_num, target_name in enumerate(args.names):
		# Load photometry
		ret = None
		for fname in args.photometry:
			ret = read_photometry(fname, target_name)
			if ret != None:
				break
		if ret == None:
			print 'Target "%s" not found.' % target_name
			return 0
		mags, errs, EBV, t_name, pix_idx = ret
		mags = dereddened_mags(mags, EBV)
		
		# Load evidences
		lnZ = np.zeros(len(mags), dtype='f8')
		if args.evidences != None:
			for fname in args.evidences:
				ret = read_evidences(fname, pix_idx)
				if ret != None:
					lnZ = ret[:]
					break
		
		print '== Target #%d ==' % target_num
		print '  name: %s' % (target_name)
		print '  E(B-V): %.2f' % (np.percentile(EBV, 95.))
		print '  # of stars: %d' % (len(mags))
		txt = ''
		
		idx = ( (mags[:,0] > 10.) & (mags[:,0] < 25.)
		      & (mags[:,1] > 10.) & (mags[:,1] < 25.) )
		mags = mags[idx]
		errs = errs[idx]
		lnZ = lnZ[idx]
		
		colors = -np.diff(mags, axis=1)
		
		idx = ( (colors[:,0] > np.percentile(colors[:,0], 0.5))
		      & (colors[:,0] < np.percentile(colors[:,0], 99.5)) )
		
		colors = colors[idx]
		mags = mags[idx]
		errs = errs[idx]
		lnZ = lnZ[idx]
		
		color_min[0], color_max[0] = np.percentile(colors[:,0], [1., 99.])
		
		# Plot color-Magnitude diagrams
		for i,c in enumerate(color_names[1:]):
			ax = axgrid[3*target_num + i]
			
			idx = ( (colors[:,i+1] > np.percentile(colors[:,i+1], 0.5))
			      & (colors[:,i+1] < np.percentile(colors[:,i+1], 99.5))
			      & (mags[:,i+1] > 10.) & (mags[:,i+1] < 25.)
			      & (mags[:,i+2] > 10.) & (mags[:,i+2] < 25.) )
			
			# Empirical
			lnZ_max = 0. #np.percentile(lnZ_tmp[idx], 97.)
			Delta_lnZ = 25.
			
			colors_tmp = colors[idx, i+1]
			
			ax.scatter(colors_tmp, colors[idx,0],
			           c=lnZ[idx], cmap='Spectral',
			           vmin=lnZ_max-Delta_lnZ, vmax=lnZ_max,
			           s=1.5, alpha=0.1, edgecolor='none')
			
			xlim = ax.get_xlim()
			ylim = ax.get_ylim()
			
			# Model
			for FeH, style in zip([0., -1., -2.], ['c-', 'y-', 'r-']):
				isochrone = templates.get_isochrone(FeH)
				ax.plot(isochrone[c], isochrone['gr'], style, lw=1, alpha=0.25)
				print isochrone[c]
				print isochrone['gr']
				print ''
			
			color_min_tmp, color_max_tmp = np.percentile(colors_tmp, [1., 99.])
			if color_min_tmp < color_min[i+1]:
				color_min[i+1] = color_min_tmp
			if color_max_tmp > color_max[i+1]:
				color_max[i+1] = color_max_tmp
			
			txt += '  %s: %d stars\n' % (c, np.sum(idx))
			txt += '      ln(Z_max) = %.2f\n' % lnZ_max
		
		print txt
		print ''
		
		# Reddening vectors
		'''R = get_reddening_vector()
		EBV_0 = np.median(EBV)
		for i in range(len(colors)):
			A_gr = EBV_0 * (R[0] - R[1])
			A_xy = EBV_0 * (R[i] - R[i+1])
			r_0 = Mr_min
			w = color_max[i] - color_min[i]
			h = Mr_max - Mr_min
			xy_0 = color_min[i] + 0.1 * w
			axgrid[4*cluster_num + i].arrow(xy_0, r_0, A_xy, A_r,
			                                head_width=0.03*w, head_length=0.01*h,
			                                color='r', alpha=0.5)
			
		'''
		
		# Format y-axes
		axgrid[3*target_num].set_ylim(color_min[0], color_max[1])
		
		axgrid[3*target_num].set_ylabel(r'$g - r$', fontsize=16)
		
		axgrid[3*target_num].yaxis.set_major_locator(MaxNLocator(nbins=5))
		axgrid[3*target_num].yaxis.set_minor_locator(AutoMinorLocator())
	
	# Format x-axes
	for i,c in enumerate(color_names[1:]):
		axgrid[i].set_xlim(color_min[i+1], color_max[i+1])
		
		color_label = r'$%s - %s$' % (c[0], c[1])
		axgrid[3*(n_targets-1) + i].set_xlabel(color_label, fontsize=16)
		axgrid[3*(n_targets-1) + i].xaxis.set_major_locator(MaxNLocator(nbins=4))
		axgrid[3*(n_targets-1) + i].xaxis.set_minor_locator(AutoMinorLocator())
	
	fig.subplots_adjust(left=0.12, right=0.85, top=0.98, bottom=0.10)
	
	cax = fig.add_axes([0.87, 0.10, 0.03, 0.88])
	norm = mplib.colors.Normalize(vmin=-25., vmax=0.)
	mappable = mplib.cm.ScalarMappable(cmap='Spectral', norm=norm)
	mappable.set_array(np.array([-25., 0.]))
	fig.colorbar(mappable, cax=cax, ticks=[0., -5., -10., -15., -20., -25.])
	
	cax.yaxis.set_label_position('right')
	cax.yaxis.tick_right()
	#cax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
	#cax.set_yticks([0., -5., -10., -15., -20., -25.])
	cax.set_ylabel(r'$\mathrm{ln} \left( Z \right)$', rotation='vertical', fontsize=16)
	
	if args.output != None:
		fig.savefig(args.output, dpi=300)
	
	if args.show:
		plt.show()
	
	return 0

if __name__ == '__main__':
	main()

