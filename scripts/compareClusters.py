#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  compareClusters.py
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

# TODO: Exclude centers of clusters

import numpy as np
import scipy, scipy.stats, scipy.special
import h5py
import time

import argparse, sys, os

import matplotlib.pyplot as plt
import matplotlib as mplib
from matplotlib.colors import LinearSegmentedColormap
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
			t_ID = item.attrs['target_ID']
			EBV_cluster = item.attrs['EBV']
			if (t_name == target_name) or (t_ID == target_name):
				mags = item['mag'][:]
				errs = item['err'][:]
				EBV_SFD = item['EBV'][:]
				FeH = item.attrs['target_FeH']
				mu = item.attrs['target_DM']
				pix_idx = int(name.split()[1])
				return mags, errs, EBV_SFD, EBV_cluster, FeH, mu, t_name, t_ID, pix_idx
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
	parser = argparse.ArgumentParser(prog='compareClusters.py',
	                                 description='Compare cluster photometry to stellar templates.',
	                                 add_help=True)
	parser.add_argument('--templates', '-t', type=str, required=True,
	                    help='Stellar templates (in ASCII format).')
	parser.add_argument('--photometry', '-ph', type=str, nargs='+', required=True,
	                    help='Bayestar input file(s) with photometry.')
	parser.add_argument('--evidences', '-ev', type=str, nargs='+', default=None,
	                    help='Bayestar output file(s) with evidences.')
	parser.add_argument('--clusters', '-c', type=str, nargs='+', required=True,
	                    help='Cluster names.')
	parser.add_argument('--output', '-o', type=str, default=None, help='Plot filename.')
	parser.add_argument('--show', '-sh', action='store_true', help='Show plot.')
	
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	args = parser.parse_args(sys.argv[offset:])
	
	n_clusters = len(args.clusters)
	
	templates = TStellarModel(args.templates)
	colors = ['gr', 'ri', 'iz', 'zy']
	n_colors = len(colors)
	
	# Set matplotlib style attributes
	mplib.rc('text', usetex=True)
	mplib.rc('xtick.major', size=6)
	mplib.rc('xtick.minor', size=4)
	mplib.rc('ytick.major', size=6)
	mplib.rc('ytick.minor', size=4)
	mplib.rc('xtick', direction='in')
	mplib.rc('ytick', direction='in')
	mplib.rc('axes', grid=False)
	
	# Create color scale
	cdict = {'red':   ((0., 1., 1.),
	                   (1., 0., 0.)),
	         
	         'green': ((0., 0., 0.),
	                   (1., 0., 0.)),
	         
	         'blue':  ((0., 0., 0.),
	                   (1., 1., 1.))}
	br_cmap = LinearSegmentedColormap('br1', cdict)
	
	# Set up figure
	fig = plt.figure(figsize=(1. + 2.5*n_colors, 2.*n_clusters), dpi=150)
	axgrid = Grid(fig, 111,
	              nrows_ncols=(n_clusters, n_colors),
	              axes_pad=0.05,
	              add_all=True,
	              label_mode='L')
	
	color_min = np.empty(len(colors), dtype='f8')
	color_max = np.empty(len(colors), dtype='f8')
	color_min[:] = np.inf
	color_max[:] = -np.inf
	
	lnZ_max = 0.
	Delta_lnZ = 10.
	
	for cluster_num, cluster_name in enumerate(args.clusters):
		# Load photometry
		ret = None
		for fname in args.photometry:
			ret = read_photometry(fname, cluster_name)
			if ret != None:
				break
		if ret == None:
			print 'Cluster "%s" not found.' % cluster_name
			return 0
		mags, errs, EBV, EBV_cluster, FeH, mu, name, ID, pix_idx = ret
		if np.isfinite(EBV_cluster):
			print 'Using E(B-V) = %.3f' % EBV_cluster
			mags = dereddened_mags(mags, float(EBV_cluster))
		else:
			mags = dereddened_mags(mags, EBV)
		
		# Load evidences
		lnZ = np.zeros(len(mags), dtype='f8')
		if args.evidences != None:
			for fname in args.evidences:
				ret = read_evidences(fname, pix_idx)
				if ret != None:
					lnZ = ret[:]
					break
		
		print '== Cluster #%d ==' % (cluster_num + 1)
		print '  ID: %s' % (ID)
		print '  name: %s' % (name)
		print '  FeH: %.2f' % (FeH)
		print '  DM: %.2f' % (mu)
		print '  E(B-V): %.2f' % (np.percentile(EBV, 95.))
		print '  # of stars: %d' % (len(mags))
		txt = ''
		
		isochrone = templates.get_isochrone(FeH)
		
		Mr_min, Mr_max = np.inf, -np.inf
		
		# Plot color-Magnitude diagrams
		for i,c in enumerate(colors):
			ax = axgrid[n_colors*cluster_num + i]
			
			# Empirical
			b1, b2 = i, i+1
			idx = ( (mags[:,b1] > 10.) & (mags[:,b1] < 25.)
			      & (mags[:,b2] > 10.) & (mags[:,b2] < 25.) )
			mags_tmp = mags[idx]
			lnZ_tmp = lnZ[idx]
			#idx = (  (mags_tmp[:,b1] > np.percentile(mags_tmp[:,b1], 1.))
			#       & (mags_tmp[:,b1] < np.percentile(mags_tmp[:,b1], 99.))
			#       & (mags_tmp[:,b2] > np.percentile(mags_tmp[:,b2], 1.))
			#       & (mags_tmp[:,b2] < np.percentile(mags_tmp[:,b2], 99.)) )
			#mags_tmp = mags_tmp[idx]
			#lnZ_tmp = lnZ_tmp[idx]
			Mr = mags_tmp[:,1] - mu
			color = mags_tmp[:,b1] - mags_tmp[:,b2]
			idx = (Mr > -5.)
			
			#lnZ_max = np.percentile(lnZ_tmp[idx], 97.)
			
			ax.scatter(color[idx], Mr[idx],
			           c=lnZ_tmp[idx], cmap=br_cmap,
			           vmin=lnZ_max-Delta_lnZ, vmax=lnZ_max,
			           s=3., alpha=0.75/np.log(np.sum(idx)), edgecolor='none')
			
			xlim = ax.get_xlim()
			ylim = ax.get_ylim()
			
			# Model
			ax.plot(isochrone[c], isochrone['Mr'], 'g-', lw=2, alpha=0.5)
			
			#print Mr[idx]
			Mr_min_tmp, Mr_max_tmp = np.percentile(Mr[idx], [1., 99.])
			if Mr_min_tmp < Mr_min:
				Mr_min = Mr_min_tmp
			if Mr_max_tmp > Mr_max:
				Mr_max = Mr_max_tmp
			
			lower, upper = 0.5, 98.
			if c == 'ri':
				lower = 1.5
			if c in ['iz', 'zy']:
				lower, upper = 10., 90.
			color_min_tmp, color_max_tmp = np.percentile(color[idx], [lower, upper])
			if color_min_tmp < color_min[i]:
				color_min[i] = color_min_tmp
			if color_max_tmp > color_max[i]:
				color_max[i] = color_max_tmp
			
			txt += '  %s: %d stars\n' % (c, np.sum(idx))
		
		print txt
		print ''
		
		# Reddening vectors
		R = get_reddening_vector()
		EBV_0 = np.median(EBV)
		for i in range(len(colors)):
			A_r = EBV_0 * R[1]
			A_xy = EBV_0 * (R[i] - R[i+1])
			r_0 = Mr_min - 0.5
			w = color_max[i] - color_min[i]
			h = Mr_max - Mr_min
			xy_0 = color_min[i] + 0.05 * w
			axgrid[n_colors*cluster_num + i].arrow(xy_0, r_0, A_xy, A_r,
			                                       head_width=0.03*w, head_length=0.01*h,
			                                       color='r', alpha=0.5)
			
		
		# Format y-axes
		axgrid[n_colors*cluster_num].set_ylim(Mr_max+0.5, Mr_min-1.0)
		
		cluster_label = ''
		if len(name) == 0:
			cluster_label = ID
		else:
			cluster_label = name
		cluster_label = cluster_label.replace(' ', '\ ')
		axgrid[n_colors*cluster_num].set_ylabel('$\mathrm{%s}$\n$M_{r}$' % cluster_label,
		                                        fontsize=16,
		                                        multialignment='center')
		axgrid[n_colors*cluster_num].yaxis.set_major_locator(MaxNLocator(nbins=5))
		axgrid[n_colors*cluster_num].yaxis.set_minor_locator(AutoMinorLocator())
	
	# Format x-axes
	for i,c in enumerate(colors):
		axgrid[i].set_xlim(color_min[i], color_max[i])
		
		color_label = r'$%s - %s$' % (c[0], c[1])
		axgrid[n_colors*(n_clusters-1) + i].set_xlabel(color_label, fontsize=16)
		axgrid[n_colors*(n_clusters-1) + i].xaxis.set_major_locator(MaxNLocator(nbins=4))
		axgrid[n_colors*(n_clusters-1) + i].xaxis.set_minor_locator(AutoMinorLocator())
	
	fig.subplots_adjust(left=0.12, right=0.85, top=0.98, bottom=0.10)
	
	cax = fig.add_axes([0.87, 0.10, 0.03, 0.88])
	norm = mplib.colors.Normalize(vmin=lnZ_max-Delta_lnZ, vmax=lnZ_max)
	mappable = mplib.cm.ScalarMappable(cmap=br_cmap, norm=norm)
	mappable.set_array(np.array([lnZ_max-Delta_lnZ, lnZ_max]))
	fig.colorbar(mappable, cax=cax, ticks=[0., -2., -4., -6., -8., -10.])
	
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

