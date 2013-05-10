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
			if (t_name == target_name) or (t_ID == target_name):
				mags = item['mag'][:]
				errs = item['err'][:]
				EBV_SFD = item['EBV'][:]
				FeH = item.attrs['target_FeH']
				mu = item.attrs['target_DM']
				return mags, errs, EBV_SFD, FeH, mu, t_name, t_ID
	return None

def dereddened_mags(mags, EBV):
	R = np.array([3.172, 2.271, 1.682, 1.322, 1.087])
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
	                                    help='Bayestar input file with photometry.')
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
	
	# Set matplotlib style attributes
	mplib.rc('text', usetex=True)
	mplib.rc('xtick.major', size=6)
	mplib.rc('xtick.minor', size=4)
	mplib.rc('ytick.major', size=6)
	mplib.rc('ytick.minor', size=4)
	mplib.rc('xtick', direction='out')
	mplib.rc('ytick', direction='out')
	mplib.rc('axes', grid=False)
	
	fig = plt.figure(figsize=(8,3*n_clusters), dpi=150)
	
	for cluster_num, cluster_name in enumerate(args.clusters):
		ret = None
		for fname in args.photometry:
			ret = read_photometry(fname, cluster_name)
			if ret != None:
				break
		if ret == None:
			print 'Cluster "%s" not found.' % cluster_name
			return 0
		mags, errs, EBV, FeH, mu, name, ID = ret
		mags = dereddened_mags(mags, EBV)
		
		print '== Cluster information =='
		print '  ID: %s' % (ID)
		print '  name: %s' % (name)
		print '  FeH: %.2f' % (FeH)
		print '  DM: %.2f' % (mu)
		print '  E(B-V): %.2f' % (np.percentile(EBV, 95.))
		print '  # of stars: %d' % (len(mags))
		txt = ''
		
		isochrone = templates.get_isochrone(FeH)
		
		for i,c in enumerate(colors):
			ax = fig.add_subplot(n_clusters, 4, 4*cluster_num + i + 1)
			
			b1, b2 = i, i+1
			idx = (mags[:,b1] > 10.) & (mags[:,b1] < 25.) & (mags[:,b2] > 10.) & (mags[:,b2] < 25.)
			mags_tmp = mags[idx]
			idx = (  (mags_tmp[:,b1] > np.percentile(mags_tmp[:,b1], 2.))
			       & (mags_tmp[:,b1] < np.percentile(mags_tmp[:,b1], 98.))
			       & (mags_tmp[:,b2] > np.percentile(mags_tmp[:,b2], 2.))
			       & (mags_tmp[:,b2] < np.percentile(mags_tmp[:,b2], 98.)) )
			mags_tmp = mags_tmp[idx]
			Mr = mags_tmp[:,1] - mu
			color = mags_tmp[:,b1] - mags_tmp[:,b2]
			idx = ((color > np.percentile(color, 2.))
			        & (color < np.percentile(color, 98.))
			        & (Mr > -5.))
			ax.scatter(color[idx], Mr[idx], c='b', s=1.5, alpha=0.8, edgecolor='none')
			
			xlim = ax.get_xlim()
			ylim = ax.get_ylim()
			
			ax.plot(isochrone[c], isochrone['Mr'], 'g-', lw=2, alpha=0.5)
			
			ax.set_xlim(xlim)
			ax.set_ylim(ylim[1], ylim[0])
			
			txt += '  %s: %d' % (c, np.sum(idx))
		
		print txt
		print ''
	
	fig.subplots_adjust(left=0.10, right=0.95, top=0.95, bottom=0.15)
	
	if args.output != None:
		fig.savefig(args.output, dpi=300)
	
	if args.show:
		plt.show()
	
	return 0

if __name__ == '__main__':
	main()

