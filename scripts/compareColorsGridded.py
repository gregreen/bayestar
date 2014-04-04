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

def read_photometry(fname, pixel):
	f = h5py.File(fname, 'r')
	dset = f['/photometry/pixel %d-%d' % (pixel[0], pixel[1])]
	
	# Load in photometry from selected target
	l = dset.attrs['l']
	b = dset.attrs['b']
	
	mags = dset['mag'][:]
	errs = dset['err'][:]
	EBV_SFD = dset['EBV'][:]
	
	f.close()
	
	return mags, errs, EBV_SFD, l, b

def read_inferences(fname, pix_idx):
	f = h5py.File(fname, 'r')
	
	dtype = [('lnZ','f8'), ('conv',np.bool),
	         ('DM','f8'), ('EBV','f8'),
	         ('Mr','f8'), ('FeH','f8')]
	
	ret = None
	
	dset = '/pixel %d-%d/stellar chains' % (pix_idx[0], pix_idx[1])
	
	try:
		lnZ = f[dset].attrs['ln(Z)'][:]
		conv = f[dset].attrs['converged'][:].astype(np.bool)
		best = f[dset][:, 1, 1:]
		
		ret = np.empty(len(lnZ), dtype=dtype)
		ret['lnZ'] = lnZ
		ret['conv'] = conv
		ret['EBV'] = best[:,0]
		ret['DM'] = best[:,1]
		ret['Mr'] = best[:,2]
		ret['FeH'] = best[:,3]
	except:
		print 'Dataset "%s" does not exist.' % dset
	
	return ret

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


class KnotLogger:
	def __init__(self, ax, marker='+', c='r', s=4):
		self.ax = ax
		self.cid = ax.figure.canvas.mpl_connect('button_press_event', self)
		
		self.x = []
		self.y = []
		
		self.marker = marker
		self.c = c
		self.s = s
	
	def __call__(self, event):
		if event.inaxes != self.ax:
			return
		self.x.append(event.xdata)
		self.y.append(event.ydata)
		if self.marker != None:
			event.inaxes.scatter([event.xdata], [event.ydata],
			                     marker=self.marker, s=self.s, c=self.c)
			self.ax.figure.canvas.draw()
	
	def get_knots(self):
		return self.x, self.y


def main():
	parser = argparse.ArgumentParser(prog='compareColorsGridded.py',
	                                 description='Compare photometric colors to model colors.',
	                                 add_help=True)
	parser.add_argument('--templates', '-t', type=str, required=True,
	                    help='Stellar templates (in ASCII format).')
	parser.add_argument('--photometry', '-ph', type=str, required=True,
	                    help='Bayestar input file with photometry.')
	parser.add_argument('--bayestar', '-bayes', type=str, default=None,
	                    help='Bayestar output file with inferences.')
	parser.add_argument('--color', '-c', type=str, default='lnZ',
	                    choices=('lnZ', 'conv', 'EBV', 'DM', 'Mr', 'FeH'),
	                    help='Field by which to color stars.')
	#parser.add_argument('--name', '-n', type=str, required=True,
	#                    help='Region name.')
	parser.add_argument('--pixel', '-pix', type=int, nargs=2, required=True,
	                    help='HEALPix nside and pixel index.')
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
	Delta_lnZ = 15.
	
	# Load photometry
	ret = read_photometry(args.photometry, args.pixel)
	if ret == None:
		print 'Pixel not found.'
		return 0
	mags, errs, EBV, l, b = ret
	mags = dereddened_mags(mags, EBV)
	colors = -np.diff(mags, axis=1)
	
	#print '  name: %s' % (args.name)
	print '  (l, b): %.2f, %.2f' % (l, b)
	print '  E(B-V)_SFD: %.4f' % (np.percentile(EBV, 95.))
	print '  # of stars: %d' % (len(mags))
	
	# Load bayestar inferences
	params = None
	vmin, vmax = None, None
	
	if args.bayestar != None:
                params = read_inferences(args.bayestar, args.pixel)
		idx = np.isfinite(params['lnZ'])
		
		n_rejected = np.sum(params['lnZ'] < np.percentile(params['lnZ'][idx], 95.) - 15.)
		pct_rejected = 100. * float(n_rejected) / np.float(len(params))
		n_nonconv = np.sum(~params['conv'])
		pct_nonconv = 100. * float(n_nonconv) / float(len(params))
		
		if args.color != None:
			vmin, vmax = np.percentile(params[args.color], [2., 98.])
		
		print vmin, vmax
		
		print '  # rejected: %d (%.2f %%)' % (n_rejected, pct_rejected)
		print '  # nonconverged: %d (%.2f %%)' % (n_nonconv, pct_nonconv)
		print '  ln(Z_max): %.2f' % (np.max(params['lnZ'][idx]))
		print '  ln(Z_95): %.2f' % (np.percentile(params['lnZ'][idx], 95.))
	
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
	              axes_pad=0.0,
	              add_all=False,
	              label_mode='L')
	
	cdict = {'red':   ((0., 1., 1.),
	                   (1., 0., 0.)),
	         
	         'green': ((0., 0., 0.),
	                   (1., 0., 0.)),
	         
	         'blue':  ((0., 0., 0.),
	                   (1., 1., 1.))}
	br_cmap = LinearSegmentedColormap('br1', cdict)
	#plt.register_cmap(br_cmap)
	
	logger = []
	cbar_ret = None
	
	# Grid of axes
	for row in xrange(3):
		color_y = colors[:,row+1]
		
		for col in xrange(row+1):
			color_x = colors[:,col]
			idx_xy = idx[col] & idx[row+1]
			
			#print colors.shape
			#print color_x.shape
			#print idx_xy.shape
			
			ax = axgrid[3*row + col]
			fig.add_axes(ax)
			
			logger.append(KnotLogger(ax, s=25))
			
			# Empirical
			c = 'k'
			if (params != None) and (args.color != None):
				#print params[args.color].shape
				c = params[args.color][idx_xy]
			
			cbar_ret = ax.scatter(color_x[idx_xy], color_y[idx_xy],
			                      c=c, #cmap=br_cmap,
			                      vmin=vmin, vmax=vmax,
			                      s=3., alpha=0.30, edgecolor='none')
			
			#idx_rej = lnZ < lnZ_max - Delta_lnZ
			#idx_tmp = idx_xy & ~idx_rej
			#ax.scatter(color_x[idx_tmp], color_y[idx_tmp],
			#           c='b', s=1.5, alpha=0.15, edgecolor='none')
			#idx_tmp = idx_xy & idx_rej
			#ax.scatter(color_x[idx_tmp], color_y[idx_tmp],
			#           c='r', s=1.5, alpha=0.15, edgecolor='none')
			
			# Model
			cx, cy = color_names[col], color_names[row+1]
			for FeH in np.linspace(-2.5, 0., 30):
				isochrone = templates.get_isochrone(FeH)
				ax.plot(isochrone[cx], isochrone[cy], 'k-', lw=1., alpha=0.03)
			
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
	x_0 = 0.10
	y_0 = 0.10
	x_1 = 0.98
	y_1 = 0.98
	
	fig.subplots_adjust(bottom=y_0, top=y_1, left=x_0, right=x_1) #right=0.85)
	
	w = (x_1 - x_0) / 3.
	h = (y_1 - y_0) / 3.
	
	cx = x_0 + 2.3*w
	cy = y_0 + h + 0.02
	cw = 0.05
	ch = 2. * h - 0.02
	
	if (params != None) and (args.color != None):
		cax = fig.add_axes([cx, cy, cw, ch])
		norm = mplib.colors.Normalize(vmin=vmax, vmax=vmax)
		mappable = mplib.cm.ScalarMappable(norm=norm)#cmap=br_cmap, norm=norm)
		mappable.set_array(np.array([vmin, vmax]))
		fig.colorbar(cbar_ret, cax=cax)#, ticks=[0., -3., -6., -9., -12., -15.])
	
	cax.yaxis.set_label_position('right')
	cax.yaxis.tick_right()
	cax.set_ylabel(r'$\mathrm{ln} \left( Z \right)$', rotation='vertical', fontsize=16)
	
	# Information on l.o.s.
	txt = '$\ell = %.2f^{\circ}$\n' % l
	txt += '$b = %.2f^{\circ}$\n' % b
	txt += '$\mathrm{E} \! \left( B \! - \! V \\right) = %.3f$' % (np.median(EBV))
	fig.text(x_0 + 1.1*w, y_0 + 2.5*h, txt, fontsize=14, ha='left', va='center')
	
	if args.output != None:
		fig.savefig(args.output, dpi=300)
	
	if args.show:
		plt.show()
	
	for i,log in enumerate(logger):
		x, y = log.get_knots()
		if len(x) != 0:
			print ''
			print 'Axis %d:' % (i + 1)
			print x
			print y
	
	return 0

if __name__ == '__main__':
	main()

