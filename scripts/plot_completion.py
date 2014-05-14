#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  plot_completion.py
#  
#  Copyright 2013-2014 Greg Green <gregorymgreen@gmail.com>
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
mplib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.colors import ListedColorMap, BoundaryNorm

import argparse, sys, time, glob
from os.path import expanduser, abspath
import os.path

import healpy as hp
import h5py

import hputils, maptools


pallette = {'orange': (0.9, 0.6, 0.),
            'sky blue': (0.35, 0.70, 0.90),
            'bluish green': (0., 0.6, 0.5),
            'yellow': (0.95, 0.9, 0.25),
            'blue': (0., 0.45, 0.7),
            'vermillion': (0.8, 0.4, 0.),
            'reddish purple': (0.8, 0.6, 0.7)}


class PixelIdentifier:
	'''
	Class that prints out the HEALPix pixel index when the user
	clicks somewhere in a figure.
	'''
	
	def __init__(self, ax, nside,
	                   nest=True,
	                   proj=hputils.Cartesian_projection()):
		self.ax = ax
		self.cid = ax.figure.canvas.mpl_connect('button_press_event', self)
		
		self.nside = nside
		self.nest = nest
		
		self.proj = proj
	
	def __call__(self, event):
		if event.inaxes != self.ax:
			return
		
		# Determine healpix index of point
		x, y = event.xdata, event.ydata
		b, l = self.proj.inv(x, y)
		pix_idx = hputils.lb2pix(self.nside, l, b, nest=self.nest)
		
		print '(%.2f, %.2f) -> %d' % (l, b, pix_idx)


class TCompletion:
	def __init__(self, indir, outdir):
		# Initialize input and output filenames
		self.infiles = glob.glob(os.path.normpath(indir + '/*.h5'))
		self.basenames = [os.path.basename(fname) for fname in self.infiles]
		self.outfiles = [outdir + '/' + fname for fname in self.basenames]
		
		# Get list of pixels
		self.nside = []
		self.pix_idx = []
		self.n_stars = []
		
		for fname in self.infiles:
			f = h5py.File(fname, 'r')
			
			for _, pixel in f['/photometry'].iteritems():
				self.nside.append(pixel.attrs['nside'])
				self.pix_idx.append(pixel.attrs['healpix_index'])
				self.n_stars.append(pixel.size)
			
			f.close()
		
		self.nside = np.array(self.nside)
		self.pix_idx = np.array(self.pix_idx)
		self.n_stars = np.array(self.n_stars)
		self.n_stars_tot = np.sum(n_stars)
		
		# Initialize output file modification times
		self.modtime = np.empty(len(self.outfiles), dtype='f8')
		self.modtime[:] = -1.
		#self.modtime = dict.fromkeys(self.outfiles, -1)
		
		# Initialize pixel statuses
		self.pix_name = ['%d-%d' % (n, i) for n, i in zip(self.nside, self.pix_idx)]
		self.idx_in_map = {name:i for i,name in enumerate(pix_name)}
		
		self.has_indiv = np.empty(self.nside.size, dtype=np.bool)
		self.has_indiv[:] = False
		
		self.has_cloud = np.empty(self.nside.size, dtype=np.bool)
		self.has_cloud[:] = False
		
		self.has_los = np.empty(self.nside.size, dtype=np.bool)
		self.has_los[:] = False
		
		#self.has_indiv = dict.fromkeys(self.pix_name, False)
		#self.has_cloud = dict.fromkeys(self.pix_name, False)
		#self.has_los = dict.fromkeys(self.pix_name, False)
		
		self.rasterizer = None
	
	def update(self):
		# Determine which output files have been updated
		mtime = np.array([os.path.getmtime(fname) for fname in self.outfiles])
		file_idx = (mtime > self.modtime)
		
		# Read information from updated files
		for i in file_idx:
			f = h5py.File(self.outifles[i], 'r')
			
			for _, pixel in f.iteritems():
				#nside, hp_idx = name.split('-')
				name = '%d-%d' % (pixel.attrs['nside'], pixel.attrs['healpix_index'])
				idx = self.idx_in_map[name]
				keys = pixel.keys()
				
				self.has_indiv[idx] = ('stellar_chains' in keys)
				self.has_cloud[idx] = ('clouds' in keys)
				self.has_los[idx] = ('los' in keys)
			
			f.close()
	
	def init_rasterizer(self, img_shape,
	                          proj=hputils.Cartesian_projection(),
	                          l_cent=0., b_cent=0):
		self.rasterizer = hputils.MapRasterizer(self.nside, self.pix_idx, img_shape,
		                                        proj=proj, l_cent=l_cent)
	
	def get_pix_status(self, method='piecewise'):
		pix_val = self.has_indiv.astype('i4')
		
		if method == 'piecewise':
			pix_val += self.has_los.astype('i4')
		elif method == 'cloud':
			pix_val += self.has_cloud.astype('i4')
		
		return pix_val
	
	def rasterize(self, method='piecewise'):
		if self.rasterizer == None:
			return None
		
		pix_val = self.get_pix_status(method=method)
		img = rasterizer.rasterize(pix_val)
	
	def get_pct_complete(self, method='piecewise'):
		n_stars_complete = None
		
		if method == 'piecewise':
			n_stars_complete = self.has_los.astype('i4') * self.n_stars
		elif method = 'cloud':
			n_stars_complete = self.has_los.astype('i4') * self.n_stars
		else:
			raise ValueError("Method '%s' not understood." % method)
		
		return 100. * float(n_stars_complete) / float(self.n_stars_tot)
	
	def to_ax(self, ax, method='piecewise',
	                    l_lines=None, b_lines=None,
	                    l_spacing=1., b_spacing=1.):
		if self.rasterizer == None:
			return
		
		# Plot map of processing status
		img = self.rasterize(method=method)
		bounds = rasterizer.get_lb_bounds()
		
		cmap = ListedColorMap(['gray', pallette['vermillion'], pallette['bluish green']])
		norm = BoundaryNorm([0,1,2,3], cmap.N)
		
		ax.imshow(img.T, extent=bounds, origin='lower', aspect='auto',
		                 interpolation='nearest', cmap=cmap, norm=norm)
		
		if (l_lines != None) and (b_lines != None):
			# Determine label positions
			l_labels, b_labels = rasterizer.label_locs(l_lines, b_lines, shift_frac=0.04)
			
			# Determine grid lines to plot
			l_lines = np.array(l_lines)
			b_lines = np.array(b_lines)
			
			idx = (np.abs(l_lines) < 1.e-5)
			l_lines_0 = l_lines[idx]
			l_lines = l_lines[~idx]
			
			idx = (np.abs(b_lines) < 1.e-5)
			b_lines_0 = b_lines[idx]
			b_lines = b_lines[~idx]
			
			x_guides, y_guides = rasterizer.latlon_lines(l_lines, b_lines,
			                                             l_spacing=l_spacing,
			                                             b_spacing=b_spacing)
			
			x_guides_l0, y_guides_l0, x_guides_b0, y_guides_b0 = None, None, None, None
			
			if l_lines_0.size != 0:
				x_guides_l0, y_guides_l0 = rasterizer.latlon_lines(l_lines_0, 0.,
				                                                   mode='meridians',
				                                                   b_spacing=0.5*b_spacing)
			
			if b_lines_0.size != 0:
				x_guides_b0, y_guides_b0 = rasterizer.latlon_lines(0., b_lines_0,
				                                                   mode='parallels',
				                                                   l_spacing=0.5*l_spacing)
			
			# Plot lines of constant l and b
			xlim = ax.get_xlim()
			ylim = ax.get_ylim()
			
			if x_guides != None:
				ax.scatter(x_guides, y_guides, s=1., c='b', edgecolor='b', alpha=0.10)
			
			if x_guides_l0 != None:
				ax.scatter(x_guides_l0, y_guides_l0, s=3., c='g', edgecolor='g', alpha=0.25)
			
			if x_guides_b0 != None:
				ax.scatter(x_guides_b0, y_guides_b0, s=3., c='g', edgecolor='g', alpha=0.25)
			
			ax.set_xlim(xlim)
			ax.set_ylim(ylim)
			
			# Label Galactic coordinates
			if l_lines != None:
				if bounds != None:
					if (bounds[2] > -80.) | (bounds[3] < 80.):
						for l, (x_0, y_0), (x_1, y_1) in l_labels:
							ax.text(x_0, y_0, r'$%d^{\circ}$' % l, fontsize=20,
							                                       ha='center',
							                                       va='center')
							ax.text(x_1, y_1, r'$%d^{\circ}$' % l, fontsize=20,
							                                       ha='center',
							                                       va='center')
				
			if b_lines != None:
				for b, (x_0, y_0), (x_1, y_1) in b_labels:
					ax.text(x_0, y_0, r'$%d^{\circ}$' % b, fontsize=20,
					                                       ha='center',
					                                       va='center')
					ax.text(x_1, y_1, r'$%d^{\circ}$' % b, fontsize=20,
					                                       ha='center',
					                                       va='center')
				
				# Expand axes limits to fit labels
				expand = 0.075
				xlim = ax.get_xlim()
				w = xlim[1] - xlim[0]
				xlim = [xlim[0] - expand * w, xlim[1] + expand * w]
				
				ylim = ax.get_ylim()
				h = ylim[1] - ylim[0]
				ylim = [ylim[0] - expand * h, ylim[1] + expand * h]
				
				ax.set_xlim(xlim)
				ax.set_ylim(ylim)
		
		


def main():
	parser = argparse.ArgumentParser(prog='plot_completion.py',
	                                 description='Represent competion of Bayestar job as a rasterized map.',
	                                 add_help=True)
	parser.add_argument('--indir', '-i', type=str, required=True,
	                                       help='Directory with Bayestar input files.')
	parser.add_argument('--outdir', '-o', type=str, required=True,
	                                       help='Directory with Bayestar output files.')
	parser.add_argument('--plot-fname', '-plt', type=str, required=True,
	                                       help='Output filename for plot.')
	parser.add_argument('--figsize', '-fs', type=int, nargs=2, default=(8, 4),
	                                       help='Figure size (in inches).')
	parser.add_argument('--dpi', '-dpi', type=float, default=200,
	                                       help='Dots per inch for figure.')
	parser.add_argument('--projection', '-proj', type=str, default='Cartesian',
	                                       choices=('Cartesian', 'Mollweide', 'Hammer', 'Eckert IV'),
	                                       help='Map projection to use.')
	parser.add_argument('--center-lb', '-cent', type=float, nargs=2, default=(0., 0.),
	                                       help='Center map on (l, b).')
	parser.add_argument('--method', '-mtd', type=str, default='both',
	                                       choices=('cloud', 'piecewise'),
	                                       help='Measure of line-of-sight completion to show.')
	parser.add_argument('--interval', '-int', type=float, default=1.,
	                                       help='Generate a picture every X hours.')
	parser.add_argument('--maxtime', '-max', type=float, default=24.,
	                                       help='Number of hours to continue monitoring job completion.')
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	args = parser.parse_args(sys.argv[offset:])
	
	
	# Parse arguments
	proj = None
	if args.projection == 'Cartesian':
		proj = hputils.Cartesian_projection()
	elif args.projection == 'Mollweide':
		proj = hputils.Mollweide_projection()
	elif args.projection == 'Hammer':
		proj = hputils.Hammer_projection()
	elif args.projection == 'Eckert IV':
		proj = hputils.EckertIV_projection()
	else:
		raise ValueError("Unrecognized projection: '%s'" % args.proj)
	
	l_cent, b_cent = args.center_lb
	
	img_shape = (int(args.figsize[0] * 0.8 * args.dpi),
	             int(args.figsize[1] * 0.8 * args.dpi))
	
	# Generate grid lines
	ls = np.linspace(-180., 180., 13)
	bs = np.linspace(-90., 90., 7)[1:-1]
	
	# Initialize completion log
	completion = TCompletion(args.indir, args.outdir)
	completion.init_rasterizer(img_shape, proj=proj, l_cent=l_cent, b_cent=b_cent)
	
	# Matplotlib settings
	mplib.rc('text', usetex=True)
	mplib.rc('xtick.major', size=6)
	mplib.rc('xtick.minor', size=2)
	mplib.rc('ytick.major', size=6)
	mplib.rc('ytick.minor', size=2)
	mplib.rc('xtick', direction='out')
	mplib.rc('ytick', direction='out')
	mplib.rc('axes', grid=False)
	
	t_start = time.time()
	
	while time.time() - t_start < 3600. * args.maxtime:
		t_next = time.time() + 3600.*args.interval
		
		print 'Updating processing status...'
		completion.update()
		
		# Plot completion map
		print 'Plotting status map...'
		
		fig = plt.figure(figsize=args.figsize, dpi=args.dpi)
		ax = fig.add_subplot(1,1,1)
		
		completion.to_ax(ax, method=method, l_lines=ls, b_lines=ls,
		                                    l_spacing=1., b_spacing=1.)
		
		# Labels, ticks, etc.
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		
		# Title
		timestr = time.strftime('%m.%d-%H:%M:%S')
		pct_complete = completion.get_pct_complete()
		ax.set_title(r'$\mathrm{Completion \ as \ of \ %s \ (%.2f \ \%%)}$' % (timestr, pct_complete),
		             fontsize=16)
		
		# Allow user to determine healpix index
		#pix_identifiers = []
		#nside_max = np.max(completion.nside)
		#pix_identifiers.append(PixelIdentifier(ax, nside_max, nest=True, proj=proj))
		
		# Save figure
		print 'Saving plot ...'
		fig.savefig(args.plot_fname, dpi=args.dpi)
		plt.close(fig)
		del img
		
		print 'Time: %s' % timestr
		print 'Sleeping ...'
		print ''
		
		t_sleep = t_next - time.time()
		t_sleep = max([60., t_sleep])
		time.sleep(t_sleep)
	
	
	return 0

if __name__ == '__main__':
	main()

