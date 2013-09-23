#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  plot_completion.py
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
mplib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from mpl_toolkits.axes_grid1 import ImageGrid

import argparse, sys, time

import healpy as hp
import h5py

import hputils, maptools


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


def main():
	parser = argparse.ArgumentParser(prog='plot_completion.py',
	                                 description='Represent competion of Bayestar job as a rasterized map.',
	                                 add_help=True)
	parser.add_argument('--infiles', '-i', type=str, nargs='+', required=True,
	                                       help='Bayestar input files.')
	parser.add_argument('--outfiles', '-o', type=str, nargs='+', required=True,
	                                       help='Bayestar output files.')
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
	                                       choices=('cloud', 'piecewise', 'both'),
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
	
	size = (int(args.figsize[0] * 0.8 * args.dpi), int(args.figsize[1] * 0.8 * args.dpi))
	
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
		# Load information on completion
		completion = maptools.job_completion_counter(args.infiles, args.outfiles)
		
		# Plot completion
		pix_identifiers = []
		nside_max = np.max(completion.nside)
		
		fig = plt.figure(figsize=args.figsize, dpi=args.dpi)
		ax = fig.add_subplot(1,1,1)
		
		img, bounds = completion.rasterize(size, method=args.method,
		                                         proj=proj,
		                                         l_cent=l_cent,
		                                         b_cent=b_cent)
		
		ax.imshow(img.T, extent=bounds, vmin=0, vmax=3,
		                 aspect='auto', origin='lower', interpolation='nearest')
		
		# Labels, ticks, etc.
		ax.set_xlabel(r'$\ell$', fontsize=16)
		ax.set_ylabel(r'$b$', fontsize=16)
		
		ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
		ax.xaxis.set_minor_locator(AutoMinorLocator())
		ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
		ax.yaxis.set_minor_locator(AutoMinorLocator())
		
		# Title
		timestr = time.strftime('%m.%d-%H:%M:%S')
		ax.set_title(r'$\mathrm{Completion \ as \ of \ %s}$' % (timestr), fontsize=16)
		
		# Allow user to determine healpix index
		pix_identifiers.append(PixelIdentifier(ax, nside_max, nest=True, proj=proj))
		
		# Save figure
		fig.savefig(args.plot_fname, dpi=args.dpi)
		plt.close(fig)
		
		print timestr
		print 'Sleeping ...'
		print ''
		
		del img
		
		time.sleep(3600. * args.interval)
	
	
	return 0

if __name__ == '__main__':
	main()

