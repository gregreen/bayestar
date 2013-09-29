#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  plotmap.py
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

import argparse, os, sys, time

import healpy as hp
import h5py

import multiprocessing
import Queue

import hputils, maptools


def plot_EBV(ax, img, bounds, **kwargs):
	# Configure plotting options
	if 'vmin' not in kwargs:
		kwargs['vmin'] = np.min(img[np.isfinite(img)])
	if 'vmax' not in kwargs:
		kwargs['vmax'] = np.max(img[np.isfinite(img)])
	if 'aspect' not in kwargs:
		kwargs['aspect'] = 'auto'
	if 'interpolation' not in kwargs:
		kwargs['interpolation'] = 'nearest'
	if 'origin' in kwargs:
		print "Ignoring option 'origin'."
	if 'extent' in kwargs:
		print "Ignoring option 'extent'."
	kwargs['origin'] = 'lower'
	kwargs['extent'] = bounds
	kwargs['cmap'] = 'binary'
	
	# Plot image in B&W
	img_res = ax.imshow(img.T, **kwargs)
	
	# Neutrally color masked regions
	kwargs['vmin'] = None
	kwargs['vmax'] = None
	kwargs['cmap'] = None
	mask = np.isnan(img.T)
	shape = (img.shape[1], img.shape[0], 4)
	mask_img = np.zeros(shape, dtype='f8')
	mask_img[:,:,0] = 0.30
	mask_img[:,:,1] = 0.70
	mask_img[:,:,2] = 0.90
	mask_img[:,:,3] = 0.95 * mask.astype('f8')
	#ax.imshow(mask_img, **kwargs)
	ax.imshow(mask_img, origin='lower', aspect='auto',
	                    interpolation='nearest',extent=bounds)
	
	#xlim = ax.get_xlim()
	#ax.set_xlim(xlim[1], xlim[0])
	
	return img_res


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


def plotter_worker(img_q, lock,
                   n_rasterizers,
                   figsize, dpi,
                   EBV_max, outfname):
	
	n_finished = 0
	
	# Plot images
	while True:
		n, mu, img = img_q.get()
		
		# Count number of rasterizer workers that have finished
		# processing their queue
		if n == 'FINISHED':
			n_finished += 1
			
			if n_finished >= n_rasterizers:
				return
			else:
				continue
		
		# Plot this image
		print 'Plotting mu = %.2f (image %d) ...' % (mu, n+1)
		
		fig = plt.figure(figsize=figsize, dpi=dpi)
		ax = fig.add_subplot(1,1,1)
		
		img = plot_EBV(ax, img, bounds, vmin=0., vmax=EBV_max)
		
		# Colorbar
		fig.subplots_adjust(bottom=0.12, left=0.12, right=0.89, top=0.88)
		cax = fig.add_axes([0.9, 0.12, 0.03, 0.76])
		cb = fig.colorbar(img, cax=cax)
		
		# Labels, ticks, etc.
		ax.set_xlabel(r'$\ell$', fontsize=16)
		ax.set_ylabel(r'$b$', fontsize=16)
		
		ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
		ax.xaxis.set_minor_locator(AutoMinorLocator())
		ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
		ax.yaxis.set_minor_locator(AutoMinorLocator())
		
		# Title
		d = 10.**(mu/5. - 2.)
		ax.set_title(r'$\mu = %.2f \ \ \ d = %.2f \, \mathrm{kpc}$' % (mu, d), fontsize=16)
		
		# Save figure
		if outfname != None:
			full_fname = '%s.%s.%s.%.5d.png' % (outfname, model, method, n)
			fig.savefig(full_fname, dpi=dpi)
		
		plt.close(fig)
		del img


def rasterizer_worker(dist_q, img_q,
                      los_coll,
                      figsize, dpi, size,
                      model, method, mask,
                      proj, l_cent, b_cent, bounds,
                      delta_mu):
	# Reseed random number generator
	t = time.time()
	t_after_dec = int(1.e9*(t - np.floor(t)))
	seed = np.bitwise_xor([t_after_dec], [os.getpid()])
	
	np.random.seed(seed=seed)
	
	# Generate images
	while True:
		try:
			n, mu = dist_q.get_nowait()
			
			# Rasterize E(B-V)
			img, bounds = los_coll.rasterize(mu, size, fit=model,
			                                           method=method,
			                                           mask_sigma=mask,
			                                           delta_mu=delta_mu,
			                                           proj=proj,
			                                           l_cent=l_cent,
			                                           b_cent=b_cent)
			
			# Put image on queue
			img_q.put((n, mu, img))
			
		except Queue.Empty:
			img_q.put('FINISHED')
			
			print 'Rasterizer finished.'
			
			return


def main():
	parser = argparse.ArgumentParser(prog='plotmap.py',
	                                 description='Generate a map of E(B-V) from bayestar output.',
	                                 add_help=True)
	parser.add_argument('input', type=str, nargs='+', help='Bayestar output files.')
	parser.add_argument('--output', '-o', type=str, help='Output filename for plot.')
	#parser.add_argument('--show', '-sh', action='store_true', help='Show plot.')
	parser.add_argument('--dists', '-d', type=float, nargs=3,
	                                     default=(4., 19., 21),
	                                     help='DM min, DM max, # of distance slices.')
	parser.add_argument('--delta-mu', '-dmu', type=float, default=None,
	                                     help='Difference in DM used to estimate rate of\n'
	                                          'reddening (default: None, i.e. calculate cumulative reddening).')
	parser.add_argument('--figsize', '-fs', type=int, nargs=2, default=(8, 4),
	                                     help='Figure size (in inches).')
	parser.add_argument('--dpi', '-dpi', type=float, default=200,
	                                     help='Dots per inch for figure.')
	parser.add_argument('--projection', '-proj', type=str, default='Cartesian',
	                                     choices=('Cartesian', 'Mollweide', 'Hammer', 'Eckert IV'),
	                                     help='Map projection to use.')
	parser.add_argument('--center-lb', '-cent', type=float, nargs=2, default=(0., 0.),
	                                     help='Center map on (l, b).')
	parser.add_argument('--bounds', '-b', type=float, nargs=4, default=None,
	                                     help='Bounds of pixels to plot (l_min, l_max, b_min, b_max).')
	parser.add_argument('--model', '-m', type=str, default='piecewise',
	                                     choices=('piecewise', 'cloud'),
	                                     help='Line-of-sight extinction model to use.')
	parser.add_argument('--mask', '-msk', type=float, default=None,
	                                     help=r'Hide parts of map where sigma_{E(B-V)} is greater than given value.')
	parser.add_argument('--method', '-mtd', type=str, default='median',
	                                     choices=('median', 'mean', 'best', 'sample', 'sigma' , '5th', '95th'),
	                                     help='Measure of E(B-V) to plot.')
	parser.add_argument('--processes', '-proc', type=int, default=1,
	                                     help='# of processes to spawn.')
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	args = parser.parse_args(sys.argv[offset:])
	
	
	# Parse arguments
	outfname = args.output
	if outfname != None:
		if outfname.endswith('.png'):
			outfname = outfname[:-4]
	
	method = args.method
	if method == '5th':
		method = 5.
	elif method == '95th':
		method = 95.
	
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
	
	size = (int(args.figsize[0] * 0.8 * args.dpi),
	        int(args.figsize[1] * 0.8 * args.dpi))
	
	mu_plot = np.linspace(args.dists[0], args.dists[1], args.dists[2])
	
	
	# Load in line-of-sight data
	fnames = args.input
	los_coll = maptools.los_collection(fnames, bounds=args.bounds)
	
	
	# Get upper limit on E(B-V)
	method_tmp = method
	
	if method == 'sample':
		method_tmp = 'median'
	
	EBV_max = None
	
	if args.delta_mu == None:
		nside_tmp, pix_idx_tmp, EBV = los_coll.gen_EBV_map(mu_plot[-1],
		                                                   fit=args.model,
		                                                   method=method_tmp,
		                                                   mask_sigma=args.mask,
		                                                   delta_mu=args.delta_mu)
		idx = np.isfinite(EBV)
		EBV_max = np.percentile(EBV[idx], 95.)
		
	else:
		EBV_max = los_coll.est_dEBV_pctile(95., delta_mu=args.delta_mu,
		                                        fit=args.model)
	
	mask = args.mask
	
	if method == 'sample':
		mask = None
	
	print 'EBV_max = %.3f' % EBV_max
	
	# Matplotlib settings
	mplib.rc('text', usetex=True)
	mplib.rc('xtick.major', size=6)
	mplib.rc('xtick.minor', size=2)
	mplib.rc('ytick.major', size=6)
	mplib.rc('ytick.minor', size=2)
	mplib.rc('xtick', direction='out')
	mplib.rc('ytick', direction='out')
	mplib.rc('axes', grid=False)
	
	
	# Plot at each distance
	pix_identifiers = []
	nside_max = los_coll.get_nside_levels()[-1]
	
	# Set up queue for rasterizer workers to pull from
	dist_q = multiprocessing.Queue()
	
	for n,mu in enumerate(mu_plot):
		dist_q.put((n, mu))
	
	# Set up results queue for rasterizer workers
	img_q = multiprocessing.Queue()
	
	# Spawn worker processes to plot images
	n_rasterizers = args.processes
	procs = []
	
	for i in xrange(n_rasterizers):
		p = multiprocessing.Process(target=rasterizer_worker,
		                            args=(dist_q, img_q,
		                                  los_coll,
		                                  args.figsize, args.dpi, size,
		                                  args.model, method, mask,
		                                  proj, l_cent, b_cent, args.bounds,
		                                  args.delta_mu)
		                           )
		
		procs.append(p)
		p.start()
	
	# Plot and save results
	n_rast_finished = 0
	n_img_finished = 0
	
	while True:
		q_in = img_q.get()
		
		# Count number of rasterizer workers that have finished
		# processing their queue
		if q_in == 'FINISHED':
			n_rast_finished += 1
			
			if n_rast_finished >= n_rasterizers:
				break
			else:
				continue
		
		n, mu, img = q_in
		
		# Plot this image
		print 'Plotting mu = %.2f (%d of %d) ...' % (mu,
		                                             n_img_finished+1,
		                                             mu_plot.size)
		
		fig = plt.figure(figsize=args.figsize, dpi=args.dpi)
		ax = fig.add_subplot(1,1,1)
		
		img = plot_EBV(ax, img, args.bounds, vmin=0., vmax=EBV_max)
		
		# Colorbar
		fig.subplots_adjust(bottom=0.12, left=0.12, right=0.89, top=0.88)
		cax = fig.add_axes([0.9, 0.12, 0.03, 0.76])
		cb = fig.colorbar(img, cax=cax)
		
		# Labels, ticks, etc.
		ax.set_xlabel(r'$\ell$', fontsize=16)
		ax.set_ylabel(r'$b$', fontsize=16)
		
		ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
		ax.xaxis.set_minor_locator(AutoMinorLocator())
		ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
		ax.yaxis.set_minor_locator(AutoMinorLocator())
		
		# Title
		d = 10.**(mu/5. - 2.)
		ax.set_title(r'$\mu = %.2f \ \ \ d = %.2f \, \mathrm{kpc}$' % (mu, d), fontsize=16)
		
		# Save figure
		if outfname != None:
			full_fname = '%s.%s.%s.%.5d.png' % (outfname, args.model, method, n)
			fig.savefig(full_fname, dpi=args.dpi)
		
		plt.close(fig)
		del img
		
		n_img_finished += 1
	
	for p in procs:
		p.join()
	
	print 'Done.'
	
	#if args.show:
	#	plt.show()
	
	
	return 0

if __name__ == '__main__':
	main()

