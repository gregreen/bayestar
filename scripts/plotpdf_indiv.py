#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#
#       plotpdf.py
#
#       Copyright 2012 Greg <greg@greg-G53JW>
#
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later version.
#
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.
#
#

import sys, argparse
from os.path import abspath, expanduser

import numpy as np
import scipy.optimize as opt
from scipy.ndimage.filters import gaussian_filter

import matplotlib as mplib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

import hdf5io

def los2ax(ax, fname, group, DM_lim, *args, **kwargs):
	chain = hdf5io.TChain(fname, '%s/los' % group)

	mu = np.linspace(DM_lim[0], DM_lim[1], chain.get_nDim())
	if 'alpha' not in kwargs:
		kwargs['alpha'] = 1. / np.power(chain.get_nSamples(), 0.55)

	# Plot all paths
	EBV_all = np.cumsum(np.exp(chain.get_samples(0)), axis=1)
	lnp = chain.get_lnp()[0, 1:]
	#lnp = np.exp(lnp - np.max(lnp))
	lnp_min, lnp_max = np.percentile(lnp, [10., 90.])
	lnp = (lnp - lnp_min) / (lnp_max - lnp_min)
	lnp[lnp > 1.] = 1.
	lnp[lnp < 0.] = 0.

	#for i,EBV in enumerate(EBV_all[1:]):
	#	c = (1.-lnp[i], 0., lnp[i])
	#	kwargs['c'] = c
	#	ax.plot(mu, EBV, *args, **kwargs)

	kwargs['c'] = 'g'
	kwargs['lw'] = 1.5
	kwargs['alpha'] = 0.5
	ax.plot(mu, EBV_all[0], *args, **kwargs)

	# Plot mean path
	#y = np.mean(EBV_all, axis=0)
	#y_err = np.std(EBV_all, axis=0)
	#ax.errorbar(mu, y, yerr=y_err, c='g', ecolor=(0., 1., 0., 0.5), alpha=0.3)

	# Plot best path
	#i = np.argsort(chain.get_lnp(0))[::-1]
	#alpha = 1.
	#for ii in i[:3]:
	#	ax.plot(mu, EBV_all[ii], 'r-', alpha=alpha)
	#	alpha *= 0.5
	#alpha = 1.
	#for ii in i[-10:]:
	#	ax.plot(mu, EBV_all[ii], 'k-', alpha=alpha)
	#	alpha *= 0.85

	ax.set_xlim(DM_lim[0], DM_lim[1])

def clouds2ax(ax, fname, group, DM_lim, *args, **kwargs):
	chain = hdf5io.TChain(fname, '%s/clouds' % group)
	mu_range = np.linspace(DM_lim[0], DM_lim[1], chain.get_nDim())
	if 'alpha' not in kwargs:
		kwargs['alpha'] = 1. / np.power(chain.get_nSamples(), 0.55)

	# Plot all paths
	N_clouds = chain.get_nDim() / 2
	N_paths = chain.get_nSamples()
	mu_tmp = np.cumsum(chain.get_samples(0)[:,:N_clouds], axis=1)
	EBV_tmp = np.cumsum(np.exp(chain.get_samples(0)[:,N_clouds:]), axis=1)

	mu_all = np.zeros((N_paths, 2*(N_clouds+1)), dtype='f8')
	EBV_all = np.zeros((N_paths, 2*(N_clouds+1)), dtype='f8')
	mu_all[:,0] = mu_range[0]
	mu_all[:,1:-1:2] = mu_tmp
	mu_all[:,2:-1:2] = mu_tmp
	mu_all[:,-1] = mu_range[-1]
	EBV_all[:,2:-1:2] = EBV_tmp
	EBV_all[:,3::2] = EBV_tmp
	#EBV_all[:,-1] = EBV_tmp[:,-1]

	lnp = chain.get_lnp()[0, 1:]
	lnp_min, lnp_max = np.percentile(lnp, [10., 90.])
	lnp = (lnp - lnp_min) / (lnp_max - lnp_min)
	lnp[lnp > 1.] = 1.
	lnp[lnp < 0.] = 0.

	for i,(mu,EBV) in enumerate(zip(mu_all[1:], EBV_all[1:])):
		c = (1.-lnp[i], 0., lnp[i])
		kwargs['c'] = c
		ax.plot(mu, EBV, *args, **kwargs)

	kwargs['c'] = 'g'
	kwargs['alpha'] = 0.5
	ax.plot(mu_all[0], EBV_all[0], *args, **kwargs)

	# Plot mean path
	#y = np.mean(EBV_all, axis=0)
	#y_err = np.std(EBV_all, axis=0)
	#ax.errorbar(mu, y, yerr=y_err, c='g', ecolor=(0., 1., 0., 0.5), alpha=0.3)

	# Plot best path
	#i = np.argsort(chain.get_lnp(0))[::-1]
	#alpha = 1.
	#for ii in i[:3]:
	#	ax.plot(mu, EBV_all[ii], 'r-', alpha=alpha)
	#	alpha *= 0.5
	#alpha = 1.
	#for ii in i[-10:]:
	#	ax.plot(mu, EBV_all[ii], 'k-', alpha=alpha)
	#	alpha *= 0.85

	ax.set_xlim(DM_lim[0], DM_lim[1])

class TClouds:
	def __init__(self, fname, group, DM_lim):
		chain = hdf5io.TChain(fname, '%s/clouds' % group)

		mu_range = np.linspace(DM_lim[0], DM_lim[1], chain.get_nDim())

		self.lnp = chain.get_lnp()[0, 1:]
		lnp_min, lnp_max = np.percentile(self.lnp, [10., 90.])
		self.color = (self.lnp - lnp_min) / (lnp_max - lnp_min)
		self.color[self.color > 1.] = 1.
		self.color[self.color < 0.] = 0.

		# Plot all paths
		self.N_clouds = chain.get_nDim() / 2
		self.N_paths = chain.get_nSamples()
		mu_tmp = np.cumsum(chain.get_samples(0)[:,:self.N_clouds], axis=1)
		EBV_tmp = np.cumsum(np.exp(chain.get_samples(0)[:,self.N_clouds:]), axis=1)

		self.mu_all = np.zeros((self.N_paths, 2*(self.N_clouds+1)), dtype='f8')
		self.EBV_all = np.zeros((self.N_paths, 2*(self.N_clouds+1)), dtype='f8')
		self.mu_all[:,0] = mu_range[0]
		self.mu_all[:,1:-1:2] = mu_tmp
		self.mu_all[:,2:-1:2] = mu_tmp
		self.mu_all[:,-1] = mu_range[-1]
		self.EBV_all[:,2:-1:2] = EBV_tmp
		self.EBV_all[:,3::2] = EBV_tmp

	def plot(self, ax, *args, **kwargs):
		if 'alpha' not in kwargs:
			kwargs['alpha'] = 1. / np.power(float(self.N_paths), 0.55)

		for i,(mu,EBV) in enumerate(zip(self.mu_all[1:], self.EBV_all[1:])):
			c = (1.-self.color[i], 0., self.color[i])
			kwargs['c'] = c
			ax.plot(mu, EBV, *args, **kwargs)

		kwargs['c'] = 'g'
		kwargs['alpha'] = 0.5
		ax.plot(self.mu_all[0], self.EBV_all[0], *args, **kwargs)


class TLOS:
	def __init__(self, fname, group, DM_lim):
		chain = hdf5io.TChain(fname, '%s/los' % group)

		self.mu = np.linspace(DM_lim[0], DM_lim[1], chain.get_nDim())
		self.alpha = 1. / np.power(chain.get_nSamples(), 0.55)
		self.EBV_all = np.cumsum(np.exp(chain.get_samples(0)), axis=1)

		self.lnp = chain.get_lnp()[0, 1:]
		lnp_min, lnp_max = np.percentile(self.lnp, [10., 90.])

		self.color = (self.lnp - lnp_min) / (lnp_max - lnp_min)
		self.color[self.color > 1.] = 1.
		self.color[self.color < 0.] = 0.

	def plot(self, ax, *args, **kwargs):
		if 'alpha' not in kwargs:
			kwargs['alpha'] = self.alpha

		# Plot all paths
		#for i,EBV in enumerate(self.EBV_all[1:]):
		#	c = (1.-self.color[i], 0., self.color[i])
		#	kwargs['c'] = c
		#	ax.plot(self.mu, EBV, *args, **kwargs)

		kwargs['c'] = 'g'
		kwargs['lw'] = 2.
		kwargs['alpha'] = 0.5

		#ax.plot(self.mu, self.EBV_all[0], *args, **kwargs)
		ax.plot(self.mu, np.median(self.EBV_all[1:], axis=0), *args, **kwargs)

		ax.set_xlim(self.mu[0], self.mu[-1])



def main():
	# Parse commandline arguments
	parser = argparse.ArgumentParser(prog='plotpdf',
	              description='Plots posterior distributions produced by galstar',
	              add_help=True)
	parser.add_argument('input', type=str, help='Bayestar output file.')
	parser.add_argument('loc', type=int, nargs=2, help='HEALPix nside and index.')
	parser.add_argument('-o', '--output', type=str, help='Filename for plot.')
	parser.add_argument('-s', '--show', action='store_true', help='Show plot.')
	parser.add_argument('-los', '--show-los', action='store_true',
	                            help='Show piecewise-linear l.o.s. reddening.')
	parser.add_argument('-cl', '--show-clouds', action='store_true',
	                            help='Show cloud model of reddening.')
	parser.add_argument('-ovcl', '--overplot-clouds', type=float, nargs='+', default=None,
	                            help='Overplot clouds at specified distance/depth pairs.')
	#parser.add_argument('-ovlos', '--overplot-los', type=float, nargs='+', default=None,
	#                            help='Overplot piecewise-linear reddening profile with'
	#                                 'specified reddening increases.')
	parser.add_argument('-g', '--grid', type=int, nargs=2,
	                            help='Grid shape (rows, columns).')
	parser.add_argument('-fig', '--figsize', type=float, nargs=2, default=(7, 5.),
	                            help='Figure width and height in inches.')
	parser.add_argument('-nfig', '--n-figures', type=int, default=None,
	                            help='Maximum # of figures to display (default: display all)')
	parser.add_argument('--stretch', type=str, choices=('linear', 'sqrt', 'log'), default='sqrt',
	                            help='Stretch for the color scale.')
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	args = parser.parse_args(sys.argv[offset:])

	if (args.output == None) and not args.show:
		print 'Either --output or --show must be given.'
		return 0

	fname = abspath(expanduser(args.input))
	group = 'pixel %d-%d' % (args.loc[0], args.loc[1])

	# Load in pdfs
	x_min, x_max = [4., 0.], [19., 5.]
	pdf_stack = None
	pdf_indiv = None
	EBV_max = None

	dset = '%s/stellar chains' % group
	chain = hdf5io.TChain(fname, dset)
	lnZ = chain.get_lnZ()[:]
	lnZ_max = np.percentile(lnZ[np.isfinite(lnZ)], 95.)
	lnZ_idx = np.ones(lnZ.shape).astype(np.bool) #(lnZ > lnZ_max - 30.)
	#print np.sum(lnZ_idx), np.sum(~lnZ_idx)

	# Load stellar pdfs
	try:
		dset = '%s/stellar pdfs' % group
		pdf = hdf5io.TProbSurf(fname, dset)
		x_min, x_max = pdf.x_min, pdf.x_max
		pdf_stack = np.sum(pdf.get_p()[lnZ_idx], axis=0)

		pdf_indiv = pdf.get_p()

	except:
		res = (501, 121)

		E_range = np.linspace(x_min[1], x_max[1], res[0]*2+1)
		DM_range = np.linspace(x_min[0], x_max[0], res[1]*2+1)

		star_samples = chain.get_samples()[:, :, 0:2]

		n_stars_tmp, n_star_samples, n_star_dim = star_samples.shape

		pdf_indiv = np.empty((n_stars_tmp, res[1], res[0]), dtype='f8')

		for i in xrange(star_samples.shape[0]):
			tmp_pdf, tmp, tmp = np.histogram2d(star_samples[i,:,0],
			                                   star_samples[i,:,1],
			                                   bins=[E_range, DM_range])
			tmp_pdf = gaussian_filter(tmp_pdf, sigma=(4, 2), mode='reflect')
			tmp_pdf = tmp_pdf.reshape([res[0], 2, res[1], 2]).mean(3).mean(1)
			tmp_pdf /= np.sum(tmp_pdf)
			pdf_indiv[i] = tmp_pdf.T

		star_samples.shape = (n_stars_tmp * n_star_samples, n_star_dim)

		pdf_stack, tmp1, tmp2 = np.histogram2d(star_samples[:,0], star_samples[:,1],
		                                       bins=[E_range, DM_range])

		pdf_stack = gaussian_filter(pdf_stack.astype('f8'),
		                            sigma=(4, 2), mode='reflect')
		pdf_stack = pdf_stack.reshape([res[0], 2, res[1], 2]).mean(3).mean(1)
		pdf_stack *= 100. / np.max(pdf_stack)
		pdf_stack = pdf_stack.T

	# Normalize peak to unity at each distance
	pdf_stack /= np.max(pdf_stack)
	norm = 1. / np.power(np.max(pdf_stack, axis=1), 0.8)
	norm[np.isinf(norm)] = 0.
	pdf_stack = np.einsum('ij,i->ij', pdf_stack, norm)

	# Determine maximum E(B-V)
	w_y = np.mean(pdf_stack, axis=0)
	y_max = 1.25 * np.max(np.where(w_y > 5.e-2)[0])
	EBV_max = y_max * (x_max[1] / pdf_stack.shape[1])

	# Set matplotlib style attributes
	mplib.rc('text', usetex=True)
	mplib.rc('xtick.major', size=5)
	mplib.rc('xtick.minor', size=2)
	mplib.rc('ytick.major', size=5)
	mplib.rc('ytick.minor', size=2)
	mplib.rc('xtick', direction='in')
	mplib.rc('ytick', direction='in')
	mplib.rc('axes', grid=True)

	# Plot individual stellar pdfs
	figs = []
	n_rows, n_cols = args.grid
	bounds = [x_min[0], x_max[0], x_min[1], x_max[1]]
	DM_lim = [x_min[0], x_max[0]]

	x_lnZ = x_min[0] + 0.125 * (x_max[0] - x_min[0])
	y_lnZ = x_max[1] - 0.125 * (x_max[1] - x_min[1])

	x_idx = x_max[0] - 0.05 * (x_max[0] - x_min[0])

	if EBV_max != None:
		y_lnZ = EBV_max - 0.125 * (EBV_max - x_min[1])

	f_stretch = lambda a: a

	if args.stretch == 'log':
		f_stretch = lambda a: np.log(a)
	elif args.stretch == 'sqrt':
		f_stretch = lambda a: np.sqrt(a)

	clouds = None
	if args.show_clouds:
		clouds = TClouds(fname, group, DM_lim)

	los = None
	if args.show_los:
		los = TLOS(fname, group, DM_lim)

	for i,p in enumerate(pdf_indiv):
		print 'Plotting axis %d of %d ...' % (i+1, pdf_indiv.shape[0])

		k = i % (n_rows * n_cols) + 1

		if k == 1:
			if args.n_figures != None:
				if len(figs) >= args.n_figures:
					break

			figs.append( plt.figure(figsize=args.figsize, dpi=150) )

			ax = figs[-1].add_subplot(1,1,1)
			ax.set_xticks([])
			ax.set_yticks([])
			ax.set_xlabel(r'$\mu$', fontsize=22)
			ax.set_ylabel(r'$\mathrm{E} \left( B - V \right)$', fontsize=22)

		ax = figs[-1].add_subplot(n_rows, n_cols, k)

		img = f_stretch(pdf_indiv[i].T)
		img[~np.isfinite(img)] = np.min(img[np.isfinite(img)])
		
		ax.imshow(img, extent=bounds, origin='lower',
				  aspect='auto', cmap='Blues', interpolation='nearest')

		c = 'k'

		if lnZ[i] < lnZ_max - 10.:
			c = 'r'

		#ax.text(x_lnZ, y_lnZ, r'$%.1f$' % lnZ[i],
		#        ha='left', va='top', fontsize=10, color=c)

		#ax.text(x_idx, y_lnZ, r'$%d$' % i,
		#        ha='right', va='top', fontsize=10, color='k')

		if args.show_los:
			los.plot(ax)

		if args.show_clouds:
			clouds.plot(ax, c='k')

		if EBV_max != None:
			ax.set_ylim(x_min[1], EBV_max)

		ax.grid(which='minor', alpha=0.05)
		ax.grid(which='major', alpha=0.25)

		# Overplot manually-specified clouds
		if args.overplot_clouds != None:
			mu_cloud = np.array(args.overplot_clouds[::2])
			EBV_cloud = np.cumsum(np.array(args.overplot_clouds[1::2]))

			# Plot all paths
			N_clouds = len(args.overplot_clouds) / 2

			mu_all = np.zeros(2*(N_clouds+1), dtype='f8')
			EBV_all = np.zeros(2*(N_clouds+1), dtype='f8')
			mu_all[0] = DM_lim[0]
			mu_all[1:-1:2] = mu_cloud
			mu_all[2:-1:2] = mu_cloud
			mu_all[-1] = DM_lim[1]
			EBV_all[2:-1:2] = EBV_cloud
			EBV_all[3::2] = EBV_cloud

			ax.plot(mu_all, EBV_all, 'k--', lw=1.25, alpha=0.5)

		#if args.overplot_los != None:
		#	mu_los = np.linspace(4., 19., len(args.overplot_los)+1)

		ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
		ax.xaxis.set_minor_locator(AutoMinorLocator())
		ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
		ax.yaxis.set_minor_locator(AutoMinorLocator())

		ax.set_xticklabels([])
		ax.set_yticklabels([])

		ax.set_xlim(x_min[0], x_max[0])

	# Save/show plot
	base_fname = args.output

	if base_fname.endswith('.png'):
		base_fname = base_fname[:-4]

	for i, fig in enumerate(figs):
		print 'Saving figure %d of %d ...' % (i+1, len(figs))

		fig.subplots_adjust(bottom=0.12, top=0.98,
		                    left=0.12, right=0.98,
		                    wspace=0., hspace=0.)

		fig.savefig('%s.%.4d.png' % (base_fname, i), transparent=True, bbox_inches='tight', dpi=400)

	if args.show:
		plt.show()

	return 0

if __name__ == '__main__':
	main()
