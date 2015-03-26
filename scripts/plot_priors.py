#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#
#       plot_priors.py
#       
#       Copyright 2011 Gregory <greg@greg-G53JW>
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


import sys, argparse
import matplotlib as mplib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator
from matplotlib.patches import Rectangle
import numpy as np
from scipy.integrate import quad
from math import pi, sqrt, log, exp, sin, cos
from os.path import abspath

from astroutils import parse_RA, parse_DEC, parse_dhms, equatorial2galactic, rad2deg, deg2rad



from model import TGalacticModel


def dV_dDM(DM, cos_l, sin_l, cos_b, sin_b, radius=1.):
	return (np.pi*radius**2.) * (1000.*2.30258509/5.) * np.exp(3.*2.30258509/5. * DM)


def Gaussian(x, mu=0., sigma=1.):
	Delta = (x-mu)/sigma
	return np.exp(-Delta*Delta/2.) / 2.50662827 / sigma


def plot_hist(ax, bin_x, p_x, var_name):
	ax.fill_between(bin_x, 0, p_x, alpha=0.8)
	ax.set_xlabel(r'$%s$'%var_name, fontsize=14)
	ax.set_ylabel(r'$\mathrm{d}n / \mathrm{d}%s$'%var_name, fontsize=14)
	return ax.get_xlim(), ax.get_ylim()


def plot_Galactic_slice(Rmax, Zmax, component=None, log_scale=True):
	# Determine density in slice
	model = TGalacticModel()
	R = np.abs(np.linspace(-Rmax*1000, Rmax*1000, 500))
	Z = np.abs(np.linspace(-Zmax*1000, Zmax*1000, 500))
	RR, ZZ = np.meshgrid(R, Z)
	rho = model.rho_rz(RR.flatten(), ZZ.flatten(), component=component)
	
	# Remove overdensity at center
	idx = np.isfinite(rho)
	rho[~idx] = np.max(rho[idx])
	rho_sorted = rho.__copy__()
	rho_sorted.sort()
	rho_max = rho_sorted[int(0.95*(rho_sorted.size-1))]
	idx = (rho > rho_max)
	rho[idx] = rho_max
	
	if log_scale:
		rho = np.log(rho)
	
	rho.shape = (R.size, Z.size)
	
	# Set matplotlib options
	mplib.rc('text',usetex=True)
	mplib.rc('xtick.major', size=6)
	mplib.rc('xtick.minor', size=4)
	mplib.rc('axes', grid=True)
	
	# Set up figure
	fig = plt.figure(figsize=(12,7), dpi=100)
	title = None
	if log_scale:
		title = r'$\mathrm{Stellar\ number\ density\ (arbitrary\ log\ scale)}$'
	else:
		r'$\mathrm{Stellar\ number\ density\ (arbitrary\ scale)}$'
	fig.suptitle(title, fontsize=22, y=0.95)
	ax = fig.add_subplot(1,1,1)
	fig.subplots_adjust(top=0.88)
	ax.imshow(rho, extent=(-Rmax, Rmax, -Zmax, Zmax), cmap='gray')
	ax.set_xlabel(r'$R \, (\mathrm{kpc})$', fontsize=18)
	ax.set_ylabel(r'$Z \, (\mathrm{kpc})$', fontsize=18)
	ax.grid(which='major', alpha=0.2)
	
	fig.savefig('../plots/log_rho.png', dpi=300)
	plt.show()


def main():
	parser = argparse.ArgumentParser(prog='plot_prior.py',
	            description='Plot galstar priors along a line of sight',
	            add_help=True)
	parser.add_argument('RA',
	                    type=str, help='Right Ascension (hh:mm:ss) or degrees')
	parser.add_argument('DEC',
	                    type=str, help='Declination (deg:mm:ss) or degrees')
	parser.add_argument('Radius',
	                    type=str,
	                    nargs='?',
	                    default="'1d'",
	                    help='Radius (hh:mm:ss) or degrees')
	parser.add_argument('--lb',
	                    action='store_true',
	                    help='Interpret RA and DEC as Galactic l and b, respectively')
	parser.add_argument('--correct',
	                    action='store_true',
	                    help='Correct for Malmquist bias.')
	#parser.add_argument('--density', action='store_true', help='Plot stellar number density rather than p(mu)')
	parser.add_argument('--output',
	                    type=str,
	                    default=None,
	                    help='Save output to file (default: open window with output)')
	if sys.argv[0] == 'python':
		offset = 2
	else:
		offset = 1
	args = []
	for arg in sys.argv[offset:]:
		args.append(arg)
		if '--' not in arg: args[-1] = "'%s'" % args[-1]
	values = parser.parse_args(args)
	
	# Get Galactic coordinates
	if values.lb:
		l = float(values.RA[1:-1])
		b = float(values.DEC[1:-1])
	else:
		# Parse RA and DEC
		RA = parse_dhms(values.RA[1:-1])
		if RA == None: RA = parse_RA(values.RA[1:-1])
		DEC = parse_dhms(values.DEC[1:-1])
		if DEC == None: DEC = parse_DEC(values.DEC[1:-1])
		l, b = equatorial2galactic(RA, DEC)
	
	# Parse Radius
	radius = parse_dhms(values.Radius[1:-1])
	if radius == None: radius = float(values.Radius[1:-1])
	radius = deg2rad(radius)
	
	print '# Galactic Coordinates (l, b): %d %d' % (round(l) , round(b))
	
	# Set up figure
	mplib.rc('text',usetex=True)
	mplib.rc('xtick.major', size=6)
	mplib.rc('xtick.minor', size=4)
	mplib.rc('axes', grid=True)
	fig = plt.figure(figsize=(8.5, 11.), dpi=100)
	fig.suptitle(r'$\mathrm{Priors \ for} \ ( \ell = %d^{\circ} , \, b = %d^{\circ} )$' % (round(l) , round(b)), fontsize=24, y=0.97)
	ax = []
	for i in range(2):
		ax.append(fig.add_subplot(2, 1, i+1))
	#fig.subplots_adjust(left=0.12, right=0.90, hspace=0.25)
	fig.subplots_adjust(left=0.14, right=0.90, top=0.88, bottom=0.08, hspace=0.25)
	
	# Intialize Galactic model
	model = TGalacticModel(fh=0.0006)
	model.L_epsilon = 500.
	
	# Precompute trigonometric functions
	l, b  = deg2rad(l), deg2rad(b)
	cos_l, sin_l, cos_b, sin_b = cos(l), sin(l), cos(b), sin(b)
	
	# Determine total number of stars along line of sight
	f_disk = lambda x: model.dn_dDM(x, cos_l, sin_l, cos_b, sin_b,
	                                radius, component='disk',
	                                correct=values.correct)
	N_disk = quad(f_disk, 0.01, 20., epsrel=1.e-5)[0]
	print '# of stars in disk: %d' % int(N_disk)
	
	f_halo = lambda x: model.dn_dDM(x, cos_l, sin_l, cos_b, sin_b,
	                                radius, component='halo',
	                                correct=values.correct)
	N_halo = quad(f_halo, 0.01, 40., epsrel=1.e-5)[0]
	print '# of stars in halo: %d' % int(N_halo)
	
	f_bulge = lambda x: model.dn_dDM(x, cos_l, sin_l, cos_b, sin_b,
	                                 radius, component='bulge',
	                                 correct=values.correct)
	N_bulge = quad(f_bulge, 0.01, 40., epsrel=1.e-5)[0]
	print '# of stars in bulge: %d' % int(N_bulge)
	
	# Calculate dn/dDM and rho
	DM_range = np.linspace(4., 20., 500)
	dn_dDM_disk, dn_dDM_halo, dn_dDM_bulge = [np.empty(500, dtype='f8') for i in xrange(3)]
	rho_disk, rho_halo, rho_bulge = [np.empty(500, dtype='f8') for i in xrange(3)]
	
	for i,DM in enumerate(DM_range):
		rho_halo[i] = model.rho(DM, cos_l, sin_l, cos_b, sin_b, component='halo')
		rho_disk[i] = model.rho(DM, cos_l, sin_l, cos_b, sin_b, component='disk')
		rho_bulge[i] = model.rho(DM, cos_l, sin_l, cos_b, sin_b, component='bulge')
		
		dn_dDM_halo[i] = 1./(N_halo+N_disk+N_bulge) * model.dn_dDM(DM,
		                                          cos_l, sin_l,
		                                          cos_b, sin_b,
		                                          radius, component='halo',
		                                          correct=values.correct)
		dn_dDM_disk[i] = 1./(N_halo+N_disk+N_bulge) * model.dn_dDM(DM,
		                                          cos_l, sin_l,
		                                          cos_b, sin_b,
		                                          radius, component='disk',
		                                          correct=values.correct)
		
		dn_dDM_bulge[i] = 1./(N_halo+N_disk+N_bulge) * model.dn_dDM(DM,
		                                          cos_l, sin_l,
		                                          cos_b, sin_b,
		                                          radius, component='bulge',
		                                          correct=values.correct)
	
	rho_bulge[~np.isfinite(rho_bulge)] = 0.
	dn_dDM_bulge[~np.isfinite(dn_dDM_bulge)] = 0.
	
	print np.sum(dn_dDM_bulge)
	
	rho = rho_disk + rho_halo + rho_bulge
	dn_dDM = dn_dDM_halo + dn_dDM_disk + dn_dDM_bulge
	
	# Plot dn/dDM
	y_min = min(dn_dDM_disk.min(), dn_dDM_halo.min())
	
	ax[0].fill_between(DM_range, y_min, dn_dDM, alpha=0.4, facecolor='k', label='__nolabel__')
	
	ax[0].fill_between(DM_range, y_min, dn_dDM_disk, alpha=0.4, facecolor='g')
	rect = Rectangle((0,0), 0, 0, facecolor='g', label=r'$\mathrm{disk}$', alpha=0.6)
	ax[0].add_patch(rect)
	
	ax[0].fill_between(DM_range, y_min, dn_dDM_halo, alpha=0.4, facecolor='b')
	rect = Rectangle((0,0), 0, 0, facecolor='b', label=r'$\mathrm{halo}$', alpha=0.6)
	ax[0].add_patch(rect)
	
	ax[0].fill_between(DM_range, y_min, dn_dDM_bulge, alpha=0.4, facecolor='r')
	rect = Rectangle((0,0), 0, 0, facecolor='r', label=r'$\mathrm{bulge}$', alpha=0.6)
	ax[0].add_patch(rect)
	
	ax[0].set_title(r'$\mathrm{Probability}$', fontsize=20)
	ax[0].set_xlabel(r'$\mu$', fontsize=20)
	ax[0].set_ylabel(r'$p(\mu)$', fontsize=18)
	ax[0].yaxis.set_major_locator(MaxNLocator(nbins=5))
	ax[0].yaxis.set_minor_locator(MaxNLocator(nbins=20))
	y_max = ax[0].get_ylim()[1]
	ax[0].set_ylim(0., y_max)
	ax[0].legend(loc='upper left')
	ax[0].get_legend().get_frame().set_alpha(0.)
	
	# Plot rho
	y_min = min(dn_dDM_disk.min(), dn_dDM_halo.min())
	ax[1].fill_between(DM_range, y_min, rho, alpha=0.4, facecolor='k')
	ax[1].fill_between(DM_range, y_min, rho_disk, alpha=0.4, facecolor='g')
	ax[1].fill_between(DM_range, y_min, rho_halo, alpha=0.4, facecolor='b')
	ax[1].fill_between(DM_range, y_min, rho_bulge, alpha=0.4, facecolor='r')
	ax[1].set_title(r'$\mathrm{Density}$', fontsize=20)
	ax[1].set_xlabel(r'$\mu$', fontsize=20)
	ax[1].set_ylabel(r'$n (\mu)$', fontsize=18)
	ax[1].set_yscale('log')
	y_min = max(dn_dDM_disk.min(), dn_dDM_halo.min())
	y_max = ax[1].get_ylim()[1]
	ax[1].set_ylim(y_min, y_max)
	
	'''
	# dn/dFeH
	FeH_range = np.linspace(-2.5, 0.5, 100)
	p_FeH_range = np.empty(100, dtype=float)
	for i,FeH in enumerate(FeH_range):
		p_FeH_range[i] = model.p_FeH_los(FeH, cos_l, sin_l, cos_b, sin_b, DM_max=20.)
	func = lambda x: model.p_FeH_los(x, cos_l, sin_l, cos_b, sin_b, DM_max=20.)
	norm = quad(func, -2.5, 0.5, epsrel=1.e-3)[0]
	for i in range(len(FeH_range)): FeH_range[i] /= norm
	ax[1].fill_between(FeH_range, 0, p_FeH_range, alpha=0.8)
	ax[1].set_xlabel(r'$[Fe/H]$', fontsize=17)
	ax[1].set_ylabel(r'$p([Fe/H])$', fontsize=16)
	y_max = ax[1].get_ylim()[1]
	ax[1].set_ylim(0., y_max)
	ax[1].set_xlim(-2.5, 0.5)
	ax[1].xaxis.set_major_locator(MaxNLocator(nbins=4, prune='lower'))
	ax[1].xaxis.set_minor_locator(MaxNLocator(nbins=20))
	'''
	
	for ax_i in ax:
		ax_i.set_xlim(4., 20.)
		ax_i.xaxis.set_major_locator(MultipleLocator(4.))
		ax_i.xaxis.set_minor_locator(MultipleLocator(1.))
		#for tick in ax_i.xaxis.get_major_ticks() + ax_i.xaxis.get_minor_ticks() + ax_i.yaxis.get_major_ticks() + ax_i.yaxis.get_minor_ticks():
		#	tick.tick2On = False
		ax_i.grid(which='minor', alpha=0.3)
	
	if values.output == None:
		plt.show()
	else:
		fn = values.output[1:-1]
		if '.' not in fn: fn += '.png' 
		fig.savefig(fn, dpi=300)
	
	return 0

if __name__ == '__main__':
	main()

