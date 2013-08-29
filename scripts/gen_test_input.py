#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  gen_test_input.py
#  
#  Copyright 2012 Greg Green <greg@greg-UX31A>
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

import os, sys, argparse

import numpy as np

from scipy.special import erf
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.integrate import quad

import matplotlib.pyplot as plt
import matplotlib as mplib

from model import TGalacticModel, TStellarModel
from wrap_bayestar import write_infile, write_true_params

fh = 0.0051

class TSample1D:
	'''
	Draw samples from a 1D probability density function.
	'''
	
	def __init__(self, f, x_min, x_max, N=100, M=1000):
		x = np.linspace(x_min, x_max, N)
		try:
			p_x = f(x)
		except:
			p_x = [f(xx) for xx in x]
		P = np.zeros(N, dtype='f4')
		for i in xrange(1,N-1):
			P[i] = P[i-1] + 0.5*p_x[i-1]+0.5*p_x[i]
		P[-1] = P[-2] + 0.5*p_x[-1]
		P /= np.sum(p_x)
		P[-1] = 1.
		if N < M:
			P_spl = InterpolatedUnivariateSpline(x, P)
			x = np.linspace(x_min, x_max, M)
			P = P_spl(x)
			P[0] = 0.
			P[-1] = 1.
		self.x = interp1d(P, x, kind='linear')
	
	def __call__(self, N=1):
		P = np.random.random(N)
		return self.x(P)
	
	def get_x(self, P):
		return self.x(P)

def mock_mags(stellarmodel, mu, Ar, Mr, FeH, mag_limit=(23., 23., 23., 23., 23.)):
	# Apparent magnitudes
	m = mu + stellarmodel.absmags(Mr, FeH)
	err = np.empty(m.size, m.dtype)
	
	# Apply extinction and add in errors
	bands = ['g','r','i','z','y']
	Ab = np.array([3.172, 2.271, 1.682, 1.322, 1.087])
	for b,A,lim in zip(bands,Ab,mag_limit):
		m[b] += Ar * A / Ab[1]
		
		err[b] = 0.02 + 0.1 * np.exp(m[b] - lim - 1.5)
		m[b] += sigma * np.random.normal(size=mu.size)
		
		idx = (m[b] > lim)
		m[b][idx] = np.nan
	
	return m

def observed(mags, mag_lim):
	pass

def err_model(mag, mag_lim):
	err = np.sqrt(0.02*0.02 + 0.2 * np.exp(2. * (mag - mag_lim + 0.1) / 0.25))
	
	idx = (err > 1.)
	err[idx] = 1.
	
	return err

def draw_from_model(l, b, N, EBV_spread=0.02,
                    mag_lim=(23., 22., 22., 21., 20.),
                    EBV_of_mu=None, EBV_uniform=False,
                    redraw=True, n_bands=4):
	dtype = [('DM', 'f8'), ('EBV', 'f8'),
	         ('Mr', 'f8'), ('FeH', 'f8'),
	         ('mag', '5f8'), ('err', '5f8')]
	ret = np.empty(N, dtype=dtype)
	
	l = np.pi/180. * l
	b = np.pi/180. * b
	cos_l, sin_l = np.cos(l), np.sin(l)
	cos_b, sin_b = np.cos(b), np.sin(b)
	
	gal_model = TGalacticModel(fh=fh)
	stellar_model = TStellarModel(os.path.expanduser('~/projects/bayestar/data/PScolors.dat'))
	R = np.array([3.172, 2.271, 1.682, 1.322, 1.087])
	
	mu_max = mag_lim[1] - gal_model.Mr_min + 3.
	mu_min = min(0., mu_max-25.)
	Mr_max = min(mag_lim[1], gal_model.Mr_max)
	
	dN_dDM = lambda mu: gal_model.dn_dDM(mu, cos_l, sin_l, cos_b, sin_b)
	
	# Set up 1D samplers
	draw_mu = TSample1D(dN_dDM, mu_min, mu_max, 500, 10000)
	draw_Mr = TSample1D(gal_model.LF, gal_model.Mr_min, Mr_max, 10000, 1)
	
	idx = np.arange(N)
	
	keep_sampling = True
	
	while keep_sampling:
		size = idx.size
		print 'Drawing %d...' % size
		
		# Draw DM and Mr
		ret['DM'][idx] = draw_mu(size)
		ret['Mr'][idx] = draw_Mr(size)
		
		# Draw EBV
		ret['EBV'][idx] = 0.
		if EBV_uniform:
			ret['EBV'][idx] += 2. * np.random.random(size=idx.size)
		else:	
			if EBV_of_mu != None:
				ret['EBV'][idx] += EBV_of_mu(ret['DM'][idx]) #+ np.random.normal(scale=EBV_spread, size=size)
			ret['EBV'][idx] += EBV_spread * np.random.chisquare(1., size)
		
		x, y, z = gal_model.Cartesian_coords(ret['DM'][idx], cos_l,
		                                     sin_l, cos_b, sin_b)
		
		# Determine which component stars belong to
		halo = np.random.random(size) < gal_model.f_halo(ret['DM'][idx], cos_l,
		                                                 sin_l, cos_b, sin_b)
		thin = ~halo & (np.random.random(size) < 0.63)
		thick = ~halo & ~thin
		
		# Assign metallicities to halo stars
		while np.any(halo):
			ret['FeH'][idx[halo]] = np.random.normal(-1.46, 0.3, size=np.sum(halo))
			halo &= (ret['FeH'][idx] <= -2.5) | (ret['FeH'][idx] >= 0.)
		
		# Assign metallicities to thin-disk stars
		while np.any(thin):
			ret['FeH'][idx[thin]] = np.random.normal(gal_model.mu_FeH_D(z[thin])-0.067,
			                                         0.2, size=np.sum(thin))
			thin &= (ret['FeH'][idx] <= -2.5) | (ret['FeH'][idx] >= 0.)
		
		# Assign metallicities to thick-disk stars
		while np.any(thick):
			ret['FeH'][idx[thick]] = np.random.normal(gal_model.mu_FeH_D(z[thick])-0.067+0.14,
			                                          0.2, size=np.sum(thick))
			thick &= (ret['FeH'][idx] <= -2.5) | (ret['FeH'][idx] >= 0.)
		
		# Calculate absolute stellar magnitudes
		absmags_tmp = stellar_model.absmags(ret['Mr'][idx], ret['FeH'][idx])
		ret['mag'][idx,0] = absmags_tmp['g']
		ret['mag'][idx,1] = absmags_tmp['r']
		ret['mag'][idx,2] = absmags_tmp['i']
		ret['mag'][idx,3] = absmags_tmp['z']
		ret['mag'][idx,4] = absmags_tmp['y']
		
		# Determine errors and apparent magnitudes
		for k in xrange(5):
			ret['mag'][idx,k] += ret['DM'][idx]
			ret['mag'][idx,k] += ret['EBV'][idx] * R[k]
			ret['err'][idx,k] = err_model(ret['mag'][idx][:,k], mag_lim[k])
			#0.02 + 0.3 * np.exp(ret['mag'][idx][:,k] - mag_lim[k])
			#idx_tmp = ret['err'][idx,k] > 1.
			#ret['err'][idx[idx_tmp],k] = 1.
		
		# Calculate observation probability
		p_obs = np.empty((size, 5), dtype='f8')
		
		for k in xrange(5):
			p_obs[:,k] = 0.5 - 0.5 * erf((ret['mag'][idx,k] - mag_lim[k] + 0.1) / 0.25)
		
		# Determine which bands stars are observed in
		obs = (p_obs > np.random.random(size=(size, 5)))
		
		# Add in errors to apparent stellar magnitudes
		ret['mag'][idx] += ret['err'][idx] * np.random.normal(size=(size,5))
		
		# Re-estimate errors based on magnitudes
		for k in xrange(5):
			ret['err'][idx,k] = err_model(ret['err'][idx,k], mag_lim[k])
		
		# Remove observations with errors above 0.2 mags
		obs = obs & (ret['err'][idx] < 0.2)
		
		for k in xrange(5):
			ret['mag'][idx[~obs[:,k]],k] = 0.
			ret['err'][idx[~obs[:,k]],k] = 1.e10
		
		# Require detection in g and at least n_bands-1 other bands
		obs = obs[:,0] & (np.sum(obs, axis=1) >= n_bands)
		
		idx = idx[~obs]
		
		if redraw:
			if idx.size == 0:
				keep_sampling = False
		else:
			keep_sampling = False
			
			ret = ret[obs]
	
	return ret


def draw_flat(N, Ar=0.5):
	dtype = [('DM', 'f8'), ('Ar', 'f8'), ('Mr', 'f8'), ('FeH', 'f8')]
	ret = np.empty(N, dtype=dtype)
	
	idx = np.ones(N, dtype=np.bool)
	while np.any(idx):
		ret['DM'][idx] = np.random.rand(N) * 13.5 + 5.5
		ret['Ar'][idx] = np.random.rand(N) * 2. * Ar
		ret['Mr'][idx] = np.random.rand(N) * 20. - 0.8
		idx = (ret['DM'] + ret['Ar'] + ret['Mr'] > 23.)
	
	ret['FeH'] = np.random.rand(N) * 2.4 - 2.45
	
	return ret


def main():
	parser = argparse.ArgumentParser(prog='gen_test_input.py',
	                                 description='Generates test input file for galstar.',
	                                 add_help=True)
	parser.add_argument('-N', type=int, default=None, help='# of stars to generate.')
	parser.add_argument('-rad', '--radius', type=float, default=None, help='Radius of beam.')
	parser.add_argument('-o', '--output', type=str, default=None,
	                    help='Output filename (creates Bayestar input file).')
	parser.add_argument('-lb', '--gal-lb', type=float, nargs=2,
	                    metavar='deg', default=(90., 10.),
	                    help='Galactic latitude and longitude, in degrees.')
	parser.add_argument('-EBV', '--mean-EBV', type=float, default=0.02,
	                    metavar='mags', help='Mean E(B-V) extinction.')
	parser.add_argument('--EBV-uniform', action='store_true',
	                    help='Draw E(B-V) from U(0,2).')
	parser.add_argument('-cl', '--clouds', type=float, nargs='+',
	                    default=None, metavar='mu Delta_EBV',
	                    help='Place clouds of reddening Delta_EBV at distances mu')
	parser.add_argument('-r', '--max-r', type=float, default=23.,
	                    metavar='mags', help='Limiting apparent r-band magnitude.')
	parser.add_argument('-lim', '--limiting-mag', metavar='mags', type=float,
	                    nargs=5, default=(22.5, 22.5, 22., 21., 20.),
	                    help='Limiting magnitudes in grizy.')
	parser.add_argument('-nb', '--n-bands', type=int, default=4,
	                    help='# of bands required to keep object.')
	parser.add_argument('-flat', '--flat', action='store_true',
	                    help='Draw parameters from flat distribution')
	parser.add_argument('-sh', '--show', action='store_true',
	                    help='Plot distribution of DM, Mr and E(B-V).')
	#parser.add_argument('-b', '--binary', action='store_true', help='Generate binary stars.')
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	args = parser.parse_args(sys.argv[offset:])
	
	# Determine number of stars to draw
	redraw = False
	N_stars = None
	
	if args.N == None:
		if args.radius == None:
			print 'Either -N or -rad must be specified'
		
		model = TGalacticModel(fh=fh)
		N_stars = model.tot_num_stars(args.gal_lb[0], args.gal_lb[1], args.radius)
		N_stars = np.random.poisson(lam=N_stars)
	else:
		if args.radius != None:
			print 'Cannot specify both -N and -rad'
		
		redraw = True
		N_stars = args.N
	
	EBV_of_mu = None
	if args.clouds != None:
		mu = np.linspace(-5., 35., 1000)
		dmu = mu[1] - mu[0]
		Delta_EBV = 0.01 * dmu * np.ones(mu.size)
		for i in range(len(args.clouds)/2):
			s = 0.05
			m = args.clouds[2*i]
			EBV = args.clouds[2*i+1]
			Delta_EBV += EBV/np.sqrt(2.*np.pi)/s*np.exp(-(mu-m)*(mu-m)/2./s/s)
		EBV = np.cumsum(Delta_EBV) * dmu
		EBV_of_mu = InterpolatedUnivariateSpline(mu, EBV)
		mu = np.linspace(4., 19., 1000) 
		EBV = EBV_of_mu(mu)
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)
		ax.plot(mu, EBV)
		plt.show()
		#print Ar
	
	params = None
	if args.flat:
		pass
		#params = draw_flat(values.N, EBV_spread=args.mean_EBV,
		#                   r_max=values.max_r, EBV_of_mu=EBV_of_mu)
	else:
		params = draw_from_model(args.gal_lb[0], args.gal_lb[1],
		                         N_stars, EBV_spread=args.mean_EBV,
		                         mag_lim=args.limiting_mag, EBV_of_mu=EBV_of_mu,
		                         EBV_uniform=args.EBV_uniform, redraw=redraw,
		                         n_bands=args.n_bands)
		print '%d stars observed' % (len(params))
	
	# Write Bayestar input file
	if args.output != None:
		mag_lim = np.array(args.limiting_mag)
		mag_lim.shape = (1, 5)
		mag_lim = np.repeat(mag_lim, len(params), axis=0)
		write_infile(args.output, params['mag'], params['err'], mag_lim,
		             l=args.gal_lb[0], b=args.gal_lb[1],
		             access_mode='w')
		
		# Write true parameter values
		write_true_params(args.output, params['DM'], params['EBV'],
		                  params['Mr'], params['FeH'],
		                  l=args.gal_lb[0], b=args.gal_lb[1])
	
	header = '''# Format:
# l  b
# DM  E(B-V)  Mr  FeH
# DM  E(B-V)  Mr  FeH
# DM  E(B-V)  Mr  FeH
# ...'''
	#print header
	#print '%.3f  %.3f' % (args.gal_lb[0], args.gal_lb[1])
	#for p in params:
	#	print '%.3f  %.3f  %.3f  %.3f' % (p['DM'], p['EBV'], p['Mr'], p['FeH']), p['mag'], p['err']
	
	if args.show:
		model = TGalacticModel(fh=fh)
		l = np.pi/180. * args.gal_lb[0]
		b = np.pi/180. * args.gal_lb[1]
		cos_l, sin_l = np.cos(l), np.sin(l)
		cos_b, sin_b = np.cos(b), np.sin(b)
		dN_dDM = lambda mu: model.dn_dDM(mu, cos_l, sin_l, cos_b, sin_b)
		
		mplib.rc('text', usetex=True)
		
		fig = plt.figure(figsize=(6,4), dpi=300)
		
		ax = fig.add_subplot(2,2,1)
		ax.hist(params['DM'], bins=100, normed=True, alpha=0.3)
		xlim = ax.get_xlim()
		x = np.linspace(xlim[0], xlim[1], 1000)
		ax.plot(x, dN_dDM(x)/quad(dN_dDM, 1., 25.)[0], 'g-', lw=1.3, alpha=0.5)
		ax.set_xlim(xlim)
		ax.set_xlabel(r'$\mu$', fontsize=14)
		
		ax = fig.add_subplot(2,2,2)
		ax.hist(params['Mr'], bins=100, normed=True, alpha=0.3)
		xlim = ax.get_xlim()
		x = np.linspace(model.Mr_min, model.Mr_max, 1000)
		ax.plot(x, model.LF(x)/quad(model.LF, x[0], x[-1], full_output=1)[0],
		                                               'g-', lw=1.3, alpha=0.5)
		ax.set_xlim(xlim)
		ax.set_xlabel(r'$M_{r}$', fontsize=14)
		
		ax = fig.add_subplot(2,2,3)
		ax.hist(params['EBV'], bins=100, normed=True, alpha=0.3)
		ax.set_xlabel(r'$\mathrm{E} \! \left( B \! - \! V \right)$', fontsize=14)
		
		ax = fig.add_subplot(2,2,4)
		ax.hist(params['FeH'], bins=100, normed=True, alpha=0.3)
		xlim = ax.get_xlim()
		x = np.linspace(xlim[0], xlim[1], 100)
		y = model.p_FeH_los(x, cos_l, sin_l, cos_b, sin_b)
		ax.plot(x, y, 'g-', lw=1.3, alpha=0.5)
		ax.set_xlabel(r'$\left[ \mathrm{Fe} / \mathrm{H} \right]$', fontsize=14)
		ax.set_xlim(xlim)
		
		fig.subplots_adjust(hspace=0.40, wspace=0.25,
		                    bottom=0.13, top=0.95,
		                    left=0.1, right=0.9)
		
		# CMD of stars
		
		fig = plt.figure(figsize=(6,4), dpi=150)
		ax = fig.add_subplot(1,1,1)
		idx = ((params['err'][:,0] < 1.e9)
		       & (params['err'][:,1] < 1.e9)
		       & (params['err'][:,2] < 1.e9))
		ax.hexbin(params['mag'][idx,0] - params['mag'][idx,2],
		          params['mag'][idx,1],
		          gridsize=100, bins='log')
		ax.set_xlabel(r'$g - i$', fontsize=14)
		ax.set_ylabel(r'$m_{r}$', fontsize=14)
		ylim = ax.get_ylim()
		ax.set_ylim(ylim[1], ylim[0])
		
		plt.show()
	
	return 0

if __name__ == '__main__':
	main()

