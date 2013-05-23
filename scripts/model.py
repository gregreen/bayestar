#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  model.py
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



import sys, argparse
from os.path import abspath, expanduser

import matplotlib as mplib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator
from matplotlib.patches import Rectangle

import numpy as np

import scipy
from scipy.integrate import quad
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline, RectBivariateSpline



class TGalacticModel:
	rho_0 = None
	R0 = None
	Z0 = None
	H1, L1 = None, None
	f, H2, L2 = None, None, None
	fh, qh, nh, fh_outer, nh_outer, Rbr = None, None, None, None, None, None
	H_mu, Delta_mu, mu_FeH_inf = None, None, None
	
	def __init__(self, R0=8000., Z0=25., L1=2150., H1=245., f=0.13,
	                   L2=3261., H2=743., fh=0.0051, qh=0.70, nh=-2.62,
	                   nh_outer=-3.8, Rbr=27.8, rho_0=0.0058, Rep=500.,
	                   H_mu=500., Delta_mu=0.55, mu_FeH_inf=-0.82,
	                   LF_fname=expanduser('~/projects/bayestar/data/PSMrLF.dat')):
		self.R0, self.Z0 = R0, Z0
		self.L1, self.H1 = L1, H1
		self.f, self.L2, self.H2 = f, L2, H2
		self.fh, self.qh, self.nh, self.nh_outer, self.Rbr = fh, qh, nh, nh_outer, Rbr*1000.
		self.Rep = Rep
		self.rho_0 = rho_0
		self.H_mu, self.Delta_mu, self.mu_FeH_inf = H_mu, Delta_mu, mu_FeH_inf
		self.fh_outer = self.fh * (self.Rbr/self.R0)**(self.nh-self.nh_outer)
		#print self.fh_outer/self.fh
		
		self.data = np.loadtxt(abspath(LF_fname),
		                       usecols=(0,1),
		                       dtype=[('Mr','f4'), ('LF','f4')],
		                       unpack=False)
		self.Mr_min = np.min(self.data['Mr'])
		self.Mr_max = np.max(self.data['Mr'])
		self.LF = interp1d(self.data['Mr'], self.data['LF'], kind='linear')
		#self.LF = InterpolatedUnivariateSpline(LF['Mr'], LF['LF'])
	
	def Cartesian_coords(self, DM, cos_l, sin_l, cos_b, sin_b):
		d = 10.**(DM/5. + 1.)
		x = self.R0 - cos_l*cos_b*d
		y = -sin_l*cos_b*d
		z = sin_b*d
		return x, y, z
	
	def rho_thin(self, r, z):
		return self.rho_0 * np.exp(-(np.abs(z+self.Z0) - np.abs(self.Z0))/self.H1 - (r-self.R0)/self.L1)
	
	def rho_thick(self, r, z):
		return self.rho_0 * self.f * np.exp(-(np.abs(z+self.Z0) - np.abs(self.Z0))/self.H2 - (r-self.R0)/self.L2)
	
	def rho_halo(self, r, z):
		r_eff2 = r*r + (z/self.qh)*(z/self.qh) + self.Rep*self.Rep
		if type(r_eff2) == np.ndarray:
			ret = np.empty(r_eff2.size, dtype=np.float64)
			idx = (r_eff2 <= self.Rbr*self.Rbr)
			ret[idx] = self.rho_0 * self.fh * np.power(r_eff2[idx]/self.R0/self.R0, self.nh/2.)
			ret[~idx] = self.rho_0 * self.fh_outer * np.power(r_eff2[~idx]/self.R0/self.R0, self.nh_outer/2.)
			return ret
		else:
			if r_eff2 <= self.Rbr*self.Rbr:
				return self.rho_0 * self.fh * (r_eff2/self.R0/self.R0)**(self.nh/2.)
			else:
				return self.rho_0 * self.fh_outer * (r_eff2/self.R0/self.R0)**(self.nh_outer/2.)
	
	def f_halo(self, DM, cos_l, sin_l, cos_b, sin_b):
		x,y,z = self.Cartesian_coords(DM, cos_l, sin_l, cos_b, sin_b)
		r = np.sqrt(x*x + y*y)
		return self.rho_rz(r, z, component='halo') / self.rho_rz(r, z, component='disk')
	
	def rho_rz(self, r, z, component=None):
		if component == 'disk':
			return self.rho_thin(r,z) + self.rho_thick(r,z)
		elif component == 'thin':
			return self.rho_thin(r,z)
		elif component == 'thick':
			return self.rho_thick(r,z)
		elif component == 'halo':
			return self.rho_halo(r,z)
		else:
			return self.rho_thin(r,z) + self.rho_thick(r,z) + self.rho_halo(r,z)
	
	def rho(self, DM, cos_l, sin_l, cos_b, sin_b, component=None):
		x,y,z = self.Cartesian_coords(DM, cos_l, sin_l, cos_b, sin_b)
		r = np.sqrt(x*x + y*y)
		return self.rho_rz(r, z, component=component)
		'''if component == 'disk':
			return self.rho_thin(r,z) + self.rho_thick(r,z)
		elif component == 'thin':
			return self.rho_thin(r,z)
		elif component == 'thick':
			return self.rho_thick(r,z)
		elif component == 'halo':
			return self.rho_halo(r,z)
		else:
			return self.rho_thin(r,z) + self.rho_thick(r,z) + self.rho_halo(r,z)'''
	
	def dn_dDM(self, DM, cos_l, sin_l, cos_b, sin_b, radius=1., component=None):
		return self.rho(DM, cos_l, sin_l, cos_b, sin_b, component) * dV_dDM(DM, cos_l, sin_l, cos_b, sin_b, radius)
	
	def dn_dDM_corr(self, DM, m_max=23.):
		Mr_max = m_max - DM
		if Mr_max < self.LF['Mr'][0]:
			return 0.
		i_max = np.argmin(np.abs(self.LF['Mr'] - Mr_max))
		return np.sum(self.LF['LF'][:i_max+1])
	
	def mu_FeH_D(self, z):
		return self.mu_FeH_inf + self.Delta_mu*np.exp(-np.abs(z)/self.H_mu)
	
	def p_FeH(self, FeH, DM, cos_l, sin_l, cos_b, sin_b):
		x,y,z = self.Cartesian_coords(DM, cos_l, sin_l, cos_b, sin_b)
		r = np.sqrt(x*x + y*y)
		rho_halo_tmp = self.rho_halo(r,z)
		f_halo = rho_halo_tmp / (rho_halo_tmp + self.rho_thin(r,z) + self.rho_thick(r,z))
		# Disk metallicity
		a = self.mu_FeH_D(z) - 0.067
		p_D = 0.63*Gaussian(FeH, a, 0.2) + 0.37*Gaussian(FeH, a+0.14, 0.2)
		# Halo metallicity
		p_H = Gaussian(FeH, -1.46, 0.3)
		return (1.-f_halo)*p_D + f_halo*p_H
	
	def p_FeH_los(self, FeH, cos_l, sin_l, cos_b, sin_b, radius=1.,
	                                          DM_min=0.01, DM_max=100.):
		func = lambda x, Z: self.p_FeH(Z, x, cos_l, sin_l, cos_b, sin_b) * self.dn_dDM(x, cos_l, sin_l, cos_b, sin_b, radius)
		normfunc = lambda x: self.dn_dDM(x, cos_l, sin_l, cos_b, sin_b, radius)
		norm = quad(normfunc, DM_min, DM_max, epsrel=1.e-5, full_output=1)[0]
		ret = np.empty(len(FeH), dtype='f8')
		for i,Z in enumerate(FeH):
			ret[i] = quad(func, DM_min, DM_max, args=Z, epsrel=1.e-2, full_output=1)[0]
		return ret / norm
		#
		#return quad(func, DM_min, DM_max, epsrel=1.e-5)[0] / quad(normfunc, DM_min, DM_max, epsrel=1.e-5)[0]


def dV_dDM(DM, cos_l, sin_l, cos_b, sin_b, radius=1.):
	return (np.pi*radius**2.) * (1000.*2.30258509/5.) * np.exp(3.*2.30258509/5. * DM)


def Gaussian(x, mu=0., sigma=1.):
	Delta = (x-mu)/sigma
	return np.exp(-Delta*Delta/2.) / 2.50662827 / sigma


class TStellarModel:
	'''
	Loads the given stellar model, and provides access to interpolated
	colors on (Mr, FeH) grid.
	'''
	
	def __init__(self, template_fname):
		self.load_templates(template_fname)
	
	def load_templates(self, template_fname):
		'''
		Load in stellar template colors from an ASCII file. The colors
		should be stored in the following format:
		
		#
		# Arbitrary comments
		#
		# Mr    FeH   gr     ri     iz     zy
		# 
		-1.00 -2.50 0.5132 0.2444 0.1875 0.0298
		-0.99 -2.50 0.5128 0.2442 0.1873 0.0297
		...
		
		or something similar. A key point is that there be a row
		in the comments that lists the names of the colors. The code
		identifies this row by the presence of both 'Mr' and 'FeH' in
		the row, as above. The file must be whitespace-delimited, and
		any whitespace will do (note that the whitespace is not required
		to be regular).
		'''
		
		f = open(abspath(template_fname), 'r')
		row = []
		self.color_name = ['gr', 'ri', 'iz', 'zy']
		for l in f:
			line = l.rstrip().lstrip()
			if len(line) == 0:	# Empty line
				continue
			if line[0] == '#':	# Comment
				if ('Mr' in line) and ('FeH' in line):
					try:
						self.color_name = line.split()[3:]
					except:
						pass
				continue
			data = line.split()
			if len(data) < 6:
				print 'Error reading in stellar templates.'
				print 'The following line does not have the correct number of entries (6 expected):'
				print line
				return 0
			row.append([float(s) for s in data])
		f.close()
		template = np.array(row, dtype=np.float64)
		
		# Organize data into record array
		dtype = [('Mr','f4'), ('FeH','f4')]
		for c in self.color_name:
			dtype.append((c, 'f4'))
		self.data = np.empty(len(template), dtype=dtype)
		
		self.data['Mr'] = template[:,0]
		self.data['FeH'] = template[:,1]
		for i,c in enumerate(self.color_name):
			self.data[c] = template[:,i+2]
		
		self.MrFeH_bounds = [[np.min(self.data['Mr']), np.max(self.data['Mr'])],
		                     [np.min(self.data['FeH']), np.max(self.data['FeH'])]]
		
		# Produce interpolating class with data
		self.Mr_coords = np.unique(self.data['Mr'])
		self.FeH_coords = np.unique(self.data['FeH'])
		
		self.interp = {}
		for c in self.color_name:
			tmp = self.data[c][:]
			tmp.shape = (len(self.FeH_coords), len(self.Mr_coords))
			self.interp[c] = RectBivariateSpline(self.Mr_coords,
			                                     self.FeH_coords,
			                                     tmp.T,
			                                     kx=3,
			                                     ky=3,
			                                     s=0)
	
	def color(self, Mr, FeH, name=None):
		'''
		Return the colors, evaluated at the given points in
		(Mr, FeH)-space.
		
		Inputs:
		    Mr    float or array of floats
		    FeH   float or array of floats
		    name  string, or list of strings, with names of colors to
		          return. By default, all colors are returned.
		
		Output:
		    color  numpy record array of colors
		'''
		
		if name == None:
			name = self.get_color_names()
		elif type(name) == str:
			name = [name]
		
		if type(Mr) == float:
			Mr = np.array([Mr])
		elif type(Mr) == list:
			Mr = np.array(Mr)
		if type(FeH) == float:
			FeH = np.array([FeH])
		elif type(FeH) == list:
			FeH = np.array(FeH)
		
		dtype = []
		for c in name:
			if c not in self.color_name:
				raise ValueError('No such color in model: %s' % c)
			dtype.append((c, 'f4'))
		ret_color = np.empty(Mr.size, dtype=dtype)
		
		for c in name:
			ret_color[c] = self.interp[c].ev(Mr, FeH)
		
		return ret_color
	
	def absmags(self, Mr, FeH):
		'''
		Return the absolute magnitude in each bandpass corresponding to
		(Mr, FeH).
		
		Inputs:
		    Mr   r-band absolute magnitude of the star(s) (float or numpy array)
		    FeH  Metallicity of the star(s) (float or numpy array)
		
		Output:
		    M    Absolute magnitude in each band for each star (numpy record array)
		'''
		
		c = self.color(Mr, FeH)
		
		dtype = [('g','f8'), ('r','f8'), ('i','f8'), ('z','f8'), ('y','f8')]
		M = np.empty(c.shape, dtype=dtype)
		
		M['r'] = Mr
		M['g'] = c['gr'] + Mr
		M['i'] = Mr - c['ri']
		M['z'] = M['i'] - c['iz']
		M['y'] = M['z'] - c['zy']
		
		return M
	
	def get_color_names(self):
		'''
		Return the names of the colors in the templates.
		
		Ex.: For PS1 colors, this would return
		     ['gr', 'ri', 'iz', 'zy']
		'''
		
		return self.color_name


def main():
	
	return 0

if __name__ == '__main__':
	main()

