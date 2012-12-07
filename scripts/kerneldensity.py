#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  kerneldensity.py
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

import numpy as np


class TKernelDensity:
	'''
	Kernel density class, using a Gaussian kernel.
	'''
	
	def __init__(self, x0, cov=None):
		'''
		Kernel density class, using a Gaussian kernel.
		
		Intiate with a numpy array of the form
		
		    x0[npoints, dimension]
		
		A kernel covariance can also be provided, as a full
		covariance matrix, a list of standard deviations, or a single
		(universal) standard deviation. If no covariance is provided,
		a covariance is guessed based on the input points.
		'''
		
		self.nkern = x0.shape[0]
		self.ndim = x0.shape[1]
		self.x0 = x0
		self.set_bandwidth(cov)
	
	def kernel(self, dist2):
		'''
		Returns the probability density (up to a normalizing factor),
		given a set of squared distances:
		
		    p ~ exp(-0.5*d^2)
		
		Each distance is given by
		
		    d^2 = x_i K^{-1}_{ij} x_j ,
		
		where K^{-1}_{ij} is the inverse of the covariance matrix.
		'''
		
		return np.exp(-0.5*dist2)
	
	def set_bandwidth(self, cov=None):
		'''
		Set the covariance of the kernel. The covariance may be provided
		as a covariance matrix, a list of standard deviations, or a
		single (universal) standard deviation. If no covariance is
		provided, a covariance is guessed based on the input points.
		'''
		
		if cov == None:
			cov = 0.05 * np.cov(self.x0, rowvar=0)
		elif type(cov) == float:
			cov = np.diag(cov*cov*np.ones(self.ndim))
		elif type(cov) == np.ndarray:
			if cov.shape == (self.ndim,):
				cov = np.diag(cov*cov)
			elif cov.shape != (self.ndim, self.ndim):
				raise ValueError("'cov' has wrong shape.")
		elif type(cov) in [list, tuple]:
			cov = np.diag(cov)
			cov = cov*cov
		else:
			raise ValueError("'cov' not of recognized type.")
		
		self.invcov = np.linalg.inv(cov)
		#self.norm = pow(2.*np.pi, -0.5*self.ndim)
	
	def __call__(self, x):
		'''
		Evaluates the kernel density at the point, or set of points, x.
		
		Inputs:
		    x  A numpy array, either of shape [npoints, dimension] or
		       [dimension] (optionally in the case of only one point).
		
		Output:
		    rho[npoints]  The density evaluated at each point.
		'''
		
		if len(x.shape) == 1:
			x.shape = (1,x.size)
		if x.shape[1] != self.ndim:
			raise ValueError("'x' has wrong number of dimensions.")
		
		npoints = x.shape[0]
		
		D = (x.reshape(1, npoints, self.ndim).repeat(self.nkern, axis=0)
		     - self.x0.reshape(self.nkern, 1, self.ndim).repeat(npoints, axis=1))
		
		dist2 = np.einsum('abi,ij,abj->ab', D, self.invcov, D)
		
		del D
		
		return np.mean(self.kernel(dist2), axis=0)


def main():
	
	return 0

if __name__ == '__main__':
	main()

