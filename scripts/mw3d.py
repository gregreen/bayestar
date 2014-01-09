#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  mw3d.py
#  
#  Copyright 2014 Greg Green <greg@greg-UX31A>
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

from mayavi import mlab
from tvtk.util.ctf import PiecewiseFunction
from tvtk.util.ctf import ColorTransferFunction

import sys, argparse

import maptools



#
# Rendering functions
#

def vol_render_pts(x, y, z, density):
	fig = mlab.figure(1, bgcolor=(1, 1, 1))
	fig.scene.disable_render = True
	
	indices = np.arange(len(density))
	
	pts = mlab.pipeline.scalar_scatter(x, y, z, density)
	
	delaunay = mlab.pipeline.delaunay3d(pts)
	vol = mlab.pipeline.volume(delaunay)
	
	fig.scene.disable_render = False
	
	return vol


def set_opacity_transfer(vol, density, opacity):
	otf = PiecewiseFunction()
	otf.add_point(0, 0)
	otf.add_point(density, opacity)
	vol._otf = otf
	vol._volume_property.set_scalar_opacity(otf)


def set_bw_colorscheme(vol, vmin, vmax):
	ctf = ColorTransferFunction()
	ctf.add_rgb_point(vmin, 0., 0., 0.) # r, g, and b are float between 0 and 1
	ctf.add_rgb_point(vmax, 0., 0., 0.)
	vol._volume_property.set_color(ctf)
	vol._ctf = ctf
	vol.update_ctf = True


def test_vol_render_pts():
	n_pts = 1000
	x, y, z = np.random.random((3, n_pts))
	#density = np.sin(x) * np.cos(y) * np.cosh(z)
	density = x*y*z
	
	vol = vol_render_pts(x, y, z, density)
	set_opacity_transfer(vol, 1., 1.)
	set_bw_colorscheme(vol, 0., 1.)
	
	print 'showing...'
	mlab.show()


def vol_render_mw(fnames, method='median', bounds=None, processes=1, max_samples=None):
	los_coll = maptools.los_collection(fnames, bounds=bounds,
	                                           processes=processes,
	                                           max_samples=max_samples)
	
	x, y, z = los_coll.get_xyz()
	rho = los_coll.get_density(method=method)
	
	x += 0.1 * np.random.random(x.shape)
	y += 0.1 * np.random.random(y.shape)
	z += 0.1 * np.random.random(z.shape)
	
	vol = vol_render_pts(x, y, z, rho)
	
	set_opacity_transfer(vol, np.max(rho), 1.)
	set_bw_colorscheme(vol, 0., 1.)
	
	print 'showing...'
	mlab.show()

def main():
	parser = argparse.ArgumentParser(prog='mw3d.py',
	                                 description='Visualize Bayestar output in 3D.',
	                                 add_help=True)
	parser.add_argument('input', type=str, nargs='+', help='Bayestar output files.')
	parser.add_argument('--bounds', '-b', type=float, nargs=4, default=None,
	                                     help='Bounds of pixels to plot (l_min, l_max, b_min, b_max).')
	parser.add_argument('--method', '-mtd', type=str, default='median',
	                                     choices=('median', 'mean', 'best', 'sample', 'sigma' , '5th', '95th'),
	                                     help='Measure of E(B-V) to plot.')
	parser.add_argument('--processes', '-proc', type=int, default=1,
	                                     help='# of processes to spawn.')
	parser.add_argument('--max-samples', '-samp', type=int, default=None,
	                                     help='Maximum # of MCMC samples to load per pixel (to limit memory usage).')
	
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	args = parser.parse_args(sys.argv[offset:])
	
	
	vol_render_mw(args.input, bounds=args.bounds, processes=args.processes,
	                          method=args.method, max_samples=args.max_samples)
	
	return 0

if __name__ == '__main__':
	main()

