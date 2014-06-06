#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  plot_region.py
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
import healpy as hp

import itertools

import matplotlib.pyplot as plt
import matplotlib as mplib

import hputils


def grab_region(nside, l, b, radius=1.):
    i_0 = hputils.lb2pix(nside, l, b)
    
    # Expand out by finding neighbors of neighbors
    scale = hp.pixelfunc.nside2resol(nside)
    n_hops = int(np.ceil(1.2 * (np.pi/180.*radius / scale + 1.)))
    print '# of hops: %d' % n_hops
    
    pix_idx = [np.array([i_0], dtype='i8')]
    
    for n in xrange(n_hops):
        i = [hp.pixelfunc.get_all_neighbours(nside, i_last, nest=True) for i_last in pix_idx]
        i = np.hstack(i).astype('i8')
        i = np.unique(i)
        
        mask = ~(i == -1)
        
        pix_idx.append(i[mask])
    
    pix_idx = np.unique(np.hstack(pix_idx))
    
    # Restrict to pixels within given radius of central (l, b)
    t_0 = np.pi/180. * (90. - b)
    p_0 = np.pi/180. * l
    theta, phi = hp.pixelfunc.pix2ang(nside, pix_idx, nest=True)
    dist = hp.rotator.angdist([t_0, p_0], [theta, phi])
    idx = dist <= np.pi/180. * radius
    
    return pix_idx[idx]


def rasterize_region(m, l, b, radius=10.,
                              proj=hputils.Hammer_projection(),
                              img_shape=(250,250)):
    nside = hp.pixelfunc.npix2nside(m.size)
    pix_idx = grab_region(nside, l, b, radius=radius)
    nside = nside * np.ones(pix_idx.size, dtype='i8')
    
    rasterizer = hputils.MapRasterizer(nside, pix_idx, img_shape,
                                       nest=True, clip=True,
                                       proj=proj, l_cent=l, b_cent=b)
    
    img = rasterizer(m[pix_idx])
    
    return img



def main():
    nside = 512
    n_pix = hp.pixelfunc.nside2npix(nside)
    
    l, b = 0., 15.
    radius = 3.
    
    print 'Generating map...'
    t_0 = np.pi/180. * (90. - b)
    p_0 = np.pi/180. * l
    theta, phi = hp.pixelfunc.pix2ang(nside, np.arange(n_pix), nest=True)
    dist = hp.rotator.angdist([t_0, p_0], [theta, phi])
    m = np.sinc(1. * dist/(np.pi/180.*radius)) #* np.sin(phi)
    
    print 'Rasterizing...'
    img = rasterize_region(m, l, b, radius=radius)
    
    print 'Plotting...'
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    im = ax.imshow(img.T, origin='lower', interpolation='nearest', aspect='auto')
    
    fig.colorbar(im)
    
    plt.show()
    
    return 0

if __name__ == '__main__':
    main()

