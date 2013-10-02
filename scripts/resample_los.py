#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  resample_los.py
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
import healpy as hp

import hputils
import maptools


def gc_dist_all(l, b):
    l = np.pi / 180. * l
    b = np.pi / 180. * b
    
    l_0 = np.reshape(l, (1, l.size))
    l_0 = np.repeat(l_0, l.size, axis=0)
    
    l_1 = np.reshape(l, (l.size, 1))
    l_1 = np.repeat(l_1, l.size, axis=1)
    
    b_0 = np.reshape(b, (1, b.size))
    b_0 = np.repeat(b_0, b.size, axis=0)
    
    b_1 = np.reshape(b, (b.size, 1))
    b_1 = np.repeat(b_1, b.size, axis=1)
    
    d = np.arccos(np.sin(b_0) * np.sin(b_1) + np.cos(b_0) * np.cos(b_1) * np.cos(l_1 - l_0))
    
    return d


def gc_dist(l_source, b_source, l_dest, b_dest):
    l_s = np.pi / 180. * l_source
    b_s = np.pi / 180. * b_source
    
    l_d = np.pi / 180. * l_dest
    b_d = np.pi / 180. * b_dest
    
    l_0 = np.reshape(l_s, (l_source.size, 1))
    l_0 = np.repeat(l_0, l_dest.size, axis=1)
    
    l_1 = np.reshape(l_d, (1, l_dest.size))
    l_1 = np.repeat(l_1, l_source.size, axis=0)
    
    b_0 = np.reshape(b_s, (b_source.size, 1))
    b_0 = np.repeat(b_0, b_dest.size, axis=1)
    
    b_1 = np.reshape(b_d, (1, b_dest.size))
    b_1 = np.repeat(b_1, b_source.size, axis=0)
    
    d = np.arccos(np.sin(b_0) * np.sin(b_1) + np.cos(b_0) * np.cos(b_1) * np.cos(l_1 - l_0))
    
    return d


def find_neighbors_naive(nside, pix_idx, n_neighbors):
    '''
    Find the neighbors of each pixel using a naive algorithm.
    
    Each pixel is defined by a HEALPix nside and pixel index (in nested
    order).
    
    Returns two arrays:
    
        neighbor_idx   (n_pix, n_neighbors)  Index of each neighbor in
                                             the nside and pix_idx
                                             arrays.
        neighbor_dist  (n_pix, n_neighbors)  Distance to each neighbor.
    '''
    
    # Determine (l, b) of all pixels
    l = np.empty(pix_idx.size, dtype='f8')
    b = np.empty(pix_idx.size, dtype='f8')
    
    nside_unique = np.unique(nside)
    
    for n in nside_unique:
        idx = (nside == n)
        
        l[idx], b[idx] = hputils.pix2lb(n, pix_idx[idx], nest=True)
    
    
    # Determine distances between all pixel pairs
    dist = gc_dist_all(l, b)
    
    # Determine closest neighbors
    sort_idx = np.argsort(dist, axis=1)
    neighbor_idx = sort_idx[:, 1:n_neighbors+1]
    
    neighbor_dist = np.sort(dist, axis=1)[:, 1:n_neighbors+1]
    
    return neighbor_idx, neighbor_dist


def find_neighbors(nside, pix_idx, n_neighbors):
    '''
    Find the neighbors of each pixel.
    
    Each pixel is defined by a HEALPix nside and pixel index (in nested
    order).
    
    Returns two arrays:
    
        neighbor_idx   (n_pix, n_neighbors)  Index of each neighbor in
                                             the nside and pix_idx
                                             arrays.
        neighbor_dist  (n_pix, n_neighbors)  Distance to each neighbor.
    '''
    
    # Determine (l, b) and which downsampled pixel
    # each (nside, pix_idx) combo belongs to
    nside_rough = np.min(nside)
    
    if nside_rough != 1:
        nside_rough /= 2
    
    l = np.empty(pix_idx.size, dtype='f8')
    b = np.empty(pix_idx.size, dtype='f8')
    
    pix_idx_rough = np.empty(pix_idx.size, dtype='i8')
    
    nside_unique = np.unique(nside)
    
    for n in nside_unique:
        idx = (nside == n)
        
        factor = (n / nside_rough)**2
        
        pix_idx_rough[idx] = pix_idx[idx] / factor
        
        l[idx], b[idx] = hputils.pix2lb(n, pix_idx[idx], nest=True)
    
    print pix_idx_rough
    
    # For each downsampled pixel, determine nearest neighbors of all subpixels
    neighbor_idx = np.empty((pix_idx.size, n_neighbors), dtype='i8')
    neighbor_dist = np.empty((pix_idx.size, n_neighbors), dtype='f8')
    
    for i_rough in np.unique(pix_idx_rough):
        rough_neighbors = hp.get_all_neighbours(nside_rough, i_rough, nest=True)
        
        idx_centers = np.argwhere(pix_idx_rough == i_rough)[:,0]
        
        tmp = [np.argwhere(pix_idx_rough == i)[:,0] for i in rough_neighbors]
        tmp.append(idx_centers)
        
        idx_search = np.hstack(tmp)
        
        #print idx_search
        
        dist = gc_dist(l[idx_centers], b[idx_centers],
                       l[idx_search], b[idx_search])
        
        tmp = np.argsort(dist, axis=1)[:, 1:n_neighbors+1]
        neighbor_idx[idx_centers, :] = idx_search[tmp]
        
        #print ''
        #print dist
        #print np.sort(dist, axis=1)
        #print tmp
        
        #print tmp
        #print pix_idx[idx_search[tmp]]
        
        #sort_idx = np.reshape(tmp, (idx_centers.size * n_neighbors,))
        #full_arr_idx = idx_search[sort_idx]
        #sort_pix_idx = np.reshape(pix_idx[full_arr_idx], (idx_centers.size, n_neighbors))
        
        #neighbor_idx[idx_centers, :] = sort_pix_idx[:,:]
        
        neighbor_dist[idx_centers, :] = np.sort(dist, axis=1)[:, 1:n_neighbors+1]
    
    return neighbor_idx, neighbor_dist


def find_nearest_neighbors(nside, pix_idx):
    # TODO
    
    # Determine mapping of pixel indices at highest resolutions to index
    # in the array pix_idx
    
    nside_max = np.max(nside)
    pix_idx_highres = []
    
    for n in np.unique(nside):
        idx = (nside == n)
        
        pix_idx
    
    
    
    
    l = np.empty(pix_idx.size, dtype='f8')
    b = np.empty(pix_idx.size, dtype='f8')
    
    neighbor_idx = np.empty((8, pix_idx.size), dtype='i8')
    neighbor_dist = np.empty((8, pix_idx.size), dtype='f8')
    
    for n in np.unique(nside):
        idx = (nside == n)
        
        l[idx], b[idx] = hputils.pix2lb(n, pix_idx[idx], nest=True)
        
        neighbor_idx[:, idx] = 1


def test_gc_dist():
    l = np.array([0., 1., 2., 359.])
    b = np.array([0., 30., 60., 90.])
    
    d_0 = gc_dist_all(l, b) * 180. / np.pi
    d_1 = gc_dist(l, b, l, b) * 180. / np.pi
    print d_1
    print d_0
    
    idx = np.argsort(d_1, axis=0)
    
    print d_1 - d_0


def test_find_neighbors():
    nside = 128
    n_neighbors = 16
    
    n_pix = hp.nside2npix(nside)
    
    pix_idx = np.arange(n_pix)
    nside_arr = np.ones(n_pix, dtype='i8') * nside
    
    #neighbor_idx, neighbor_dist = find_neighbors_naive(nside_arr, pix_idx,
    #                                                   n_neighbors)
    neighbor_idx, neighbor_dist = find_neighbors(nside_arr, pix_idx,
                                                 n_neighbors)
    
    #print neighbor_idx
    #print neighbor_dist
    
    import matplotlib.pyplot as plt
    
    m = np.zeros(n_pix)
    idx = np.random.randint(n_pix)
    #idx = 1
    
    print idx
    print neighbor_idx[idx, :]
    print neighbor_dist[idx, :]
    
    m[idx] = 2
    m[neighbor_idx[idx, :]] = 1
    
    hp.visufunc.mollview(m, nest=True)
    
    plt.show()


def test_find_neighbors_adaptive_res():
    n_neighbors = 8
    
    
    import glob
    
    fnames = glob.glob('/home/greg/projects/bayestar/output/allsky.016??.h5')
    
    los_coll = maptools.los_collection(fnames)
    
    nside, pix_idx = los_coll.nside, los_coll.pix_idx
    
    neighbor_idx, neighbor_dist = find_neighbors_naive(nside, pix_idx,
                                                 n_neighbors)
    
    idx = np.random.randint(pix_idx.size)
    pix_val = np.zeros(pix_idx.size)
    pix_val[idx] = 2
    pix_val[neighbor_idx[idx, :]] = 1
    
    nside_max, pix_idx_exp, pix_val_exp = maptools.reduce_to_single_res(pix_idx,
                                                                        nside,
                                                                        pix_val)
    
    size = (600, 400)
    
    img, bounds, xy_bounds = hputils.rasterize_map(pix_idx, pix_val,
                                                   nside_max, size)
    
    # Plot nearest neighbors
    
    import matplotlib.pyplot as plt
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    ax.imshow(img, extent=bounds, origin='lower', aspect='auto',
                                  interpolation='nearest')
    
    plt.show()


def main():
    #test_gc_dist()
    test_find_neighbors_adaptive_res()
    
    return 0

if __name__ == '__main__':
    main()

