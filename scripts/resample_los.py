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
import model


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
    
    #d = np.arccos(np.sin(b_0) * np.sin(b_1) + np.cos(b_0) * np.cos(b_1) * np.cos(l_1 - l_0))
    d = np.arcsin(np.sqrt(np.sin(0.5*(b_1-b_0))**2 + np.cos(b_0) * np.cos(b_1) * np.sin(0.5*(l_1-l_0))**2))
    
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
    
    #d = np.arccos(np.sin(b_0) * np.sin(b_1) + np.cos(b_0) * np.cos(b_1) * np.cos(l_1 - l_0))
    d = np.arcsin(np.sqrt(np.sin(0.5*(b_1-b_0))**2 + np.cos(b_0) * np.cos(b_1) * np.sin(0.5*(l_1-l_0))**2))
    
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
    
    # For each downsampled pixel, determine nearest neighbors of all subpixels
    neighbor_idx = -np.ones((pix_idx.size, n_neighbors), dtype='i8')
    neighbor_dist = np.inf * np.ones((pix_idx.size, n_neighbors), dtype='f8')
    
    for i_rough in np.unique(pix_idx_rough):
        rough_neighbors = hp.get_all_neighbours(nside_rough, i_rough, nest=True)
        
        idx_centers = np.argwhere(pix_idx_rough == i_rough)[:,0]
        
        tmp = [np.argwhere(pix_idx_rough == i)[:,0] for i in rough_neighbors]
        tmp.append(idx_centers)
        
        idx_search = np.hstack(tmp)
        
        dist = gc_dist(l[idx_centers], b[idx_centers],
                       l[idx_search], b[idx_search])
        
        tmp = np.argsort(dist, axis=1)[:, 1:n_neighbors+1]
        fill = idx_search[tmp]
        neighbor_idx[idx_centers, :fill.shape[1]] = fill
        
        fill = np.sort(dist, axis=1)[:, 1:n_neighbors+1]
        neighbor_dist[idx_centers, :fill.shape[1]] = fill
    
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
    n_neighbors = 36
    n_disp = 5
    processes = 4
    bounds = [60., 80., -5., 5.]
    
    import glob
    
    fnames = glob.glob('/n/fink1/ggreen/bayestar/output/l70/l70.*.h5')
    
    los_coll = maptools.los_collection(fnames, bounds=bounds,
                                               processes=processes)
    
    nside, pix_idx = los_coll.nside, los_coll.pix_idx
    
    print 'Finding neighbors ...'
    neighbor_idx, neighbor_dist = find_neighbors(nside, pix_idx,
                                                       n_neighbors)
    print 'Done.'
    
    # Highlight a couple of nearest-neighbor sections
    pix_val = np.zeros(pix_idx.size)
    
    l = 0.025 * np.pi / 180.
    print l
	
    for k in xrange(n_disp):
	    idx = np.random.randint(pix_idx.size)
	    pix_val[idx] = 2
	    pix_val[neighbor_idx[idx, :]] = 1
	    
	    d = neighbor_dist[idx, :]
	    print 'Center pixel: %d' % idx
	    print d
	    print np.exp(-(d/l)**2.)
	    print ''
    
    nside_max, pix_idx_exp, pix_val_exp = maptools.reduce_to_single_res(pix_idx,
                                                                        nside,
                                                                        pix_val)
    
    size = (2000, 2000)
    
    img, bounds, xy_bounds = hputils.rasterize_map(pix_idx_exp, pix_val_exp,
                                                   nside_max, size)
    
    # Plot nearest neighbors
    
    import matplotlib.pyplot as plt
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    ax.imshow(img, extent=bounds, origin='lower', aspect='auto',
                                  interpolation='nearest')
    
    plt.show()


def get_prior_ln_Delta_EBV(nside, pix_idx, n_regions=30):
    # Find (l, b) for each pixel
    l = np.empty(pix_idx.size, dtype='f8')
    b = np.empty(pix_idx.size, dtype='f8')
    
    for n in np.unique(nside):
        idx = (nside == n)
        
        l[idx], b[idx] = hputils.pix2lb(n, pix_idx[idx], nest=True)
    
    # Determine priors in each pixel
    gal_model = model.TGalacticModel()
    
    ln_Delta_EBV = np.empty((pix_idx.size, n_regions+1), dtype='f8')
    
    for i,(ll,bb) in enumerate(zip(l, b)):
        ret = gal_model.EBV_prior(ll, bb, n_regions=n_regions)
        
        ln_Delta_EBV[i, :] = ret[1]
    
    return ln_Delta_EBV


class map_resampler:
    def __init__(self, fnames, bounds=None,
                               processes=1,
                               n_neighbors=32,
                               corr_length_core=0.5,
                               corr_length_tail=1.,
                               max_corr=0.05,
                               tail_weight=0.1,
                               dist_floor=0.5):
        
        self.n_neighbors = n_neighbors
        
        self.corr_length_core = corr_length_core
        self.corr_length_tail = corr_length_tail
        self.max_corr = max_corr
        self.tail_weight = tail_weight
        
        self.los_coll = maptools.los_collection(fnames, bounds=bounds,
                                                        processes=processes)
        
        self.nside = self.los_coll.nside
        self.pix_idx = self.los_coll.pix_idx
        
        print 'Finding neighbors ...'
        self.neighbor_idx, self.neighbor_ang_dist = find_neighbors(self.nside,
                                                                   self.pix_idx,
                                                                   self.n_neighbors)
        
        # Determine difference from priors
        self.delta = np.log(self.los_coll.los_delta_EBV)
        self.n_pix, self.n_samples, self.n_slices = self.delta.shape
        
        print 'Calculating priors ...'
        ln_Delta_EBV_prior = get_prior_ln_Delta_EBV(self.nside,
                                                    self.pix_idx,
                                                    n_regions=self.n_slices-1)
        
        #print ''
        #print 'priors:'
        #print ln_Delta_EBV_prior[0, :]
        
        for n in xrange(self.n_samples):
            self.delta[:, n, :] -= ln_Delta_EBV_prior
        
        self.delta /= 1.5  # Standardize to units of the std. dev. on the priors
        
        #print ''
        #print 'delta:'
        #print self.delta
        
        # Distance in pc to each bin
        slice_dist = np.power(10., self.los_coll.los_mu_anchor/5. + 1.)
        slice_dist = np.hstack([[0.], slice_dist])
        self.bin_dist = 0.5 * (slice_dist[:-1] + slice_dist[1:])
        
        # Determine physical distance of each voxel to its neighbors
        # shape = (pix, neighbor, slice)
        print 'Determining neighbor correlation weights ...'
        self.neighbor_dist = np.einsum('ij,k->ijk', self.neighbor_ang_dist, self.bin_dist)
        self.neighbor_dist = np.sqrt(self.neighbor_dist * self.neighbor_dist + dist_floor * dist_floor)
        self.neighbor_corr = self.corr_of_dist(self.neighbor_dist)
        
        #print ''
        #print 'dist:'
        #print self.bin_dist
        #print ''
        #print 'corr:'
        #print self.neighbor_corr
        #print ''
        
        # Set initial state
        print 'Randomizing initial state ...'
        self.randomize()
        self.chain = []
        self.update_order = np.arange(self.n_pix)
    
    def corr_of_dist(self, d):
    	core = np.exp(-0.5 * (d/self.corr_length_core)**2)
    	tail = 1. / np.cosh(d / self.corr_length_core)
    	
    	return self.max_corr * ((1. - self.tail_weight) * core + self.tail_weight * tail)
    
    def randomize(self):
        self.sel_idx = np.random.randint(self.n_samples, size=self.n_pix)
    
    def update_pixel(self, idx):
        n_idx = self.neighbor_idx[idx, :]
        delta = self.delta[idx, :, :]  # (sample, slice)
        n_delta = self.delta[n_idx, self.sel_idx[n_idx], :]  # (neighbor, slice)
        n_corr = self.neighbor_corr[idx, :, :]  # (neighbor, slice)
        
        #print delta.shape
        #print n_delta.shape
        #print n_corr.shape
        
        p = np.exp(0.5 * np.einsum('ij,nj->i', delta, n_delta * n_corr))
        
        if idx == 5:
        	#print delta
        	#print n_delta
        	#print n_corr
        	p_norm = np.sort(p / np.sum(p))[::-1]
        	#print p_norm
        	print np.min(p_norm), np.percentile(p_norm, 5.), np.median(p_norm), np.percentile(p_norm, 95.), np.max(p_norm)
        
        P = np.cumsum(p) / np.sum(p)
        
        new_sample = np.sum(P < np.random.random())
        
        self.sel_idx[idx] = new_sample
    
    def round_robin(self):
        np.random.shuffle(self.update_order)
        
        for n in self.update_order:
            self.update_pixel(n)
        
        print self.sel_idx[:5]
        
        self.chain.append(self.sel_idx)
    
    def clear_chain(self):
        self.chain = []
    
    def save_resampled(self, fname, n_samples=None):
        n_chain = len(self.chain)
        
        if n_samples == None:
            n_samples = n_chain
        elif n_samples > n_chain:
            n_samples = n_chain
        
        chain_idx = np.arange(n_chain)[:n_samples]
        
        
        
        f = h5py.File(fname, 'w')
        dset = f.create_dataset('Delta_EBV')
        f.close()


def test_map_resampler():
    n_steps = 100
    n_neighbors = 36
    processes = 4
    bounds = [60., 80., -5., 5.]
    
    import glob
    
    fnames = glob.glob('/n/fink1/ggreen/bayestar/output/l70/l70.*.h5')
    
    resampler = map_resampler(fnames, bounds=bounds,
                                      processes=processes,
                                      n_neighbors=n_neighbors)
    
    print 'Resampling map ...'
    
    for n in xrange(n_steps):
        print 'step %d' % n
        
        resampler.round_robin()

def main():
    #test_gc_dist()
    #test_find_neighbors_adaptive_res()
    
    test_map_resampler()
    
    return 0

if __name__ == '__main__':
    main()

