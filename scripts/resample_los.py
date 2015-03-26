#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  resample_los.py
#  
#  Copyright 2013-2014 Greg Green <greg@greg-UX31A>
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

import matplotlib
matplotlib.use('Agg')

import numpy as np
import healpy as hp
import h5py

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
    n_neighbors = 128
    n_disp = 4
    processes = 4
    bounds = None
    
    import glob
    
    fnames = ['/n/fink1/ggreen/bayestar/output/allsky_2MASS/Orion_500samp.h5']
    
    resampler = MapResampler(fnames, bounds=bounds, processes=processes,
                                     n_neighbors=n_neighbors)
    
    #mapper = maptools.LOSMapper(fnames, bounds=bounds,
    #                                    processes=processes)
    
    #nside, pix_idx = mapper.data.nside[0], mapper.data.pix_idx[0]
    
    #print 'Finding neighbors ...'
    #neighbor_idx, neighbor_dist = find_neighbors(nside, pix_idx,
    #                                                    n_neighbors)
    #print 'Done.'
    
    nside = resampler.mapper.data.nside[0]
    pix_idx = resampler.mapper.data.pix_idx[0]
    neighbor_idx = resampler.neighbor_idx
    neighbor_corr = resampler.neighbor_corr
    
    # Highlight a couple of nearest-neighbor sections
    pix_val = np.empty((3, pix_idx.size))
    pix_val[:] = np.nan
    
    for k in xrange(n_disp):
	    idx = np.random.randint(pix_idx.size)
	    pix_val[:, idx] = np.nan
	    
	    for d in xrange(3):
	        tmp = neighbor_corr[idx, :, 4*d]
	        pix_val[d, neighbor_idx[idx, :]] = tmp
	        print tmp
    
    pix_val_min = np.nanmin(pix_val)
    pix_val_max = np.nanmax(pix_val)
    pix_val_min = max([1.e-5 * pix_val_max, pix_val_min])
    
    print pix_val_min, pix_val_max
    
    idx = np.isnan(pix_val)
    pix_val[idx] = pix_val_min
    
    
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(7,10), dpi=150)
    
    for d in xrange(3):
        nside_max, pix_idx_exp, pix_val_exp = maptools.reduce_to_single_res(pix_idx,
                                                                            nside,
                                                                            pix_val[d])
        
        size = (2000, 2000)
        
        img, bounds, xy_bounds = hputils.rasterize_map(pix_idx_exp, pix_val_exp,
                                                       nside_max, size)
        
        # Plot nearest neighbors
        
        ax = fig.add_subplot(3,1,d+1)
        
        dist = np.power(10., (4. + 3.*d)/5. + 1.)
        ax.set_title(r'$d = %d \, \mathrm{pc}$' % dist, fontsize=18)
        
        im = ax.imshow(np.log10(img), extent=bounds, origin='lower',
                                      aspect='auto', interpolation='nearest',
                                      vmin=np.log10(pix_val_min),
                                      vmax=np.log10(pix_val_max))
    
    fig.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.80)
    
    ax = fig.add_axes([0.83, 0.05, 0.04, 0.90])
    
    cbar = fig.colorbar(im, cax=ax)
    cbar.set_label(r'$\Sigma^{-1} \ \mathrm{log}_{10} \left( \mathrm{coefficient} \right)$', fontsize=18)
    
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


class MapResampler:
    def __init__(self, fnames, bounds=None,
                               processes=1,
                               n_neighbors=32,
                               corr_length_core=0.10,
                               corr_length_tail=1.00,
                               max_corr=0.25,
                               tail_weight=0.50,
                               dist_floor=0.25):
        
        self.n_neighbors = n_neighbors
        
        self.corr_length_core = corr_length_core
        self.corr_length_tail = corr_length_tail
        self.max_corr = max_corr
        self.tail_weight = tail_weight
        
        '''
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8,4), dpi=200)
        d = np.linspace(0., 20., 1000)
        c = self.corr_of_dist(d)
        ax = fig.add_subplot(1,1,1)
        ax.semilogy(d, c)
        plt.show()
        '''
        
        self.mapper = maptools.LOSMapper(fnames, bounds=bounds,
                                                   processes=processes)
        
        self.nside = self.mapper.data.nside[0]
        self.pix_idx = self.mapper.data.pix_idx[0]
        
        print 'Finding neighbors ...'
        self.neighbor_idx, self.neighbor_ang_dist = find_neighbors(self.nside,
                                                                   self.pix_idx,
                                                                   self.n_neighbors)
        
        # Determine difference from priors
        los_EBV = self.mapper.data.los_EBV[0]
        print 'los_EBV.shape =', los_EBV.shape
        
        tmp = np.diff(los_EBV, axis=2)
        print 'min, median, max:', np.min(tmp), np.median(tmp), np.max(tmp)
        
        idx = (tmp == 0.)
        idx = np.any(idx, axis=2)
        idx_0, idx_1 = np.where(idx)
        print 'Has zero jump:'
        print los_EBV[idx_0[0], idx_1[0]]
        print ''
        print np.sum(idx)
        
        slice_0 = np.reshape(los_EBV[:,:,0], (los_EBV.shape[0], los_EBV.shape[1], 1))
        print 'slice_0.shape =', slice_0.shape
        
        self.los_delta_EBV = np.concatenate([slice_0, np.diff(los_EBV, axis=2)], axis=2)
        print 'self.los_delta_EBV.shape =', self.los_delta_EBV.shape
        
        self.delta = np.log(self.los_delta_EBV)
        self.n_pix, self.n_samples, self.n_slices = self.delta.shape
        
        self.delta[~np.isfinite(self.delta)] = 0.
        
        print 'Calculating priors ...'
        ln_Delta_EBV_prior = get_prior_ln_Delta_EBV(self.nside,
                                                    self.pix_idx,
                                                    n_regions=self.n_slices-1)
        
        #print ''
        #print 'priors:'
        print ln_Delta_EBV_prior[0, :]
        
        print '# of non-finite entries in ln_Delta_EBV_prior:', np.sum(~np.isfinite(ln_Delta_EBV_prior))
        print '# of non-finite entries in delta:', np.sum(~np.isfinite(self.delta))
        print '# of non-finite entries in los_EBV:', np.sum(~np.isfinite(los_EBV))
        
        self.delta[:,:,:] -= ln_Delta_EBV_prior[:,np.newaxis,:]
        #for n in xrange(self.n_samples):
        #    self.delta[:, n, :] -= ln_Delta_EBV_prior
        
        self.delta /= 1.5  # Standardize to units of the std. dev. on the priors
        
        #print ''
        #print 'delta:'
        #print self.delta
        
        # Distance in pc to each bin
        slice_dist = np.power(10., self.mapper.los_DM_anchor/5. + 1.)
        slice_dist = np.hstack([[0.], slice_dist])
        self.bin_dist = 0.5 * (slice_dist[:-1] + slice_dist[1:])
        
        # Determine physical distance of each voxel to its neighbors
        # shape = (pix, neighbor, slice)
        print 'Determining neighbor correlation weights ...'
        self.neighbor_dist = np.einsum('ij,k->ijk', self.neighbor_ang_dist, self.bin_dist)
        self.neighbor_dist = np.sqrt(self.neighbor_dist**2. + dist_floor**2.)
        self.neighbor_corr = self.corr_of_dist(self.neighbor_dist)
        #self.neighbor_corr = self.hard_sphere_corr(self.neighbor_dist, self.corr_length_core)
        
        ang_dist = 1. * hp.nside2resol(512)
        shape = self.neighbor_ang_dist.shape
        self.neighbor_ang_dist.shape = (shape[0], shape[1], 1)
        self.neighbor_ang_dist = np.repeat(self.neighbor_ang_dist, self.bin_dist.size, axis=2)
        self.neighbor_corr = 0.5 / np.cosh(self.neighbor_ang_dist/ang_dist)
        
        #print ''
        #print 'dist:'
        #print self.bin_dist
        #print ''
        #print 'corr:'
        #print self.neighbor_corr
        #print ''
        
        # Set initial state
        print 'Randomizing initial state ...'
        self.beta = 1.
        self.chain = []
        self.randomize()
        self.log_state()
        self.update_order = np.arange(self.n_pix)
    
    def set_temperature(self, T):
        self.beta = 1. / T
    
    def corr_of_dist(self, d):
    	core = np.exp(-0.5 * (d/self.corr_length_core)**2)
    	tail = 1. / np.cosh(d / self.corr_length_core)
    	
    	return self.max_corr * ((1. - self.tail_weight) * core + self.tail_weight * tail)
    
    def hard_sphere_corr(self, d, d_max):
        return self.corr_max * (d < d_max)
    
    def randomize(self):
        self.sel_idx = np.random.randint(self.n_samples, size=self.n_pix)
    
    def update_pixel(self, idx, downhill=False):
        '''
        Update one pixel, using a Gibbs or downhill step.
        '''
        
        n_idx = self.neighbor_idx[idx, :]
        delta = self.delta[idx, :, :]  # (sample, slice)
        n_delta = self.delta[n_idx, self.sel_idx[n_idx], :]  # (neighbor, slice)
        n_corr = self.neighbor_corr[idx, :, :]  # (neighbor, slice)
        
        #print delta.shape
        #print n_delta.shape
        #print n_corr.shape
        
        p = np.einsum('ij,nj->i', delta, n_delta * n_corr)
        p -= np.einsum('ij,nj->i', delta*delta, n_corr)
        p -= np.sum(n_delta*n_delta * n_corr)
        
	p *= 0.5 * self.beta
        #p = np.exp(0.5 * self.beta * p)
        
        #p = np.exp(0.5 * np.einsum('ij,nj->i', delta, n_delta * n_corr))
        
        if idx == 0:
        	print '\ndelta'
        	print delta
        	print '\nn_delta'
        	print n_delta
        	print '\nneighbor_dist'
        	print self.neighbor_dist[idx, :, :]
        	print '\nn_corr'
        	print n_corr
        	#print p_norm
        	print '\np summary'
        	print np.sum(np.isnan(p))
        	print np.min(p), np.percentile(p, 5.), np.median(p), np.percentile(p, 95.), np.max(p)
        	print ''
        
        if downhill:
            new_sample = np.argmax(p)
            self.sel_idx[idx] = new_sample
        else:
            p -= np.max(p)
            P = np.cumsum(np.exp(p))
            new_sample = np.sum(P < np.random.random() * P[-1])
            self.sel_idx[idx] = new_sample
    
    def round_robin(self, downhill=False):
        '''
        Update all pixels in a random order and add
        the resulting state to the chain.
        '''
        
        np.random.shuffle(self.update_order)
        
        for n in self.update_order:
            self.update_pixel(n, downhill=downhill)
        
        self.log_state()
        
        print self.sel_idx[:5]
        print np.array(self.chain)[:,0]
        print ''
    
    def clear_chain(self):
        self.chain = []
    
    def log_state(self):
        '''
        Add the current state of the map to the chain.
        '''
        
        self.chain.append(self.sel_idx.copy())
    
    def save_resampled(self, fname, n_samples=None):
        '''
        Save the resampled map to an HDF5 file, with
        one dataset containing the map samples, and
        another dataset containing the pixel locations.
        '''
        
        n_chain = len(self.chain)
        
        if n_samples == None:
            n_samples = n_chain
        elif n_samples > n_chain:
            n_samples = n_chain
        
        # Pick a set of samples to return
        chain_idx = np.arange(n_chain)
        #np.random.shuffle(chain_idx)
        #chain_idx = chain_idx[:n_samples]
        
        print chain_idx.shape
        
        # Translate chain sample indices to pixel sample indices
        sample_idx = np.array(self.chain)[chain_idx]
        
        print sample_idx.shape
        
        # Create a data cube with the chosen samples
        #   (sample, pixel, slice)
        data = np.empty((n_chain, self.n_pix, self.n_slices), dtype='f8')
        
        m = np.arange(self.n_pix)
        
        print data.shape
        
        #print self..los_delta_EBV.shape
        
        #los_EBV = self.mapper.data.los_EBV[0]
        #slice_0 = np.reshape(los_EBV[:,:,0], (los_EBV.shape[0], los_EBV.shape[1], 1))
        #los_delta_EBV = np.concatenate([slice_0, np.diff(los_EBV, axis=2)], axis=2)
        
        for n,idx in enumerate(sample_idx):
            data[n, :, :] = self.los_delta_EBV[m, idx, :]
        
        # Store locations to a record array
        loc = np.empty(self.n_pix, dtype=[('nside', 'i4'), ('pix_idx', 'i8')])
        loc['nside'][:] = self.nside
        loc['pix_idx'][:] = self.pix_idx
        
        # Write to file
        f = h5py.File(fname, 'w')
        
        dset = f.create_dataset('/Delta_EBV', data.shape, 'f4',
                                             chunks=(2, data.shape[1], data.shape[2]),
                                             compression='gzip',
                                             compression_opts=9)
        dset[:,:,:] = data[:,:,:]
        
        dset.attrs['DM_min'] = self.mapper.los_DM_anchor[0]
        dset.attrs['DM_max'] = self.mapper.los_DM_anchor[-1]
        
        dset = f.create_dataset('/location', loc.shape, loc.dtype,
                                            compression='gzip', compression_opts=9)
        dset[:] = loc[:]
        
        f.close()


def test_map_resampler():
    n_steps = 5
    n_neighbors = 12
    processes = 4
    bounds = None #[60., 80., -5., 5.]
    
    import glob
    
    fnames = ['/n/fink1/ggreen/bayestar/output/allsky_2MASS/Orion_500samp.h5']
    
    resampler = MapResampler(fnames, bounds=bounds,
                                     processes=processes,
                                     n_neighbors=n_neighbors)
    
    print 'Resampling map ...'
    
    for n in xrange(5):
        print 'downhill step %d' % n
        resampler.round_robin(downhill=True)
    
    for n in xrange(n_steps):
        print 'Gibbs step %d' % n
        resampler.round_robin(downhill=False)
    
    outfname = '/n/home09/ggreen/projects/bayestar/output/Orion_resampled_ang1_corr50_neighbors12.h5'
    resampler.save_resampled(outfname)


def test_plot_resampled_map():
    infname = '/n/home09/ggreen/projects/bayestar/output/Orion_resampled_ang1_corr50_neighbors12.h5'
    plot_fname = '/n/pan1/www/ggreen/maps/Orion_resampled/Orion_ang1_corr50_neighbors12'
    size = (2000, 2000)
    
    # Load in chain
    f = h5py.File(infname, 'r')
    
    loc = f['/location'][:]
    chain = f['/Delta_EBV'][:,:,:] # (sample, pixel, slice)
    
    DM_min = f['/Delta_EBV'].attrs['DM_min']
    DM_max = f['/Delta_EBV'].attrs['DM_max']
    
    f.close()
    
    nside = loc[:]['nside']
    pix_idx = loc[:]['pix_idx']
    
    # Rasterize each sample and plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as PathEffects
    
    mu = np.linspace(4., 19., 31)
    d = np.power(10., mu/5. + 1.)
    
    for n,sample in enumerate(chain):
        print 'Plotting sample %d ...' % n
        fig = plt.figure(figsize=(16,12), dpi=150)
        
        for k in xrange(12):
            pix_val = np.sum(sample[:, k:(k+1)], axis=1) #np.sum(sample[:, 5], axis=1)
            
            nside_max, pix_idx_exp, pix_val_exp = maptools.reduce_to_single_res(pix_idx,
                                                                                nside,
                                                                                pix_val)
            
            
            img, bounds, xy_bounds = hputils.rasterize_map(pix_idx_exp, pix_val_exp,
                                                           nside_max, size)
            
            ax = fig.add_subplot(3,4,k+1)
            
            ax.imshow(np.sqrt(img.T), extent=bounds, origin='lower', aspect='auto',
                             interpolation='nearest', vmin=0., vmax=1.3)
            
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            x = xlim[0] + 0.02 * (xlim[1] - xlim[0])
            y = ylim[0] + 0.98 * (ylim[1] - ylim[0])
            
            txt = ax.text(x, y, '$\mathbf{%d - %d \, pc}$' % (d[k], d[(k+1)]),
                          fontsize=20, color='k',
                          ha='left', va='top',
                          path_effects=[PathEffects.withStroke(linewidth=0.1, foreground='w')])
            txt.set_bbox(dict(color='w', alpha=0.75, edgecolor='w'))
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        
        fig.subplots_adjust(wspace=0.02, hspace=0.02,
                            left=0.02, right=0.98,
                            bottom=0.02, top=0.98)
        
        fname = '%s.%.5d.png' % (plot_fname, n)
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        
        plt.close(fig)
        del img
    
    print 'Done.'


def main():
    #test_gc_dist()
    #test_find_neighbors_adaptive_res()
    
    test_map_resampler()
    test_plot_resampled_map()
    
    return 0


if __name__ == '__main__':
    main()

