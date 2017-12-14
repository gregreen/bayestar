#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  reliable_dists.py
#  
#  Copyright 2014 greg <greg@greg-UX301LAA>
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

import matplotlib
matplotlib.use('Agg')
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as PathEffects
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable

import healpy as hp
import h5py
import glob
import os.path

import hputils, maptools, model


def minmax_reliable_dists(n_close=2., n_far=10.,
                          pct_close=0., pct_far=0.,
                          n_blocks=20, block=0):
    out_dir = os.path.expanduser('~/BMK/output-rw')
    infiles = glob.glob(os.path.join(out_dir, 'test_BMK_rw_???.00000.h5'))
    print 'Output directory: {:s}'.format(out_dir)
    #infiles = infiles[:n_blocks]
    
    n_files = len(infiles)
    n_per_block = int(np.ceil(n_files / float(n_blocks)))
    
    k_start = block * n_per_block
    k_end = (block+1) * n_per_block
    
    print '{:d} files total.'.format(n_files)
    print '{:d} files per block.'.format(n_per_block)
    print 'Block {:02d}/{:02d}: [{:d}, {:d})'.format(
        block, n_blocks, k_start, k_end)
    
    infiles.sort()
    infiles = infiles[k_start:k_end]
    #infiles = infiles[:1]
    
    loc_block_list = []
    DM_block_list = []
    loc_dtype = [('nside', 'i4'),
                 ('healpix_index', 'i8'),
                 ('n_stars', 'i8'),
                 ('n_good', 'i8'),
                 ('n_dwarfs', 'i8'),
                 ('D_KL_avg', 'f4')]
    DM_dtype = [('DM_min', 'f4'), ('DM_max', 'f4')]
    
    for j, infile in enumerate(infiles):
        print 'Block {:02d}/{:02d}: {:d} of {:d}: Loading {:s} ...'.format(
            block, n_blocks, j+1, len(infiles), infile)
        
        tmp_locs = []
        
        KL_fname = os.path.splitext(infile)[0] + '_KL.h5'
        
        f = h5py.File(infile, 'r')
        f_KL = h5py.File(KL_fname, 'r')
        
        keys = f['/locs'].keys()
        
        loc_block = np.empty(len(keys), dtype=loc_dtype)
        DM_block = np.empty((len(keys), 2), dtype='f4')
        DM_block[:] = np.nan
        
        for k, pix_label in enumerate(keys):
            nside, healpix_idx = [int(s) for s in pix_label.split('-')]
            loc_block['nside'][k] = nside
            loc_block['healpix_index'][k] = healpix_idx
            
            data = f['/samples/' + pix_label][:]
            attrib = f['/locs/' + pix_label][:]
            
            idx = np.nonzero((   (attrib['conv'] == 1)
                               & (attrib['lnZ'] > -10.)
                               & (attrib['rw_chisq_min'] < 1.)
                             ))[0]
            
            n_stars = idx.size
            loc_block['n_good'][k] = n_stars
            loc_block['n_stars'][k] = attrib.size
            
            KL_locs = f_KL['locs'][pix_label]
            loc_block['D_KL_avg'][k] = np.nanmean(KL_locs['D_KL'][:])
            
            if n_stars == 0:
                loc_block['n_dwarfs'][k] = 0
                continue
            
            #print '    loc, nstars: (%d, %d), %d' % (nside, healpix_idx, n_stars)
            
            threshold_close = max([n_close, pct_close/100.*n_stars])
            threshold_far = max([n_far, pct_far/100.*n_stars])
            
            Mr = data['Mr'][:]
            giant_idx = Mr < 4.
            ln_w = data['ln_w'][:]
            ln_w[giant_idx] = -1.e10
            
            w = np.exp(ln_w[idx]).flatten()
            DM = data['DM'][idx].flatten()
            Mr = data['Mr'][idx].flatten()
            
            sort_idx = np.argsort(DM)
            
            W = np.cumsum(w[sort_idx])
            
            loc_block['n_dwarfs'][k] = W[-1]
            
            DM = DM[sort_idx]
            idx_min = np.sum(W < threshold_close)
            idx_max = np.sum(W < W[-1] - threshold_far) - 1
            
            if (idx_min >= 0) and (idx_min < DM.size):
                DM_block[k,0] = DM[idx_min]
            
            if (idx_max >= 0) and (idx_max < DM.size):
                DM_block[k,1] = DM[idx_max]
            
            #DM_block[k,1] = DM[idx_max]
            
            #print 'n_stars/n_effective:      %6d  %6d' % (n_stars, W[-1])
            #print 'min/max (00000, 00000): %
            
            #l, b = hputils.pix2lb_scalar(nside, healpix_idx)
            #print 'min/max (%6.1f, %6.1f): %6.2f  %6.2f' % (l, b, DM_block[k,0], DM_block[k,1])
        
        f.close()
        f_KL.close()
        
        loc_block_list.append(loc_block)
        DM_block_list.append(DM_block)
    
    loc_data = np.hstack(loc_block_list)
    DM_data = np.concatenate(DM_block_list, axis=0)
    
    print 'Block {:02d}/{:02d}: # of NaNs: {:d}'.format(
        block, n_blocks, np.sum(np.isnan(DM_data)))
    print 'Block {:02d}/{:02d}: Writing data ...'.format(
        block, n_blocks)
    
    out_fname = os.path.join(
        out_dir,
        'reliable_dists.{:02d}.h5'.format(block))
    f = h5py.File(out_fname, 'w')
    f.create_dataset('/locs', data=loc_data,
                     chunks=True, compression='gzip',
                     compression_opts=3)
    f.create_dataset('/distmod', data=DM_data,
                     chunks=True, compression='gzip',
                     compression_opts=3) #scaleoffset=2)
    f.close()


def main():
    n_workers = 12
    
    from multiprocessing import Pool
    pool = Pool(n_workers)
    
    for n in xrange(n_workers):
        kw = {'n_blocks': n_workers, 'block': n}
        pool.apply_async(minmax_reliable_dists, (), kw)
    
    pool.close()
    pool.join()
    
    #minmax_reliable_dists(block=0, n_blocks=500)
    
    return 0

if __name__ == '__main__':
    main()

