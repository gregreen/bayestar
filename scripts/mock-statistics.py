#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  mock-statistics.py
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

from scipy.interpolate import interp2d, RectBivariateSpline
import scipy.ndimage.interpolation as interp
import scipy.stats

import matplotlib.pyplot as plt
import matplotlib as mplib
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

import argparse, sys
from os.path import abspath

import h5py

import hdf5io


def main():
    parser = argparse.ArgumentParser(
                  prog='mock-statistics.py',
                  description='Compares results from Bayestar for mock data '
                              'with true stellar parameters, printing out '
                              'a comparison in each M_r bin.',
                  add_help=True)
    parser.add_argument('input', type=str, help='Bayestar input file with true parameters.')
    parser.add_argument('output', type=str, help='Bayestar output file with surfaces.')
    parser.add_argument('index', type=int, help='HEALPix index of pixel.')
    if 'python' in sys.argv[0]:
        offset = 2
    else:
        offset = 1
    args = parser.parse_args(sys.argv[offset:])
    
    
    # Read in chain and convergence information
    print 'Loading samples and convergence information...'
    group = 'pixel %d' % (args.index)
    dset = '%s/stellar chains' % group
    chain = hdf5io.TChain(args.output, dset)
    lnp = chain.get_lnp()[:]
    lnZ = chain.get_lnZ()[:]
    conv = chain.get_convergence()[:]
    tmp_samples = chain.get_samples()[:]
    samples = np.empty(tmp_samples.shape, dtype='f8')   # shape = (star, sample, parameter)
    samples[:,:,0] = tmp_samples[:,:,1]
    samples[:,:,1] = tmp_samples[:,:,0]
    samples[:,:,2] = tmp_samples[:,:,2]
    samples[:,:,3] = tmp_samples[:,:,3]
    
    lnp_norm = np.empty(lnp.shape, dtype='f8')
    lnp_norm[:] = lnp[:]
    lnZ.shape = (lnZ.size, 1)
    lnp_norm -= np.repeat(lnZ, lnp.shape[1], axis=1)
    lnZ.shape = (lnZ.size)
    
    lnZ_max = np.percentile(lnZ[np.isfinite(lnZ)], 0.95)
    lnZ_idx = (lnZ > lnZ_max - 15.)
    
    mean = np.mean(samples, axis=1)
    
    mean.shape = (mean.shape[0], 1, mean.shape[1])
    Delta = np.repeat(mean, samples.shape[1], axis=1)
    mean.shape = (mean.shape[0], mean.shape[2])
    Delta -= samples
    cov = np.einsum('ijk,ijl->ikl', Delta, Delta) / float(samples.shape[1])
    
    # Read in true parameter values
    print 'Loading true parameter values...'
    f = h5py.File(args.input, 'r')
    dset = f['/parameters/pixel %d' % (args.index)]
    
    fields = ['DM', 'EBV', 'Mr', 'FeH']
    #dtype = [(field, 'f8') for field in fields]
    truth = np.empty((len(dset), 4), dtype='f8')
    
    for i,field in enumerate(fields):
        truth[:,i] = dset[field][:]
    
    # Read in detection information
    dset = f['/photometry/pixel %d' % (args.index)]
    mag_errs = dset['err'][:]
    
    det_idx = (np.sum(mag_errs > 1.e9, axis=1) == 0)
    
    f.close()
    
    # Mask stars based on convergence and detection
    #mask_idx = np.ones(p.shape[0]).astype(np.bool)
    mask_idx = det_idx & conv #& lnZ_idx
    mean = mean[mask_idx]
    lnp = lnp[mask_idx]
    conv = conv[mask_idx]
    truth = truth[mask_idx]
    cov = cov[mask_idx]
    samples = samples[mask_idx]
    
    print 'Filtered out %d stars.' % (np.sum(~mask_idx))
    
    # Center values in chains on true values
    n_stars, n_samples, n_params = samples.shape
    
    print truth.shape
    truth_expanded = np.reshape(truth, (n_stars, 1, n_params))
    truth_expanded = np.repeat(truth_expanded, n_samples, axis=1)
    samples_centered = samples - truth_expanded
    
    # Determine spread of DM and E(B-V) for each population
    Mr_bin_min = np.array([-1., 4.])
    Mr_bin_max = np.array([4., 12.])
    
    Mr_bin_min = np.append(Mr_bin_min, -np.inf)
    Mr_bin_max = np.append(Mr_bin_max, np.inf)
    
    for Mr_min, Mr_max in zip(Mr_bin_min, Mr_bin_max):
        idx = (truth[:,2] >= Mr_min) & (truth[:,2] < Mr_max)
        
        samples_bin = samples_centered[idx]
        
        DM_pctiles = np.percentile(samples_bin[:,:,0], [2.28, 15.87, 50., 84.13, 97.72])
        EBV_pctiles = np.percentile(samples_bin[:,:,1], [2.28, 15.87, 50., 84.13, 97.72])
        
        DM_mean, DM_std = np.mean(samples_bin[:,:,0]), np.std(samples_bin[:,:,0])
        EBV_mean, EBV_std = np.mean(samples_bin[:,:,1]), np.std(samples_bin[:,:,1])
        
        print '%.1f < M_r < %.1f:' % (Mr_min, Mr_max)
        print '    # of stars: %d' % (np.sum(idx))
        print '    Delta DM %%iles: %.3f %.3f %.3f %.3f %.3f' % (DM_pctiles[0], DM_pctiles[1], DM_pctiles[2], DM_pctiles[3], DM_pctiles[4])
        print '    Delta E(B-V) %%iles: %.3f %.3f %.3f %.3f %.3f' % (EBV_pctiles[0], EBV_pctiles[1], EBV_pctiles[2], EBV_pctiles[3], EBV_pctiles[4])
        print '    Delta DM = %.3f +- %.3f' % (DM_mean, DM_std)
        print '    Delta E(B-V) = %.3f +- %.3f' % (EBV_mean, EBV_std)
        print ''
    
    return 0

if __name__ == '__main__':
    main()

