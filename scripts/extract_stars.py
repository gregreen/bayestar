#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  extract_stars.py
#
#  Extracts stellar fits from Bayestar output files.
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
import h5py as h5py
import glob


def load_stars(infname, outfname):
    l = []
    b = []
    SFD = []
    
    chain = []
    conv = []
    lnZ = []
    
    in_pixname = []
    out_pixname = []
    
    # Load stellar locations
    f = h5py.File(infname, 'r')
    
    in_pixname = f['/photometry'].keys()
    in_pixname = np.array(in_pixname)
    in_idx = np.argsort(in_pixname)
    
    for name in in_pixname[in_idx]:
        dset = f['/photometry/%s' % name]
        l.append(dset['l'][:])
        b.append(dset['b'][:])
        SFD.append(dset['EBV'][:])
    
    f.close()
    
    # Load Markov chains for stars
    f = h5py.File(outfname, 'r')
    
    out_pixname = f.keys()
    out_pixname = np.array(out_pixname)
    out_idx = np.argsort(out_pixname)
    
    if out_pixname[out_idx] != in_pixname[in_idx]:
        print 'Pixel names in input and output files do not match.'
        return None
    
    for name in out_pixname[out_idx]:
        dset = f['/%s/stellar chains' % name]
        
        chain.append(dset[:])
        conv.append(dset.attrs['converged'])
        lnZ.append(dset.attrs['ln(Z)'])
        
        idx = np.isfinite(lnZ[-1])
        
        if np.sum(idx) != 0:
            lnZ[-1] -= np.percentile(lnZ[-1][idx], 98.)
    
    f.close()
    
    # Combine information into fewer arrays
    l = np.hstack(l)
    b = np.hstack(b)
    SFD = np.hstack(SFD)
    chain = np.hstack(chain)
    conv = np.hstack(conv)
    lnZ = np.hstack(lnZ)
    
    dtype = [('l', 'f4'), ('b', 'f4'), ('SFD', 'f4'),
             ('conv', 'i1'), ('lnZ', 'f4'),
             ('EBV', 'f4'), ('DM', 'f4'),
             ('Mr', 'f4'), ('FeH', 'f4')]
    
    data = np.empty(l.size, dtype=dtype)
    data['l'][:] = l[:]
    data['b'][:] = b[:]
    data['SFD'][:] = SFD[:]
    data['conv'][:] = conv[:]
    data['lnZ'][:] = lnZ[:]
    data['EBV'][:] = chain[:, 1, 1]
    data['DM'][:] = chain[:, 1, 2]
    data['Mr'][:] = chain[:, 1, 3]
    data['FeH'][:] = chain[:, 1, 4]
    
    return data, chain


def reduce_stellar_data(infnames, outfnames, reduced_fname):
    data = []
    chain = []
    
    for infn, outfn in zip(infnames, outfnames):
        ret = load_stars(infn, outfn)
        
        if ret != None:
            data.append(ret[0])
            chain.append(ret[1])
    
    data = np.hstack(data)
    chain = np.hstack(chain)
    
    f = h5py.File(reduced_fname, 'w')
    f.create_dataset('/stellar_data', data=data, chunks=True,
                                 compression='gzip', compression_opts=9)
    dset = f.create_dataset('/markov_chains', data=chain, chunks=True,
                                 compression='gzip', compression_opts=9)
    dset.attrs['parameter_names'] = 'EBV, DM, Mr, FeH'
    f.close()


def unpack_dset(dset, max_samples=None):
    samples, lnp, GR = None, None, None
    
    if max_samples == None:
        samples = dset[:, 2:, 1:].astype('f4')
        lnp = dset[:, 2:, 0].astype('f4')
        GR = dset[:, 0, 1:].astype('f4')
    else:
        samples = dset[:, 2:max_samples+2, 1:].astype('f4')
        lnp = dset[:, 2:max_samples+2, 0].astype('f4')
        GR = dset[:, 0, 1:].astype('f4')
    
    return samples, lnp, GR


def read_reduced_file(fname):
    '''
    Example of how to read the files produced by this program.
    
    This function returns basic data about all the stars in <stellar_data>,
    which contains (l, b)-coordinates for each star, SFD reddening at the
    location of the star, whether the stellar fit converged, the
    Bayesian evidence for the stellar fit, and the best fit parameters
    for the star.
    
    The function also returns the Markov-chain samples for each star,
    ln(p) for each sample, and the Gelman-Rubin statistic for each
    inferred variable (a value above ~1.1 indicates poor convergence).
    '''
    
    f = h5py.File(fname, 'r')
    
    stellar_data = f['/stellar_data'][:]
    samples, lnp, GR = unpack_dset(f['/markov_chains'])
    
    f.close()
    
    return stellar_data, samples, lnp, GR


def main():
    #infnames = sorted(glob.glob('/n/fink1/ggreen/bayestar/input/allsky/allsky.*.h5'))
    #outfnames = sorted(glob.glob('/n/fink1/ggreen/bayestar/output/allsky/allsky.*.h5'))
    #reduced_fname = '/n/fink1/ggreen/bayestar/output/stellar_data.h5'
    
    infnames = ['/home/greg/projects/bayestar/input/90_10_n150_cl10_05.h5']
    outfnames = ['/home/greg/projects/bayestar/output/90_10_n150_cl10_05.h5']
    reduced_fname = '/home/greg/projects/bayestar/output/90_10_n150_cl10_05_stellar_data.h5'
    
    if len(infnames) != len(outfnames):
        print 'Input filenames do not match output filenames.'
        return 0
    
    reduce_stellar_data(infnames, outfnames, reduced_fname)
    
    return 0

if __name__ == '__main__':
    main()

