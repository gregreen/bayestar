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
import pyfits
import glob
import argparse, sys


pctiles = [10., 15.87, 50., 84.13, 90.]


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
    
    if np.any(out_pixname[out_idx] != in_pixname[in_idx]):
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
    chain = np.concatenate(chain, axis=0)
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
    
    # Compute percentiles of chain
    pctile_data = np.array(np.percentile(chain[:, 2:, 1:], pctiles, axis=1))
    pctile_data = np.swapaxes(pctile_data, 0, 1)
    pctile_data = np.swapaxes(pctile_data, 1, 2)
    
    return data, pctile_data


def reduce_stellar_data(infnames, outfnames, reduced_fname):
    # Load and combine data from all Bayestar input/output files
    data = []
    pctile_data = []
    chain = []
    
    n_files = len(outfnames)
    
    for k, (infn, outfn) in enumerate(zip(infnames, outfnames)):
        print 'Loading %s (%d of %d) ...' % (outfn, k, n_files)
    
    ret = load_stars(infn, outfn)
        
        if ret != None:
            data.append(ret[0])
            pctile_data.append(ret[1])
            #chain.append(ret[1])
    
    data = np.hstack(data)
    pctile_data = np.concatenate(pctile_data, axis=0)
    #chain = np.concatenate(chain, axis=0)
    
    # Write stellar data to file
    f = h5py.File(reduced_fname, 'w')
    
    f.create_dataset('/stellar_data', data=data, chunks=True,
                                 compression='gzip', compression_opts=9)
    
    #dset = f.create_dataset('/markov_chains', data=chain[:,:22,:], chunks=True,
    #                             compression='gzip', compression_opts=9)
    #dset.attrs['parameter_names'] = 'EBV, DM, Mr, FeH'
    #dset.attrs['axes'] = '(star, sample, parameter) **SEE README**'
    
    dset = f.create_dataset('/percentiles', data=pctile_data, chunks=True,
                                 compression='gzip', compression_opts=9)
    dset.attrs['parameter_names'] = 'EBV, DM, Mr, FeH'
    dset.attrs['axes'] = '(star, parameter, percentile)'
    dset.attrs['percentiles'] = '(' + ', '.join(['%.2f' % p for p in pctiles]) + ')'
    
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


def h52fits(infname, outfname):
    f = h5py.File(infname, 'r')
    
    stellar_data = f['/stellar_data'][:]
    pctile_data = f['/percentiles'][:]
    param_names = f['/percentiles'].attrs['parameter_names']
    axes = f['/percentiles'].attrs['axes']
    pctiles = f['/percentiles'].attrs['percentiles']
    
    f.close()
    
    hdu0 = pyfits.PrimaryHDU()
    
    hdu1 = pyfits.BinTableHDU(stellar_data)
    hdu1.header['CONTENTS'] = 'STELLAR DATA'
    
    hdu2 = pyfits.ImageHDU(pctile_data)
    hdu2.header['AXIS1'] = 'STAR'
    hdu2.header['AXIS2'] = 'PARAMETER'
    hdu2.header['AXIS3'] = 'PERCENTILE'
    hdu2.header['PARAM1'] = 'EBV'
    hdu2.header['PARAM2'] = 'DM'
    hdu2.header['PARAM3'] = 'Mr'
    hdu2.header['PARAM4'] = 'FeH'
    hdu2.header['PCTILE1'] = 10.
    hdu2.header['PCTILE2'] = 15.87
    hdu2.header['PCTILE3'] = 50.
    hdu2.header['PCTILE4'] = 84.13
    hdu2.header['PCTILE5'] = 90.
    hdu2.header['CONTENTS'] = 'PARAM PERCENTILES'
    
    hdulist = pyfits.core.HDUList([hdu0, hdu1, hdu2])
    hdulist.writeto(outfname, clobber=True)



def main():
    parser = argparse.ArgumentParser(prog='extract_stars.py',
                                     description='Extract stellar fits from Bayestar output.',
                                     add_help=True)
    parser.add_argument('--input', '-i', type=str, help='Bayestar input files (can include wildcards).')
    parser.add_argument('--output', '-o', type=str, help='Bayestar output files (can include wildcards).')
    parser.add_argument('--stellar', '-s', type=str, help='Filename to which to write stellar data.')
    
    if 'python' in sys.argv[0]:
        offset = 2
    else:
        offset = 1
    args = parser.parse_args(sys.argv[offset:])
    
    infnames = sorted(glob.glob(args.input))
    outfnames = sorted(glob.glob(args.output))
    
    if len(infnames) != len(outfnames):
        print 'Input filenames do not match output filenames.'
        return 0
    
    h5_fname = args.stellar + '.h5'
    fits_fname = args.stellar + '.fits'
    
    print 'Extracting and writing stellar data as HDF5 ...'
    reduce_stellar_data(infnames, outfnames, h5_fname)
    
    print 'Producing FITS output ...'
    h52fits(h5_fname, fits_fname)
    
    print 'Done.'
    
    return 0

if __name__ == '__main__':
    main()

