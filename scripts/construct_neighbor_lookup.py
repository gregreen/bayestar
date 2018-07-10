#!/usr/bin/env python


from __future__ import print_function, division

import numpy as np
from sklearn.neighbors import NearestNeighbors
from argparse import ArgumentParser
from glob import glob
import h5py
import healpy as hp
import re
from progressbar import ProgressBar

from healtree import loc2digits


def main():
    parser = ArgumentParser(
        description='Construct neighbor lookup table from set of input files.',
        add_help=True)
    parser.add_argument(
        '-i', '--input',
        metavar='INPUT.h5',
        type=str,
        nargs='+',
        required=True,
        help='Input filenames.')
    parser.add_argument(
        '--neighbors',
        metavar='OUTPUT.h5',
        type=str,
        required=True,
        help='Output filename for neighbor lookup table.')
    parser.add_argument(
        '--pixels',
        metavar='OUTPUT.h5',
        type=str,
        required=True,
        help='Output filename for pixel lookup table.')
    parser.add_argument(
        '--index-regex',
        metavar='PATTERN',
        type=str,
        default=r'[^0-9]([0-9]{5})[^0-9]',
        help=('Regex pattern to use to extract index '
              'from input filenames.'))
    parser.add_argument(
        '-n',
        metavar='N',
        type=int,
        default=4,
        help='# of neighbors to store.')
    parser.add_argument(
        '--nside',
        metavar='N',
        type=int,
        default=32,
        help='NSIDE level at which to group pixels.')
    args = parser.parse_args()
    
    input_fname = []
    for fn_pattern in args.input:
        input_fname += glob(fn_pattern)
    
    #file_idx = [re.search(args.index_regex, fn).group(1) for fn in input_fname]
    
    # Load in input pixel locations
    pix_idx = []
    nside = []
    file_idx = []
    
    print('Reading input files ...')
    bar = ProgressBar(max_value=len(input_fname))
    bar.update(0)
    
    for k,fn in enumerate(input_fname):
        with h5py.File(fn, 'r') as f:
            # Transform filename into index
            f_idx = int(re.search(args.index_regex, fn).group(1))
            
            for key in f['/photometry'].keys():
                n,i = key.split()[1].split('-')
                pix_idx.append(int(i))
                nside.append(int(n))
                file_idx.append(f_idx)
        
        bar.update(k)
    
    # Convert pixel (nside, pix_idx) to (x,y,z)
    file_idx = np.array(file_idx)
    nside = np.array(nside)
    pix_idx = np.array(pix_idx)
    xyz = np.empty((pix_idx.size, 3), dtype='f8')
    pix_idx_lowres = np.empty(pix_idx.size, dtype='i4')
    
    print('Calculating positions of pixels...')
    
    for n in np.unique(nside):
        idx = (nside == n)
        xyz[idx,0], xyz[idx,1], xyz[idx,2] = hp.pixelfunc.pix2vec(n, pix_idx[idx], nest=True)
        
        # Determine low-resolution pixel indices
        pix_idx_lowres[idx] = pix_idx[idx] // (n // args.nside)**2
    
    # Construct nearest-neighbor lookup
    print('Looking up nearest neighbors ...')
    
    nbrs = NearestNeighbors(n_neighbors=args.n+1, algorithm='auto').fit(xyz)
    match_dist, match_idx = nbrs.kneighbors(xyz)
    
    # Write nearest-neighbor and pixel lookup tables
    with h5py.File(args.neighbors, 'w') as f_n, \
         h5py.File(args.pixels, 'w') as f_p:
        # Loop over lower-resolution pixels
        pix_idx_lowres_unique = np.unique(pix_idx_lowres)
        
        print('Writing lookup tables ...')
        bar = ProgressBar(max_value=len(pix_idx_lowres_unique))
        bar.update(0)
        
        for k,i in enumerate(pix_idx_lowres_unique):
            idx = (pix_idx_lowres == i)
            
            digits = loc2digits(args.nside, i)
            dset_name = r'/' + r'/'.join([str(d) for d in digits])
            
            # Neighbors
            data = np.empty((np.sum(idx), args.n+1, 2), dtype='i4')
            data[:,:,0] = n
            data[:,:,1] = pix_idx[match_idx[idx]]
            
            dset = f_n.create_dataset(
                dset_name,
                data=data,
                chunks=True,
                compression='gzip',
                compression_opts=3)
            
            # Pixels
            data = np.empty((np.sum(idx), 3), dtype='i4')
            data[:,0] = n
            data[:,1] = pix_idx[idx]
            data[:,2] = file_idx[idx]
            
            dset = f_p.create_dataset(
                dset_name,
                data=data,
                chunks=True,
                compression='gzip',
                compression_opts=3)
            
            bar.update(k)
    
    return 0


if __name__ == '__main__':
    main()
