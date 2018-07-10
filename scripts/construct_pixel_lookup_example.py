#!/usr/bin/env python


import numpy as np
import healpy as hp
import h5py
from collections import deque

from progressbar import ProgressBar

from healtree import *


def main():
    tree = healtree_init()
    
    nside = 512
    n_pix = hp.pixelfunc.nside2npix(nside)
    pix_idx = np.arange(n_pix, dtype='i8')
    
    nside_lowres = 32
    pix_idx_lowres = pix_idx // (nside // nside_lowres)**2
    
    f = h5py.File('output/pixel_lookup.h5', 'w')
    
    i_unique = np.unique(pix_idx_lowres)
    
    bar = ProgressBar(max_value=i_unique.size)
    bar.update(0)
    
    for i in i_unique:
        idx = (pix_idx_lowres == i)
        
        data = np.empty((np.sum(idx),3), dtype='i4')
        data[:,0] = nside
        data[:,1] = pix_idx[idx]
        data[:,2] = np.random.randint(10, size=np.sum(idx))
        
        digits = loc2digits(nside_lowres, i)
        dset = r'/' + r'/'.join([str(d) for d in digits])
        
        f.create_dataset(dset, data=data, chunks=True, compression='gzip', compression_opts=3)
        bar.update(i)
    
    f.close()
    
    return 0


if __name__ == '__main__':
    main()
