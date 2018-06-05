#!/usr/bin/env python


import numpy as np
import healpy as hp
import h5py
from collections import deque

from progressbar import ProgressBar

from healtree import *


#def tree2h5(

def main():
    tree = healtree_init()
    
    nside = 512
    n_pix = hp.pixelfunc.nside2npix(nside)
    pix_idx = np.arange(n_pix, dtype='i8')
    
    nside_lowres = 32
    pix_idx_lowres = pix_idx // (nside // nside_lowres)**2
    
    f = h5py.File('output/neighbor_lookup.h5', 'w')
    
    i_unique = np.unique(pix_idx_lowres)
    
    bar = ProgressBar(max_value=i_unique.size)
    bar.update(0)
    
    for i in i_unique:
        idx = (pix_idx_lowres == i)
        
        data = np.empty((np.sum(idx),9,2), dtype='i4')
        data[:,:,0] = nside
        
        for j,k in enumerate(pix_idx[idx]):
            data[j,0,1] = k
            data[j,1:,1] = hp.pixelfunc.get_all_neighbours(nside, k)
        
        digits = loc2digits(nside_lowres, i)
        dset = r'/' + r'/'.join([str(d) for d in digits])
        
        f.create_dataset(dset, data=data, chunks=True, compression='gzip', compression_opts=3)
        bar.update(i)
    
    f.close()
    
    #for i in pix_idx:
    #    # Retrieve existing node
    #    node, (n_ret, i_ret) = healtree_get(tree, nside, i)
    #    if (n_ret != nside) or (i_ret != i) or (node is None):
    #        node = deque()
    #        healtree_set(tree, nside, i, node)
    #    
    #    # Append this pixel's neighbors
    #    neighbors = hp.pixelfunc.get_all_neighbours(nside, i)
    #    neighbors = [[nside, i]] + [[nside,k] for k in neighbors] 
    #    node.append(neighbors)
    #
    #print(tree)
    #
    #datasets = healtree_map_flat(tree, np.array)
    #print(datasets)
    
    return 0


if __name__ == '__main__':
    main()
