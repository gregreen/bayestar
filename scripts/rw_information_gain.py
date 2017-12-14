#!/usr/bin/env python

from __future__ import print_function, division
import numpy as np
import numpy.lib.recfunctions
import h5py
import os.path
from contextlib import closing
from progressbar import ProgressBar
import time
from multiprocessing import Pool
from glob import glob


def entropy(ln_p, renormalize=False, axis=-1):
    """
    Calculate the entropy (in nats) of a discrete probability
    distribution.
    """
    
    H = -1. * np.sum(np.exp(ln_p) * ln_p, axis=axis)
    
    if not renormalize:
        return H
    
    N = np.sum(np.exp(ln_p), axis=axis)
    
    return H / N + np.log(N)


def KL_divergence(ln_p, renormalize=False, axis=-1):
    """
    Calculate the Kullback-Leibler divergence between
    a discrete probability distribution and a uniform
    distribution.
    """
    
    H = entropy(ln_p, renormalize=renormalize, axis=axis)
    
    if axis == -1:
        return np.log(ln_p.size) - H
    
    return np.log(ln_p.shape[axis]) - H


def add_KL_divergence_to_file(in_fname, out_fname):
    with closing(h5py.File(out_fname, 'w')) as f_out:
        with closing(h5py.File(in_fname, 'r')) as f_in:
            for key in f_in['samples']:
                D_KL = KL_divergence(f_in['samples'][key]['ln_w'][:], axis=1)
                
                data = np.lib.recfunctions.append_fields(
                    f_in['locs'][key][:],
                    'D_KL',
                    D_KL,
                    dtypes='f4',
                    usemask=False,
                    asrecarray=True)
                
                f_out.create_dataset(
                    'locs/{:s}'.format(key),
                    data=data,
                    chunks=True,
                    compression='gzip',
                    compression_opts=3)
                
                # f_out.create_dataset(
                #     'samples/{:s}'.format(key),
                #     data=f_in['samples'][key][:],
                #     chunks=True,
                #     compression='gzip',
                #     compression_opts=3)
        

def process_file(fns):
    fn_in, fn_out = fns
    #print('{:s} -> {:s}'.format(fn_in, fn_out))
    add_KL_divergence_to_file(fn_in, fn_out)
    #time.sleep(0.1)

    
def process_files(in_fnames, n_workers=10):
    out_fnames = [os.path.splitext(fn)[0] + '_KL.h5' for fn in in_fnames]
    
    pool = Pool(n_workers)
    
    bar = ProgressBar(max_value=len(in_fnames), redirect_stdout=True)
    
    for _ in bar(pool.imap_unordered(process_file, zip(in_fnames, out_fnames))):
        pass
    
    print('Done.')


def avg_KL_per_pix(fname):
    nside = []
    pix_idx = []
    KL_avg = []
    lnZ_avg = []
    
    with closing(h5py.File(fname, 'r')) as f:
        pass
    

def main():
    base_dir = os.path.expanduser('~/BMK/output-rw/')
    in_fnames = glob(os.path.join(base_dir, 'test_BMK_rw_???.00000.h5'))
    
    process_files(in_fnames)
    
    # in_fname = os.path.expanduser('~/BMK/output-rw/test_BMK_rw_000.00000.h5')
    # out_fname = os.path.expanduser('~/BMK/output-rw/KL_000.h5')
    # add_KL_divergence_to_file(in_fname, out_fname)
    
    # n = 10

    # p = np.random.random(n)
    # Hp = entropy(np.log(p), renormalize=True) * np.log2(np.e)
    # print('H(p) = {:.5f}'.format(Hp))
    # 
    # p /= np.sum(p)
    # Hp = entropy(np.log(p)) * np.log2(np.e)
    # print('H(p) = {:.5f}'.format(Hp))

    # p = np.ones(n)
    # Hp = entropy(np.log(p), renormalize=True) * np.log2(np.e)
    # print('H(p) = {:.5f}'.format(Hp))
    
    return 0


if __name__ == '__main__':
    main()
