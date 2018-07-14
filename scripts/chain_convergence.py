#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt

from argparse import ArgumentParser


def load_chain(fname, dset):
    with h5py.File(fname, 'r') as f:
        # dset.shape = (chain, GR+best+sample, lnp+parameter)
        dset = f[dset][0,2:,2:]
    return dset


def PCA(data):
    # Calculate the eigendecomposition of the covariance matrix
    C = np.cov(data, rowvar=False)
    eival, eivec = np.linalg.eigh(C)
    
    # Sort the eigenvalues/eigenvectors (largest to smallest)
    idx = np.argsort(eival)[::-1]
    eival = eival[idx]
    eivec = eivec[:,idx]
    eivec = eivec[:,:] / np.linalg.norm(eivec, axis=1)[:,None]
    
    # Transform the data to the new coordinate system
    d_transf = np.dot(data - np.mean(data, axis=0), eivec)

    # Returns the (eigenvalues, eigenvectors, transormed data)
    return eival, eivec, d_transf


def autocorr(data):
    acorr = []
    for dim in data.T:
        acorr.append(np.correlate(dim, dim, mode='same'))
    return np.vstack(acorr)


def main():
    parser = ArgumentParser(
        description='Analyze convergence of MCMC chain.',
        add_help=True)
    parser.add_argument(
        '-i', '--input',
        metavar='file.h5',
        type=str,
        required=True,
        help='Input filename')
    parser.add_argument(
        '-d', '--dataset',
        metavar='/path/to/dataset',
        type=str,
        required=True,
        help='Location of chain dataset inside file.')
    parser.add_argument(
        '-o', '--output',
        metavar='plot.png',
        type=str,
        required=True,
        help='Filename to save figure to.')
    args = parser.parse_args()

    chain = load_chain(args.input, args.dataset)
    eival, eivec, chain_transf = PCA(chain)
    #print(eival)
    
    #acorr = autocorr(chain_transf[:,:1] - np.mean(chain_transf[:,:1], axis=0))

    fig = plt.figure(figsize=(8,10), dpi=200)
    
    # Principal component coefficients vs. time
    ax = fig.add_subplot(3,1,1)
    
    x = np.arange(chain_transf.shape[0])
    
    alpha = np.abs(eival)
    alpha /= np.max(alpha)

    for k in range(10):
        y = np.sqrt(eival[k]) * chain_transf[:,k]
        ax.plot(x, y, c='b', alpha=alpha[k])

    ax.axhline(y=0., c='k', alpha=0.25)
    
    ax.set_xlabel(r'$\mathrm{sample\ \#}$', fontsize=10)
    ax.set_ylabel(r'$\mathrm{principal\ component\ strength}$', fontsize=10)
    
    # Principal components
    ax = fig.add_subplot(3,1,2)
    
    x = np.arange(eivec.shape[1])
    for k in range(10):
        y = np.sqrt(eival[k]) * eivec[:,k]
        ax.plot(x, y, c='b', alpha=alpha[k])
    
    ax.set_xlabel(r'$\mathrm{distance\ bin}$', fontsize=10)
    ax.set_ylabel(r'$\mathrm{principal\ component}$', fontsize=10)

    # Raw data
    ax = fig.add_subplot(3,1,3)
    x = np.arange(chain.shape[1])
    y0 = np.mean(chain, axis=0)
    for k in range(chain.shape[0]):
        y = chain[k] - y0
        ax.plot(x, y, c='b', alpha=0.1)
    
    ax.set_xlabel(r'$\mathrm{distance\ bin}$', fontsize=10)
    ax.set_ylabel(r'$\mathrm{raw\ data\ \left( mean\ subtracted \right)}$', fontsize=10)
    
    # Save figure
    fig.savefig(args.output, bbox_inches='tight', dpi=200)
    
    return 0


if __name__ == '__main__':
    main()
