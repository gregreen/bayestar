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


def autocorr_1d(y):
    """
    Calculates the autocorrelation of a 1-dimensional
    signal, y.
    
    Inputs:
        y (array-like): 1-dimensional signal.
    
    Returns:
        Autocorrelation as a function of displacement,
        and an estimate of the autocorrelation time,
        based on the smallest displacement with a
        negative autocorrelation.

    From the StackOverflow answer by unutbu:
    <https://stackoverflow.com/a/14298647/1103939>
    """
    n = len(y)
    y0 = np.mean(y)
    sigma2 = np.var(y)
    dy = y - y0
    r = np.correlate(dy, dy, mode='full')[-n:]
    r /= (sigma2 * np.arange(n, 0, -1))
    tau = np.where(r < 0)[0][0]
    return r, tau


def rel2abs_coords(ax, x, y):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    w = xlim[1] - xlim[0]
    h = ylim[1] - ylim[0]
    return (xlim[0] + x*w, ylim[0] + y*h)


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
    
    fig = plt.figure(figsize=(8,10), dpi=200)
    
    # Autocorrelation of principal component coefficients
    ax = fig.add_subplot(4,1,1)
    
    acorr, tau = autocorr_1d(chain_transf[:,0])
    n = acorr.size // 2
    dt = np.arange(n)
    ax.plot(dt, acorr[:n], c='b', alpha=1.0)
    ax.axhline(y=0., c='k', alpha=0.25)
    
    x, y = rel2abs_coords(ax, 0.98, 0.98)
    ax.text(x, y, r'$\tau = {:d}$'.format(tau),
            ha='right', va='top', fontsize=12)
    x, y = rel2abs_coords(ax, 0.98, 0.85)
    ax.text(x, y, r'$n_{{\tau}} = {:.1f}$'.format(acorr.size/tau),
            ha='right', va='top', fontsize=12)
    
    ax.set_xlabel(r'$\Delta t$', fontsize=10)
    ax.set_ylabel(r'$\mathrm{autocorrelation}$', fontsize=10)
    
    # Principal component coefficients vs. time
    ax = fig.add_subplot(4,1,2)
    
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
    ax = fig.add_subplot(4,1,3)
    
    x = np.arange(eivec.shape[1])
    for k in range(10):
        y = np.sqrt(eival[k]) * eivec[:,k]
        ax.plot(x, y, c='b', alpha=alpha[k])
    
    ax.set_xlabel(r'$\mathrm{distance\ bin}$', fontsize=10)
    ax.set_ylabel(r'$\mathrm{principal\ component}$', fontsize=10)

    # Raw data
    ax = fig.add_subplot(4,1,4)
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
