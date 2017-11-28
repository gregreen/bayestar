#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import h5py

import matplotlib
import matplotlib.pyplot as plt

import scipy.ndimage.measurements

import os


script_dir = os.path.dirname(os.path.realpath(__file__))
output_dir = os.path.abspath(os.path.join(script_dir, '..', 'output'))
plot_dir = os.path.abspath(os.path.join(script_dir, '..', 'plots'))

def exact2ax(dset, ax):
    img = dset['surf'].T

    dm_spacing = (dset['dmod'][-1] - dset['dmod'][0]) / (dset['dmod'].size-1)
    E_spacing = (dset['col'][-1] - dset['col'][0]) / (dset['col'].size-1)
    dm_bounds = (dset['dmod'][0]-0.5*dm_spacing, dset['dmod'][-1]+0.5*dm_spacing)
    E_bounds = (dset['col'][0]-0.5*E_spacing, dset['col'][-1]+0.5*E_spacing)
    extent = dm_bounds + E_bounds

    print(extent)

    ax.imshow(img, interpolation='nearest', origin='lower',
                   cmap='Blues', extent=extent, aspect='auto')

def bayestar2ax(dset, idx, ax):
    img = dset[idx,:,:]

    extent = (dset.attrs['min'][1], dset.attrs['max'][1],
              dset.attrs['min'][0], dset.attrs['max'][0])

    # print(extent)

    ax.imshow(img, interpolation='nearest', origin='lower',
                   cmap='Blues', extent=extent, aspect='auto')


def bayestar_exact_diff(dset_exact, dset_bayestar, idx, ax):
    extent_bayestar = (
        dset_bayestar.attrs['min'][1], dset_bayestar.attrs['max'][1],
        dset_bayestar.attrs['min'][0], dset_bayestar.attrs['max'][0])

    dm_spacing = (dset_exact['dmod'][-1] - dset_exact['dmod'][0]) / (dset_exact['dmod'].size-1)
    E_spacing = (dset_exact['col'][-1] - dset_exact['col'][0]) / (dset_exact['col'].size-1)
    dm_bounds = (dset_exact['dmod'][0]-0.5*dm_spacing, dset_exact['dmod'][-1]+0.5*dm_spacing)
    E_bounds = (dset_exact['col'][0]-0.5*E_spacing, dset_exact['col'][-1]+0.5*E_spacing)
    extent_exact = dm_bounds + E_bounds


    assert(np.allclose(np.array(extent_bayestar), np.array(extent_exact)))
    img = dset_bayestar[idx,:,:]
    img /= np.sum(img)
    vmax = np.max(img)

    com_bayestar = scipy.ndimage.measurements.center_of_mass(img)

    img_exact = dset_exact['surf'].T / np.sum(dset_exact['surf'])

    com_exact = scipy.ndimage.measurements.center_of_mass(img_exact)

    img -= img_exact

    ax.imshow(img, interpolation='nearest', origin='lower',
                   cmap='bwr_r', vmin=-vmax, vmax=vmax,
                   extent=extent_bayestar, aspect='auto')

    print(np.array(com_bayestar) - np.array(com_exact))
    print(com_bayestar)

    return com_bayestar[1], com_exact[1]


def plot_pdf_comparison(fname_exact, fname_bayestar, out_fname):
    xlim = (4., 19.)
    ylim = (0., 2.5)

    n_rows = 4

    delta_DM = []

    with h5py.File(fname_exact, 'r') as f_exact:
        with h5py.File(fname_bayestar, 'r') as f_bayestar:
            dset_bayestar = f_bayestar['/pixel 512-1/stellar pdfs']
            dset_exact = f_exact['/default']

            n_stars = dset_bayestar.shape[0]
            n_figs = int(np.ceil(n_stars / n_rows))

            for fig_idx in range(n_figs):
                fig = plt.figure(figsize=(12,4*n_rows), dpi=100)

                for row_idx in range(n_rows):
                    star_idx = fig_idx * n_rows + row_idx

                    if star_idx >= n_stars:
                        break

                    ax = fig.add_subplot(n_rows, 3, 3*row_idx+1)
                    exact2ax(dset_exact[star_idx], ax)
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)

                    ax = fig.add_subplot(n_rows, 3, 3*row_idx+2)
                    bayestar2ax(dset_bayestar, star_idx, ax)
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    ax.set_yticklabels([])

                    ax = fig.add_subplot(n_rows, 3, 3*row_idx+3)
                    dm0,dm1 = bayestar_exact_diff(dset_exact[star_idx],
                                                  dset_bayestar,
                                                  star_idx, ax)
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    ax.set_yticklabels([])

                    delta_DM.append(dm0-dm1)

                fig.subplots_adjust(wspace=0.)

                fig.savefig(out_fname.format(fig_idx),
                            bbox_inches='tight',
                            transparent=False,
                            dpi=100)
                plt.close(fig)

    print(delta_DM)
    print(np.mean(delta_DM))
    print(np.median(delta_DM))


def main():
    l, b = 45, 45
    name = 'test-l{:d}-b{:d}'.format(l, b)

    fname_exact = os.path.join(output_dir, '{}-out.h5'.format(name))
    # fname_bayestar = os.path.join(output_dir, 'test-l0-b0-real-1M-err0-nomaglim.h5')
    fname_bayestar = os.path.join(output_dir, '{}.h5'.format(name))
    # out_fname = os.path.join(plot_dir, 'test-l0-b0-real-1M-err0-nomaglim-comparison-{:d}.png')
    out_fname = os.path.join(plot_dir, 'comp-{}-{{:d}}.png'.format(name))

    plot_pdf_comparison(fname_exact, fname_bayestar, out_fname)

    return 0


if __name__ == '__main__':
    main()
