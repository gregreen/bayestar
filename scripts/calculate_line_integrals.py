#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import h5py
from argparse import ArgumentParser


def calc_los_integrals(pdf, E):
    # pdf.shape = (star, E, mu)
    # E.shape = (mu,)
    return np.sum(pdf[:,E.astype(int),np.arange(E.size)], axis=1)


def calc_los_integrals_smooth(pdf, E):
    E = E.astype(int)

    line_int = 0
    line_int += pdf[:,E[0],0]

    for d in range(1, E.size):
        n_E = E[d] - E[d-1] + 1
        for EE in range(E[d-1], E[d]+1):
            line_int += pdf[:,EE,d] / n_E

    return line_int


def main():
    parser = ArgumentParser(
        description='Calculate line integrals and show best/worst-fitting stars',
        add_help=True)
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Bayestar output file.')
    parser.add_argument(
        '-l', '--loc',
        type=int,
        nargs=2,
        default=(512, 1),
        help='Healpix nside and index of pixel to load.')
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Plot filename.')
    parser.add_argument(
        '-cat', '--catalog',
        type=str,
		help='Overplot true stellar (DM, EBV) from mock catalog.')
    parser.add_argument(
        '-cl', '--cloud',
        type=float,
        nargs='+',
        help='(distmod, reddening) pairs for true l.o.s. reddening profile.')
    args = parser.parse_args()

    if (args.cloud is not None) and (len(args.cloud) % 2):
        print('Length of "--cloud" argument must be multiple of two.')

    with h5py.File(args.input, 'r') as f:
        group = 'pixel {}-{}'.format(*args.loc)
        los_data = f['{}/discrete-los'.format(group)][0,:,:]

        dset = f['{}/stellar pdfs'.format(group)]
        star_data = dset[:]
        img_min = dset.attrs['min']
        img_max = dset.attrs['max']

    print(np.sum(np.sum(star_data, axis=1), axis=1))

    mu_lim = [img_min[1], img_max[1]]
    E_lim = [img_min[0], img_max[0]]

    dmu = (mu_lim[1] - mu_lim[0]) / star_data.shape[2]
    dE = (E_lim[1] - E_lim[0]) / star_data.shape[1]
    print('dmu = {}'.format(dmu))
    print('dE = {}'.format(dE))

    # Load catalog of true parameters
    if args.catalog is not None:
        with h5py.File(args.catalog, 'r') as f:
            cat = f['parameters/{}'.format(group)][:]

    # line_int = calc_los_integrals_smooth(star_data, los_data[2,1:])
    line_int = calc_los_integrals(star_data, los_data[2,1:])

    E_true = np.zeros(120)

    if args.cloud is not None:
        E_sum = 0.
        for mu,E in zip(args.cloud[0::2], args.cloud[1::2]):
            j = int(round((mu - mu_lim[0]) / dmu))
            E_sum += E
            k = int((E_sum - E_lim[0]) / dE)
            E_true[j:] = k

    # E_true[72+0:] += 50
    # E_true[24:] += 100

    # line_int_true = calc_los_integrals_smooth(star_data, E_true)
    line_int_true = calc_los_integrals(star_data, E_true)

    delta_lnL = np.log(line_int / line_int_true)
    
    idx = np.isnan(delta_lnL) & (line_int < line_int_true)
    if np.any(idx):
        delta_lnL[idx] = -np.inf
    
    idx = np.isnan(delta_lnL) & (line_int > line_int_true)
    if np.any(idx):
        delta_lnL[idx] = np.inf

    n_positive = np.sum(delta_lnL > 0)
    print('# favored by inferred l.o.s.: {:d} / {:d}'.format(
        n_positive,
        len(delta_lnL)
    ))
    print('Delta ln(L): {:.3g}'.format(np.sum(delta_lnL)))
    # for log10_eps in range(-10, 0, 1):
    #     eps = 10.**log10_eps
    #     delta_lnL_eps = np.log((line_int+eps) / (line_int_true+eps))
    #     print('  eps = 10^{: <3d} : {: >8.5f}'.format(
    #         log10_eps,
    #         np.sum(delta_lnL_eps)
    #     ))

    idx = np.argsort(delta_lnL)

    fig = plt.figure(figsize=(16,12), dpi=100)
    extent = (mu_lim[0], mu_lim[1], E_lim[0], E_lim[1])

    dmu = (mu_lim[1] - mu_lim[0]) / star_data.shape[2]
    dE = (E_lim[1] - E_lim[0]) / star_data.shape[1]

    mu = np.linspace(mu_lim[0]+0.5*dmu, mu_lim[1]-0.5*dmu, E_true.size)
    E0 = E_lim[0] + dE * E_true
    E1 = E_lim[0] + dE * los_data[2, 1:]

    E1_samples = E_lim[0] + dE * los_data[3:, 1:]

    n_cols = 8

    y_max = 1.5 * max(E0[-1], E1[-1])

    gs1 = GridSpec(
        2, n_cols,
        hspace=0., wspace=0.,
        left=0.08, right=0.98,
        bottom=0.30, top=0.95)

    gs2 = GridSpec(
        1, 2,
        left=0.08, right=0.98,
        bottom=0.08, top=0.25)

    for row in range(2):#(-1,3,2):
        col_mult = [-1, 1][row]
        idx_offset = [-1, 0][row]

        for col in range(n_cols):
            # print(row)
            # print(row*col + (row-1)//2)
            ax = fig.add_subplot(gs1[row, col])
            # ax = fig.add_subplot(2, n_cols, n_cols*row+col+1)
            i = idx[col_mult*col + idx_offset]
            # ax = fig.add_subplot(2, n_cols, (n_cols//2)*(row+1)+col+1)
            # i = idx[row*col + (row-1)//2]

            img = star_data[i]

            print(i)
            print(np.argmax(img))

            vmin = 0.
            vmax = np.max(img)
            print(vmax)

            ax.imshow(
                img,
                extent=extent,
                origin='lower',
                interpolation='nearest',
                aspect='auto',
                vmin=vmin,
                vmax=vmax,
                cmap='Blues')

            # Indiciate true stellar (DM, EBV)
            if args.catalog is not None:
                ax.scatter(cat['DM'][i], cat['EBV'][i],
                           edgecolor='none', facecolor='g',
                           alpha=1.0, s=20)

            ax.set_xlim(mu_lim)
            ax.set_ylim(E_lim)

            for E_s in E1_samples[:10]:
                ax.plot(mu, E_s+0.5*dE, lw=0.5, c='k', alpha=0.1)

            ax.plot(mu, E0+0.5*dE, lw=1, c='g')
            ax.plot(mu, E1+0.5*dE, lw=1, c='r')

            ax.set_ylim(0., y_max)

            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            x_txt = xlim[0] + 0.04 * (xlim[1] - xlim[0])
            y_txt = ylim[1] - 0.03 * (ylim[1] - ylim[0])
            label = r'${:.3g}$'.format(delta_lnL[i])
            label = label.replace('inf', r'\infty')
            ax.text(
                x_txt, y_txt, label,
                ha='left', va='top',
                fontsize=14)

            ax.set_xticks(np.arange(5, 20, 3))

            if col == 0:
                title = [r'disfavor \ true', r'favor \ true'][row]
                title = r'$\mathrm{{ {} }}$'.format(title)
                ax.set_ylabel(title, fontsize=16)
            else:
                ax.set_yticklabels([])

            if row != 1:
                ax.set_xticklabels([])

    title = '{:.3g}'.format(np.sum(delta_lnL))
    title = title.replace('inf', r'\infty')
    title = (
        r'$\Delta \ln \mathcal{{L}} = {:s}'
      + r' \ \left( \mathrm{{inferred \, - \, true}} \right)$'
    ).format(title)
    fig.suptitle(title, fontsize=14)

    # fig.subplots_adjust(
    #     hspace=0., wspace=0.,
    #     left=0.08, right=0.98,
    #     bottom=0.1, top=0.95)

    # Histogram of Delta ln(L)
    idx = np.isfinite(delta_lnL)
    bin_max = 1.1 * np.nanmax(np.abs(delta_lnL[idx]))

    ax = fig.add_subplot(gs2[0,0])
    counts = ax.hist(
        delta_lnL,
        bins=20,
        range=(-bin_max, bin_max),
        edgecolor=(0., 0., 0., 0.25),
        log=False)[0]
    ax.set_ylim(0, 1.1*np.max(counts))
    ax.set_xlabel(r'$\Delta \ln \mathcal{L}$', fontsize=16)
    ax.set_ylabel(r'$\mathrm{counts}$', fontsize=16)

    ax = fig.add_subplot(gs2[0,1])
    counts = ax.hist(
        delta_lnL,
        bins=20,
        range=(-bin_max, bin_max),
        edgecolor=(0., 0., 0., 0.25),
        log=True)[0]
    ax.set_ylim(0, 1.5*np.max(counts))
    ax.set_xlabel(r'$\Delta \ln \mathcal{L}$', fontsize=16)
    ax.set_ylabel(r'$\mathrm{counts}$', fontsize=16)

    if args.output:
        fig.savefig(args.output, bbox_inches='tight', transparent=False, dpi=100)
    else:
        plt.show()

    return 0


if __name__ == '__main__':
    main()
