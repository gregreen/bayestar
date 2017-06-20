#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  reweight_stellar_inferences.py
#  
#  Copyright 2014 greg <greg@greg-UX301LA>
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

import h5py

import collections

import multiprocessing
import Queue

import sys, os, argparse
import glob


def load_stars(in_dset, out_dset):
    '''
    Load information on individual stars from a photometry dataset in
    a Bayestar input file, and an individual-star output dataset in a
    Bayestar output file.
    
    Both <in_dset> and <out_dset> should be h5py dataset objects.
    '''
    
    # Load stellar locations
    l = in_dset['l'][:]
    b = in_dset['b'][:]
    SFD = in_dset['EBV'][:]
    nside = in_dset.attrs['nside']
    hp_idx = in_dset.attrs['healpix_index']
    
    # Load Markov chains
    stellar_chain = out_dset[:, 2:, 1:]
    conv = out_dset.attrs['converged'][:]
    lnZ = out_dset.attrs['ln(Z)'][:]
    
    return nside, hp_idx, l, b, SFD, conv, lnZ, stellar_chain


def load_los(out_dset):
    '''
    Load line-of-sight fit information from a dataset in a Bayestar
    output file.
    
    The input <out_dset> should be an h5py dataset object.
    '''
    
    tmp = out_dset[0, :, :]
    
    los_EBV = np.cumsum(np.exp(tmp[2:, 1:]), axis=1)
    los_lnp = tmp[2:, 0]
    los_GR = tmp[0, 1:]
    
    DM_anchors = np.linspace(out_dset.attrs['DM_min'],
                             out_dset.attrs['DM_max'],
                             los_EBV.shape[1])
    
    return los_EBV, los_lnp, los_GR, DM_anchors


def calc_sigma_E(nside, E):
    '''
    Calculate the smoothing kernel (in mags) for each color excess E.
    
    Inputs:
        nside    The scale of the pixel, in HEALPix nside.
        E        An array containing color excesses (in mags).
    
    Output:
        sigma_E  An array of the same shape as <E>, containing the
                 smoothing width for each color excess value in <E>.
    '''
    
    a = [0.880, -2.963]
    b = [0.578, -1.879]
    
    l = 180. * 60. / (np.sqrt(3.*np.pi) * float(nside))
    log_l = np.log10(l)
    
    alpha = 10.**(a[0] * log_l + a[1])
    beta = 10.**(b[0] * log_l + b[1])
    
    pct_sigma = alpha * E + beta
    
    idx = pct_sigma < 0.1
    pct_sigma[idx] = 0.1
    
    idx = pct_sigma > 0.25
    pct_sigma[idx] = 0.25
    
    return E * pct_sigma


def reweight_samples(stellar_data, los_data): #, n_sigma_warn=1.):
    '''
    Assign a new weight to each sample in the stellar inference Markov
    Chains, conditioned on the entire line-of-sight fit.
    '''
    
    nside, hp_idx, l, b, SFD, conv, lnZ, stellar_chain = stellar_data
    los_EBV, los_lnp, los_GR, DM_anchors = los_data
    
    #k_outlier = [1, 5, 8]
    #stellar_chain[k_outlier, :, 0] += 0.5
    
    # Calculate E(DM) for each stellar sample / los sample combination
    shape = stellar_chain.shape[:2]
    DM = stellar_chain[:,:,1].flatten()
    low_idx = np.digitize(DM, DM_anchors) - 1
    mask_idx = (DM < DM_anchors[0]) | (DM >= DM_anchors[-1])
    low_idx[mask_idx] = -1
    #low_idx.shape = shape
    #DM.shape = shape
    
    low_DM = DM_anchors[low_idx]
    high_DM = DM_anchors[low_idx+1]
    a = (DM - low_DM) / (high_DM - low_DM)
    #a.shape = shape
    
    #print 'Interpolation weights:'
    #print a
    #print ''
    
    # Interpolated E(DM), with shape (star, stellar sample, los sample)
    EBV = np.einsum('j,kj->jk', 1.-a, los_EBV[:, low_idx])
    EBV += np.einsum('j,kj->jk', a, los_EBV[:, low_idx+1])
    EBV[mask_idx,:] = 1000. #-100.
    EBV.shape = (shape[0], shape[1], EBV.shape[1])
    
    
    # Calculate Delta E, sigma_E for each sample
    sigma_EBV = calc_sigma_E(nside, EBV)
    sigma_EBV[~np.isfinite(sigma_EBV)] = 1.e-10
    
    EBV_sample = stellar_chain[:,:,0]
    Delta_EBV = EBV[:, :] - EBV_sample[:,:,np.newaxis]
    
    w = -0.5 * (Delta_EBV/sigma_EBV)**2
    
    chisq_min = -0.5 * np.nanmax(np.nanmax(w, axis=1), axis=1)
    
    w -= np.log(EBV[:,:])
    
    #outlier_flag = (chisq_min > n_sigma_warn**2.)
    
    #print 'Weights:'
    #exp_w = np.exp(w)
    #print np.min(exp_w), np.median(exp_w), np.max(exp_w)
    #print ''
    
    # Normalize weights to unity in each line-of-sight sample
    w_max = np.nanmax(w, axis=1)
    norm = w_max + np.log(np.sum(np.exp(w - w_max[:,np.newaxis,:]), axis=1))
    w = w - norm[:,np.newaxis,:]
    
    # Calculate the evidence for each star
    #norm_max = np.nanmax(norm, axis=1)
    #lnZ_new = norm_max + np.log(np.sum(np.exp(norm - norm_max[:,np.newaxis]), axis=1))
    #lnZ_new -= np.log(los_EBV.shape[0])
    #lnZ_new += lnZ
    #idx = np.isfinite(lnZ_new)
    #lnZ_new -= np.percentile(lnZ_new[idx], 98.)
    
    #w = np.einsum('ik,ijk->ijk', 1./norm, w)
    
    #print 'Normalizations in each line-of-sight sample:'
    #print norm
    #print np.sum(~np.isfinite(norm))
    #print np.min(norm), np.median(norm), np.max(norm)
    #print ''
    
    #print 'Normalized Weights:'
    #exp_w = np.exp(w)
    #print np.min(exp_w), np.median(exp_w), np.max(exp_w)
    #print ''
    
    # Marginalize over the line-of-sight fit
    # Note: This renders the covariance between the inferences for
    # different stars inaccessible.
    w_max = np.nanmax(w, axis=2)
    w = w - w_max[:,:,np.newaxis]
    w = w_max + np.log(np.sum(np.exp(w), axis=2))
    
    #print 'w_max:'
    #print w_max
    #print np.min(w_max), np.median(w_max), np.max(w_max)
    #print ''
    
    # Normalize weights to unity for each star
    w_max = np.nanmax(w, axis=1)
    norm = w_max + np.log(np.sum(np.exp(w - w_max[:,np.newaxis]), axis=1))
    w = w - norm[:,np.newaxis]
    #w = np.einsum('i,ij->ij', 1./norm, w)
    
    #print 'Normalization for each star:'
    #print norm
    #print np.sum(~np.isfinite(norm))
    #print np.log(500.)
    #print ''
    
    #
    # Plot reweighted samples
    #
    
    '''
    import matplotlib.pyplot as plt
    from matplotlib import rc
    
    rc('text', usetex=True)
    
    y_max = max([np.percentile(stellar_chain[:,:,0], 98.),
                np.percentile(los_EBV[:,-1], 95.)])
    y_max += 0.1
    
    xlim = [DM_anchors[0], DM_anchors[-1]]
    ylim = [0., y_max]
    
    for k, (chain, chain_w) in enumerate(zip(stellar_chain, w)):
        fig = plt.figure(figsize=(10,5))#, dpi=200)
        ax = fig.add_subplot(1,1,1)
        
        for E in los_EBV:
            ax.plot(DM_anchors, E, alpha=0.1, c='k')
        
        c = chain_w - np.max(chain_w)
        c[c < -10] = -10.
        
        if np.any(np.isnan(c)):
            c = 'k'
        
        cax = ax.scatter(chain[:,1], chain[:,0], c=c, lw=0)
        
        cbar = fig.colorbar(cax)
        cbar.set_label(r'$\ln \left(\mathrm{Sample \ Weight} \right)$', fontsize=20)
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        ax.set_xlabel(r'$\mu$', fontsize=20)
        ax.set_ylabel(r'$\mathrm{E} \left( B - V \right)$', fontsize=20)
        
        #x = xlim[0] + 0.05 * (xlim[1] - xlim[0])
        #y = ylim[0] + 0.95 * (ylim[1] - ylim[0])
        
        #txt = r'$\chi^{2}_{\mathrm{min}} = %.2f$' % (chisq_min[k])
        #ax.text(x, y, txt, fontsize=16, ha='left', va='top')
        
        fig.savefig('/home/greg/projects/bayestar/plots/reweighted_samples/reweighted_%05d_v2.png' % k,
                    dpi=150, bbox_inches='tight')
        
        plt.close(fig)
    '''
    
    # w: shape = (star, stellar sample)
    return w, chisq_min


def reweight_pixels(in_fname, out_fname, write_buffer):
    '''
    Re-weight the stellar samples in all the pixels contained in a
    Bayestar input/output file pair.
    '''
    
    f_in = h5py.File(in_fname)
    f_out = h5py.File(out_fname)
    
    # Get a list of pixels in the input/output file pair
    in_pixname = f_in['/photometry'].keys()
    in_pixname = np.array(in_pixname)
    idx = np.argsort(in_pixname)
    in_pixname = in_pixname[idx]
    
    out_pixname = f_out.keys()
    out_pixname = np.array(out_pixname)
    idx = np.argsort(out_pixname)
    out_pixname = out_pixname[idx]
    
    if np.any(out_pixname != in_pixname):
        print 'Pixel names in input and output files do not match.'
        return None
    
    # Loop through pixels, reweighting samples and passing the results
    # to the write buffer
    
    for in_pix, out_pix in zip(in_pixname, out_pixname):
        in_dset = f_in['/photometry/%s' % in_pix]
        out_star_dset = f_out['/%s/stellar chains' % out_pix]
        out_los_dset = f_out['/%s/los' % out_pix]
        
        stellar_data = load_stars(in_dset, out_star_dset)
        los_data = load_los(out_los_dset)
        
        w, chisq_min = reweight_samples(stellar_data, los_data)
        
        write_buffer.store(stellar_data, los_data, w, chisq_min)
    
    f_in.close()
    f_out.close()


class ReweightedWriteBuffer:
    def __init__(self, fname, write_threshold=20):
        self.f = h5py.File(fname, 'w')
        
        self.loc_data = collections.deque([])
        self.sample_data = collections.deque([])
        self.hp_pos = collections.deque([])
        
        self.write_threshold = write_threshold
    
    def __del__(self):
        self._cleanup()
    
    def store(self, stellar_data, los_data, w, chisq_min):
        nside, hp_idx, l, b, SFD, conv, lnZ, stellar_chain = stellar_data
        los_EBV, los_lnp, los_GR, DM_anchors = los_data
        
        self.hp_pos.append((nside, hp_idx))
        
        # Information about the locations and fit convergence of stars
        dtype = [('l', 'f4'), ('b', 'f4'),
                 ('SFD', 'f4'), ('conv', 'u1'),
                 ('lnZ', 'f4'), ('rw_chisq_min', 'f4')]
        loc_data = np.empty(l.size, dtype=dtype)
        loc_data['l'] = l[:]
        loc_data['b'] = b[:]
        loc_data['SFD'] = SFD[:]
        loc_data['conv'] = conv[:]
        loc_data['lnZ'] = lnZ[:]
        loc_data['rw_chisq_min'] = chisq_min[:]
        
        self.loc_data.append(loc_data)
        
        # Re-weighted Markov-Chain samples
        dtype = [('ln_w', 'f4'),
                 ('EBV', 'f4'), ('DM', 'f4'),
                 ('Mr', 'f4'), ('FeH', 'f4')]
        sample_data = np.empty(w.shape, dtype=dtype)
        sample_data['ln_w'] = w[:,:]
        sample_data['EBV'] = stellar_chain[:,:,0]
        sample_data['DM'] = stellar_chain[:,:,1]
        sample_data['Mr'] = stellar_chain[:,:,2]
        sample_data['FeH'] = stellar_chain[:,:,3]
        
        self.sample_data.append(sample_data)
        
        if len(self.hp_pos) >= self.write_threshold:
            self._flush()
    
    def _flush(self):
        while True:
            try:
                loc_data = self.loc_data.popleft()
                sample_data = self.sample_data.popleft()
                nside, hp_idx = self.hp_pos.popleft()
                
                name = '%d-%d' % (nside, hp_idx)
                
                dset = self.f.create_dataset('/locs/%s' % name,
                                             data=loc_data,
                                             chunks=True,
                                             compression='gzip',
                                             compression_opts=9)
                dset.attrs['nside'] = nside
                dset.attrs['healpix_index'] = hp_idx
                
                dset = self.f.create_dataset('/samples/%s' % name,
                                             data=sample_data,
                                             chunks=True,
                                             compression='gzip',
                                             compression_opts=9)
                dset.attrs['nside'] = nside
                dset.attrs['healpix_index'] = hp_idx
                
            except IndexError:
                break
    
    def _cleanup(self):
        self._flush()
        self.f.close()


def reweight_worker(ID, fname_queue, rw_fname):
    write_buffer = ReweightedWriteBuffer(rw_fname)
    
    while True:
        try:
            in_fname, out_fname = fname_queue.get_nowait()
            
            _, tail = os.path.split(out_fname)
            print 'Worker %d processing %s ...' % (ID, tail)
            
            reweight_pixels(in_fname, out_fname, write_buffer)
            
        except Queue.Empty:
            print 'Worker %d finished.' % ID
            return


def process_files(in_fnames, out_fnames, rw_fname_base, n_procs=1):
    fname_queue = multiprocessing.Queue()
    
    for fname_pair in zip(in_fnames, out_fnames):
        fname_queue.put(fname_pair)
    
    procs = []
    
    for k in xrange(n_procs):
        rw_fname = '%s.%05d.h5' % (rw_fname_base, k)
        procs.append( multiprocessing.Process(target=reweight_worker,
                                      args=(k, fname_queue, rw_fname)) )
    
    for p in procs:
        p.start()
    
    for p in procs:
        p.join()
    
    print 'All workers have finished.'


def test_calc_sigma_E():
    E = np.linspace(0., 2., 11)
    
    for nside in [128, 256, 512, 1024, 2048]:
        sigma_E = calc_sigma_E(nside, E)
        
        print 'nside = %d:' % nside
        print ' E    sigma'
        print '============'
        for EE,ss in zip(E, sigma_E):
            print '%.2f\t%.2f' % (EE, ss)
        print ''


def main():
    parser = argparse.ArgumentParser(prog='reweight_stellar_inferences.py',
                                     description='Assign new weights to individual-star Markov Chains,\n'
                                                 'based on line-of-sight inference.',
                                     add_help=True)
    parser.add_argument('--input', '-i', type=str, help='Bayestar input files (can include wildcards).')
    parser.add_argument('--output', '-o', type=str, help='Bayestar output files (can include wildcards).')
    parser.add_argument('--reweighted', '-rw', type=str, help='Filename to which to write reweighted stellar data.')
    parser.add_argument('--procs', '-p', type=int, help='# of processes to use.')
    
    if 'python' in sys.argv[0]:
        offset = 2
    else:
        offset = 1
    args = parser.parse_args(sys.argv[offset:])
    
    in_fnames = sorted(glob.glob(args.input))
    out_fnames = sorted(glob.glob(args.output))
    
    if len(in_fnames) != len(out_fnames):
        print 'Input filenames do not match output filenames.'
        return 0
    
    rw_fname_base = args.reweighted
    
    if rw_fname_base.endswith('.h5'):
        rw_fname_base = rw_fname_base[:-3]
    
    process_files(in_fnames, out_fnames, rw_fname_base, n_procs=args.procs)
    
    return 0

if __name__ == '__main__':
    main()

