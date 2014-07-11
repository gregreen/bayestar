#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  maptools.py
#  
#  Copyright 2013-2014 Greg Green <greg@greg-UX31A>
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
from scipy.ndimage.filters import gaussian_filter

import matplotlib as mplib
#mplib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as PathEffects

import healpy as hp
import h5py

import os, sys, glob, time

import multiprocessing
import Queue

import hputils


####################################################################################
#
# I/O Auxiliary Functions
#
#   Auxiliary functions needed for I/O of Bayestar output files.
#   Used by LOSData, in particular.
#
####################################################################################

def unpack_dset(dset, max_samples=None):
    samples, lnp, GR = None, None, None    
    
    if max_samples == None:
        samples = dset[:, 1:, 1:].astype('f4')
        lnp = dset[:, 1:, 0].astype('f4')
        GR = dset[:, 0, 1:].astype('f4')
    else:
        samples = dset[:, 1:max_samples+1, 1:].astype('f4')
        lnp = dset[:, 1:max_samples+1, 0].astype('f4')
        GR = dset[:, 0, 1:].astype('f4')
    
    return samples, lnp, GR


def load_output_file_raw(f, bounds=None,
                            max_samples=None,
                            load_stacked_pdfs=False):
    # Define output
    pix_idx = []
    nside = []
    
    cloud_mu = []
    cloud_delta_EBV = []
    cloud_lnp = []
    cloud_GR = []
    cloud_mask = []
    
    los_EBV = []
    los_lnp = []
    los_GR = []
    los_mask = []
    
    star_stack = []
    n_stars = []
    
    DM_min, DM_max = 4., 19.
    EBV_min, EBV_max = 0., 5.
    
    for name,item in f.iteritems():
        # Load pixel position
        try:
            pix_idx_tmp = int(item.attrs['healpix_index'][0])
            nside_tmp = int(item.attrs['nside'][0])
        except:
            continue
        
        # Check if pixel is in bounds
        if bounds != None:
            l, b = hputils.pix2lb_scalar(nside_tmp, pix_idx_tmp, nest=True)
            
            if not hputils.lb_in_bounds(l, b, bounds):
                continue
            #if (     (l < bounds[0]) or (l > bounds[1])
            #      or (b < bounds[2]) or (b > bounds[3])  ):
            #    continue
        
        # Pixel location
        pix_idx.append(pix_idx_tmp)
        nside.append(nside_tmp)
        
        # Cloud model
        if 'clouds' in item:
            samples_tmp, lnp_tmp, GR_tmp = unpack_dset(item['clouds'],
                                                       max_samples=max_samples)
            
            n_clouds = samples_tmp.shape[2] / 2
            
            cloud_mu.append(samples_tmp[:, :, :n_clouds])
            cloud_delta_EBV.append(samples_tmp[:, :, n_clouds:])
            cloud_lnp.append(lnp_tmp)
            cloud_GR.append(GR_tmp)
            
            del samples_tmp, lnp_tmp, GR_tmp
            
            cloud_mask.append(True)
        else:
            cloud_mask.append(False)
        
        # Piecewise-linear model
        if 'los' in item:
            samples_tmp, lnp_tmp, GR_tmp = unpack_dset(item['los'],
                                                       max_samples=max_samples)
            
            DM_min = float(item['los'].attrs['DM_min'])
            DM_max = float(item['los'].attrs['DM_max'])
            
            los_EBV.append(samples_tmp)
            los_lnp.append(lnp_tmp)
            los_GR.append(GR_tmp)
            
            del samples_tmp, lnp_tmp, GR_tmp
            
            los_mask.append(True)
        else:
            los_mask.append(False)
        
        
        # Load stellar chains and create stacked image
        if load_stacked_pdfs:
            dset = item['stellar chains']
            
            star_samples = dset[:, 1:, 1:3]
            conv = dset.attrs['converged'].astype(np.bool)
            lnZ = dset.attrs['ln(Z)']
            
            n_stars.append(star_samples.shape[0])
            
            idx = conv & (np.percentile(lnZ, 98.) - lnZ < 5.)
            
            stack_tmp = None
            
            try:
                dset = item['stellar pdfs']
                stack_tmp = dset[idx, :, :]
                stack_tmp = np.sum(stack_tmp, axis=0)
            except:
                #print 'Using chains...'
                star_samples = star_samples[idx]
            
                n_stars_tmp, n_star_samples, n_star_dim = star_samples.shape
                star_samples.shape = (n_stars_tmp * n_star_samples, n_star_dim)
                
                res = (501, 121)
                
                E_range = np.linspace(EBV_min, EBV_max, res[0]*2+1)
                DM_range = np.linspace(DM_min, DM_max, res[1]*2+1)
                
                stack_tmp, tmp1, tmp2 = np.histogram2d(star_samples[:,0], star_samples[:,1],
                                                       bins=[E_range, DM_range])
                
                stack_tmp = gaussian_filter(stack_tmp.astype('f8'),
                                            sigma=(4, 2), mode='reflect')
                stack_tmp = stack_tmp.reshape([res[0], 2, res[1], 2]).mean(3).mean(1)
            
            stack_tmp *= 100. / np.max(stack_tmp)
            stack_tmp = stack_tmp.astype('f2')
            stack_tmp.shape = (1, stack_tmp.shape[0], stack_tmp.shape[1])
            
            star_stack.append(stack_tmp)
            
        else:
            n_stars.append(item['stellar chains'].shape[0])
    
    # Concatenate output
    if len(pix_idx) == 0:
        return None
    
    # Pixel information
    pix_idx = np.array(pix_idx)
    nside = np.array(nside)
    cloud_mask = np.array(cloud_mask)
    los_mask = np.array(los_mask)
    n_stars = np.array(n_stars)
    
    pix_info = (pix_idx, nside, cloud_mask, los_mask, n_stars)
    
    # Cloud information
    cloud_info = None
    
    if np.sum(cloud_mask) > 0:
        cloud_mu = np.cumsum( np.concatenate(cloud_mu), axis=2)
        cloud_delta_EBV = np.exp( np.concatenate(cloud_delta_EBV) )
        cloud_lnp = np.concatenate(cloud_lnp)
        cloud_GR = np.concatenate(cloud_GR)
        
        cloud_info = (cloud_mu, cloud_delta_EBV, cloud_lnp, cloud_GR)
    
    # Piecewise-linear model information
    los_info = None
    
    if np.sum(los_mask) > 0:
        los_EBV = np.cumsum(np.exp( np.concatenate(los_EBV) ), axis=2)
        los_lnp = np.concatenate(los_lnp)
        los_GR = np.concatenate(los_GR)
        
        los_info = (los_EBV, los_lnp, los_GR)
    
    # Stacked stellar surfaces
    if load_stacked_pdfs:
        star_stack = np.concatenate(star_stack)
    else:
        star_stack = None
    
    # Limits on DM and E(B-V) (for l.o.s. fits and stacked surfaces)
    DM_EBV_lim = (DM_min, DM_max, EBV_min, EBV_max)
    
    return pix_info, cloud_info, los_info, star_stack, DM_EBV_lim


def load_output_file_unified(f, bounds=None,
                                max_samples=None,
                                load_stacked_pdfs=False):
    # Pixel locations
    print 'Loading locations'
    dset = f['locations']
    nside = dset['nside'][:]
    pix_idx = dset['healpix_index'][:]
    cloud_mask = dset['cloud_mask'][:].astype(np.bool)
    los_mask = dset['piecewise_mask'][:].astype(np.bool)
    n_stars = dset['n_stars'][:]
    
    # Cloud model
    has_cloud = 'cloud' in f
    
    if has_cloud:
        print 'Loading cloud model...'
        tmp_samples, cloud_lnp, cloud_GR = unpack_dset(f['cloud'],
                                                       max_samples=max_samples)
        
        n_clouds = tmp_samples.shape[2] / 2
        cloud_mu = tmp_samples[:, :, :n_clouds]
        cloud_delta_EBV = tmp_samples[:, :, n_clouds:]
    
    # Piecewise-linear model
    has_los = 'piecewise' in f
    
    DM_min, DM_max = 4., 19.
    EBV_min, EBV_max = 0., 5.
    
    if has_los:
        print 'Loading piecewise-linear model...'
        los_EBV, los_lnp, los_GR = unpack_dset(f['piecewise'],
                                               max_samples=max_samples)
        
        DM_min = float(f['piecewise'].attrs['DM_min'])
        DM_max = float(f['piecewise'].attrs['DM_max'])
    
    # Stacked pdfs
    star_stack = None
    
    if load_stacked_pdfs:
        print 'Loading stacked pdfs...'
        dset = f['stacked_pdfs']
        
        star_stack = dset[:]
        
        EBV_min = float(dset.attrs['EBV_min'])
        EBV_max = float(dset.attrs['EBV_max'])
    
    # Filter out-of-bounds pixels
    if bounds != None:
        print 'Filtering pixels by bounds...'
        l = np.empty(nside.size, dtype='f8')
        b = np.empty(nside.size, dtype='f8')
        
        for n in np.unique(nside):
            idx = (nside == n)
            l[idx], b[idx] = hputils.pix2lb(n, pix_idx[idx], nest=True)
        
        idx = hputils.lb_in_bounds(l, b, bounds)
        #idx = (  (l >= bounds[0]) & (l <= bounds[1])
        #       & (b >= bounds[2]) & (b <= bounds[3])  )
        
        nside = nside[idx]
        pix_idx = pix_idx[idx]
        cloud_mask = cloud_mask[idx]
        los_mask = los_mask[idx]
        n_stars = n_stars[idx]
        
        if has_cloud:
            cloud_mu = cloud_mu[idx]
            cloud_delta_EBV = cloud_delta_EBV[idx]
            cloud_GR = cloud_GR[idx]
            cloud_lnp = cloud_lnp[idx]
        
        if has_los:
            los_EBV = los_EBV[idx]
            los_GR = los_GR[idx]
            los_lnp = los_lnp[idx]
        
        if load_stacked_pdfs:
            star_stack = star_stack[idx]
    
    print 'Returning data...'
    # Return
    if len(pix_idx) == 0:
        return None
    
    pix_info = (pix_idx, nside, cloud_mask, los_mask, n_stars)
    
    # Cloud information
    cloud_info = None
    
    if has_cloud and (np.sum(cloud_mask) > 0):
        cloud_info = (cloud_mu, cloud_delta_EBV, cloud_lnp, cloud_GR)
    
    # Piecewise-linear model information
    los_info = None
    
    if has_los and (np.sum(los_mask) > 0):
        los_info = (los_EBV, los_lnp, los_GR)
    
    # Limits on DM and E(B-V) (for l.o.s. fits and stacked surfaces)
    DM_EBV_lim = (DM_min, DM_max, EBV_min, EBV_max)
    
    return pix_info, cloud_info, los_info, star_stack, DM_EBV_lim


def load_output_file(fname, bounds=None,
                            max_samples=None,
                            load_stacked_pdfs=False):
    '''
    Loads Bayestar output, either from the files that Bayestar
    itself outputs ("raw" files), or from more compact files
    that contain only information on the line-of-sight fits
    ("unified" files).
    
    Input:
        fname              Filename of Bayestar output file to load.
        
        bounds             Galactic (l, b) bounds of pixels to load.
                           Pixels outside of these bounds will not
                           be loaded. Default: None (load all pixels).
        
        max_samples        Maximum number of Markov-Chain samples
                           to load for each cloud or line-of-sight
                           fit. Default: None (load all samples).
        
        load_stacked_pdfs  If True, an array containing stacked
                           stellar probability surfaces will
                           be returned for each pixel.
    
    Ouput:
        The following are returned in a tuple:
        
          pix_info = (pix_idx, nside, cloud_mask, los_mask, n_stars)
          
          cloud_info = (cloud_mu, cloud_delta_EBV, cloud_lnp, cloud_GR)
          
          los_info = (los_EBV, los_lnp, los_GR)
          
          star_stack = numpy array of shape (n_stars, EBV pixels, DM pixels)
          
          DM_EBV_lim = (DM_min, DM_max, EBV_min, EBV_max)
        
        If any information is missing (e.g. line-of-sight information), the
        corresponding tuple (e.g. los_info) will be replaced by None.
        
        If no pixels are in bounds, then the output of the function is None.
    '''
    
    f = None
    
    try:
        f = h5py.File(fname, 'r')
    except:
        raise IOError('Unable to open %s.' % fname)
    
    if 'locations' in f: # Unified filetype
        ret = load_output_file_unified(f, bounds=bounds,
                                          max_samples=max_samples,
                                          load_stacked_pdfs=load_stacked_pdfs)
    else:  # Native Bayestar output
        ret = load_output_file_raw(f, bounds=bounds,
                                      max_samples=max_samples,
                                      load_stacked_pdfs=load_stacked_pdfs)
    
    f.close()
    
    return ret


def load_output_worker(fname_q, output_q, *args, **kwargs):
    data = LOSData()
    
    # Process input files from queue
    while True:
        fname = fname_q.get()
        
        if fname != 'STOP':
            print 'Loading %s ...' % fname
            
            data.append(load_output_file(fname, *args, **kwargs))
            fname_q.task_done()
            
        else:
            print 'Putting data on queue ...'
            
            data.put_on_queue(output_q)
            output_q.put('DONE')
            
            print 'Data on queue.'
            
            fname_q.task_done()
            
            return


def load_multiple_outputs(fnames, processes=1,
                                  bounds=None,
                                  max_samples=None,
                                  load_stacked_pdfs=False):
    '''
    Load multiple Bayestar output files.
    
    Spawns one or more processes, as specified by
    <processes>.
    '''
    
    # Special case if only one process is requested
    if (processes == 1) or (len(fnames) == 1):
        data = LOSData()
        
        for fn in fnames:
            data.append(load_output_file(fn, bounds=bounds,
                                             max_samples=max_samples,
                                             load_stacked_pdfs=load_stacked_pdfs))
        
        data.concatenate()
        
        return data
    
    # Set up Queues for filenames and output data
    fname_q = multiprocessing.JoinableQueue()
    
    for fname in fnames:
        fname_q.put(fname)
    
    output_q = multiprocessing.Queue()
    
    # Spawn a set of processes to load data from files
    print 'Spawning processes ...'
    
    procs = []
    
    for i in xrange(processes):
        p = multiprocessing.Process(target=load_output_worker,
                                    args=(fname_q, output_q),
                                    kwargs = {'bounds':bounds,
                                              'max_samples':max_samples,
                                              'load_stacked_pdfs':load_stacked_pdfs}
                                   )
        p.daemon = True
        procs.append(p)
        
        fname_q.put('STOP')
    
    for p in procs:
        p.start()
    
    print 'Joining filename Queue ...'
    
    fname_q.join()
    
    # Combine output from separate processes
    data = LOSData()
    
    n_proc_done = 0
    
    print 'Loading output blocks ...'
    
    while n_proc_done < processes:
        ret = output_q.get()
        
        if ret == 'DONE':
            n_proc_done += 1
        elif ret != None:
            data.append(ret)
    
    print 'Concatening output from workers ...'
    
    data.concatenate()
    
    return data


####################################################################################
#
# LOS Data
#
#   Loads line-of-sight data from multiple files.
#
####################################################################################

class LOSData:
    '''
    Container for line-of-sight fit data from multiple
    pixels.
    
    Separate chunks of data (from <load_output_file>)
    can be appended to the object, and transformed into
    a more compact representation by calling the
    class method <concatenate>.
    '''
    
    def __init__(self):
        self.pix_idx = []
        self.nside = []
        self.cloud_mask = []
        self.los_mask = []
        self.n_stars = []
        
        self.cloud_mu = []
        self.cloud_delta_EBV = []
        self.cloud_lnp = []
        self.cloud_GR = []
        
        self.los_EBV = []
        self.los_lnp = []
        self.los_GR = []
        
        self.star_stack = []
        
        self.DM_EBV_lim = None
        
        self._has_pixels = False
        self._has_cloud = False
        self._has_los = False
        self._has_stack = False
        
        self._compact = False
    
    def append(self, output):
        '''
        Add a block of output containing
        
            pix_info
            cloud_info
            los_info
            star_stack
            DM_EBV_lim
        
        to the data contained in the class.
        
        The data can be passed directly from the
        method <load_output_file>.
        '''
        
        if output == None:
            return
        
        self._compact = False
        
        pix_info, cloud_info, los_info, stack_tmp, DM_EBV_lim = output
        
        # Pixel info
        self._has_pixels = True
        self.pix_idx.append(pix_info[0])
        self.nside.append(pix_info[1])
        self.cloud_mask.append(pix_info[2])
        self.los_mask.append(pix_info[3])
        self.n_stars.append(pix_info[4])
        
        # Cloud info
        if cloud_info != None:
            self._has_cloud = True
            self.cloud_mu.append(cloud_info[0])
            self.cloud_delta_EBV.append(cloud_info[1])
            self.cloud_lnp.append(cloud_info[2])
            self.cloud_GR.append(cloud_info[3])
        else:
            self._has_cloud = False
        
        if los_info != None:
            self._has_los = True
            self.los_EBV.append(los_info[0])
            self.los_lnp.append(los_info[1])
            self.los_GR.append(los_info[2])
        else:
            self._has_los = False
        
        if stack_tmp != None:
            self._has_stack = True
            self.star_stack.append(stack_tmp)
        else:
            self._has_stack = False
        
        if DM_EBV_lim != None:
            self.DM_EBV_lim = DM_EBV_lim
    
    def concatenate(self):
        '''
        Concatenate arrays from separate output blocks, so that the
        data is stored more compactly.
        '''
        
        if not self._has_pixels:
            return
        
        self.pix_idx = [np.concatenate(self.pix_idx)]
        self.nside = [np.concatenate(self.nside)]
        self.cloud_mask = [np.concatenate(self.cloud_mask)]
        self.los_mask = [np.concatenate(self.los_mask)]
        self.n_stars = [np.concatenate(self.n_stars)]
        
        if self._has_cloud:
            self.cloud_mu = [np.concatenate(self.cloud_mu)]
            self.cloud_delta_EBV = [np.concatenate(self.cloud_delta_EBV)]
            self.cloud_lnp = [np.concatenate(self.cloud_lnp)]
            self.cloud_GR = [np.concatenate(self.cloud_GR)]
        
        if self._has_los:
            self.los_EBV = [np.concatenate(self.los_EBV)]
            self.los_lnp = [np.concatenate(self.los_lnp)]
            self.los_GR = [np.concatenate(self.los_GR)]
        
        if self._has_stack:
            self.star_stack = [np.concatenate(self.star_stack)]
        
        self._compact = True
    
    def put_on_queue(self, output_q, block_size=1.e9):
        '''
        Put data on a multiprocessing Queue, for
        transmission back to the master process.
        
        The maximum block size is given by
        
            block_size,
        
        which defaults to 1.e9 (Bytes).
        '''
        
        if not self._has_pixels:
            return
        
        if not self._compact:
            self.concatenate()
        
        arrs = (self.pix_idx, self.nside, self.cloud_mask, self.los_mask, self.n_stars)
        pix_bytes = sum([a[0].itemsize * a[0].size / a[0].shape[0] for a in arrs])
        
        if self._has_cloud:
            arrs = (self.cloud_mu, self.cloud_delta_EBV, self.cloud_lnp, self.cloud_GR)
            pix_bytes += sum([a[0].itemsize * a[0].size / a[0].shape[0] for a in arrs])
        
        if self._has_los:
            arrs = (self.los_EBV, self.los_lnp, self.los_GR)
            pix_bytes += sum([a[0].itemsize * a[0].size / a[0].shape[0] for a in arrs])
        
        if self._has_stack:
            arrs = (self.star_stack)
            pix_bytes = sum([a[0].itemsize * a[0].size / a[0].shape[0] for a in arrs])
        
        print 'Pixel size: %d B' % pix_bytes
        
        block_len = int(float(block_size) / float(pix_bytes))
        
        if block_len < 1:
            block_len = 1
        
        print 'Block length: %d pixels' % block_len
        
        s_idx = 0
        
        while s_idx < self.pix_idx[0].size:
            e_idx = s_idx + block_len
            
            pix_info = (self.pix_idx[0][s_idx:e_idx],
                        self.nside[0][s_idx:e_idx],
                        self.cloud_mask[0][s_idx:e_idx],
                        self.los_mask[0][s_idx:e_idx],
                        self.n_stars[0][s_idx:e_idx])
            
            cloud_info = None
            if self._has_cloud:
                cloud_info = (self.cloud_mu[0][s_idx:e_idx],
                              self.cloud_delta_EBV[0][s_idx:e_idx],
                              self.cloud_lnp[0][s_idx:e_idx],
                              self.cloud_GR[0][s_idx:e_idx])
            
            los_info = None
            if self._has_los:
                los_info = (self.los_EBV[0][s_idx:e_idx],
                            self.los_lnp[0][s_idx:e_idx],
                            self.los_GR[0][s_idx:e_idx])
            
            stack_tmp = None
            if self._has_stack:
                stack_tmp = self.star_stack[0][s_idx:e_idx]
            
            output = (pix_info, cloud_info, los_info, stack_tmp, self.DM_EBV_lim)
            
            output_q.put(output)
            
            s_idx = e_idx
    
    def sort(self):
        if not self._compact:
            self.concatenate()
        
        nside_max = np.max(self.nside[0])
        
        sort_key = 4**(nside_max - self.nside[0]) * self.pix_idx[0]
        idx = np.argsort(sort_key)
        
        self.pix_idx = [self.pix_idx[0][idx]]
        self.nside = [self.nside[0][idx]]
        self.cloud_mask = [self.cloud_mask[0][idx]]
        self.los_mask = [self.los_mask[0][idx]]
        self.n_stars = [self.n_stars[0][idx]]
        
        if self._has_cloud:
            self.cloud_mu = [self.cloud_mu[0][idx]]
            self.cloud_delta_EBV = [self.cloud_delta_EBV[0][idx]]
            self.cloud_lnp = [self.cloud_lnp[0][idx]]
            self.cloud_GR = [self.cloud_GR[0][idx]]
        
        if self._has_los:
            self.los_EBV = [self.los_EBV[0][idx]]
            self.los_lnp = [self.los_lnp[0][idx]]
            self.los_GR = [self.los_GR[0][idx]]
        
        if self._has_stack:
            self.star_stack = [self.star_stack[0][idx]]
    
    def expand_missing(self):
        if not self._compact:
            self.concatenate()
        
        n_pix = self.nside[0].size
        
        arr_list = []
        
        if self._has_cloud:
            idx = self.cloud_mask[0]
            
            arr_list += [(self.cloud_mu, idx),
                         (self.cloud_delta_EBV, idx),
                         (self.cloud_lnp, idx),
                         (self.cloud_GR, idx)]
        
        if self._has_los:
            idx = self.los_mask[0]
            
            arr_list += [(self.los_EBV, idx),
                         (self.los_lnp, idx),
                         (self.los_GR, idx)]
        
        #print 'Expanding from %d to %d pixels.' % (np.sum(self.los_mask[0]), self.los_mask[0].size)
        #print self.los_EBV[0].shape
        
        for arr, idx in arr_list:
            s_new = [n_pix] + [a for a in arr[0].shape[1:]]
            tmp = np.empty(s_new, dtype='f8')
            tmp[:] = np.nan
            tmp[idx] = arr[0]
            
            arr[0] = tmp
        
        #print self.los_EBV[0].shape
    
    def require(self, require='piecewise'):
        idx = None
        
        if require == 'piecewise':
            idx = self.los_mask
        elif require == 'cloud':
            idx = self.cloud_mask
        else:
            raise ValueError("Unrecognized option: '%s'" % require)
        
        if idx != None:
            self.pix_idx = [self.pix_idx[0][idx]]
            self.nside = [self.nside[0][idx]]
            self.cloud_mask = [self.cloud_mask[0][idx]]
            self.los_mask = [self.los_mask[0][idx]]
            self.n_stars = [self.n_stars[0][idx]]
            
            if self._has_cloud:
                self.cloud_mu = [self.cloud_mu[0][idx]]
                self.cloud_delta_EBV = [self.cloud_delta_EBV[0][idx]]
                self.cloud_lnp = [self.cloud_lnp[0][idx]]
                self.cloud_GR = [self.cloud_GR[0][idx]]
            
            if self._has_los:
                self.los_EBV = [self.los_EBV[0][idx]]
                self.los_lnp = [self.los_lnp[0][idx]]
                self.los_GR = [self.los_GR[0][idx]]
            
            if self._has_stack:
                self.star_stack = [self.star_stack[0][idx]]
    
    def save_unified(self, fname, save_stacks=False, save_cloud=True, save_piecewise=True):
        if not self._compact:
            self.concatenate()
        
        f = h5py.File(fname, 'w')
        
        # Pixel locations
        shape = (self.pix_idx[0].size,)
        
        dtype = [('nside', 'i4'),
                 ('healpix_index', 'i8'),
                 ('cloud_mask', 'i1'),
                 ('piecewise_mask', 'i1'),
                 ('n_stars', 'i4')]
        
        data = np.empty(shape=shape, dtype=dtype)
        
        data['nside'][:] = self.nside[0][:]
        data['healpix_index'][:] = self.pix_idx[0][:]
        data['piecewise_mask'][:] = self.los_mask[0][:]
        data['cloud_mask'][:] = self.cloud_mask[0][:]
        data['n_stars'][:] = self.n_stars[0][:]
        
        dset = f.create_dataset('locations', shape=shape, dtype=dtype,
                                             compression='gzip', compression_opts=9,
                                             chunks=True)
        dset[:] = data[:]
        
        # Cloud model
        if self._has_cloud and save_cloud:
            n_pix, n_samples, n_clouds = self.cloud_mu[0].shape
            shape = (n_pix, n_samples+1, 2*n_clouds+1)
            
            dset = f.create_dataset('cloud', shape=shape, dtype='f4',
                                             compression='gzip', compression_opts=9,
                                             chunks=True)
            
            dset[:, 1:, 1:n_clouds+1] = self.cloud_mu[0][:]
            dset[:, 1:, n_clouds+1:] = self.cloud_delta_EBV[0][:]
            dset[:, 1:, 0] = self.cloud_lnp[0][:]
            dset[:, 0, 1:] = self.cloud_GR[0][:]
        
        # Piecewise-linear model
        if self._has_los and save_piecewise:
            n_pix, n_samples, n_slices = self.los_EBV[0].shape
            shape = (n_pix, n_samples+1, n_slices+1)
            
            dset = f.create_dataset('piecewise', shape=shape, dtype='f4',
                                                 compression='gzip', compression_opts=9,
                                                 chunks=True)
            
            dset[:, 1:, 1:] = self.los_EBV[0][:]
            dset[:, 1:, 0] = self.los_lnp[0][:]
            dset[:, 0, 1:] = self.los_GR[0][:]
            
            f['piecewise'].attrs['DM_min'] = self.DM_EBV_lim[0]
            f['piecewise'].attrs['DM_max'] = self.DM_EBV_lim[1]
        
        # Stacked pdfs
        if save_stacks and self._has_stack:
            shape = self.star_stack[0].shape
            
            dset = f.create_dataset('stacked_pdfs', shape=shape, dtype='f4',
                                                    compression='gzip', compression_opts=9,
                                                    chunks=True)
            
            dset[:] = self.star_stack[0][:].astype('f4')
            
            dset.attrs['EBV_min'] = self.DM_EBV_lim[2]
            dset.attrs['EBV_max'] = self.DM_EBV_lim[3]
            dset.attrs['DM_min'] = self.DM_EBV_lim[0]
            dset.attrs['DM_max'] = self.DM_EBV_lim[1]
        
        f.close()
    
    def get_pix_idx(self):
        return self.pix_idx[0]
    
    def get_nside(self):
        return self.nside[0]
    
    def get_cloud_mask(self):
        return self.cloud_mask[0]
    
    def get_los_mask(self):
        return self.los_mask[0]
    
    def get_cloud_mu(self):
        return self.cloud_mu[0]
    
    def get_n_los_slices(self):
        if not self._has_los:
            return 0
        
        return self.los_EBV[0].shape[2]
    
    def get_los_DM_range(self):
        if not self._has_los:
            return np.array([np.nan])
        
        n_slices = self.get_n_los_slices()
        
        return np.linspace(self.DM_EBV_lim[0], self.DM_EBV_lim[1], n_slices)
    
    def get_nside_levels(self):
        '''
        Returns the unique nside values present in the
        map.
        '''
        
        if not self._compact:
            self.concatenate()
        
        return np.unique(self.nside[0])


####################################################################################
#
# Auxiliary Mapping Functions
#
#   Used, in particular, by LOSMapper class.
#
####################################################################################


def reduce_to_single_res(pix_idx, nside, pix_val, nside_max=None):
    '''
    Converts a multi-resolution map to a single-resolution map, at the
    highest resolution present in the original map.
    '''
    
    nside_unique = np.unique(nside)
    
    if nside_max == None:
        nside_max = np.max(nside_unique)
    
    #idx = (nside == nside_max)
    #pix_idx_exp = [pix_idx[idx]]
    #pix_val_exp = [pix_val[idx]]
    
    pix_idx_exp = []
    pix_val_exp = []
    
    for n in nside_unique:#[:-1]:
        n_rep = (nside_max / n)**2
        
        idx = (nside == n)
        n_pix = np.sum(idx)
        
        pix_idx_n = np.repeat(n_rep * pix_idx[idx], n_rep, axis=0)
        
        pix_adv = np.mod(np.arange(n_rep * n_pix), n_rep)
        pix_idx_n += pix_adv
        
        pix_val_n = np.repeat(pix_val[idx], n_rep, axis=0)
        
        pix_idx_exp.append(pix_idx_n)
        pix_val_exp.append(pix_val_n)
    
    pix_idx_exp = np.concatenate(pix_idx_exp, axis=0)
    pix_val_exp = np.concatenate(pix_val_exp, axis=0)
    
    return nside_max, pix_idx_exp, pix_val_exp


def resample_at_res(pix_idx, nside, pix_val, nside_target):
    '''
    Resamples a multi-resolution map at a particular resolution.
    '''
    
    nside_max, pix_idx_exp, pix_val_exp = reduce_to_single_res(pix_idx, nside, pix_val)
    
    m_hires = np.empty(hp.pixelfunc.nside2npix(nside_max), dtype=pix_val.dtype)
    m_hires[:] = np.nan
    m_hires[pix_idx_exp] = pix_val_exp[:]
    
    m_resamp = hp.pixelfunc.ud_grade(m_hires, nside_target, False, 'NESTED')
    
    return m_resamp


def take_measure(EBV, method):
    if method == 'median':
        return np.median(EBV[:, 1:], axis=1)
    elif method == 'mean':
        return np.mean(EBV[:, 1:], axis=1)
    elif method == 'best':
        return EBV[:, 0]
    elif method == 'sample':
        n_pix, n_samples = EBV.shape
        j = np.random.randint(1, high=n_samples, size=n_pix)
        i = np.arange(n_pix)
        return EBV[i, j]
    elif method == 'sigma':
        high = np.percentile(EBV[:, 1:], 84.13, axis=1)
        low = np.percentile(EBV[:, 1:], 15.87, axis=1)
        return 0.5 * (high - low)
    elif type(method) == float:
        return np.percentile(EBV[:, 1:], method, axis=1)
    else:
        raise ValueError('method not implemented: "%s"' % (str(method)))


def take_measure_nd(x, method, axis=1):
    xp = np.rollaxis(x, axis)
    
    if method == 'median':
        return np.median(xp, axis=0)
    elif method == 'mean':
        return np.mean(EBV, axis=0)
    elif method == 'sample':
        n_samples = xp.shape[0]
        n_pix = xp.shape[1]
        j = np.random.randint(0, high=n_samples, size=n_pix)
        k = np.arange(n_pix)
        return xp[j, k]
    elif method == 'sigma':
        high = np.percentile(xp, 84.13, axis=0)
        low = np.percentile(xp, 15.87, axis=0)
        return 0.5 * (high - low)
    elif type(method) == float:
        return np.percentile(xp, method, axis=0)
    else:
        raise ValueError('method not implemented: "%s"' % (str(method)))


####################################################################################
#
# LOS Mapper
#
#   Contains methods for mapping line-of-sight data.
#
####################################################################################

class LOSMapper:
    
    def __init__(self, fnames, **kwargs):
        self.data = load_multiple_outputs(fnames, **kwargs)
        self.data.expand_missing()
        
        self.los_DM_anchor = self.data.get_los_DM_range()
        self.n_slices = self.data.get_n_los_slices()
        self.los_dmu = self.los_DM_anchor[1] - self.los_DM_anchor[0]
    
    
    def calc_cloud_EBV(self, mu):
        '''
        Returns an array of E(B-V) evaluated at
        distance modulus mu, with
        
            shape = (n_pixels, n_samples)
        
        The first sample is the best fit sample.
        The rest are drawn from the posterior.
        
        Uses the cloud model.
        '''
        
        foreground = (self.data.cloud_mu[0] < mu)
        
        return np.sum(foreground * self.data.cloud_delta_EBV[0], axis=2)
    
    
    def calc_piecewise_EBV(self, mu):
        '''
        Returns an array of E(B-V) evaluated at
        distance modulus mu, with
        
            shape = (n_pixels, n_samples)
        
        The first sample is the best fit sample.
        The rest are drawn from the posterior.
        
        Uses the piecewise-linear model.
        '''
        
        low_idx = np.sum(mu > self.los_DM_anchor) - 1
        
        if low_idx >= self.n_slices - 1:
            return self.data.los_EBV[0][:,:,-1]
        elif low_idx < 0:
            return self.data.los_EBV[0][:,:,0]
        
        low_mu = self.los_DM_anchor[low_idx]
        high_mu = self.los_DM_anchor[low_idx+1]
        
        a = (mu - low_mu) / (high_mu - low_mu)
        EBV_interp = (1. - a) * self.data.los_EBV[0][:,:,low_idx]
        EBV_interp += a * self.data.los_EBV[0][:,:,low_idx+1]
        
        return EBV_interp
    
    
    def est_dEBV_pctile(self, pctile, delta_mu=0.1, fit='piecewise'):
        '''
        Estimate the requested percentile of
        
            dE(B-V) / dDM
        
        over the whole map.
        
        Use the distance modulus step <delta_mu>
        to estimate the derivative.
        '''
        
        if fit == 'piecewise':
            if self.los_dmu > 0.:
                return np.percentile(np.diff(self.data.los_EBV[0]), pctile) / self.los_dmu    # TODO: Include first distance bin
            else:
                s = np.power(10., self.los_DM_anchor - 2.)
                return np.percentile(np.diff(self.data.los_EBV[0]) / np.diff(s), pctile)
        elif fit == 'cloud':
            return np.percentile(self.data.cloud_delta_EBV[0], pctile) / delta_mu
        else:
            raise ValueError('Unrecognized fit type: "%s"' % fit)
    
    
    def gen_EBV_map(self, mu, fit='piecewise',
                              method='median',
                              mask_sigma=None,
                              delta_mu=None,
                              reduce_nside=True):
        '''
        Returns an array of E(B-V) evaluated at
        distance modulus mu, with
        
            shape = (n_pixels,)
        
        Also returns an array of HEALPix pixel indices,
        and a single nside parameter, equal to the
        highest nside resolution in the map.
        
        The order of the output is
        
            nside, pix_idx, EBV
        
        The fit option can be either 'piecewise' or 'cloud',
        depending on which type of fit the map should use.
        
        The method option determines which measure of E(B-V)
        is returned. The options are
        
            'median', 'mean', 'best',
            'sample', 'sigma', float (percentile)
        
        'sample' generates a random map, drawn from the
        posterior. 'sigma' returns the percentile-equivalent
        of the standard deviation (half the 84.13%% - 15.87%% range).
        If method is a float, then the corresponding
        percentile map is returned.
        
        If mask_sigma is a float, then pixels where sigma is
        greater than the provided threshold will be masked out.
        '''
        
        EBV = None
        
        if fit == 'piecewise':
            EBV = self.calc_piecewise_EBV(mu)
        elif fit == 'cloud':
            EBV = self.calc_cloud_EBV(mu)
        else:
            raise ValueError('Unrecognized fit type: "%s"' % fit)
        
        # Calculate rate of reddening (dEBV/dDM or dEBV/ds), if requested
        if delta_mu != None:
            dmu = delta_mu
            if dmu < 0.:
                d1 = 10.**(mu/5.-2.)
                d2 = d1 + dmu
                dmu = mu - 5. * (2. + np.log10(d2))
                #print mu, dmu, d1, d2, delta_mu
            
            if fit == 'piecewise':
                EBV -= self.calc_piecewise_EBV(mu-dmu)
            elif fit == 'cloud':
                EBV -= self.calc_cloud_EBV(mu-dmu)
            
            EBV /= np.abs(delta_mu)
        
        # Mask regions with high uncertainty
        if mask_sigma != None:
            sigma = take_measure(EBV, 'sigma')
            mask_idx = (sigma > mask_sigma)
        
        # Reduce EBV in each pixel to one value
        EBV = take_measure(EBV, method)
        
        if mask_sigma != None:
            EBV[mask_idx] = np.nan
        
        if reduce_nside:
            # Reduce to one HEALPix nside resolution
            mask = self.data.los_mask[0]
            pix_idx = self.data.pix_idx[0][mask]
            nside = self.data.nside[0][mask]
            
            nside, pix_idx, EBV = reduce_to_single_res(pix_idx, nside, EBV)
            
            return nside, pix_idx, EBV
            
        else:
            return self.data.nside[0], self.data.pix_idx[0], EBV
    
    
    def gen_rasterizer(self, img_shape,
                             clip=True,
                             proj=hputils.Cartesian_projection(),
                             l_cent=0., b_cent=0.):
        '''
        Return a class that rasterizes a map with the same layout as this
        los_coll object (same nside and healpix index values). The
        class which is returned is a MapRasterizer object from hputils.
        '''
        
        return hputils.MapRasterizer(self.data.nside[0],
                                     self.data.pix_idx[0],
                                     img_shape,
                                     clip=clip,
                                     proj=proj,
                                     l_cent=l_cent,
                                     b_cent=b_cent)
    
    def get_nside_levels(self):
        '''
        Returns the unique nside values present in the
        map.
        '''
        
        return self.data.get_nside_levels()




####################################################################################
#
# 3D Mapper
#
#   Maps pixel data to a Cartesian grid.
#
####################################################################################

class Mapper3D:
    #def __init__(self, data):
    def __init__(self, nside, pix_idx, los_EBV, DM_min, DM_max):
        #self.data = data
        
        # Calculate the density of the map in every voxel,
        # indexed by (pixel, sample, distance)
        self.n_dist_bins = los_EBV.shape[2]
        self.DM_min, self.DM_max = DM_min, DM_max
        self.dDM = (self.DM_max - self.DM_min) / float(self.n_dist_bins - 1)
        
        mu = np.linspace(self.DM_min, self.DM_max, self.n_dist_bins)
        r = np.power(10., mu/5. + 1.)
        dr = np.diff(r)
        
        self.density = np.einsum('k,ijk->ijk', 1./dr, np.diff(los_EBV, axis=2))
        idx = ~np.isfinite(self.density)
        self.density[idx] = 0.
        
        # Calculate a mapping from a nested healpix index
        # (at the highest resolution present in the map) to
        # the index of the pixel in the map
        
        self.nside_max = np.max(nside)
        n_hires = hp.pixelfunc.nside2npix(self.nside_max)
        self.hires2mapidx = np.empty(n_hires, dtype='i8')
        self.hires2mapidx[:] = -1
        
        #print 'nside_max: %d' % (self.nside_max)
        
        for n in np.unique(nside):
            idx = np.nonzero(nside == n)[0]
            #print 'max(pix_idx) = %d' % (np.max(pix_idx[idx]))
            #print 'nside = %d' % n
            pix_idx_hires = (self.nside_max/n)**2 * pix_idx[idx]
            
            for k in xrange((self.nside_max/n)**2):
                self.hires2mapidx[pix_idx_hires+k] = idx
        
        #print '%d < hires2mapidx < %d' % (np.min(self.hires2mapidx), np.max(self.hires2mapidx))
    
    def _dist2bin(self, r):
        '''
        Convert from distance (in pc) to the lower distance bin index.
        '''
        
        mu = 5. * np.log10(r / 10.)
        bin = np.floor((mu - self.DM_min) / self.dDM)
        
        #for i in xrange(10):
        #    print mu.flatten()[1000*i], bin.flatten()[1000*i]
        
        return bin.astype('i4')
    
    def Cartesian2idx(self, x, y, z):
        '''
        Convert from a heliocentric position (x, y, z) to
        an index in the map.
        '''
        
        #r = np.sqrt(x**2 + y**2 + z**2)
        #l = np.arctan2(y, x)
        #theta = np.pi/2. - np.arcsin(z/r)
        #hires_idx = hp.pixelfunc.ang2pix(self.nside_max, theta, l, nest=True)
        
        #t, p = hp.pixelfunc.vec2ang(x, y, z)
        #l = 180./np.pi * p
        #b = 90. - 180./np.pi * t
        
        hires_idx = hp.pixelfunc.vec2pix(self.nside_max, x, y, z, nest=True)
        
        #idx = ~np.isfinite(hires_idx.flatten()) | (hires_idx.flatten() < 0)
        #print '# of bad coordinates: %d' % (np.sum(idx))
        #print ''
        #for xx,yy,zz,ii in zip(x.flatten()[idx], y.flatten()[idx], z.flatten()[idx], hires_idx.flatten()[idx]):
        #    print '(%.2f, %2f, %2f) --> %d' % (xx, yy, zz, ii)
        #print ''
        
        return self.hires2mapidx[hires_idx]
    
    def _unit_ortho(self, alpha, beta, n_x, n_y, n_z, scale):
        ijk = np.indices([2*n_x+1, 2*n_y+1, 2])
        ijk[0] -= n_x
        ijk[1] -= n_y
        
        # Unit vector matrix matrix
        ca, sa = np.cos(np.pi/180. * alpha), np.sin(np.pi/180. * alpha)
        cb, sb = np.cos(np.pi/180. * beta), np.sin(np.pi/180. * beta)
        
        u = np.array([[ca*cb, -sb, sa*cb],
                      [ca*sb,  cb, sa*sb],
                      [  -sa,   0, ca]])
        
        # Grid of points
        pos = np.einsum('dn,nijk->dijk', u, ijk)
        
        ray_dir = np.diff(pos, axis=3)[:,:,:,0]
        
        pos = pos[:,:,:,0] - n_z*ray_dir
        
        for k in xrange(3):
            pos[k] *= scale[k]
            ray_dir[k] *= scale[k]
        
        return pos, ray_dir
    
    def _unit_pinhole(self, alpha, beta, n_x, n_y, fov, r_0, ray_step, dist_init):
        # Generate tangent plane and normal vectors to plane
        pos, dpos = self._unit_ortho(alpha, beta, n_y, n_x, -1, (1.,1.,1.))
        
        # Offset plane from camera by appropriate amount to cover given field of view
        #dx = np.max(pos[0,:,:])
        dr = float(n_x) / np.tan(np.pi/180.*0.5*fov)
        pos += dr * dpos
        #print '\ndr = %.4f\n' % dr
        
        # Ray vectors are normalized tangent plane
        norm = np.sqrt(np.sum(pos**2, axis=0))
        ray_dir = ray_step * np.einsum('ij,dij->dij', 1./norm, pos)
        
        for i in xrange(3):
            pos[i,:,:] = r_0[i]
        
        pos += ray_dir * dist_init
        
        #print ''
        #print 'dr'
        #print dr
        #print ''
        #print 'pos'
        #print pos[:,::100,::100]
        #print pos[:,::100,::100]
        #print pos[:,::100,::100]
        #print ''
        #print 'ray_dir'
        #print ray_dir[:,::100,::100]
        #print ''
        
        return pos, ray_dir
    
    def _pos2map(self, pos):
        idx = self.Cartesian2idx(*pos)
        r = np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
        dist_bin = self._dist2bin(r)
        
        return idx, dist_bin
    
    def _calc_slice(self, map_val, pos):
        map_idx, dist_bin = self._pos2map(pos)
        
        idx = (map_idx != -1) & (dist_bin >= 0) & (dist_bin < self.density.shape[2])
        
        if np.sum(~idx) != 0:
            map_idx[~idx] = -1
            dist_bin[~idx] = -1
        
        #if np.sum(idx) != 0:
        #    print np.min(dist_bin[idx]), np.max(dist_bin[idx])
        #    print np.percentile(dist_bin[idx], [10., 30., 50., 70., 90.])
        
        return map_val[map_idx, dist_bin] * idx
    
    def proj_map_in_slices(self, camera, steps, reduction, *args, **kwargs): #alpha, beta, n_x, n_y, n_z, scale):
        verbose = kwargs.pop('verbose', False)
        
        if verbose:
            t_start = time.time()
            print '[.....................]',
            print '\b'*23,
        
        map_val = take_measure_nd(self.density, reduction)
        
        if camera == 'orthographic':
            pos, u = self._unit_ortho(*args, **kwargs)
            #pos, u = self._unit_ortho(alpha, beta, n_x, n_y, n_z)
        elif camera == 'pinhole':
            pos, u = self._unit_pinhole(*args, **kwargs)
        else:
            raise ValueError('Unrecognized camera: "%s" (choose "orthographic" or "pinhole")')
        
        img = self._calc_slice(map_val, pos)
        
        for k in xrange(steps):
            if verbose:
                if k % int((steps) / 20) == 0:
                    sys.stdout.write('>')
                    sys.stdout.flush()
            
            pos += u
            img += self._calc_slice(map_val, pos)
        
        #img /= 2.*n_z + 1.
        
        if verbose:
            dt = time.time() - t_start
            sys.stdout.write('] %.1f s \n' % dt)
            sys.stdout.flush()
        
        return img
    
    def _grid_ortho(self, alpha, beta, n_x, n_y, n_z):
        '''
        Compute rays for orthographic projection.
        '''
        
        # Increments of unit vectors for each point in grid
        ijk = np.indices([2*n_x+1, 2*n_y+1, 2*n_z+1])
        ijk[0] -= n_x
        ijk[1] -= n_y
        ijk[2] -= n_z
        
        # Unit vector matrix matrix
        ca, sa = np.cos(np.pi/180. * alpha), np.sin(np.pi/180. * alpha)
        cb, sb = np.cos(np.pi/180. * beta), np.sin(np.pi/180. * beta)
        
        u = np.array([[ca*cb, -sb, sa*cb],
                      [ca*sb,  cb, sa*sb],
                      [  -sa,   0, ca]])
        
        # Grid of points
        pos = np.einsum('dn,nijk->dijk', u, ijk)
        
        #print u[:,2]
        #print np.diff(pos, axis=3)
        #print ''
        
        return pos
    
    def _grid_mappos(self, alpha, beta, n_x, n_y, n_z, scale):
        pos = scale * self._grid_ortho(alpha, beta, n_x, n_y, n_z)
        idx = self.Cartesian2idx(*pos)
        r = np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
        dist_bin = self._dist2bin(r)
        
        return idx, dist_bin
    
    def project_map(self, alpha, beta, n_x, n_y, n_z, scale):
        map_val = np.median(self.density, axis=1)
        
        map_idx, dist_bin = self._grid_mappos(alpha, beta, n_x, n_y, n_z, scale)
        
        idx = (map_idx != -1) & (dist_bin >= 0) & (dist_bin < self.density.shape[2])
        map_idx[~idx] = -1
        dist_bin[~idx] = -1
        
        img = np.sum(map_val[map_idx, dist_bin] * idx, axis=2)
        
        return img
        





####################################################################################
#
# LOS Differencer
#
#   Contains methods for differencing two 3D maps, potentially with different
#   pixelizations.
#
####################################################################################

class LOSDifferencer:
    def __init__(self, nside_1, pix_idx_1,
                       nside_2, pix_idx_2):
        nside_max = max([np.max(nside_1), np.max(nside_2)])
        
        #print nside_max
        #print pix_idx_1
        #print pix_idx_2
        
        src_idx = []
        dest_idx = []
        
        for pix_idx, nside in [(pix_idx_1, nside_1), (pix_idx_2, nside_2)]:
            src_tmp = np.arange(pix_idx.size)
            _, dest_tmp, src_tmp = reduce_to_single_res(pix_idx, nside,
                                                        src_tmp, nside_max=nside_max)
            src_idx.append(src_tmp)
            dest_idx.append(dest_tmp)
        
        #print ''
        #print 'dest'
        #print dest_idx
        #print ''
        #print 'src'
        #print src_idx
        #print ''
        
        self.pix_idx = np.intersect1d(dest_idx[0], dest_idx[1])
        
        #print 'pix_idx'
        #print self.pix_idx
        #print ''
        
        self.src_idx = []
        
        for src, dest in zip(src_idx, dest_idx):
            idx = np.argsort(dest)
            dest = dest[idx]
            src = src[idx]
            
            #print dest
            
            idx = np.searchsorted(dest, self.pix_idx)
            
            #print idx
            
            self.src_idx.append(src[idx])
        
        self.nside = nside_max
    
    def gen_rasterizer(self, img_shape, clip=True,
                       proj=hputils.Cartesian_projection(),
                       l_cent=0., b_cent=0.):
        nside = self.nside * np.ones(self.pix_idx.size, dtype='i8')
        
        return hputils.MapRasterizer(nside, self.pix_idx,
                                     img_shape, clip=clip, proj=proj,
                                     l_cent=l_cent, b_cent=b_cent)
    
    def pix_diff(self, pix_val_1, pix_val_2):
        return pix_val_2[self.src_idx[1]] - pix_val_1[self.src_idx[0]]






#
# Test functions
#

def test_load():
    fname = '/n/fink1/ggreen/bayestar/output/nogiant/AquilaSouthLarge2/AquilaSouthLarge2.00000.h5'
    
    pix_info, cloud_info, los_info, star_stack, DM_EBV_lim = load_output_file(fname, load_stacked_pdfs=True)
    
    pix_idx, nside, cloud_mask, los_mask, n_stars = pix_info
    
    print pix_idx
    print nside
    print star_stack.shape
    print DM_EBV_lim
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    ax.imshow(np.sqrt(star_stack[0]), extent=DM_EBV_lim, origin='lower',
              aspect='auto', cmap='Blues', interpolation='nearest')
    
    plt.show()


def test_load_multiple():
    #fnames = ['/n/fink1/ggreen/bayestar/output/AquilaSouthLarge2/AquilaSouthLarge2.%.05d.h5' % i for i in xrange(25)]
    fnames = glob.glob('/n/fink1/ggreen/bayestar/output/AquilaSouthLarge2/AquilaSouthLarge2.*.h5')
    
    mapper = LOSMapper(fnames, processes=8, load_stacked_pdfs=True)
    data = mapper.data
    
    # Single line of sight
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    ax.imshow(np.sqrt(data.star_stack[0][0]), extent=data.DM_EBV_lim, origin='lower',
              aspect='auto', cmap='Blues', interpolation='nearest')
    
    DM = data.get_los_DM_range()
    
    lnp = data.los_lnp[0][0, 1:]
    lnp_min, lnp_max = np.percentile(lnp, [10., 90.])
    lnp = (lnp - lnp_min) / (lnp_max - lnp_min)
    lnp[lnp > 1.] = 1.
    lnp[lnp < 0.] = 0.
    rgb_col = np.array([1.-lnp, 0.*lnp, lnp]).T
    
    
    for c, EBV in zip(rgb_col, data.los_EBV[0][0, 1:, :]):
        ax.plot(DM, EBV, c=c, lw=1.5, alpha=0.02)
    
    ax.plot(DM, data.los_EBV[0][0, 0, :], 'g', lw=2., alpha=0.50)
    
    plt.show()
    
    # Map
    size = (500, 500)
    mu = 15.
    
    rasterizer = mapper.gen_rasterizer(size)
    
    bounds = rasterizer.get_lb_bounds()
    
    tmp, tmp, pix_val = mapper.gen_EBV_map(mu, fit='cloud',
                                               method='median',
                                               reduce_nside=False)
    img = rasterizer(pix_val)
    
    EBV_max = 0.70 #np.percentile(pix_val[np.isfinite(pix_val)], 95.)
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, axisbg=(0.6, 0.8, 0.95, 0.95))
    
    ax.imshow(img.T, aspect='auto', origin='lower',
                     cmap='binary', interpolation='nearest',
                     vmin=0., vmax=EBV_max,
                     extent=bounds)
    plt.show()


def test_plot_comparison():
    #img_path = '/nfs_pan1/www/ggreen/cloudmaps/AqS_2MASS_losbounded'
    img_name = 'l30_comp'
    img_path = '/n/fink1/ggreen/bayestar/movies/allsky_2MASS/l30/'
    n_frames = 500
    method = 'sample'
    
    bounds = [30., 40., -3., 9.]
    figsize = figsize=(32, 10.8)
    size = (1200, 600)
    
    EBV_max = 3.80
    diff_max = 0.40
    
    #fnames_1 = glob.glob('/n/fink1/ggreen/bayestar/output/AquilaSouthLarge2/AquilaSouthLarge2.*.h5')
    fnames_2 = glob.glob('/n/fink1/ggreen/bayestar/output/allsky_2MASS/allsky_2MASS.*.h5')
    fnames_1 = glob.glob('/n/fink1/ggreen/bayestar/output/allsky/allsky.*.h5')
    #fnames_2 = glob.glob('/n/fink1/ggreen/bayestar/output/AqS_2MASS_smE/AqS_2MASS.*.h5')
    #fnames_1 = ['/n/fink1/ggreen/bayestar/output/AquilaSouthLarge2/AquilaSouthLarge2.%.5d.h5' % i for i in xrange(25)]
    #fnames_2 = ['/n/fink1/ggreen/bayestar/output/gbright_giant/AquilaSouthLarge2/AquilaSouthLarge2.%.5d.h5' % i for i in xrange(25)]
    
    #label_1 = r'$\mathrm{PS1}$'
    label_1 = r'$\mathrm{PS1}$'
    label_2 = r'$\mathrm{PS1 + 2MASS}$'
    
    mapper_1 = LOSMapper(fnames_1, processes=4, load_stacked_pdfs=False, bounds=bounds)
    mapper_2 = LOSMapper(fnames_2, processes=4, load_stacked_pdfs=False, bounds=bounds)
    
    #mapper_1.data.sort()
    #mapper_2.data.sort()
    
    tmp, tmp, val_end_1 = mapper_1.gen_EBV_map(19., fit='piecewise',
                                                    method='median',
                                                    reduce_nside=False)
    tmp, tmp, val_end_2 = mapper_2.gen_EBV_map(19., fit='piecewise',
                                                    method='median',
                                                    reduce_nside=False)
    
    print mapper_1.data.pix_idx[0].shape
    print mapper_2.data.pix_idx[0].shape
    
    rasterizer_1 = mapper_1.gen_rasterizer(size)
    rasterizer_2 = mapper_2.gen_rasterizer(size)
    
    bounds_1 = rasterizer_1.get_lb_bounds()
    bounds_2 = rasterizer_2.get_lb_bounds()
    
    differencer = LOSDifferencer(mapper_1.data.nside[0], mapper_1.data.pix_idx[0],
                                 mapper_2.data.nside[0], mapper_2.data.pix_idx[0])
    rasterizer_3 = differencer.gen_rasterizer(size)
    bounds_3 = rasterizer_3.get_lb_bounds()
    
    bounds = []
    bounds.append(min(bounds_1[0], bounds_2[0], bounds_3[0]))
    bounds.append(max(bounds_1[1], bounds_2[1], bounds_3[1]))
    bounds.append(min(bounds_1[2], bounds_2[2], bounds_3[2]))
    bounds.append(max(bounds_1[3], bounds_2[3], bounds_3[3]))
    
    mu_range = np.linspace(4., 19., n_frames)
    
    Delta = np.empty((3, n_frames), dtype='f8')
    Delta[:] = np.nan
    
    med = np.empty((2, 3, n_frames), dtype='f8')
    med[:] = np.nan
    
    for i, mu in enumerate(mu_range):
        tmp, tmp, pix_val_1 = mapper_1.gen_EBV_map(mu, fit='piecewise',
                                                       method=method,
                                                       reduce_nside=False)
        tmp, tmp, pix_val_2 = mapper_2.gen_EBV_map(mu, fit='piecewise',
                                                       method=method,
                                                       reduce_nside=False)
        pix_val_3 = differencer.pix_diff(pix_val_1, pix_val_2)
        
        img_1 = rasterizer_1(pix_val_1)
        img_2 = rasterizer_2(pix_val_2)
        img_3 = rasterizer_3(pix_val_3)
        
        med[0, :, i] = np.percentile(pix_val_1 / val_end_1, [10., 50., 90.])
        med[1, :, i] = np.percentile(pix_val_2 / val_end_2, [10., 50., 90.])
        Delta[:, i] = np.percentile(pix_val_3, [10., 50., 90.])
        
        print 'Plotting figure %d (mu = %.3f), Delta E(B-V) = %.3f' % (i, mu, Delta[1,i])
        
        fig = plt.figure(figsize=figsize, dpi=100)
        
        ax_1 = fig.add_subplot(1,3,1, axisbg=(0.6, 0.8, 0.95, 0.95))
        ax_2 = fig.add_subplot(1,3,2, axisbg=(0.6, 0.8, 0.95, 0.95))
        ax_3 = fig.add_subplot(1,3,3)
        
        fig.subplots_adjust(left=0.08, right=0.96,
                            top=0.95, bottom=0.32,
                            wspace=0.)
        
        ax_2.set_yticklabels([])
        ax_3.set_yticklabels([])
        
        ax_1.xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax_2.xaxis.set_major_locator(ticker.MaxNLocator(5, prune='lower'))
        ax_3.xaxis.set_major_locator(ticker.MaxNLocator(5, prune='lower'))
        
        ax_1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax_2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax_3.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        
        ax_1.yaxis.set_major_locator(ticker.MaxNLocator(5))
        ax_2.yaxis.set_major_locator(ticker.MaxNLocator(5))
        ax_3.yaxis.set_major_locator(ticker.MaxNLocator(5))
        
        ax_1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax_2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax_3.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        
        d = 10.**(mu / 5. - 2.)
        
        fig.suptitle(r'$\mu = %.2f \ (d = %.2f \, \mathrm{kpc})$' % (mu, d),
                     fontsize=20, ha='center')
        ax_1.set_ylabel(r'$b$', fontsize=18)
        ax_2.set_xlabel(r'$\ell$', fontsize=18)
        
        im_1 = ax_1.imshow(img_1.T, aspect='auto', origin='lower',
                                    cmap='binary', interpolation='nearest',
                                    vmin=0., vmax=EBV_max,
                                    extent=bounds_1)
        ax_2.imshow(img_2.T, aspect='auto', origin='lower',
                             cmap='binary', interpolation='nearest',
                             vmin=0., vmax=EBV_max,
                             extent=bounds_2)
        im_3 = ax_3.imshow(img_3.T, aspect='auto', origin='lower',
                                    cmap='RdBu', interpolation='nearest',
                                    vmin=-diff_max, vmax=diff_max,
                                    extent=bounds_3)
        
        for ax in [ax_1, ax_2, ax_3]:
            ax.set_xlim(bounds[:2])
            ax.set_ylim(bounds[2:])
        
        w = bounds[1] - bounds[0]
        h = bounds[3] - bounds[2]
        x_txt = bounds[0] + 0.05 * w
        y_txt = bounds[3] - 0.05 * h
        
        for a,lbl in zip([ax_1, ax_2], [label_1, label_2]):
            txt = a.text(x_txt, y_txt, lbl,
                         ha='left', va='top', fontsize=22,
                         path_effects=[PathEffects.withStroke(linewidth=2, foreground='k')])
            #txt.set_path_effects([PathEffects.Stroke(linewidth=2, foreground='w')])
        
        #
        # Colorbars
        #
        
        cax_1 = fig.add_axes([0.08, 0.24, 0.42, 0.025])
        cax_2 = fig.add_axes([0.54, 0.24, 0.42, 0.025])
        
        fig.colorbar(im_1, cax=cax_1, orientation='horizontal')
        #fig.colorbar(im_3, cax=cax_2, orientation='horizontal')
        
        x = np.linspace(-diff_max, diff_max, 1000)
        x.shape = (1, x.size)
        
        cax_2.imshow(x, origin='lower', cmap='RdBu',
                        interpolation='bilinear', aspect='auto',
                        vmin=-diff_max, vmax=diff_max,
                        extent=[-diff_max, diff_max, -1., 1.])
        
        cax_2.axvline(x=Delta[1,i], lw=5., c='k', alpha=0.5)
        
        cax_2.set_yticks([])
        cax_2.xaxis.set_major_locator(ticker.MaxNLocator(N=10))
        
        cax_1.set_xlabel(r'$\mathrm{E} \left( B - V \right)$', fontsize=16)
        cax_2.set_xlabel(r'$\Delta \mathrm{E} \left( B - V \right)$', fontsize=16)
        
        #
        # Difference vs. distance
        #
        
        ax = fig.add_axes([0.08, 0.06, 0.88, 0.12])
        
        ax.fill_between(mu_range[:i+1], Delta[0,:i+1], Delta[2,:i+1],
                        facecolor='b', alpha=0.5)
        ax.plot(mu_range[:i+1], Delta[1,:i+1], c='b', alpha=0.5)
        
        ax.axhline(0., c='k', lw=2., alpha=0.5)
        
        ax.set_ylim(-1.5*diff_max, 1.5*diff_max)
        
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        
        ax.set_xlabel(r'$\mu$', fontsize=18)
        ax.set_ylabel(r'$\Delta \mathrm{E} \left( B - V \right)$', fontsize=16)
        
        #
        # E(B-V) vs. distance
        #
        
        ax2 = ax.twinx()
        
        ax2.fill_between(mu_range[:i+1], med[0,0,:i+1], med[0,2,:i+1],
                         facecolor='r', alpha=0.15)
        ax2.plot(mu_range[:i+1], med[0,1,:i+1], c='r', lw=2., alpha=0.5)
        
        ax2.fill_between(mu_range[:i+1], med[1,0,:i+1], med[1,2,:i+1],
                         facecolor='g', alpha=0.15)
        ax2.plot(mu_range[:i+1], med[1,1,:i+1], c='g', lw=2., alpha=0.5)
        
        ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
        ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        
        ax2.set_ylim(0., 1.1)
        
        ax2.set_ylabel(r'$\mathrm{scaled \ } \mathrm{E} \left( B - V \right)$', fontsize=14)
        
        ax.set_xlim(mu_range[0], mu_range[-1])
        
        #print 'Saving to %s/2MASS_res_comp.%.5d.png ...' % (img_path, i)
        fig.savefig('%s/%s.%.5d.png' % (img_path, img_name, i), dpi=100)
        plt.close(fig)


def test_unified():
    in_fname = glob.glob('/n/fink1/ggreen/bayestar/output/AquilaSouthLarge2/AquilaSouthLarge2.*.h5')
    out_fname = '/n/fink1/ggreen/bayestar/output/AquilaSouthLarge2/AquilaSouthLarge2_unified.h5'
    img_path = '/nfs_pan1/www/ggreen/cloudmaps/AquilaSouthLarge2/comp2'
    
    print 'Loading Bayestar output files ...'
    mapper_1 = LOSMapper(in_fname, processes=8)
    
    print 'Saving as unified file ...'
    mapper_1.data.save_unified(out_fname)
    
    print 'Loading unified file ...'
    mapper_2 = LOSMapper([out_fname], processes=1)
    
    print 'Plotting comparison ...'
    mapper_1.data.sort()
    mapper_2.data.sort()
    
    size = (500, 500)
    rasterizer = mapper_1.gen_rasterizer(size)
    bounds = rasterizer.get_lb_bounds()
    
    mu_range = np.linspace(4., 19., 31)
    
    Delta = np.empty((3, 500), dtype='f8')
    Delta[:] = np.nan
    
    for i, mu in enumerate(mu_range):
        tmp, tmp, pix_val_1 = mapper_1.gen_EBV_map(mu, fit='piecewise',
                                                       method='median',
                                                       reduce_nside=False)
        tmp, tmp, pix_val_2 = mapper_2.gen_EBV_map(mu, fit='piecewise',
                                                       method='median',
                                                       reduce_nside=False)
        img_1 = rasterizer(pix_val_1)
        img_2 = rasterizer(pix_val_2)
        img_3 = rasterizer(pix_val_2 - pix_val_1)
        
        tmp = np.percentile(pix_val_2 - pix_val_1, [10., 50., 90.])
        
        Delta[0, i] = tmp[0]
        Delta[1, i] = tmp[1]
        Delta[2, i] = tmp[2]
        
        print 'Plotting figure %d (mu = %.3f), Delta E(B-V) = %.3f' % (i, mu, tmp[1])
        
        EBV_max = 0.70
        diff_max = 0.20
        
        fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
        
        ax_1 = fig.add_subplot(1,3,1, axisbg=(0.6, 0.8, 0.95, 0.95))
        ax_2 = fig.add_subplot(1,3,2, axisbg=(0.6, 0.8, 0.95, 0.95))
        ax_3 = fig.add_subplot(1,3,3)
        
        fig.subplots_adjust(left=0.08, right=0.96,
                            top=0.95, bottom=0.32,
                            wspace=0.)
        
        ax_2.set_yticklabels([])
        ax_3.set_yticklabels([])
        
        ax_1.xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax_2.xaxis.set_major_locator(ticker.MaxNLocator(5, prune='lower'))
        ax_3.xaxis.set_major_locator(ticker.MaxNLocator(5, prune='lower'))
        
        ax_1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax_2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax_3.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        
        ax_1.yaxis.set_major_locator(ticker.MaxNLocator(5))
        ax_2.yaxis.set_major_locator(ticker.MaxNLocator(5))
        ax_3.yaxis.set_major_locator(ticker.MaxNLocator(5))
        
        ax_1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax_2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax_3.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        
        d = 10.**(mu / 5. - 2.)
        
        fig.suptitle(r'$\mu = %.2f \ (d = %.2f \, \mathrm{kpc})$' % (mu, d),
                     fontsize=20, ha='center')
        ax_1.set_ylabel(r'$b$', fontsize=18)
        ax_2.set_xlabel(r'$\ell$', fontsize=18)
        
        im_1 = ax_1.imshow(img_1.T, aspect='auto', origin='lower',
                                    cmap='binary', interpolation='nearest',
                                    vmin=0., vmax=EBV_max,
                                    extent=bounds)
        ax_2.imshow(img_2.T, aspect='auto', origin='lower',
                             cmap='binary', interpolation='nearest',
                             vmin=0., vmax=EBV_max,
                             extent=bounds)
        im_3 = ax_3.imshow(img_3.T, aspect='auto', origin='lower',
                                    cmap='RdBu', interpolation='nearest',
                                    vmin=-diff_max, vmax=diff_max,
                                    extent=bounds)
        
        #
        # Colorbars
        #
        
        cax_1 = fig.add_axes([0.08, 0.24, 0.42, 0.025])
        cax_2 = fig.add_axes([0.54, 0.24, 0.42, 0.025])
        
        fig.colorbar(im_1, cax=cax_1, orientation='horizontal')
        #fig.colorbar(im_3, cax=cax_2, orientation='horizontal')
        
        x = np.linspace(-diff_max, diff_max, 1000)
        x.shape = (1, x.size)
        
        cax_2.imshow(x, origin='lower', cmap='RdBu',
                        interpolation='bilinear', aspect='auto',
                        vmin=-diff_max, vmax=diff_max,
                        extent=[-diff_max, diff_max, -1., 1.])
        
        cax_2.axvline(x=Delta[1,i], lw=5., c='k', alpha=0.5)
        
        cax_2.set_yticks([])
        cax_2.xaxis.set_major_locator(ticker.MaxNLocator(N=10))
        
        cax_1.set_xlabel(r'$\mathrm{E} \left( B - V \right)$', fontsize=16)
        cax_2.set_xlabel(r'$\Delta \mathrm{E} \left( B - V \right)$', fontsize=16)
        
        #
        # Difference vs. distance
        #
        
        ax = fig.add_axes([0.08, 0.06, 0.88, 0.12])
        
        ax.fill_between(mu_range[:i+1], Delta[0,:i+1], Delta[2,:i+1],
                        facecolor='b', alpha=0.5)
        ax.plot(mu_range[:i+1], Delta[1,:i+1], c='b', alpha=0.5)
        
        ax.axhline(0., c='k', lw=2., alpha=0.5)
        
        ax.set_xlim(mu_range[0], mu_range[-1])
        ax.set_ylim(-1.5*diff_max, 1.5*diff_max)
        
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        
        ax.set_xlabel(r'$\mu$', fontsize=18)
        ax.set_ylabel(r'$\Delta \mathrm{E} \left( B - V \right)$', fontsize=16)
        
        fig.savefig('%s/unified_test.%.5d.png' % (img_path, i), dpi=100)
        plt.close(fig)

def test_ortho_proj():
    #in_fname = glob.glob('/n/fink1/ggreen/bayestar/output/allsky/allsky.00000.h5')
    #mapper = LOSMapper(in_fname, processes=8, max_samples=25)
    #mapper3d = Mapper3D(mapper.data.nside[0], mapper.data.pix_idx[0],
    #                    mapper.data.los_EBV[0], mapper.data.DM_EBV_lim[0], mapper.data.DM_EBV_lim[1])
    
    '''
    f = h5py.File('/n/fink1/ggreen/bayestar/output/allsky/n64.h5')
    los_EBV = f['/EBV'][:]
    f.close()
    
    idx = (los_EBV < 0.)
    los_EBV[idx] = np.nan
    los_EBV.shape = (los_EBV.shape[0], 1, los_EBV.shape[1])
    
    pix_idx = np.arange(los_EBV.shape[0])
    nside = 64 * np.ones(los_EBV.shape[0], dtype='i8')
    '''
    
    mapper = LOSMapper(['/n/fink1/ggreen/bayestar/output/allsky/compact_5samp.h5'])
    nside = mapper.data.nside[0]
    pix_idx = mapper.data.pix_idx[0]
    los_EBV = mapper.data.los_EBV[0]
    
    mapper3d = Mapper3D(nside, pix_idx, los_EBV, 4., 19.)
    
    n_frames = 200
    alpha = 100. * (1. - np.abs(np.linspace(1., -1., n_frames))**1.5)
    beta = np.linspace(0., 180, alpha.size)
    
    block = 9
    n_per_block = 10
    
    k_start = block * n_per_block
    alpha = alpha[k_start:k_start+20]
    beta = beta[k_start:k_start+20]
    
    n_x, n_y, n_z = 1200, 1200, 1200
    scale = (7., 7., 7.)
    
    extent = [scale[0]*n_x, -scale[0]*n_x, -scale[1]*n_y, scale[1]*n_y]
    vmax_lst = []
    
    np.seterr(all='ignore')
    
    for k, (a, b) in enumerate(zip(alpha, beta)):
        print 'Rasterizing frame %d: ' % (k + k_start),
        img = mapper3d.proj_map_in_slices('orthographic', n_z, 'sample', a, b, n_x, n_y, n_z, scale)
        
        vmax_lst.append(1.2*np.percentile(img, 99.5))
        #w = np.exp(-np.arange(len(vmax_lst)))
        #vmax = np.sum(w * np.array(vmax_lst)[::-1]) / np.sum(w)
        vmax = vmax_lst[-1]
        
        fig = plt.figure(figsize=(15,15), dpi=300)
        ax = fig.add_subplot(1,1,1)
        
        ax.imshow(img, origin='lower', cmap='binary',
                       interpolation='bilinear', aspect='auto',
                       extent=extent, vmin=0., vmax=vmax)
        
        ax.set_xlim(ax.get_xlim()[::-1])
        
        ax.set_title(r'$\left( \alpha, \beta \right) = \left( %.1f^{\circ} \ %.1f^{\circ} \right)$' % (a, b), fontsize=20)
        
        fig.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.95)
        
        fig.savefig('/n/pan1/www/ggreen/3d/mw_ortho/mw_ortho.%05d.png' % (k+k_start), dpi=300)
        
        plt.close(fig)
        del img


def pm_ang_formatter(theta, pos):
    if np.abs(theta) < 1.e-5:
        return r'$+0^{\circ}$'
    elif theta > 0.:
        return r'$+%d^{\circ}$' % theta
    else:
        return r'$%d^{\circ}$' % theta


def camera_z_bobbing(dn, z_0, x_0, y_0, dx_stare, dz_stare, beta_0):
    # Calculate z-coordinate of camera in each frame
    n = np.arange(dn)
    dz = np.cos( np.pi * (n/float(dn)) )
    z1 = 0.5 * z_0 * (1. - dz)
    
    n2 = np.arange(dn + int(1./10.*dn))
    dz2 = np.cos( np.pi * (n2/float(n2.size)) )
    dz2 = dz2[:dn]
    dz2 -= dz2[-1]
    dz2 += 0.5 * (dz2[-2] - dz2[-1])
    dz2 *= 1./dz2[0]
    z2 = z_0 * np.hstack([dz2, -dz2[::-1]])
    
    # Construct camera positions in (x,y,z)
    r_0 = np.zeros((5*dn, 3), dtype='f8')
    
    r_0[:,0] = x_0
    r_0[:,1] = y_0
    r_0[:,2] = np.hstack([np.zeros(dn), z1, z2, z1-z_0])
    
    # Orient camera to face point offset from initial position by dx_stare and dz_stare
    alpha = 90. - 180./np.pi * np.arctan(-(r_0[:,2]-dz_stare) / dx_stare)
    beta = beta_0 * np.ones(r_0.shape[0])
    
    return r_0, alpha, beta


def bob_from_zero():
    #
    # Camera properties
    #
    
    block = 61        # Which block of frames to generate
    n_per_block = 2   # Number of frames in each block
    dk = 0
    
    dn = 50     # One fifth the total number of frames
    
    z_0 = 500.     # Height (in pc) of bobbing above plane of Galaxy
    x_0 = 0.       # Offset (in pc) from Sun (+ = towards GC)
    y_0 = 0.       # Offset (in pc) from Sun (tangential to Sun's orbit)
    
    dx_stare = 1000.     # x-offset (in pc) in plane at which to point camera
    dz_stare = -100.     # z-offset (in pc) above plane at which to point camera
    
    beta_0 = 180.  # beta-angle for camera (0 = towards GC)
    
    n_x, n_y, n_z = 800, 400, 1200  # Image (x,y), and number of steps along each ray
    dr = 5.                         # Size (in pc) of each step along rays
    fov = 90.                       # Horizontal field of view (in degrees)
    dpi = 100                       # DPI of frames


def pinhole_proj_movie(r_0, alpha, beta, n_pix, dr, fov, dpi, block, n_per_block, dk, out_fname):
    k_start = n_per_block * block
    r_0 = r_0[k_start+dk:k_start+n_per_block]
    alpha = alpha[k_start+dk:k_start+n_per_block]
    beta = beta[k_start+dk:k_start+n_per_block]
    
    # Load in 3D map
    mapper = LOSMapper(['/n/fink1/ggreen/bayestar/output/allsky/compact_5samp.h5'])
    nside = mapper.data.nside[0]
    pix_idx = mapper.data.pix_idx[0]
    los_EBV = mapper.data.los_EBV[0]
    
    mapper3d = Mapper3D(nside, pix_idx, los_EBV, 4., 19.)
    
    # Generate frames
    extent = [fov/2., -fov/2., float(n_y)/float(n_x)*fov/2., -float(n_y)/float(n_x)*fov/2.]
    vmax_lst = []
    
    np.seterr(all='ignore')
    
    for k, (r, a, b) in enumerate(zip(r_0, alpha, beta)):
        print 'Rasterizing frame %d: ' % (k + k_start),
        img = mapper3d.proj_map_in_slices('pinhole', n_z, 'sample', a, b, n_x, n_y, fov, r, dr)
        
        vmax_lst.append(1.2*np.percentile(img, 99.5))
        #w = np.exp(-np.arange(len(vmax_lst)))
        #vmax = np.sum(w * np.array(vmax_lst)[::-1]) / np.sum(w)
        vmax = vmax_lst[-1]
        
        fig = plt.figure(figsize=(15,15.*float(n_y)/float(n_x)), dpi=dpi)
        ax = fig.add_subplot(1,1,1)
        
        ax.imshow(img, origin='lower', cmap='binary',
                       interpolation='bilinear', aspect='auto',
                       extent=extent, vmin=0., vmax=vmax)
        
        ax.set_xlim(ax.get_xlim()[::-1])
        ax.set_ylim(ax.get_ylim()[::-1])
        
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(pm_ang_formatter))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(pm_ang_formatter))
        
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        
        ax.set_title(r'$\left( \alpha, \, \beta, \, x, \, y,  \, z \right) = \left( %.1f^{\circ} \ %.1f^{\circ} \ \ %d \, \mathrm{pc} \ %d \, \mathrm{pc} \ %d \, \mathrm{pc} \right)$' % (a, b, r[0], r[1], r[2]), fontsize=20)
        
        fig.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.95)
        
        fig.savefig('/n/pan1/www/ggreen/3d/anticenter/%s.%05d.png' % (out_fname, k+k_start+dk), dpi=dpi)
        
        plt.close(fig)
        del img


def test_save_lowres_map():
    fnames = glob.glob('/n/fink1/ggreen/bayestar/output/allsky/compact.*.h5')
    #fnames = ['/n/fink1/ggreen/bayestar/output/allsky/compact.00.h5']
    
    pix_nside = []
    pix_idx = []
    los_EBV = []
    
    for k,fn in enumerate(fnames):
        print 'Loading file %d of %d ...' % (k+1, len(fnames))
        
        f = h5py.File(fn, 'r')
        
        tmp = f['/locations'][:]
        pix_nside.append(tmp['nside'][:])
        pix_idx.append(tmp['healpix_index'][:])
        
        los_EBV.append(np.median(f['/piecewise'][:,2:,1:], axis=1))
        
        f.close()
    
    pix_nside = np.concatenate(pix_nside)
    pix_idx = np.concatenate(pix_idx)
    los_EBV = np.concatenate(los_EBV, axis=0)
    
    #mapper = LOSMapper(fnames, processes=1, max_samples=10)
    
    EBV = []
    nside = 64
    mu = np.linspace(4., 19., 31)
    
    f = h5py.File('/n/fink1/ggreen/bayestar/output/allsky/n64.h5', 'w')
    
    shape = (hp.pixelfunc.nside2npix(nside), len(mu))
    dset = f.create_dataset('/EBV', dtype='f4', shape=shape, chunks=True,
                                    compression='gzip', compression_opts=9)
    
    for k,d in enumerate(mu):
        print 'Writing DM = %.2f (%d of %d) ...' % (d, k+1, len(mu))
        dset[:,k] = resample_at_res(pix_idx, pix_nside, los_EBV[:,k], nside)
        #n, i, E = mapper.gen_EBV_map(d)
        #dset[:,k] = resample_at_res(i, n, E, nside)
    
    f.close()


def main():
    #test_save_lowres_map()
    test_pinhole_proj()
    
    return 0


if __name__ == '__main__':
    main()
