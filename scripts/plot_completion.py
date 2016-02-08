#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  plot_completion.py
#  
#  Copyright 2013-2014 Greg Green <gregorymgreen@gmail.com>
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

import matplotlib as mplib
mplib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.colors import ListedColormap, BoundaryNorm

import argparse, sys, time, glob
from os.path import expanduser, abspath
import os.path
import subprocess

import cPickle
import gzip
import ujson as json

import healpy as hp
import h5py

import hputils, maptools


pallette = {'orange': (0.9, 0.6, 0.),
            'sky blue': (0.35, 0.70, 0.90),
            'bluish green': (0., 0.6, 0.5),
            'yellow': (0.95, 0.9, 0.25),
            'blue': (0., 0.45, 0.7),
            'vermillion': (0.8, 0.4, 0.),
            'reddish purple': (0.8, 0.6, 0.7)}


# Trap SEGFAULTs
import signal
def segfault_handler(signal, frame):
    print 'SEGFAULT trapped.'
    raise IOError('SEGFAULT likely caused by failed HDF5 read.')

signal.signal(signal.SIGSEGV, segfault_handler)


# Resample a multi-resolution HEALPix image to one NSIDE level
def multires2nside(nside_multires, pix_idx_multires,
                   pix_val_multires, nside_target,
                   fill=np.nan):
    
    npix_hires = hp.pixelfunc.nside2npix(nside_target)
    pix_val_hires = np.empty(npix_hires, dtype=pix_val_multires.dtype)
    pix_val_hires[:] = fill
    
    for nside in np.unique(nside_multires):
        # Get indices of all pixels at current nside level
        idx = (nside_multires == nside)
        
        # Extract E(B-V) of each selected pixel
        pix_val_n = pix_val_multires[idx]
        
        # Determine nested index of each selected pixel in upsampled map
        mult_factor = (nside_target/nside)**2
        pix_idx_n = pix_idx_multires[idx] * mult_factor
        
        # Write the selected pixels into the upsampled map
        for offset in range(mult_factor):
            pix_val_hires[pix_idx_n+offset] = pix_val_n[:]
    
    return pix_val_hires


def downsample_by_2(pix_val):
    return 0.25 * (pix_val[::4] + pix_val[1::4] + pix_val[2::4] + pix_val[3::4])


class PixelIdentifier:
    '''
    Class that prints out the HEALPix pixel index when the user
    clicks somewhere in a figure.
    '''
    
    def __init__(self, ax, nside,
                       nest=True,
                       proj=hputils.Cartesian_projection()):
        self.ax = ax
        self.cid = ax.figure.canvas.mpl_connect('button_press_event', self)
        
        self.nside = nside
        self.nest = nest
        
        self.proj = proj
    
    def __call__(self, event):
        if event.inaxes != self.ax:
            return
        
        # Determine healpix index of point
        x, y = event.xdata, event.ydata
        b, l = self.proj.inv(x, y)
        pix_idx = hputils.lb2pix(self.nside, l, b, nest=self.nest)
        
        print '(%.2f, %.2f) -> %d' % (l, b, pix_idx)


def match_indices(large_arr, small_arr):
    sort_idx = np.argsort(large_arr)
    large_arr_idx = np.searchsorted(large_arr, small_arr, sorter=sort_idx)
    
    filt_idx = large_arr_idx < large_arr.size
    large_arr_idx = large_arr_idx[filt_idx]
    
    match_idx = np.nonzero(small_arr[filt_idx] == large_arr[sort_idx][large_arr_idx])[0]
    
    return sort_idx[large_arr_idx][match_idx], match_idx


class TCompletion:
    def __init__(self, indir, outdir):
        # Try to load from pre-existing pickle
        self.pickle_fname = os.path.normpath(outdir + '/status.pickle.gz')
        success = self.unpickle(self.pickle_fname)
        if success:
            return
        else:
            print 'Could not load from pickle. Loading manually.'
        
        # Initialize input and output filenames
        self.indir = indir
        self.outdir = outdir
        self.infiles = glob.glob(os.path.normpath(indir + '/*.h5'))
        self.infiles = sorted(self.infiles)
        self.basenames = [os.path.basename(fname) for fname in self.infiles]
        self.outfiles = [os.path.normpath(outdir + '/' + fname) for fname in self.basenames]
        
        # Get list of pixels
        self.nside = []
        self.pix_idx = []
        self.n_stars = []
        self.infile_idx = []
        
        for k,fname in enumerate(self.infiles):
            print 'Processing input file %d of %d...' % (k, len(self.infiles))
            
            f = h5py.File(fname, 'r')
            
            for _, pixel in f['/photometry'].iteritems():
                self.nside.append(pixel.attrs['nside'])
                self.pix_idx.append(pixel.attrs['healpix_index'])
                self.n_stars.append(pixel.size)
                self.infile_idx.append(k)
            
            f.close()
        
        self.nside = np.array(self.nside).astype('i4')
        self.pix_idx = np.array(self.pix_idx).astype('i8')
        self.n_stars = np.array(self.n_stars)
        self.n_stars_tot = np.sum(self.n_stars)
        self.infile_idx = np.array(self.infile_idx)
        
        # Initialize output file modification times
        self.modtime = np.empty(len(self.outfiles), dtype='f8')
        self.modtime[:] = -1.
        #self.modtime = dict.fromkeys(self.outfiles, -1)
        
        # Initialize pixel statuses
        self.pix_name = ['%d-%d' % (n, i) for n, i in zip(self.nside, self.pix_idx)]
        self.idx_in_map = {name:i for i,name in enumerate(self.pix_name)}
        
        self.has_indiv = np.empty(self.nside.size, dtype=np.bool)
        self.has_indiv[:] = False
        
        self.has_cloud = np.empty(self.nside.size, dtype=np.bool)
        self.has_cloud[:] = False
        
        self.has_los = np.empty(self.nside.size, dtype=np.bool)
        self.has_los[:] = False
        
        self.EBV = np.empty((self.nside.size, 3), dtype='f2')
        self.EBV[:] = np.nan
        
        self.progress = [
            [], # Time of update (in hours)
            []  # % complete
        ]
        
        #self.has_indiv = dict.fromkeys(self.pix_name, False)
        #self.has_cloud = dict.fromkeys(self.pix_name, False)
        #self.has_los = dict.fromkeys(self.pix_name, False)
        
        self.rasterizer = None
        
        self.pickle(self.pickle_fname)
    
    def pickle(self, fname):
        data = {}
        data['indir'] = self.indir
        data['outdir'] = self.outdir
        data['infiles'] = self.infiles
        data['outfiles'] = self.outfiles
        data['modtime'] = self.modtime
        data['nside'] = self.nside
        data['pix_idx'] = self.pix_idx
        data['n_stars'] = self.n_stars
        data['n_stars_tot'] = self.n_stars_tot
        data['infile_idx'] = self.infile_idx
        data['pix_name'] = self.pix_name
        data['idx_in_map'] = self.idx_in_map
        data['has_indiv'] = self.has_indiv
        data['has_cloud'] = self.has_cloud
        data['has_los'] = self.has_los
        data['EBV'] = self.EBV
        data['progress'] = self.progress
        
        f = gzip.open(fname, 'wb')
        cPickle.dump(data, f)
        f.close()
    
    def unpickle(self, fname):
        f = None
        try:
            f = gzip.open(fname, 'rb')
        except:
            return False
        
        data = cPickle.load(f)
        f.close()
        
        self.indir = data['indir']
        self.outdir = data['outdir']
        self.infiles = data['infiles']
        self.outfiles = data['outfiles']
        self.modtime = data['modtime']
        self.nside = data['nside']
        self.pix_idx = data['pix_idx']
        self.n_stars = data['n_stars']
        self.n_stars_tot = data['n_stars_tot']
        self.infile_idx = data['infile_idx']
        self.pix_name = data['pix_name']
        self.idx_in_map = data['idx_in_map']
        self.has_indiv = data['has_indiv']
        self.has_cloud = data['has_cloud']
        self.has_los = data['has_los']
        self.EBV = data['EBV']
        
        #self.progress = [[],[]]
        self.progress = data['progress']
        
        return True
    
    def get_rate(self, t_smooth=1.):
        if len(self.progress[0]) < 2:
            return None, None
        
        t_status = np.array(self.progress[0])
        pct_complete = np.array(self.progress[1])
        rate = 24. * np.diff(pct_complete) / np.diff(t_status)
        w = np.exp(-0.5 * (t_status[-1] - t_status[:-1])**2. / t_smooth**2.)
        
        rate = np.sum(rate * w) / np.sum(w)
        t_remaining = (100. - pct_complete[-1]) / rate
        
        if not np.isfinite(t_remaining):
            t_remaining = -1.
        
        return rate, t_remaining
    
    def gen_status_json(self, fname):
        n_pix = self.nside.size
        pix_status = np.zeros(n_pix, dtype='f8')
        pix_status[self.has_indiv] = 1.
        pix_status[self.has_los] = 2.
        pix_status[self.get_defunct_pix()] = 3.
        
        nside_hires = np.max(self.nside)
        
        # Pixel status
        pix_status_hires = multires2nside(
            self.nside,
            self.pix_idx,
            pix_status,
            nside_hires
        )
        
        pix_status_lowres = downsample_by_2(downsample_by_2(pix_status_hires))
        nside_lowres = nside_hires / 4
        
        pix_status_lowres[~np.isfinite(pix_status_lowres)] = -1.
        
        # E(B-V)
        EBV_hires = multires2nside(
            self.nside,
            self.pix_idx,
            self.EBV[:,-1],
            nside_hires
        )
        
        EBV_lowres = downsample_by_2(downsample_by_2(EBV_hires))
        EBV_lowres[~np.isfinite(EBV_lowres)] = -1.
        
        s = json.dumps({
            'nside': nside_lowres,
            'order': 'nest',
            'vmin': 0.,
            'vmax': 3.,
            'EBV': np.round(EBV_lowres.astype('f8'), decimals=2).tolist(),
            'pixval': np.round(pix_status_lowres, decimals=1).tolist()
        })
        
        f = open(fname, 'w')
        f.write(s)
        f.close()
        
    
    def get_mtime_safe(self, fname):
        try:
            return os.path.getmtime(fname)
        except:
            return -1.
    
    def update(self):
        # Determine which output files have been updated
        mtime = np.array([self.get_mtime_safe(fname) for fname in self.outfiles])
        file_idx = np.nonzero((mtime > self.modtime) & (mtime > 0.))[0]
        self.modtime = mtime
        
        # Read information from updated files
        for i in file_idx:
            print 'Processing output file %d (%s) ...' % (i, self.outfiles[i])
            
            try:
                f = h5py.File(self.outfiles[i], 'r')
            except:
                print 'Failed to open %s. Skipping.' % (self.outfiles[i])
                continue
            
            for name, pixel in f.iteritems():
                _, nside, hp_idx = name.replace('-', ' ').split()
                name_tmp = '%d-%d' % (int(nside), int(hp_idx))
                # name = '%d-%d' % (pixel.attrs['nside'], pixel.attrs['healpix_index'])
                idx = self.idx_in_map[name_tmp]
                keys = pixel.keys()
                
                self.has_indiv[idx] = ('stellar chains' in keys)
                self.has_cloud[idx] = ('clouds' in keys)
                self.has_los[idx] = ('los' in keys)
                
                if 'los' in keys:
                    try:
                        self.EBV[idx,:] = np.cumsum(np.exp(pixel['los'][0, 1, :]))[[10, 15, -1]]
                    except Exception as error:
                        print str(error)
                        print 'Failed to load E(B-V) from pixel.'
            
            f.close()
        
        self.progress[0].append(time.time()/3600.)
        self.progress[1].append(self.get_pct_complete())
        
        print 'Done processing output files.'
        print 'Writing updates to disk ...'
        
        self.pickle(self.pickle_fname)
    
    def init_rasterizer(self, img_shape,
                              proj=hputils.Cartesian_projection(),
                              l_cent=0., b_cent=0):
        self.rasterizer = hputils.MapRasterizer(self.nside, self.pix_idx, img_shape,
                                                proj=proj, l_cent=l_cent)
    
    def get_pix_status(self, method='piecewise'):
        pix_val = self.has_indiv.astype('i4')
        
        if method == 'piecewise':
            pix_val += self.has_los.astype('i4')
        elif method == 'cloud':
            pix_val += self.has_cloud.astype('i4')
        else:
            raise ValueError("Unrecognized method: '%s'" % method)
        
        return pix_val
    
    def rasterize(self, dt=4., method='piecewise', t=None):
        if self.rasterizer == None:
            return None
        
        pix_val = None
        
        if t != None:
            pix_val = self._rasterize_historical(t)
        else:
            pix_val = self.get_pix_status(method=method)
            idx = self.get_defunct_pix(dt=dt, method=method)
            pix_val[idx] = -1.
        
        fg_img = self.rasterizer.rasterize(pix_val)
        bg_img = self.rasterizer.rasterize(self.n_stars * self.nside**2)
        
        return pix_val, bg_img, fg_img
    
    def _rasterize_historical(self, t):
        pass
    
    def get_pct_complete(self, method='piecewise'):
        n_stars_complete = None
        
        if method == 'piecewise':
            n_stars_complete = self.has_los.astype('i4') * self.n_stars
        elif method == 'cloud':
            n_stars_complete = self.has_los.astype('i4') * self.n_stars
        else:
            raise ValueError("Method '%s' not understood." % method)
        
        return 100. * float(np.sum(n_stars_complete)) / float(self.n_stars_tot)
    
    def get_incomplete_idx(self, method='piecewise'):
        #pix_val = self.get_pix_status(method=method)
        #idx = (pix_val < 2)
        
        idx = self.get_incomplete_pix(method=method)
        f_idx = self.infile_idx[idx]
        f_idx = np.unique(f_idx)
        
        return f_idx
    
    def get_incomplete_pix(self, method='piecewise'):
        pix_val = self.get_pix_status(method=method)
        idx = (pix_val < 2)
        
        return idx
    
    def get_defunct_idx(self, dt=4., method='piecewise'):
        '''
        Return indices of defunct output files (partially complete,
        but not modified in last dt hours).
        '''
        
        # Get list of output files last modified more than dt hours ago
        t = time.time()
        idx = (self.modtime > 0.) & (t - self.modtime > 3600.*dt)
        
        #print 'idx', np.sum(idx)
        
        # Get list of output files that are incomplete
        f_idx = self.get_incomplete_idx(method=method)
        
        #print 'f_idx', np.sum(f_idx)
        
        # AND lists together (incomplete output files last modified more than dt hours ago)
        m_idx = np.zeros(idx.size, dtype=np.bool)
        m_idx[f_idx] = True
        idx &= m_idx
        
        #print 'm_idx', np.sum(m_idx)
        
        # Convert bool array to array of indices
        idx = np.nonzero(idx)[0]
        
        #print 'idx', np.sum(idx)
        
        return idx
    
    def get_defunct_pix(self, dt=4., method='piecewise'):
        '''
        Get indices of defunct pixels (partially complete, but
        not modified in last dt hours).
        '''
        
        #dfct_f_idx = self.get_defunct_idx(dt=dt, method=method)
        #print dfct_f_idx
        
        inc_pix_idx = self.get_incomplete_pix(method=method)
        #print 'inc_pix_idx: %d' % (inc_pix_idx.size)
        #print np.nonzero(inc_pix_idx)[0]
        
        inc_f_idx = self.infile_idx[inc_pix_idx]
        #print 'inc_f_idx: %d' % (inc_f_idx.size)
        #print inc_f_idx
        
        mtime = self.modtime[inc_f_idx]
        idx = (mtime > 0.) & (time.time() - mtime > 3600.*dt)
        #print 'Delta t: %d' % (np.sum(idx))
        #print time.time() - mtime[idx]
        
        #print inc_f_idx
        
        #idx, _ = match_indices(inc_f_idx, dfct_f_idx)
        
        dfct_pix_idx = np.nonzero(inc_pix_idx)[0][idx]
        #print 'dfct_pix_idx: %d' % (dfct_pix_idx.size)
        #print dfct_pix_idx
        
        print 'defunct pixels: %d' % np.sum(idx)
        
        return dfct_pix_idx
    
    def to_ax(self, ax, method='piecewise',
                        l_lines=None, b_lines=None,
                        l_spacing=1., b_spacing=1.,
                        t=None):
        if self.rasterizer == None:
            return
        
        # Plot map of processing status
        pix_status, bg_img, fg_img = self.rasterize(method=method, t=t)
        bounds = self.rasterizer.get_lb_bounds()
        
        n_active = 1.1 * np.sum(pix_status == 1)
        
        cmap = ListedColormap([(0.95, 0.40, 0.40), (0., 1., 0.), pallette['sky blue']])
        norm = BoundaryNorm([-1,1,2,3], cmap.N)
        
        ax.imshow(np.sqrt(bg_img.T), extent=bounds, origin='lower', aspect='auto',
                                     interpolation='nearest', cmap='binary')
        ax.imshow(fg_img.T, extent=bounds, origin='lower', aspect='auto',
                            interpolation='nearest', cmap=cmap, norm=norm, alpha=0.5)
        
        if (l_lines != None) and (b_lines != None):
            # Determine label positions
            l_labels, b_labels = self.rasterizer.label_locs(l_lines, b_lines, 
                                                            shift_frac=0.04)
            
            # Determine grid lines to plot
            l_lines = np.array(l_lines)
            b_lines = np.array(b_lines)
            
            idx = (np.abs(l_lines) < 1.e-5)
            l_lines_0 = l_lines[idx]
            l_lines = l_lines[~idx]
            
            idx = (np.abs(b_lines) < 1.e-5)
            b_lines_0 = b_lines[idx]
            b_lines = b_lines[~idx]
            
            x_guides, y_guides = self.rasterizer.latlon_lines(l_lines, b_lines,
                                                              l_spacing=l_spacing,
                                                              b_spacing=b_spacing)
            
            x_guides_l0, y_guides_l0, x_guides_b0, y_guides_b0 = None, None, None, None
            
            if l_lines_0.size != 0:
                x_guides_l0, y_guides_l0 = self.rasterizer.latlon_lines(l_lines_0, 0.,
                                                                   mode='meridians',
                                                                   b_spacing=0.5*b_spacing)
            
            if b_lines_0.size != 0:
                x_guides_b0, y_guides_b0 = self.rasterizer.latlon_lines(0., b_lines_0,
                                                                   mode='parallels',
                                                                   l_spacing=0.5*l_spacing)
            
            # Plot lines of constant l and b
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            if x_guides != None:
                ax.scatter(x_guides, y_guides, s=1., c='b', edgecolor='b', alpha=0.10)
            
            if x_guides_l0 != None:
                ax.scatter(x_guides_l0, y_guides_l0, s=3., c='g', edgecolor='g', alpha=0.25)
            
            if x_guides_b0 != None:
                ax.scatter(x_guides_b0, y_guides_b0, s=3., c='g', edgecolor='g', alpha=0.25)
            
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            
            # Label Galactic coordinates
            #if l_lines != None:
            #    if (bounds[2] > -80.) | (bounds[3] < 80.):
            #        for l, (x_0, y_0), (x_1, y_1) in l_labels:
            #            ax.text(x_0, y_0, r'$%d^{\circ}$' % l, fontsize=20,
            #                                                   ha='center',
            #                                                   va='center')
            #            ax.text(x_1, y_1, r'$%d^{\circ}$' % l, fontsize=20,
            #                                                   ha='center',
            #                                                   va='center')
            
            #if b_lines != None:
            #    for b, (x_0, y_0), (x_1, y_1) in b_labels:
            #        ax.text(x_0, y_0, r'$%d^{\circ}$' % b, fontsize=20,
            #                                               ha='center',
            #                                               va='center')
            #        ax.text(x_1, y_1, r'$%d^{\circ}$' % b, fontsize=20,
            #                                               ha='center',
            #                                               va='center')
            #    
            #    # Expand axes limits to fit labels
            #    expand = 0.075
            #    xlim = ax.get_xlim()
            #    w = xlim[1] - xlim[0]
            #    xlim = [xlim[0] - expand * w, xlim[1] + expand * w]
            #    
            #    ylim = ax.get_ylim()
            #    h = ylim[1] - ylim[0]
            #    ylim = [ylim[0] - expand * h, ylim[1] + expand * h]
            #    
            #    ax.set_xlim(xlim)
            #    ax.set_ylim(ylim)
        
        return n_active
        
        


def main():
    parser = argparse.ArgumentParser(prog='plot_completion.py',
                                     description='Represent status of a Bayestar job as a rasterized map.',
                                     add_help=True)
    parser.add_argument('--indir', '-i', type=str, required=True,
                                           help='Directory with Bayestar input files.')
    parser.add_argument('--outdir', '-o', type=str, required=True,
                                           help='Directory with Bayestar output files.')
    parser.add_argument('--plot-fname', '-plt', type=str, required=False, default=None,
                                           help='Output filename for plot.')
    parser.add_argument('--json-dir', '-jsdir', type=str, required=False, default=None,
                                           help='Directory to put JSON status files.')
    parser.add_argument('--figsize', '-fs', type=int, nargs=2, default=(8, 4),
                                           help='Figure size (in inches).')
    parser.add_argument('--dpi', '-dpi', type=float, default=200,
                                           help='Dots per inch for figure.')
    parser.add_argument('--projection', '-proj', type=str, default='Cartesian',
                                           choices=('Cartesian', 'Mollweide', 'Hammer', 'Eckert IV'),
                                           help='Map projection to use.')
    parser.add_argument('--center-lb', '-cent', type=float, nargs=2, default=(0., 0.),
                                           help='Center map on (l, b).')
    parser.add_argument('--method', '-mtd', type=str, default='piecewise',
                                           choices=('cloud', 'piecewise'),
                                           help='Measure of line-of-sight completion to show.')
    parser.add_argument('--interval', '-int', type=float, default=1.,
                                           help='Generate a picture every X hours.')
    parser.add_argument('--maxtime', '-max', type=float, default=24.,
                                           help='Number of hours to continue monitoring job completion.')
    parser.add_argument('--defunct-time', '-dt', type=float, default=4.,
                                           help='Inactive time (in hours) after which to consider job defunct.')
    if 'python' in sys.argv[0]:
        offset = 2
    else:
        offset = 1
    args = parser.parse_args(sys.argv[offset:])
    
    
    # Parse arguments
    proj = None
    if args.projection == 'Cartesian':
        proj = hputils.Cartesian_projection()
    elif args.projection == 'Mollweide':
        proj = hputils.Mollweide_projection()
    elif args.projection == 'Hammer':
        proj = hputils.Hammer_projection()
    elif args.projection == 'Eckert IV':
        proj = hputils.EckertIV_projection()
    else:
        raise ValueError("Unrecognized projection: '%s'" % args.proj)
    
    # Initialize completion log
    print 'Loading completion counter...'
    completion = TCompletion(args.indir, args.outdir)
    
    if args.plot_fname:
        print 'Initializing rasterizer...'
        completion.init_rasterizer(img_shape, proj=proj, l_cent=l_cent, b_cent=b_cent)
    
    l_cent, b_cent = args.center_lb
    
    img_shape = (int(args.figsize[0] * 0.8 * args.dpi),
                 int(args.figsize[1] * 0.8 * args.dpi))
    
    # Generate grid lines
    ls = np.linspace(-180., 180., 19)
    bs = np.linspace(-90., 90., 19)[1:-1]
    
    # Matplotlib settings
    #mplib.rc('text', usetex=True)
    mplib.rc('xtick.major', size=6)
    mplib.rc('xtick.minor', size=2)
    mplib.rc('ytick.major', size=6)
    mplib.rc('ytick.minor', size=2)
    mplib.rc('xtick', direction='out')
    mplib.rc('ytick', direction='out')
    mplib.rc('axes', grid=False)
    
    t_start = time.time()
    
    #pct_complete_hist = []
    
    while time.time() - t_start < 3600. * args.maxtime:
        t_next = time.time() + 3600.*args.interval
        
        print 'Updating processing status...'
        completion.update()
        
        timestr = time.strftime('%m.%d-%H:%M:%S')
        
        # Percentage complete, estimated time remaining
        pct_complete = completion.get_pct_complete()
        rate, time_remaining = completion.get_rate()
        
        print 'Percent complete: {:.2f}%'.format(pct_complete)
        print 'Rate: {:.2f}%/day'.format(rate)
        print 'Time remaining: {:.2f} days'.format(time_remaining)
        
        #pct_complete_hist.append(pct_complete)
        
        #rate = None
        #time_remaining = None
        
        #if len(pct_complete_hist) > 1:
        #    pct_comp_tmp = np.array(pct_complete_hist)
        #    rate = 24. * np.diff(pct_comp_tmp) / args.interval
        #    dist = np.arange(rate.size)[::-1]
        #    w = np.exp(-dist.astype('f8'))
        #    rate = np.sum(rate * w) / np.sum(w)
        #    
        #    time_remaining = (100. - pct_complete) / rate
        #    
        #    if not np.isfinite(time_remaining):
        #        time_remaining = -1.
        
        # Dump JSON for completion webpage
        if args.json_dir:
            print 'Writing status to JSON...'
            completion.gen_status_json(os.path.join(args.json_dir, 'status_map.json'))
            
            n_jobs_running = subprocess.check_output(r'sacct --format="State"', shell=True).count('RUNNING')
            n_defunct_files = len(completion.get_defunct_idx())
            
            fairshare = None
            try:
                fairshare = float(subprocess.check_output(r'sshare --users="ggreen" --Users --format="FairShare" -nP', shell=True))
            except:
                print 'Failed to query fairshare.'
            
            f = open(os.path.join(args.json_dir, 'status_metadata.json'), 'w')
            f.write(json.dumps({
                'n_jobs_running': n_jobs_running,
                'n_defunct_files': n_defunct_files,
                'timestamp': timestr,
                'time_remaining': time_remaining,
                'rate': rate,
                'pct_complete': pct_complete,
                'fairshare': fairshare
            }))
            f.close()
        
        # Write incomplete and defunct files to an ASCII file
        print 'Outputting status to ASCII files...'
        f_idx = completion.get_incomplete_idx(method=args.method)
        f = open(os.path.normpath(args.outdir + '/incomplete.log'), 'w')
        f.write('\n'.join([str(i) for i in  f_idx]))
        f.close()
        
        f_idx = completion.get_defunct_idx(dt=args.defunct_time, method=args.method)
        f = open(os.path.normpath(args.outdir + '/defunct.log'), 'w')
        f.write('\n'.join([str(i) for i in f_idx]))
        f.close()
        
        # Plot completion map
        if args.plot_fname:
            print 'Plotting status map...'
            
            fig = plt.figure(figsize=args.figsize, dpi=args.dpi)
            ax = fig.add_subplot(1,1,1)
            
            n_active = completion.to_ax(ax, method=args.method, l_lines=ls, b_lines=bs,
                                                                l_spacing=0.5, b_spacing=0.5)
            
            # Labels, ticks, etc.
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            
            # Title
            rate_str = ''
            
            if time_remaining:
                rate_str = ', \ %.2f \%% / day \ (%.1f \ days \ remaining)' % (rate, time_remaining)
            
            title = r'$\mathrm{Status \ as \ of \ %s \ (%.2f \ \%%)%s}$' % (timestr, pct_complete, rate_str)
            ax.set_title(title, fontsize=32)
            
            fig.tight_layout()
            #fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.90)
            # Allow user to determine healpix index
            #pix_identifiers = []
            #nside_max = np.max(completion.nside)
            #pix_identifiers.append(PixelIdentifier(ax, nside_max, nest=True, proj=proj))
            
            # Save figure
            print 'Saving plot ...'
            plt_fname = args.plot_fname
            if plt_fname.endswith('.png'):
                plt_fname = plt_fname[:-4]
            
            fig.savefig(plt_fname + '.png', dpi=args.dpi)
            
            small_dpi = min(args.dpi, 1200./args.figsize[0])
            fig.savefig(plt_fname + '_thumb.png', dpi=small_dpi)
            
            plt.close(fig)
            del fig
        
        t_sleep = t_next - time.time()
        t_sleep = max([60., t_sleep])
        
        print 'Time: %s' % timestr
        print 'Sleeping for %d s...' % (t_sleep)
        print ''
        
        time.sleep(t_sleep)
    
    
    return 0

if __name__ == '__main__':
    main()

