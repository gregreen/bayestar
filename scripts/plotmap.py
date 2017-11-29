#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  plotmap.py
#  
#  Copyright 2013 Greg Green <greg@greg-UX31A>
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
#mplib.use('TkAgg')
#mplib.use('GTKAgg')
mplib.use('Agg')
mplib.rc('text', usetex=True)
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator, MultipleLocator, FuncFormatter, FixedLocator
from matplotlib.colors import LogNorm
import matplotlib.patheffects as patheffects
from mpl_toolkits.axes_grid1 import ImageGrid

import argparse, os, sys, time

import healpy as hp
import h5py

import multiprocessing
import Queue

import hputils, maptools


def plot_EBV(ax, img, bounds, **kwargs):
    # Configure plotting options
    if 'norm' not in kwargs:
        if 'vmin' not in kwargs:
            kwargs['vmin'] = np.min(img[np.isfinite(img)])
        if 'vmax' not in kwargs:
            kwargs['vmax'] = np.max(img[np.isfinite(img)])
    if 'aspect' not in kwargs:
        kwargs['aspect'] = 'auto'
    if 'interpolation' not in kwargs:
        kwargs['interpolation'] = 'nearest'
    if 'origin' in kwargs:
        print "Ignoring option 'origin'."
    if 'extent' in kwargs:
        print "Ignoring option 'extent'."
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'binary'
    kwargs['origin'] = 'lower'
    kwargs['extent'] = bounds
    
    # Plot image in B&W
    img_res = ax.imshow(img.T, **kwargs)
    
    return img_res


def plotter_worker(img_q, lock,
                   n_rasterizers,
                   figsize, dpi,
                   EBV_max, outfname):
    
    n_finished = 0
    
    # Plot images
    while True:
        n, mu, img = img_q.get()
        
        # Count number of rasterizer workers that have finished
        # processing their queue
        if n == 'FINISHED':
            n_finished += 1
            
            if n_finished >= n_rasterizers:
                return
            else:
                continue
        
        # Plot this image
        print 'Plotting mu = %.2f (image %d) ...' % (mu, n+1)
        
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(1,1,1)
        
        img = plot_EBV(ax, img, bounds, vmin=0., vmax=EBV_max)
        
        # Colorbar
        fig.subplots_adjust(bottom=0.12, left=0.12, right=0.89, top=0.88)
        cax = fig.add_axes([0.9, 0.12, 0.03, 0.76])
        cb = fig.colorbar(img, cax=cax)
        
        # Labels, ticks, etc.
        ax.set_xlabel(r'$\ell$', fontsize=16)
        ax.set_ylabel(r'$b$', fontsize=16)
        
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        # Title
        d = 10.**(mu/5. - 2.)
        ax.set_title(r'$\mu = %.2f \ \ \ d = %.2f \, \mathrm{kpc}$' % (mu, d), fontsize=16)
        
        # Save figure
        full_fname = '%s.%s.%s.%.5d.png' % (outfname, model, method, n)
        
        lock.acquire()
        fig.savefig(full_fname, dpi=dpi)
        lock.release()
        
        plt.close(fig)
        del img


def rasterizer_worker(dist_q, img_q,
                      los_coll,
                      figsize, dpi, size,
                      model, method, mask,
                      proj, l_cent, b_cent, bounds,
                      delta_mu):
    # Reseed random number generator
    t = time.time()
    t_after_dec = int(1.e9*(t - np.floor(t)))
    seed = np.bitwise_xor([t_after_dec], [os.getpid()])
    
    np.random.seed(seed=seed)
    
    # Generate images
    while True:
        try:
            n, mu = dist_q.get_nowait()
            
            # Rasterize E(B-V)
            img, bounds, xy_bounds = los_coll.rasterize(mu, size,
                                                            fit=model,
                                                            method=method,
                                                            mask_sigma=mask,
                                                            delta_mu=delta_mu,
                                                            proj=proj,
                                                            l_cent=l_cent,
                                                            b_cent=b_cent)
            
            # Put image on queue
            img_q.put((n, mu, img, bounds, xy_bounds))
            
        except Queue.Empty:
            img_q.put('FINISHED')
            
            print 'Rasterizer finished.'
            
            return


class PixelPlotter:
    def __init__(self, data, model='piecewise'):
        self.data = data
        self.model = model
        
        self.fig = None
        self.idx_current = None
        self.shown = False
    
    def __call__(self, map_idx):
        self.plot_pixel(map_idx)
    
    def setup_figure(self):
        if self.fig != None:
            return
        
        plt.ion()
        
        # Set up figure
        self.fig = plt.figure(figsize=(8,5), dpi=150)
        self.ax = self.fig.add_subplot(1,1,1)
        self.fig.subplots_adjust(left=0.12, bottom=0.12, top=0.85)
        
        # Plot dummy stacked stellar pdfs
        shape = self.data.star_stack[0].shape[1:]
        img = np.zeros(shape, dtype='f8')
        bounds = self.data.DM_EBV_lim
        self.im = self.ax.imshow(np.sqrt(img), extent=bounds, origin='lower',
                            aspect='auto', cmap='Blues', interpolation='nearest')
        
        # Plot dummy samples
        shape = self.data.los_EBV[0].shape[1:]
        EBV_all = np.zeros(shape, dtype='f8')
        alpha = 1. / np.power(shape[0], 0.55)
        
        self.sample_plt = []
        mu = self.data.get_los_DM_range()
        for i,EBV in enumerate(EBV_all[1:]):
            self.sample_plt.append(self.ax.plot(mu, EBV, alpha=alpha)[0])
        
        # Plot dummy best fit
        self.best_plt = self.ax.plot(mu, EBV_all[0, :], 'g', lw=2, alpha=0.5)[0]
        
        self.ax.set_xlim(mu[0], mu[-1])
        self.ax.set_ylim(0., 1.)
        
        # Add labels
        self.ax.set_xlabel(r'$\mu$', fontsize=16)
        self.ax.set_ylabel(r'$\mathrm{E} \left( B - V \right)$', fontsize=16)
        
        #title_txt = '$\mathrm{nside} = %d, \ \mathrm{i} = %d$\n' % (nside, pix_idx)
        #title_txt += '$\ell = %.2f, \ b = %.2f$' % (l, b)
        
        self.ax.set_title(r'$$', fontsize=16, multialignment='center')
        
        ylim = self.ax.get_ylim()
        y_txt = ylim[0] + 0.95 * (ylim[1] - ylim[0])
        x_txt = mu[0] + 0.05 * (mu[-1] - mu[0])
        self.txt = self.ax.text(x_txt, y_txt, r'$$', fontsize=16,
                                                     multialignment='left',
                                                     va='top')
    
    def plot_pixel(self, map_idx):
        if map_idx == -1:
            return
        if map_idx == self.idx_current:
            return
        self.idx_current = map_idx
        
        self.setup_figure()
        
        # Load and stretch stacked stellar pdfs
        star_stack = self.data.star_stack[0][map_idx].astype('f8')
        
        star_stack /= np.max(star_stack)
        norm1 = 1. / np.power(np.max(star_stack, axis=0), 0.8)
        norm2 = 1. / np.power(np.sum(star_stack, axis=0), 0.8)
        norm = np.sqrt(norm1 * norm2)
        norm[np.isinf(norm)] = 0.
        star_stack = np.einsum('ij,j->ij', star_stack, norm)
        
        # Determine maximum E(B-V)
        w_y = np.mean(star_stack, axis=1)
        y_max = np.max(np.where(w_y > 1.e-2)[0])
        EBV_stack_max = y_max * (5. / star_stack.shape[0])
        
        # Load piecewise-linear profiles
        EBV_all = self.data.los_EBV[0][map_idx, :, :]
        
        nside = self.data.nside[0][map_idx]
        pix_idx = self.data.pix_idx[0][map_idx]
        l, b = hputils.pix2lb_scalar(nside, pix_idx, nest=True, use_negative_l=True)
        
        EBV_los_max = 1.5 * np.percentile(EBV_all[:, -1], 95.)
        EBV_max = min([EBV_los_max, EBV_stack_max])
        
        # Load ln(p), if available
        lnp = None
        lnp_txt = None
        
        if self.data.los_lnp != []:
            lnp = self.data.los_lnp[0][map_idx, :] / self.data.n_stars[0][map_idx]
            GR = self.data.los_GR[0][map_idx, :]
            
            lnp_min, lnp_max = np.percentile(lnp[1:], [10., 90.])
            GR_max = np.max(GR)
            
            lnp_txt =  '$\ln \, p_{\mathrm{best}} = %.2f$\n' % lnp[0]
            lnp_txt += '$\ln \, p_{90\%%} = %.2f$\n' % lnp_max
            lnp_txt += '$\ln \, p_{10\%%} = %.2f$\n' % lnp_min
            lnp_txt += '$\mathrm{GR}_{\mathrm{max}} = %.3f$' % GR_max
            
            lnp = (lnp - lnp_min) / (lnp_max - lnp_min)
            lnp[lnp > 1.] = 1.
            lnp[lnp < 0.] = 0.
        else:
            lnp = [0. for EBV in EBV_all]
        
        # Plot stacked stellar pdfs
        self.im.set_data(np.sqrt(star_stack))
        self.im.autoscale()
        
        # Plot samples
        for i,EBV in enumerate(EBV_all[1:]):
            c = (1.-lnp[i+1], 0., lnp[i+1])
            self.sample_plt[i].set_ydata(EBV)
            self.sample_plt[i].set_color(c)
            #ax.plot(mu, EBV, c=c, alpha=alpha)
        
        # Plot best fit
        self.best_plt.set_ydata(EBV_all[0, :])
        
        self.ax.set_ylim(0., EBV_max)
        
        # Add labels
        self.ax.set_xlabel(r'$\mu$', fontsize=16)
        self.ax.set_ylabel(r'$\mathrm{E} \left( B - V \right)$', fontsize=16)
        
        title_txt = '$\mathrm{nside} = %d, \ \mathrm{i} = %d$\n' % (nside, pix_idx)
        title_txt += '$\ell = %.2f, \ b = %.2f$' % (l, b)
        
        self.ax.set_title(title_txt, fontsize=16, multialignment='center')
        
        if lnp_txt != None:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            y_txt = ylim[0] + 0.95 * (ylim[1] - ylim[0])
            x_txt = xlim[0] + 0.05 * (xlim[-1] - xlim[0])
            self.txt.set_text(lnp_txt)
            self.txt.set_x(x_txt)
            self.txt.set_y(y_txt)
            #ax.text(x_txt, y_txt, lnp_txt, fontsize=16,
            #                               multialignment='left',
            #                               va='top')
        
        print 'drawing'
        self.ax.figure.canvas.draw()
        #self.fig.show()
        #plt.draw()
        #plt.show()
        
        if not self.shown:
            plt.show()
            self.shown = True


def rasterizer_plotter_worker(dist_q, lock,
                              mapper,
                              figsize, dpi, size,
                              model, method, mask,
                              proj, l_cent, b_cent, bounds,
                              l_lines, b_lines,
                              delta_mu, EBV_max,
                              DM_min, DM_max,
                              outfname,
                              **kwargs):
    # Reseed random number generator
    t = time.time()
    t_after_dec = int(1.e9*(t - np.floor(t)))
    seed = np.bitwise_xor([t_after_dec], [os.getpid()])
    
    np.random.seed(seed=seed)
    
    # Read misc. options
    font_scaling = np.sqrt((figsize[0]/12.) * (figsize[1]/10.))
    
    show = kwargs.get('show', False)
    meridian_style = kwargs.get('meridian_style', 70.)
    parallel_style = kwargs.get('parallel_style', 'lh')
    grat_fontsize = kwargs.get('grat_fontsize', 14.*font_scaling)
    
    # Set up rasterizer
    rasterizer = mapper.gen_rasterizer(size, proj=proj,
                                             l_cent=l_cent,
                                             b_cent=b_cent)
    bounds = rasterizer.get_lb_bounds()
    print 'bounds:', bounds
    
    stroke = [patheffects.withStroke(linewidth=0.5, foreground='w', alpha=0.75)]
    
    first_img = True
    pix_identifier = []
    
    if show:
        pix_plotter = PixelPlotter(mapper.data)
    
    # Generate images
    while True:
        try:
            n, mu = dist_q.get_nowait()
            
            # Rasterize E(B-V)
            tmp, tmp, pix_val = mapper.gen_EBV_map(mu, fit=model,
                                                       method=method,
                                                       mask_sigma=mask,
                                                       delta_mu=delta_mu,
                                                       reduce_nside=False)
            
            img = rasterizer(pix_val)
            
            # Plot this image
            print 'Plotting mu = %.2f (image %d) ...' % (mu, n+1)
            
            fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='#D1E1ED')
            ax = fig.add_subplot(1,1,1)
            
            if method == 'sigma':
                img = plot_EBV(ax, img, bounds,
                               norm=LogNorm(vmin=0.001, vmax=EBV_max),
                               cmap='hsv')
            else:
                img = plot_EBV(ax, img, bounds, vmin=0., vmax=EBV_max)
            
            fig.subplots_adjust(bottom=0.15, left=0.05,
                                right=0.84, top=0.95)
            cax = fig.add_axes([0.85, 0.25, 0.025, 0.60])
            dax = fig.add_axes([0.15, 0.04, 0.59,  0.04])
            
            # Colorbar
            cb = fig.colorbar(img, cax=cax)
            
            clabel = r'$\mathrm{E} \left( B - V \right)$'
            if delta_mu != None:
                if delta_mu > 0.:
                    clabel = r'$\mathrm{d} \mathrm{E} \left( B - V \right) / \mathrm{d} \mu$'
                else:
                    clabel = r'$\mathrm{d} \mathrm{E} \left( B - V \right) / \mathrm{d} s \ \left( \mathrm{mags} / \mathrm{kpc} \right)$'
            
            fontsize = 20. * font_scaling
            cb.set_label(clabel, fontsize=fontsize, rotation=270)
            cb.ax.get_yaxis().labelpad = 30
            cb.ax.tick_params(labelsize=0.75*fontsize)
            
            cb.set_ticks(MaxNLocator(nbins=5))
            #cb.update_ticks()
            
            # Labels, ticks, etc.
            ax.set_xticks([])
            ax.set_yticks([])
            
            #ax.set_xlabel(r'$\ell$', fontsize=16)
            #ax.set_ylabel(r'$b$', fontsize=16)
            
            #ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            #ax.xaxis.set_minor_locator(AutoMinorLocator())
            #ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            #ax.yaxis.set_minor_locator(AutoMinorLocator())
            
            # Distance Label
            d = 10.**(mu/5. - 2.)
            fontsize = 19 * font_scaling
            #ax.set_title(r'$\mu = %.2f \ \ \ d = %.2f \, \mathrm{kpc}$' % (mu, d),
            #             fontsize=fontsize)
            dimg = np.zeros(1000, dtype='f8')
            dimg[:] = -1000.
            didx = int(np.round((1000.)*((mu-DM_min) / (DM_max-DM_min))))
            didx = np.arange(didx)
            dimg[didx] = didx
            #dimg[:didx] = 1.
            dimg.shape = (dimg.size, 1)
            
            dax.imshow(dimg.T, origin='lower', aspect='auto', interpolation='none',
                       cmap='Blues', extent=(DM_min, DM_max, 0, 1), vmin=-200., vmax=1000.)
            
            dax.set_yticks([])
            dax.set_yticklabels([])
            dax.xaxis.set_major_locator(MaxNLocator(nbins=6))
            dax.xaxis.set_minor_locator(AutoMinorLocator())
            dax.tick_params(axis='x', labelsize=0.75*fontsize)
            dax.tick_params(axis='x', which='minor', length=4., width=1.5)
            dax.tick_params(axis='x', which='major', length=8., width=2.)
            
            dax.set_xlabel(r'$\mathrm{Distance \ Modulus} \ \left( \mathrm{mags} \right)$',
                           fontsize=fontsize)
            
            def dist_formatter(dm, pos):
                d_kpc = 10.**(dm/5.-2.)
                return r'$%g$' % d_kpc
            
            d_minor = np.hstack([np.arange(0., 1., 0.1), + np.arange(1., 20., 1.)])
            d_minor = 5. * (np.log10(d_minor) + 2.)
            idx = (d_minor < DM_min) | (d_minor > DM_max)
            d_minor = d_minor[~idx]
            
            dax2 = dax.twiny()
            dax2.set_xlim(dax.get_xlim())
            dax2.xaxis.set_major_locator(MultipleLocator(base=5.))
            dax2.xaxis.set_minor_locator(FixedLocator(d_minor))
            dax2.xaxis.set_major_formatter(FuncFormatter(dist_formatter))
            dax2.tick_params(axis='x', labelsize=0.75*fontsize)
            dax2.tick_params(axis='x', which='minor', length=4., width=1.5)
            dax2.tick_params(axis='x', which='major', length=8., width=2.)
            
            dax2.set_xlabel(r'$\mathrm{Distance} \ \left( \mathrm{kpc} \right)$',
                           fontsize=fontsize)
            
            ax.axis('off')
            
            # Plot graticules
            if (l_lines != None) and (b_lines != None):
                #xlim = ax.get_xlim()
                #ylim = ax.get_ylim()
                
                thick_lw = 2.5 * (figsize[0]*dpi/1920.)
                label_pad = 15. * (figsize[0]*dpi/1920.)
                dtheta = np.sqrt(np.abs((bounds[1]-bounds[0])*(bounds[3]-bounds[2])))
                dtheta0 = np.sqrt(360.*180.)
                label_dist = 3.5 * dtheta / dtheta0
                
                #print('label_dist = {0}'.format(label_dist))
                
                bb = hputils.plot_graticules(ax, rasterizer, l_lines, b_lines,
                                             parallel_style=parallel_style,
                                             meridian_style=meridian_style,
                                             fontsize=grat_fontsize,
                                             txt_path_effects=stroke,
                                             x_excise=10., y_excise=6.,
                                             thick_c='#0C518A',
                                             thin_c='#0C518A',
                                             thick_alpha=0.1,
                                             thin_alpha=0.15,
                                             thick_lw=thick_lw,
                                             thin_lw=0.3*thick_lw,
                                             label_dist=label_dist,
                                             label_pad=label_pad,
                                             label_ang_tol=20.,
                                             return_bbox=True)
                
                xlim = bb[0], bb[1]
                ylim = bb[2], bb[3]
                
                # Expand axes limits to fit labels
                expand = 0.025
                #xlim = ax.get_xlim()
                w = xlim[1] - xlim[0]
                xlim = [xlim[0] - expand * w, xlim[1] + expand * w]
                
                #ylim = ax.get_ylim()
                h = ylim[1] - ylim[0]
                ylim = [ylim[0] - expand * h, ylim[1] + expand * h]
                
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            
            # Save figure
            full_fname = '%s.%s.%s.%.5d.png' % (outfname, model, method, n)
            
            if first_img:
                lock.acquire()
                fig.savefig(full_fname, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
                lock.release()
                
                first_img = False
            else:
                fig.savefig(full_fname, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
            
            if show:
                # Add pixel identifier to allow user to find info on
                # individual HEALPix pixels
                pix_identifier.append(hputils.PixelIdentifier(ax, rasterizer,
                                                              lb_bounds=True,
                                                              event_type='motion_notify_event',
                                                              event_key='alt'))
                pix_identifier[-1].attach_obj(pix_plotter)
            else:
                plt.close(fig)
                del img
            
        except Queue.Empty:
            print 'Rasterizer finished.'
            
            if show:
                print 'Showing ...'
                plt.show()
            
            print 'Plots closed.'
            
            return

def parse_str_or_float(s):
    try:
        return float(s)
    except:
        return s

def remove_nones(kw):
    for key in kw.keys():
        if kw[key] is None:
            kw.pop(key)

def main():
    parser = argparse.ArgumentParser(prog='plotmap.py',
                                     description='Generate a map of E(B-V) from bayestar output.',
                                     add_help=True)
    parser.add_argument('input', type=str, nargs='+', help='Bayestar output files.')
    parser.add_argument('--output', '-o', type=str, help='Output filename for plot.')
    parser.add_argument('--show', '-sh', action='store_true', help='Show plot.')
    parser.add_argument('--dists', '-d', type=float, nargs=3,
                                         default=(4., 19., 21),
                                         help='DM min, DM max, # of distance slices.')
    parser.add_argument('--dist-step', '-ds', type=str, default='log',
                                         choices=('log', 'linear'),
                                         help='Step logarithmically in distance (linearly in\n'
                                              'distance modulus) or linearly in distance.')
    parser.add_argument('--delta-mu', '-dmu', type=float, default=None,
                                         help='Difference in DM used to estimate rate of\n'
                                              'reddening (default: None, i.e. calculate cumulative reddening).')
    parser.add_argument('--figsize', '-fs', type=float, nargs=2, default=(8, 4),
                                         help='Figure size (in inches).')
    parser.add_argument('--dpi', '-dpi', type=float, default=200,
                                         help='Dots per inch for figure.')
    parser.add_argument('--projection', '-proj', type=str, default='Cartesian',
                                         choices=('Cartesian', 'Mollweide', 'Hammer', 'Eckert IV', 'Gnomonic', 'Stereographic'),
                                         help='Map projection to use.')
    parser.add_argument('--center-lb', '-cent', type=float, nargs=2, default=(0., 0.),
                                         help='Center map on (l, b).')
    parser.add_argument('--bounds', '-b', type=float, nargs=4, default=None,
                                         help='Bounds of pixels to plot (l_min, l_max, b_min, b_max).')
    parser.add_argument('--model', '-m', type=str, default='piecewise',
                                         choices=('piecewise', 'cloud'),
                                         help='Line-of-sight extinction model to use.')
    parser.add_argument('--mask', '-msk', type=float, default=None,
                                         help=r'Hide parts of map where sigma_{E(B-V)} is greater than given value.')
    parser.add_argument('--method', '-mtd', type=str, default='median',
                                         choices=('median', 'mean', 'best', 'sample', 'sigma' , '5th', '95th'),
                                         help='Measure of E(B-V) to plot.')
    parser.add_argument('--processes', '-proc', type=int, default=1,
                                         help='# of processes to spawn.')
    parser.add_argument('--max-samples', '-samp', type=int, default=None,
                                         help='Maximum # of MCMC samples to load per pixel (to limit memory usage).')
    parser.add_argument('--l-lines', '-ls', type=float, nargs='+', default=None,
                                         help='Galactic longitudes at which to draw lines.')
    parser.add_argument('--b-lines', '-bs', type=float, nargs='+', default=None,
                                         help='Galactic latitudes at which to draw lines.')
    parser.add_argument('--meridian_style', type=parse_str_or_float, default=None,
                                         help='Meridian style.')
    parser.add_argument('--parallel_style', type=parse_str_or_float, default=None,
                                         help='Parallel style.')
    parser.add_argument('--grat_fontsize', '-gfs', type=float, default=None,
                                         help='Galactic latitudes at which to draw lines.')
    parser.add_argument('--EBV-max', '-Em', type=float, default=None,
                                         help='Saturation limit for E(B-V) color scale.')
    
    if 'python' in sys.argv[0]:
        offset = 2
    else:
        offset = 1
    args = parser.parse_args(sys.argv[offset:])
    
    
    # Parse arguments
    outfname = args.output
    if outfname != None:
        if outfname.endswith('.png'):
            outfname = outfname[:-4]
    
    method = args.method
    if method == '5th':
        method = 5.
    elif method == '95th':
        method = 95.
    
    proj = None
    if args.projection == 'Cartesian':
        proj = hputils.Cartesian_projection()
    elif args.projection == 'Mollweide':
        proj = hputils.Mollweide_projection()
    elif args.projection == 'Hammer':
        proj = hputils.Hammer_projection()
    elif args.projection == 'Eckert IV':
        proj = hputils.EckertIV_projection()
    elif args.projection == 'Gnomonic':
        proj = hputils.Gnomonic_projection()
    elif args.projection == 'Stereographic':
        proj = hputils.Stereographic_projection()
    else:
        raise ValueError("Unrecognized projection: '%s'" % args.proj)
    
    l_cent, b_cent = args.center_lb
    
    size = (int(args.figsize[0] * 0.8 * args.dpi),
            int(args.figsize[1] * 0.8 * args.dpi))
    
    mu_plot = None
    delta_mu = args.delta_mu
    
    if args.dist_step == 'log':
        mu_plot = np.linspace(args.dists[0], args.dists[1], args.dists[2])
    else:
        d_0 = 10.**(args.dists[0] / 5. + 1.)
        d_1 = 10.**(args.dists[1] / 5. + 1.)
        d_plot = np.linspace(d_0, d_1, args.dists[2])
        mu_plot = 5. * (np.log10(d_plot) - 1.)
        
        #if delta_mu != None:
        #    delta_mu = -delta_mu
    
    
    # Load in line-of-sight data
    fnames = args.input
    
    mapper = maptools.LOSMapper(fnames, bounds=args.bounds,
                                        max_samples=args.max_samples,
                                        processes=args.processes,
                                        load_stacked_pdfs=args.show)
    
    # Get upper limit on E(B-V)
    method_tmp = method
    
    if method == 'sample':
        method_tmp = 'median'
    
    EBV_max = None
    
    if args.EBV_max == None:
        EBV_max = -np.inf
        
        if args.delta_mu == None:
            mu_eval = None
            
            if method == 'sigma':
                mu_eval = np.array(mapper.data.get_los_DM_range())
                idx = (mu_eval >= args.dists[0]) & (mu_eval <= args.dists[1])
                mu_eval = mu_eval[idx]
                
            else:
                mu_eval = [mu_plot[-1]]
            
            for mu in mu_eval:
                print 'Determining max E(B-V) from mu = %.2f ...' % mu
                
                nside_tmp, pix_idx_tmp, EBV = mapper.gen_EBV_map(mu,
                                                                 fit=args.model,
                                                                 method=method_tmp,
                                                                 mask_sigma=args.mask,
                                                                 delta_mu=delta_mu)
                idx = np.isfinite(EBV)
                EBV_max_tmp = np.percentile(EBV[idx], 99.)
                
                if EBV_max_tmp > EBV_max:
                    EBV_max = EBV_max_tmp
            
        else:
            EBV_max = mapper.est_dEBV_pctile(99., delta_mu=delta_mu,
                                                  fit=args.model)
    else:
        EBV_max = args.EBV_max
    
    mask = args.mask
    
    if method == 'sample':
        mask = None
    
    print 'EBV_max = %.3f' % EBV_max
    
    
    # Matplotlib settings
    #mplib.rc('text', usetex=False) # TODO: Set to True once LaTeX is fixed on CentOS 6
    mplib.rc('xtick.major', size=6)
    mplib.rc('xtick.minor', size=2)
    mplib.rc('ytick.major', size=6)
    mplib.rc('ytick.minor', size=2)
    mplib.rc('xtick', direction='out')
    mplib.rc('ytick', direction='out')
    mplib.rc('axes', grid=False)
    
    
    # Plot at each distance
    pix_identifiers = []
    nside_max = mapper.get_nside_levels()[-1]
    
    # Set up queue for rasterizer workers to pull from
    dist_q = multiprocessing.Queue()
    
    for n,mu in enumerate(mu_plot):
        dist_q.put((n, mu))
    
    # Set up results queue for rasterizer workers
    img_q = multiprocessing.Queue()
    lock = multiprocessing.Lock()
    
    # Spawn worker processes to plot images
    n_rasterizers = args.processes
    procs = []
    
    kwargs = {
        'show': args.show,
        'meridian_style': args.meridian_style,
        'parallel_style': args.parallel_style,
        'grat_fontsize': args.grat_fontsize
    }
    remove_nones(kwargs)
    
    for i in xrange(n_rasterizers):
        p = multiprocessing.Process(target=rasterizer_plotter_worker,
                                    args=(dist_q, lock,
                                          mapper,
                                          args.figsize, args.dpi, size,
                                          args.model, method, mask,
                                          proj, l_cent, b_cent, args.bounds,
                                          args.l_lines, args.b_lines,
                                          delta_mu, EBV_max,
                                          mu_plot[0], mu_plot[-1],
                                          outfname),
                                    kwargs=kwargs
                                   )
        procs.append(p)
    
    for p in procs:
        p.start()
    
    for p in procs:
        p.join()
    
    print 'Done.'
    
    #if args.show:
    #   plt.show()
    
    
    return 0

if __name__ == '__main__':
    main()

