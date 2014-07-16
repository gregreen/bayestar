#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  proj3d.py
#  
#  Copyright 2014 Greg Green <greg@greg-UX31A>
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
import matplotlib.ticker as ticker
import matplotlib.patheffects as PathEffects

import os, sys, glob, time

import multiprocessing
import Queue

import maptools


def pm_ang_formatter(theta, pos):
    if np.abs(theta) < 1.e-5:
        return r'$+0^{\circ}$'
    elif theta > 0.:
        return r'$+%d^{\circ}$' % theta
    else:
        return r'$%d^{\circ}$' % theta


def camera_z_bobbing(dn=50,
                     z_0=100., x_0=0., y_0=0.,
                     dx_stare=1000., dz_stare=-100.,
                     beta_0=180.):
    '''
    Generate a set of camera positions/orientations that bob above and
    below the plane of the Galaxy.
    
    Inputs:
        dn         One fifth the total number of frames in the sequence
        
        z_0        Amplitude (in pc) of bobbing above plane of Galaxy
        x_0        Offset (in pc) from Sun (+ = towards GC)
        y_0        Offset (in pc) from Sun (tangential to Sun's orbit)
        
        dx_stare   Distance (in pc) in plane at which to point camera
        dz_stare   z-offset (in pc) above plane at which to point camera
        
        beta_0     Azimuthal orientation of camera (0 = towards GC)
    
    Outputs a dictionary with three entries:
        'r'        n x 3 array containing (x,y,z)-coordinates of camera in each frame
        'a'        Polar angle of camera in each frame
        'b'        Azimuthal angle of camera in each frame
    '''
    
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
    
    cam_pos = {'r':r_0, 'a':alpha, 'b':beta}
    
    return cam_pos


def gen_frame_worker(frame_q, lock,
                     map_fname, base_fname,
                     camera_pos, camera_props,
                     dpi, n_averaged):
    # Reseed random number generator
    t = time.time()
    t_after_dec = int(1.e9*(t - np.floor(t)))
    seed = np.bitwise_xor([t_after_dec], [os.getpid()])
    
    np.random.seed(seed=seed)
    
    # Load in 3D map
    if isinstance(map_fname, str):
        map_fname = [map_fname]
    
    mapper = maptools.LOSMapper(map_fname, max_samples=4)
    nside = mapper.data.nside[0]
    pix_idx = mapper.data.pix_idx[0]
    los_EBV = mapper.data.los_EBV[0]
    DM_min, DM_max = mapper.data.DM_EBV_lim[:2]
    
    mapper3d = maptools.Mapper3D(nside, pix_idx, los_EBV, DM_min, DM_max)
    
    # Read camera properties
    fov = camera_props['fov']
    n_x = camera_props['n_x']
    n_y = camera_props['n_y']
    n_z = camera_props['n_z']
    dr = camera_props['dr']
    z_0 = camera_props['z_0']
    
    extent = [fov/2., -fov/2., float(n_y)/float(n_x)*fov/2., -float(n_y)/float(n_x)*fov/2.]
    
    # Generate images
    first_img = True
    np.seterr(all='ignore')
    
    while True:
        try:
            k = frame_q.get_nowait()
            
            t_start = time.time()
            
            a = camera_pos['a'][k]
            b = camera_pos['b'][k]
            r = camera_pos['r'][k]
            
            print 'Projecting frame %d ...' % k
            img = np.empty((n_averaged, 2*n_y+1, 2*n_x+1), dtype='f8')
            
            for i in xrange(n_averaged):
                img[i] = mapper3d.proj_map_in_slices('stereo', n_z, 'sample',
                                                      a, b, n_x, n_y, fov, r, dr, z_0)
            
            img = np.median(img, axis=0)
            
            vmax = 1.2 * np.percentile(img, 99.5)
            
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
            
            plt_fname = '%s.%05d.png' % (base_fname, k)
            
            if first_img:
                lock.acquire()
                fig.savefig(plt_fname, dpi=dpi, bbox_inches='tight')
                lock.release()
                first_img = False
            else:
                fig.savefig(plt_fname, dpi=dpi, bbox_inches='tight')
            
            plt.close(fig)
            del img
            
            t_end = time.time()
            
            print 't = %.1f s' % (t_end - t_start)
        
        except Queue.Empty:
            print 'Worker finished.'
            return


def gen_movie_frames(camera_pos, camera_props, map_fname, frame_fname,
                     n_procs=1, dpi=100, n_averaged=1):
    # Set up queue for workers to pull frame numbers from
    frame_q = multiprocessing.Queue()
    
    n_frames = len(camera_pos['a'])
    
    for k in xrange(n_frames):
        frame_q.put(k)
    
    # Set up lock to allow first image to be written without interference btw/ processes
    lock = multiprocessing.Lock()
    
    # Spawn worker processes to plot images
    procs = []
    
    for i in xrange(n_procs):
        p = multiprocessing.Process(target=gen_frame_worker,
                                    args=(frame_q, lock,
                                          map_fname, frame_fname,
                                          camera_pos, camera_props,
                                          dpi, n_averaged)
                                   )
        procs.append(p)
    
    for p in procs:
        p.start()
    
    for p in procs:
        p.join()
    
    print 'Done.'



def bobbing_movie():
    camera_props = {'n_x': 600,
                    'n_y': 300,
                    'n_z': 2000,
                    'fov': 90.,
                    'dr': 3.,
                    'z_0': 15.}
    
    camera_pos = camera_z_bobbing(dn=40,
                                  z_0=2000., x_0=0., y_0=0.,
                                  dx_stare=1000., dz_stare=-100.,
                                  beta_0=180.)
    
    map_fname = '/n/fink1/ggreen/bayestar/output/allsky_2MASS/compact_10samp.h5'
    frame_fname = '/n/pan1/www/ggreen/3d/allsky_2MASS/z_bobbing/z2000'
    
    gen_movie_frames(camera_pos, camera_props, map_fname, frame_fname, n_procs=8, dpi=120)


def zoom_out_movie():
    camera_props = {'n_x': 1000,
                    'n_y': 750,
                    'n_z': 5000,
                    'fov': 90.,
                    'dr': 1.8,
                    'z_0': 20.}
    
    n_frames = 500
    
    d = 0.5 + 0.5 * np.tanh(np.linspace(-2.5, 2.5, n_frames))
    #d = np.linspace(0., 1., n_frames)**2.
    
    r_0 = np.zeros((n_frames, 3), dtype='f8')
    r_0[:,0] = 4000. * d
    r_0[:,2] = 1500. * d
    
    a = 90. + 180./np.pi * np.arctan(r_0[-1,2] / r_0[-1,0])
    b = 180.
    
    a = np.linspace(90., a, n_frames)
    b = b * np.ones(n_frames)
    
    camera_pos = {'r': r_0,
                  'a': a,
                  'b': b}
    
    map_fname = '/n/fink1/ggreen/bayestar/output/allsky_2MASS/compact_10samp.h5'
    frame_fname = '/n/pan1/www/ggreen/3d/allsky_2MASS/zoom_out/zoom_slow'
    
    gen_movie_frames(camera_pos, camera_props, map_fname, frame_fname, n_procs=8, dpi=120)


def GC_perspective_movie():
    camera_props = {'n_x': 800,
                    'n_y': 800,
                    'n_z': 5000,
                    'fov': 90.,
                    'dr': 2.,
                    'z_0': 1000.}
    
    n_frames = 501
    R = 5000.
    b = 130.
    
    r_0 = np.zeros((n_frames, 3), dtype='f8')
    
    t_0 = 35.
    theta_0 = t_0 * np.linspace(0., 1., n_frames/4)**2.
    
    theta = np.pi/180. * np.linspace(-45., 45., n_frames)
    phi = np.pi/180. * (b - 180.)
    r_0[:,0] = R * np.cos(theta) * np.cos(phi)
    r_0[:,1] = R * np.cos(theta) * np.sin(phi)
    r_0[:,2] = R * np.sin(theta)
    
    x = np.sqrt(r_0[:,0]**2 + r_0[:,1]**2)
    a = 90. + 180./np.pi * np.arctan(r_0[:,2] / (x+0.15*R))
    b = b * np.ones(n_frames)
    
    camera_pos = {'r': r_0,
                  'a': a,
                  'b': b}
    
    map_fname = '/n/fink1/ggreen/bayestar/output/allsky_2MASS/compact_10samp.h5'
    frame_fname = '/n/pan1/www/ggreen/3d/allsky_2MASS/GC_perspective/GC_perspective'
    
    gen_movie_frames(camera_pos, camera_props, map_fname, frame_fname, n_procs=8, dpi=120)


def local_rot_movie():
    camera_props = {'n_x': 600,
                    'n_y': 450,
                    'n_z': 2000,
                    'fov': 90.,
                    'dr': 3.,
                    'z_0': 15.}
    
    n_frames = 1000
    
    x_0 = -250.
    y_0 = -50.
    z_0 = -50.
    
    d_0 = np.sqrt(x_0**2 + y_0**2)
    d = d_0 + 50. * np.linspace(0., 1., n_frames)**(0.15)
    
    phi_0 = np.arctan2(-x_0, -y_0)
    phi = np.linspace(phi_0, phi_0 + 4.*np.pi, n_frames)
    
    theta = np.linspace(0., 2.*np.pi, n_frames)
    
    x = x_0 + d * np.sin(phi)
    y = y_0 + d * np.cos(phi)
    z = z_0 - z_0 * np.cos(theta)
    
    b = 180. + 180./np.pi * np.arctan2(y-y_0, x-x_0)
    
    dx = np.sqrt((y-y_0)**2 + (x-x_0)**2)
    a = 90. + 180./np.pi * np.arctan2(z-z_0, dx)
    
    r = np.empty((n_frames, 3), dtype='f8')
    r[:,0] = x
    r[:,1] = y
    r[:,2] = z
    
    camera_pos = {'r': r, 'a': a, 'b': b}
    
    print r
    print ''
    print a
    print ''
    print b
    
    #fig = plt.figure()
    #ax = fig.add_subplot(1,1,1)
    #ax.plot(x, y)
    #plt.show()
    
    #return
    
    map_fname = '/n/fink1/ggreen/bayestar/output/allsky_2MASS/compact_10samp.h5'
    frame_fname = '/n/pan1/www/ggreen/3d/allsky_2MASS/local_rot/local_rot'

    gen_movie_frames(camera_pos, camera_props, map_fname, frame_fname,
                     n_procs=12, dpi=120, n_averaged=5)


def local_dust_movie():
    camera_props = {'n_x': 800,
                    'n_y': 600,
                    'n_z': 2000,
                    'fov': 90.,
                    'dr': 3.,
                    'z_0': 25.}
    
    n_frames = 200
    
    # Zoom out
    r_0 = np.zeros((n_frames/4, 3), dtype='f8')
    r_0[:,0] = np.linspace(0., 150., r_0.shape[0])
    r_0[:,2] = np.linspace(0., 55., r_0.shape[0])
    
    a_0 = 180./np.pi * np.arctan((r_0[-1,2] + 20.) / r_0[-1,0])
    b_0 = 180.
    
    a_0 = np.linspace(90. + a_0/2., 90. + a_0, r_0.shape[0])
    b_0 = b_0 * np.ones(r_0.shape[0])
    
    # Rotate around azimuthally, while bobbing in z
    phi = 25. * np.pi/180. * np.sin(np.linspace(0., 2.*np.pi, int(3./4.*n_frames)))
    theta = np.linspace(0., 2.*np.pi, phi.size)
    #phi = np.linspace(0., 2.*np.pi, int(3./4.*n_frames))
    R = r_0[-1,0]
    Z = r_0[-1,2]
    
    r_1 = np.zeros((phi.size, 3), dtype='f8')
    r_1[:,0] = R * np.cos(phi)
    r_1[:,1] = R * np.sin(phi)
    r_1[:,2] = Z + 20. * np.sin(theta)
    #r_1[:,2] = Z * np.cos(phi/2.)
    
    a_1 = 90. + 180./np.pi * np.arctan((r_1[:,2] + 20.) / R)
    b_1 = np.mod(180. + 180./np.pi * phi, 360.)
    
    camera_pos = {'r': np.concatenate([r_0, r_1], axis=0),
                  'a': np.hstack([a_0, a_1]),
                  'b': np.hstack([b_0, b_1])}
    
    map_fname = '/n/fink1/ggreen/bayestar/output/allsky_2MASS/compact_10samp.h5'
    frame_fname = '/n/pan1/www/ggreen/3d/allsky_2MASS/local_dust/local_dust'
    
    gen_movie_frames(camera_pos, camera_props, map_fname, frame_fname,
                     n_procs=6, dpi=120)


def indiv_frames():
    camera_props = {'n_x': 1000,
                    'n_y': 750,
                    'n_z': 2000,
                    'fov': 90.,
                    'dr': 3.,
                    'z_0': 25.}
    
    r = np.zeros((4, 3), dtype='f8')
    a = np.empty(r.shape[0], dtype='f8')
    b = np.empty(r.shape[0], dtype='f8')
    
    r[0,:] = [104., 146., 109.]
    a[0] = 110.
    b[0] = 190.
    
    r[1,:] = [0., 146., 150.]
    a[1] = 115.
    b[1] = 220.
    
    r[2,:] = [104., 146., 109.]
    a[2] = 110.
    b[2] = 205.
    
    r[3,:] = [104., 146., 109.]
    a[3] = 110.
    b[3] = 200.
    
    camera_pos = {'r': r,
                  'a': a,
                  'b': b}
    
    map_fname = '/n/fink1/ggreen/bayestar/output/allsky/compact_5samp.h5'
    frame_fname = '/n/pan1/www/ggreen/3d/anticenter/arm'
    
    gen_movie_frames(camera_pos, camera_props, map_fname, frame_fname,
                     n_procs=4, dpi=150, n_averaged=7)


def main():
    local_rot_movie()
    #zoom_out_movie()
    #bobbing_movie()
    #local_dust_movie()
    #indiv_frames()
    #GC_perspective_movie()
    
    return 0


if __name__ == '__main__':
    main()
