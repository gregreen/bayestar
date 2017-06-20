#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  hputils.py
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
import healpy as hp

def gal2equ(l, b):
    '''
    Convert (l, b) to (RA, Dec).
    
    All inputs and outputs are in degrees.
    '''
    
    rot = hp.rotator.Rotator(coord=['G', 'Equatorial'])
    
    t_g = np.radians(90. - b)
    p_g = np.radians(l)
    
    t_e, p_e = rot(t_g, p_g)
    ra = np.degrees(p_e)
    dec = 90. - np.degrees(t_e)
    
    return ra, dec


def lb2pix(nside, l, b, nest=True):
    '''
    Convert (l, b) to pixel index.
    '''
    
    theta = np.pi/180. * (90. - b)
    phi = np.pi/180. * l
    
    idx = (b >= -90.) & (b <= 90.)
    
    pix_idx = np.empty(l.shape, dtype='i8')
    pix_idx[idx] = hp.pixelfunc.ang2pix(nside, theta[idx], phi[idx], nest=nest)
    pix_idx[~idx] = -1
    
    return pix_idx


def pix2lb(nside, ipix, nest=True, use_negative_l=False):
    '''
    Convert pixel index to (l, b).
    '''
    
    theta, phi = hp.pixelfunc.pix2ang(nside, ipix, nest=nest)
    
    l = 180./np.pi * phi
    b = 90. - 180./np.pi * theta
    
    idx = (l > 180.)
    l[idx] = l[idx] - 360.
    
    return l, b

def pix2lb_scalar(nside, ipix, nest=True, use_negative_l=False):
    '''
    Convert pixel index to (l, b).
    
    Takes scalar input (no arrays).
    '''
    
    theta, phi = hp.pixelfunc.pix2ang(nside, ipix, nest=nest)
    
    l = 180./np.pi * phi
    b = 90. - 180./np.pi * theta
    
    if l > 180.:
        l -= 360.
    
    return l, b

def wrap_longitude(lon, delta_lon, degrees=True):
    '''
    Shift longitudes by delta_lon, and wrap
    back to range [0, 360].
    
    If degrees=False, then radians are assumed.
    '''
    
    lon_shifted = lon + delta_lon
    
    if degrees:
        return np.mod(lon_shifted, 360.)
    else:
        return np.mod(lon_shifted, 2. * np.pi)


def lb_in_bounds(l, b, bounds):
    '''
    Determine whether the given (l, b) coordinates
    are within the provided bounds.
    
    The bounds are in the format:
    
        l_0, l_1, b_0, b_1
    
    l and b can be either floats or numpy arrays. In the
    first case, a boolean will be returned, and in the
    second, a numpy boolean array will be returned.
    '''
    
    l_0 = np.mod(bounds[0], 360.)
    l_1 = np.mod(bounds[1] - l_0, 360.)
    
    l_p = np.mod(l - l_0, 360.)
    
    return (l_p >= 0.) & (l_p <= l_1) & (b >= bounds[2]) & (b <= bounds[3])


def shift_lon_lat(lon, lat, delta_lon, delta_lat,
                  degrees=True, clip=False):
    '''
    Shift latitudes and longitudes, but do not
    move them off edges map, and do not wrap
    longitude.
    '''
    
    lon_shifted = lon + delta_lon
    lat_shifted = lat + delta_lat
    
    if clip:
        idx = (lon_shifted > 360.)
        lon_shifted[idx] = 360.
        
        idx = (lon_shifted < 0.)
        lon_shifted[idx] = 0.
        
        idx = (lat_shifted > 90.)
        lat_shifted[idx] = 90.
        
        idx = (lat_shifted < -90.)
        lat_shifted[idx] = -90.
    
    return lon_shifted, lat_shifted


class Mollweide_projection:
    '''
    The Mollweide projection of the sphere onto a flat plane.
    
    Pseudocylindrical, equal-area.
    '''
    
    def __init__(self, lam_0=180.):
        '''
        lam_0 is the central longitude of the map.
        '''
        
        self.lam_0 = np.pi/180. * lam_0
    
    def proj(self, phi, lam, iterations=15, ret_bounds=False):
        '''
        Mollweide projection.
        
        phi = latitude
        lam = longitude
        '''
        
        theta = self.Mollweide_theta(phi, iterations)
        
        x = 2. * np.sqrt(2.) * (lam - self.lam_0) * np.cos(theta) / np.pi
        y = np.sqrt(2.) * np.sin(theta)
        
        #x = 180. * (lam - self.lam_0) * np.cos(theta) / np.pi
        #y = 90. * np.sin(theta)
        
        if ret_bounds:
            return x, y, np.zeros(x.shape, dtype=np.bool)
        
        return x, y
    
    def inv(self, x, y):
        '''
        Inverse Mollweide projection.
        
        Returns (phi, lam), given (x, y).
        
        phi = latitude
        lam = longitude
        
        x and y can be floats or numpy float arrays.
        '''
        
        theta = np.arcsin(y / np.sqrt(2.))
        
        phi = np.arcsin((2. * theta + np.sin(2. * theta)) / np.pi)
        lam = self.lam_0 + np.pi * x / (2. * np.sqrt(2.) * np.cos(theta))
        
        #theta = np.arcsin(y / 90.)
        
        #phi = np.arcsin((2. * theta + np.sin(2. * theta)) / np.pi)
        
        #lam = self.lam_0 + np.pi * x / (180. * np.cos(theta))
        
        out_of_bounds = (lam < 0.) | (lam > 2.*np.pi) | (phi < -np.pi) | (phi > np.pi)
        
        return phi, lam, out_of_bounds
    
    def Mollweide_theta(self, phi, iterations):
        theta = np.arcsin(2. * phi / np.pi)
        sin_phi = np.sin(phi)
        
        for i in xrange(iterations):
            theta -= 0.5 * (2. * theta + np.sin(2. * theta) - np.pi * sin_phi) / (1. + np.cos(2. * theta))
        
        idx = np.isnan(theta)
        theta[idx] = np.sign(sin_phi) * 0.5 * np.pi
        
        return theta


class EckertIV_projection:
    '''
    The Eckert IV projection of the sphere onto a flat plane.
    
    Pseudocylindrical, equal-area.
    '''
    
    def __init__(self, lam_0=180.):
        '''
        lam_0 is the central longitude of the map.
        '''
        
        self.lam_0 = np.pi/180. * lam_0
        
        self.x_scale = 180. / 2.65300085635
        self.y_scale = 90. / 1.32649973731
        
        self.a = np.sqrt(np.pi * (4. + np.pi))
        self.b = np.sqrt(np.pi / (4. + np.pi))
        self.c = 2. + np.pi / 2.
        
        #self.a = 2. / np.sqrt(np.pi * (4. + np.pi))
        #self.b = 2. * np.sqrt(np.pi / (4. + np.pi))
        #self.d = np.sqrt((4. + np.pi) / np.pi)
        #self.e = np.sqrt(np.pi * (4. + np.pi))
    
    def proj(self, phi, lam, iterations=10, ret_bounds=False):
        '''
        Eckert IV projection.
        
        phi = latitude
        lam = longitude
        '''
        
        theta = self.EckertIV_theta(phi, iterations)
        
        x = self.x_scale * 2. / self.a * (lam - self.lam_0) * (1. + np.cos(theta))
        y = self.y_scale * 2. * self.b * np.sin(theta)
        
        if ret_bounds:
            return x, y, np.zeros(x.shape, dtype=np.bool)
        
        return x, y
    
    def inv(self, x, y):
        '''
        Inverse Eckert projection.
        
        Returns (phi, lam), given (x, y).
        
        phi = latitude
        lam = longitude
        
        x and y can be floats or numpy float arrays.
        '''
        
        theta = np.arcsin((y / self.y_scale) / 2. / self.b)
        
        phi = np.arcsin((theta + 0.5 * np.sin(2. * theta) + 2. * np.sin(theta)) / self.c)
        
        lam = self.lam_0 + self.a / 2. * (x / self.x_scale) / (1. + np.cos(theta))
        
        out_of_bounds = (lam < 0.) | (lam > 2.*np.pi) | (phi < -np.pi) | (phi > np.pi)
        
        return phi, lam, out_of_bounds
    
    def EckertIV_theta(self, phi, iterations):
        theta = phi / 2.
        sin_phi = np.sin(phi)
        
        for i in xrange(iterations):
            theta -= (theta + 0.5 * np.sin(2. * theta) + 2. * np.sin(theta) - self.c * sin_phi) / (2. * np.cos(theta) * (1. + np.cos(theta)))
        
        return theta


class Hammer_projection:
    '''
    The Hammer projection of the sphere onto a flat plane.
    
    Equal-area. Similar to the Mollweide projection, but with curved
    parallels to reduce distortion at the outer limbs.
    '''
    
    def __init__(self, lam_0=180.):
        '''
        lam_0 is the central longitude of the map.
        '''
        
        self.lam_0 = np.pi/180. * lam_0
    
    def proj(self, phi, lam, ret_bounds=False):
        '''
        Hammer projection.
        
        phi = latitude
        lam = longitude
        '''
        
        denom = np.sqrt(1. + np.cos(phi) * np.cos((lam - self.lam_0)/2.))
        
        x = 2. * np.sqrt(2.) * np.cos(phi) * np.sin((lam - self.lam_0)/2.) / denom
        y = np.sqrt(2.) * np.sin(phi) / denom
        
        if ret_bounds:
            return x, y, np.zeros(x.shape, dtype=np.bool)
        
        return x, y
    
    def inv(self, x, y):
        '''
        Inverse Hammer projection.
        
        Returns (phi, lam, out_of_bounds), given (x, y).
        
        phi = latitude
        lam = longitude
        
        out_of_bounds = True if pixel is not in standard range (ellipse
                        that typically bounds the Hammer projection)
        
        x and y can be floats or numpy float arrays.
        '''
        
        z = np.sqrt(1. - np.power(x/4., 2.) - np.power(y/2., 2.))
        
        lam = self.lam_0 + 2. * np.arctan(z * x / (2. * (2. * np.power(z, 2.) - 1.)))
        
        phi = np.arcsin(z * y)
        
        out_of_bounds = (0.25 * x*x + y*y) > 2.
        
        return phi, lam, out_of_bounds


class Cartesian_projection:
    '''
    The Cartesian projection of the sphere onto a flat plane.
    '''
    
    def __init__(self, lam_0=180.):
        self.lam_0 = np.pi / 180. * lam_0
    
    def proj(self, phi, lam, ret_bounds=False):
        x = 180./np.pi * (lam - self.lam_0)
        y = 180./np.pi * phi
        
        if ret_bounds:
            return x, y, np.zeros(x.shape, dtype=np.bool)
        
        return x, y
    
    def inv(self, x, y):
        lam = self.lam_0 + np.pi/180. * x
        phi = np.pi/180. * y
        
        out_of_bounds = (lam < 0.) | (lam > 2.*np.pi) | (phi < -np.pi) | (phi > np.pi)
        
        return phi, lam, out_of_bounds


class Gnomonic_projection:
    '''
    Gnomonic projection of the sphere into a flat plane.
    '''
    
    def __init__(self, phi_0=0., lam_0=180., fov=90.):
        self.phi_0 = np.radians(phi_0)
        self.lam_0 = np.radians(lam_0)
        self.fov = np.radians(fov)
        
        self.cp0 = np.cos(self.phi_0)
        self.sp0 = np.sin(self.phi_0)
    
    def proj(self, phi, lam, ret_bounds=False):
        p = phi
        l = lam
        
        cp = np.cos(p)
        sp = np.sin(p)
        
        cDl = np.cos(l - self.lam_0)
        sDl = np.sin(l - self.lam_0)
        
        a = 1. / (self.sp0 * sp + self.cp0 * cp * cDl)
        
        x = a * cp * sDl
        y = a * (self.cp0 * sp - self.sp0 * cp * cDl)
        
        if ret_bounds:
            out_of_bounds = (np.arccos(1./a) > self.fov)
            return x, y, out_of_bounds
        
        return x, y
    
    def inv(self, x, y):
        rho = np.sqrt(x**2. + y**2.)
        c = np.arctan(rho)
        
        cc = np.cos(c)
        sc = np.sin(c)
        
        phi = np.arcsin(cc * self.sp0 + y * sc * self.cp0 / rho)
        lam = self.lam_0 + np.arctan2((x * sc), (rho * self.cp0 * cc - y * self.sp0 * sc))
        
        out_of_bounds = (np.abs(c) > self.fov)
        
        return phi, lam, out_of_bounds


class Stereographic_projection:
    '''
    Stereographic projection of the sphere into a flat plane.
    '''
    
    def __init__(self, phi_0=0., lam_0=180., fov=180.):
        self.phi_0 = np.radians(phi_0)
        self.lam_0 = np.radians(lam_0)
        self.fov = np.radians(fov)
        
        self.cp0 = np.cos(self.phi_0)
        self.sp0 = np.sin(self.phi_0)
    
    def proj(self, phi, lam, ret_bounds=False):
        cp = np.cos(phi)
        sp = np.sin(phi)
        
        cDl = np.cos(lam - self.lam_0)
        sDl = np.sin(lam - self.lam_0)
        
        a = 2. / (1. + self.sp0 * sp + self.cp0 * cp * cDl)
        
        x = a * cp * sDl
        y = a * (self.cp0 * sp - self.sp0 * cp * cDl)
        
        if ret_bounds:
            rho = np.sqrt(x**2. + y**2.)
            c = 2. * np.arctan(0.5 * rho)
            out_of_bounds = (c > self.fov)
            
            return x, y, out_of_bounds
        
        return x, y
    
    def inv(self, x, y):
        rho = np.sqrt(x**2. + y**2.)
        c = 2. * np.arctan(0.5 * rho)
        
        cc = np.cos(c)
        sc = np.sin(c)
        
        phi = np.arcsin(cc * self.sp0 + y * sc * self.cp0 / rho)
        lam = self.lam_0 + np.arctan2((x * sc), (rho * self.cp0 * cc - y * self.sp0 * sc))
        
        out_of_bounds = (np.abs(c) > self.fov)
        
        return phi, lam, out_of_bounds


def Euler_rotation_vec(x, y, z, alpha, beta, gamma, inverse=False):
    if inverse:
        alpha *= -1.
        beta *= -1.
        gamma *= -.1
    
    X = np.array([[1., 0.,             0.           ],
                  [0., np.cos(gamma), -np.sin(gamma)],
                  [0., np.sin(gamma),  np.cos(gamma)]])
    
    Y = np.array([[ np.cos(beta), 0., np.sin(beta)],
                  [ 0.,           1., 0.          ],
                  [-np.sin(beta), 0., np.cos(beta)]])
    
    Z = np.array([[np.cos(alpha), -np.sin(alpha), 0.],
                  [np.sin(alpha),  np.cos(alpha), 0.],
                  [0.,             0.,            1.]])
    
    shape = x.shape
    
    vec = np.empty((x.size, 3), dtype='f8')
    vec[:,0] = x.flatten()
    vec[:,1] = y.flatten()
    vec[:,2] = z.flatten()
    
    if inverse:
        vec = np.einsum('ij,jk,kl,ml->mi', Z, Y, X, vec)
    else:
        vec = np.einsum('ij,jk,kl,ml->mi', X, Y, Z, vec)
    
    x = np.reshape(vec[:,0], shape)
    y = np.reshape(vec[:,1], shape)
    z = np.reshape(vec[:,2], shape)
    
    return x, y, z


def ang2vec(theta, phi):
    cos_theta = np.cos(theta)
    
    x = np.cos(phi) * cos_theta
    y = np.sin(phi) * cos_theta
    z = np.sin(theta)
    
    return x, y, z


def vec2ang(x, y, z):
    r = np.sqrt(x*x + y*y + z*z)
    
    phi = np.arctan2(y, x)
    theta = np.arcsin(z/r)
    
    return theta, phi


def Euler_rotation_ang(theta, phi, alpha, beta, gamma,
                       degrees=False, inverse=False):
    x, y, z = None, None, None
    
    if degrees:
        x, y, z = ang2vec(np.pi/180. * theta, np.pi/180. * phi)
        
        alpha *= np.pi/180.
        beta *= np.pi/180.
        gamma *= np.pi/180.
        
    else:
        x, y, z = ang2vec(theta, phi)
    
    x, y, z = Euler_rotation_vec(x, y, z, alpha, beta, gamma,
                                          inverse=inverse)
    
    t, p = vec2ang(x, y, z)
    
    if degrees:
        t *= 180. / np.pi
        p *= 180. / np.pi
    
    return t, p


def rasterize_map_old(pix_idx, pix_val,
                      nside, size,
                      nest=True, clip=True,
                      proj=Cartesian_projection(),
                      l_cent=0., b_cent=0.):
    '''
    Rasterize a healpix map.
    '''
    
    pix_scale = 180./np.pi * hp.nside2resol(nside)
    
    # Determine pixel centers
    l_0, b_0 = pix2lb(nside, pix_idx, nest=nest, use_negative_l=True)
    
    # Rotate coordinate system to center (l_0, b_0)
    if (l_cent != 0.) | (b_cent != 0.):
        b_0, l_0 = Euler_rotation_ang(b_0, l_0, -l_cent, b_cent, 0.,
                                                degrees=True)
        #l_0 = np.mod(l_0, 360.)
    
    lam_0 = 180. - l_0
    
    # Determine display-space bounds
    shift = [(0., 0.), (1., 0.), (0., 1.), (-1., 0.), (0., -1.)]
    x_min, x_max, y_min, y_max = [], [], [], []
    
    #for (s_x, s_y) in shift:
    for s_x in np.linspace(-pix_scale, pix_scale, 3):
        for s_y in np.linspace(-pix_scale, pix_scale, 3):
            lam, b = shift_lon_lat(lam_0, b_0, 0.75*s_x, 0.75*s_y, clip=True)
            
            x_0, y_0 = proj.proj(np.pi/180. * b, np.pi/180. * lam)
            
            x_min.append(np.min(x_0))
            x_max.append(np.max(x_0))
            y_min.append(np.min(y_0))
            y_max.append(np.max(y_0))
    
    x_min = np.min(x_min)
    x_max = np.max(x_max)
    y_min = np.min(y_min)
    y_max = np.max(y_max)
    
    # Make grid of display-space pixels
    x_size, y_size = size
    
    x, y = np.mgrid[0:x_size, 0:y_size].astype(np.float32) + 0.5
    x = x_min + (x_max - x_min) * x / float(x_size)
    y = y_min + (y_max - y_min) * y / float(y_size)
    
    # Convert display-space pixels to (l, b)
    b, lam, mask = proj.inv(x, y)
    l = 180. - 180./np.pi * lam
    b *= 180./np.pi
    
    # Rotate back to original (l, b)-space
    if (l_cent != 0.) | (b_cent != 0.):
        b, l = Euler_rotation_ang(b, l, -l_cent, b_cent, 0.,
                                        degrees=True, inverse=True)
    
    # Determine bounds in (l, b)-space
    #if clip:
    #l_min, l_max = np.min(l[~mask]), np.max(l[~mask])
    #b_min, b_max = np.min(b[~mask]), np.max(b[~mask])
    #else:
    #    l_min, l_max = np.min(l), np.max(l)
    #    b_min, b_max = np.min(b), np.max(b)
    
    # Convert (l, b) to healpix indices
    disp_idx = lb2pix(nside, l, b, nest=nest)
    
    # Generate full map
    n_pix = hp.pixelfunc.nside2npix(nside)
    pix_idx_full = np.arange(n_pix)
    pix_val_full = np.empty(n_pix, dtype='f8')
    pix_val_full[:] = np.nan
    pix_val_full[pix_idx] = pix_val[:]
    
    # Grab pixel values
    img = None
    good_idx = None
    
    if len(pix_val.shape) == 1:
        img = pix_val_full[disp_idx]
        
        if clip:
            img[mask] = np.nan
        
        good_idx = np.isfinite(img)
        
        img.shape = (x_size, y_size)
        
    elif len(pix_val.shape) == 2:
        img = pix_val[:,disp_idx]
        
        if clip:
            img[:,mask] = np.nan
        
        good_idx = np.any(np.isfinite(img), axis=0)
        
        img.shape = (img.shape[0], x_size, y_size)
        
    else:
        raise Exception('pix_val must be either 1- or 2-dimensional.')
    
    l_min, l_max = np.min(l[good_idx]), np.max(l[good_idx])
    b_min, b_max = np.min(b[good_idx]), np.max(b[good_idx])
    
    bounds = (l_max, l_min, b_min, b_max)
    
    return img, bounds


def latlon_lines(ls, bs,
                 l_spacing=1., b_spacing=1.,
                 proj=Cartesian_projection(),
                 l_cent=0., b_cent=0.,
                 bounds=None, xy_bounds=None,
                 mode='both'):
    '''
    Return the x- and y- positions of points along a grid of parallels
    and meridians.
    '''
    
    if mode not in ['both', 'parallels', 'meridians']:
        raise ValueError("Unrecognized mode: '%s'\n"
                         "Must be 'both', 'parallels' or 'meridians'" % mode)
    
    # Construct a set of points along the meridians and parallels
    l = []
    b = []
    
    if mode in ['both', 'parallels']:
        l_row = np.arange(-180., 180.+l_spacing/2., l_spacing)
        b_row = np.ones(l_row.size)
        
        for b_val in bs:
            b.append(b_val * b_row)
            l.append(l_row)
    
    if mode in ['both', 'meridians']:
        b_row = np.arange(-90., 90.+b_spacing/2., b_spacing)
        l_row = np.ones(b_row.size)
        
        for l_val in ls:
            l.append(l_val * l_row)
            b.append(b_row)
    
    l = np.hstack(l)
    b = np.hstack(b)
    
    # Rotate coordinate system to center (l_0, b_0)
    if (l_cent != 0.) | (b_cent != 0.):
        b, l = Euler_rotation_ang(b, l, -l_cent, b_cent, 0.,
                                                degrees=True)
    
    lam = 180. - l
    
    # Project to (x, y)
    x, y, oob = proj.proj(np.pi/180. * b, np.pi/180. * lam, ret_bounds=True)
    x = x[~oob]
    y = y[~oob]
    
    # Scale (x, y) to display bounds
    if (bounds != None) and (xy_bounds != None):
        x_scale = (bounds[1] - bounds[0]) / (xy_bounds[1] - xy_bounds[0])
        y_scale = (bounds[3] - bounds[2]) / (xy_bounds[3] - xy_bounds[2])
        
        x = bounds[0] + (x - xy_bounds[0]) * x_scale
        y = bounds[2] + (y - xy_bounds[2]) * y_scale
    
    return x, y


def rasterize_map(pix_idx, pix_val,
                  nside, size,
                  nest=True, clip=True,
                  proj=Cartesian_projection(),
                  l_cent=0., b_cent=0.,
                  l_lines=None, b_lines=None,
                  l_spacing=1., b_spacing=1.):
    '''
    Rasterize a healpix map.
    '''
    
    pix_scale = 180./np.pi * hp.nside2resol(nside)
    
    # Determine pixel centers
    l_0, b_0 = pix2lb(nside, pix_idx, nest=nest, use_negative_l=True)
    
    # Rotate coordinate system to center (l_0, b_0)
    if (l_cent != 0.) | (b_cent != 0.):
        b_0, l_0 = Euler_rotation_ang(b_0, l_0, -l_cent, b_cent, 0.,
                                                degrees=True)
        #l_0 = np.mod(l_0, 360.)
    
    lam_0 = 180. - l_0
    
    # Determine display-space bounds
    shift = [(0., 0.), (1., 0.), (0., 1.), (-1., 0.), (0., -1.)]
    x_min, x_max, y_min, y_max = [], [], [], []
    
    #for (s_x, s_y) in shift:
    for s_x in np.linspace(-pix_scale, pix_scale, 3):
        for s_y in np.linspace(-pix_scale, pix_scale, 3):
            lam, b = shift_lon_lat(lam_0, b_0, 0.75*s_x, 0.75*s_y, clip=True)
            
            x_0, y_0 = proj.proj(np.pi/180. * b, np.pi/180. * lam)
            
            x_min.append(np.min(x_0))
            x_max.append(np.max(x_0))
            y_min.append(np.min(y_0))
            y_max.append(np.max(y_0))
    
    x_min = np.min(x_min)
    x_max = np.max(x_max)
    y_min = np.min(y_min)
    y_max = np.max(y_max)
    
    # Make grid of display-space pixels
    x_size, y_size = size
    
    x, y = np.mgrid[0:x_size, 0:y_size].astype(np.float32) + 0.5
    x = x_min + (x_max - x_min) * x / float(x_size)
    y = y_min + (y_max - y_min) * y / float(y_size)
    
    # Convert display-space pixels to (l, b)
    b, lam, mask = proj.inv(x, y)
    l = 180. - 180./np.pi * lam
    b *= 180./np.pi
    
    # Rotate back to original (l, b)-space
    if (l_cent != 0.) | (b_cent != 0.):
        b, l = Euler_rotation_ang(b, l, -l_cent, b_cent, 0.,
                                        degrees=True, inverse=True)
    
    # Determine bounds in (l, b)-space
    #if clip:
    #l_min, l_max = np.min(l[~mask]), np.max(l[~mask])
    #b_min, b_max = np.min(b[~mask]), np.max(b[~mask])
    #else:
    #    l_min, l_max = np.min(l), np.max(l)
    #    b_min, b_max = np.min(b), np.max(b)
    
    # Convert (l, b) to healpix indices
    disp_idx = lb2pix(nside, l, b, nest=nest)
    
    # Generate full map
    n_pix = hp.pixelfunc.nside2npix(nside)
    pix_idx_full = np.arange(n_pix)
    pix_val_full = np.empty(n_pix, dtype='f8')
    pix_val_full[:] = np.nan
    pix_val_full[pix_idx] = pix_val[:]
    
    # Grab pixel values
    img = None
    good_idx = None
    
    if len(pix_val.shape) == 1:
        img = pix_val_full[disp_idx]
        
        if clip:
            img[mask] = np.nan
        
        good_idx = np.isfinite(img)
        
        img.shape = (x_size, y_size)
        
    elif len(pix_val.shape) == 2:
        img = pix_val[:,disp_idx]
        
        if clip:
            img[:,mask] = np.nan
        
        good_idx = np.any(np.isfinite(img), axis=0)
        
        img.shape = (img.shape[0], x_size, y_size)
        
    else:
        raise Exception('pix_val must be either 1- or 2-dimensional.')
    
    l_min, l_max = np.min(l[good_idx]), np.max(l[good_idx])
    b_min, b_max = np.min(b[good_idx]), np.max(b[good_idx])
    
    bounds = (l_max, l_min, b_min, b_max)
    xy_bounds = (x_min, x_max, y_min, y_max)
    
    if (l_lines != None) and (b_lines != None):
        x, y = latlon_lines(l_lines, b_lines,
                            l_spacing=l_spacing, b_spacing=b_spacing,
                            proj=proj,
                            l_cent=l_cent, b_cent=b_cent,
                            bounds=bounds, xy_bounds=xy_bounds)
        
        return img, bounds, x, y
    else:
        return img, bounds, xy_bounds


class MapRasterizer:
    '''
    A class that rasterizes a multi-resolution HEALPix map with a given
    set of (nside, pixel index) pairs. Pre-computes mapping between
    display-space pixels and HEALPix pixels, so that maps with different
    pixel intensities can be rasterized quickly.
    '''
    
    def __init__(self, nside, pix_idx, img_shape,
                       nest=True, clip=True,
                       proj=Cartesian_projection(),
                       l_cent=0., b_cent=0.):
        '''
        Pre-computes mapping between (nside, pix_idx) pairs and
        display-space pixels.
        '''
        
        self.img_shape = img_shape
        self.clip = clip
        self.proj = proj
        self.l_cent = l_cent
        self.b_cent = b_cent
        
        
        #
        # Determine display-space bounds
        #
        
        shift = [(0., 0.), (1., 0.), (0., 1.), (-1., 0.), (0., -1.)]
        x_min, x_max, y_min, y_max = [], [], [], []
        
        nside_unique = np.unique(nside)
        
        for n in nside_unique:
            pix_scale = 180./np.pi * hp.nside2resol(n)
                
            # Determine pixel centers
            idx = (nside == n)
            
            if np.sum(idx) == 0:
                continue
            
            l_0, b_0 = pix2lb(n, pix_idx[idx], nest=nest, use_negative_l=True)
            
            # Rotate coordinate system to center (l_0, b_0)
            if (l_cent != 0.) | (b_cent != 0.):
                b_0, l_0 = Euler_rotation_ang(b_0, l_0, -l_cent, b_cent, 0.,
                                                        degrees=True)
            
            lam_0 = 180. - l_0
            
            # Compute bounds for given pixel shift
            for s_x in np.linspace(-pix_scale, pix_scale, 3):
                for s_y in np.linspace(-pix_scale, pix_scale, 3):
                    lam, b = shift_lon_lat(lam_0, b_0, 0.75*s_x, 0.75*s_y, clip=True)
                    
                    x_0, y_0, idx_out = proj.proj(np.pi/180. * b, np.pi/180. * lam,
                                                  ret_bounds=True)
                    
                    if np.sum(~idx_out) != 0:
                        x_min.append(np.min(x_0[~idx_out]))
                        x_max.append(np.max(x_0[~idx_out]))
                        y_min.append(np.min(y_0[~idx_out]))
                        y_max.append(np.max(y_0[~idx_out]))
                    
                    del x_0
                    del y_0
                    del lam
                    del b
                    del idx_out
        
        x_min = np.min(x_min)
        x_max = np.max(x_max)
        y_min = np.min(y_min)
        y_max = np.max(y_max)
        
        
        #
        # Make grid of display-space pixels
        #
        
        x_size, y_size = img_shape
        
        x, y = np.mgrid[0:x_size, 0:y_size].astype(np.float32) + 0.5
        x = ( x_min + (x_max - x_min) * x / float(x_size) ).flatten()
        y = ( y_min + (y_max - y_min) * y / float(y_size) ).flatten()
        
        # Convert display-space pixels to (l, b)
        b, lam, self.clip_mask = proj.inv(x, y)
        l = 180. - 180./np.pi * lam
        b *= 180./np.pi
        del lam
        
        # Rotate back to original (l, b)-space
        if (l_cent != 0.) | (b_cent != 0.):
            b, l = Euler_rotation_ang(b, l, -l_cent, b_cent, 0.,
                                            degrees=True, inverse=True)
        
        
        #
        # Determine mapping from image index to map index
        #
        
        #image_idx = np.arange(l.size, dtype='i8')
        input_idx = np.arange(pix_idx.size, dtype='i8')
        self.map_idx_full = np.empty(l.size, dtype='i8')
        self.map_idx_full[:] = -1
        
        for n in nside_unique:
            idx = (nside == n)
            
            if np.sum(idx) == 0:
                continue
            
            # Determine healpix index of each remaining image pixel
            disp_idx = lb2pix(n, l, b, nest=nest)
            
            # Generate full map
            n_pix = hp.pixelfunc.nside2npix(n)
            healpix_2_map = np.empty(n_pix, dtype='i8')
            healpix_2_map[:] = -1
            healpix_2_map[pix_idx[idx]] = input_idx[idx]
            
            # Update map indices
            map_idx_tmp = healpix_2_map[disp_idx]
            mask = ~(map_idx_tmp == -1)
            
            self.map_idx_full[mask] = map_idx_tmp[mask]
            
            del healpix_2_map
            del map_idx_tmp
        
        self.good_idx = ~(self.map_idx_full == -1)
        self.map_idx = self.map_idx_full[self.good_idx]
        
        l_0 = np.mod(l_cent, 360.)
        dl = np.mod(l - l_0, 360.)
        dl[dl > 180.] -= 360.
        #dl = np.mod(l - l_cent, 360.) - 180.
        dl_min, dl_max = np.min(dl[self.good_idx]), np.max(dl[self.good_idx])
        l_min = l_cent + dl_min
        l_max = l_cent + dl_max
        
        print('l_cent = {0:.2f}'.format(l_cent))
        print('dl_min/max = ({0:.2f}, {1:.2f})'.format(dl_min, dl_max))
        print('l_min/max = ({0:.2f}, {1:.2f})'.format(l_min, l_max))
        
        b_min, b_max = np.min(b[self.good_idx]), np.max(b[self.good_idx])
        
        self.lb_bounds = (l_max, l_min, b_min, b_max)
        self.xy_bounds = (x_min, x_max, y_min, y_max)
        
        self.x_scale = (self.lb_bounds[1] - self.lb_bounds[0]) / (self.xy_bounds[1] - self.xy_bounds[0])
        self.y_scale = (self.lb_bounds[3] - self.lb_bounds[2]) / (self.xy_bounds[3] - self.xy_bounds[2])
    
    def rasterize(self, pix_val):
        '''
        Rasterize the given HEALpix pixel intensity values.
        
        pix_val is assumed to correspond exactly to the nside and
        pix_idx arrays, in terms of the pixels it represents.
        '''
        
        # Grab pixel values
        img = np.empty(self.img_shape[0] * self.img_shape[1], dtype='f8')
        img[:] = np.nan
        
        img[self.good_idx] = pix_val[self.map_idx]
        
        if self.clip:
            img[self.clip_mask] = np.nan
        
        img.shape = (self.img_shape[0], self.img_shape[1])
        
        return img
    
    def __call__(self, pix_val):
        return self.rasterize(pix_val)
    
    def proj_lb(self, l, b):
        '''
        Project (l,b)-coordinates to the image space.
        '''
        
        print 'l/b:'
        print l
        print b
        
        # Rotate coordinate system to center (l_0, b_0)
        if (self.l_cent != 0.) | (self.b_cent != 0.):
            b, l = Euler_rotation_ang(b, l, -self.l_cent, self.b_cent, 0.,
                                                              degrees=True)
        
        lam = 180. - l
        
        x, y = self.proj.proj(np.radians(b), np.radians(lam))
        print 'unscaled:'
        print x
        print y
        x, y = self._scale_disp_coords(x, y)
        print 'scaled:'
        print x
        print y
        ''
        #x_scale = (self.lb_bounds[1] - self.lb_bounds[0]) / (self.xy_bounds[1] - self.xy_bounds[0])
        #y_scale = (self.lb_bounds[3] - self.lb_bounds[2]) / (self.xy_bounds[3] - self.xy_bounds[2])
        
        #x = self.lb_bounds[0] + (x - self.xy_bounds[0]) * x_scale
        #y = self.lb_bounds[2] + (y - self.xy_bounds[2]) * y_scale
        
        return x, y
    
    def xy2idx(self, x, y, lb_bounds=True):
        '''
        Return the map index of the raster position (x, y).
        
        Returns -1 if (x, y) is off the map.
        
        If lb_bounds is True, then assume that lb_bounds was used to set
        extent of image (in imshow).
        
        This is useful for letting the user click on the image of the
        map to retrieve information on a given pixel.
        '''
        
        bounds = self.xy_bounds
        
        if lb_bounds:
            bounds = self.lb_bounds
        
        dx = (bounds[1] - bounds[0]) / float(self.img_shape[0])
        dy = (bounds[3] - bounds[2]) / float(self.img_shape[1])
        
        x_idx = int(np.floor((x - bounds[0]) / dx))
        y_idx = int(np.floor((y - bounds[2]) / dy))
        
        if (    (x_idx < 0) or (x_idx >= self.img_shape[0])
             or (y_idx < 0) or (y_idx >= self.img_shape[1])):
            return -1
        
        idx = x_idx * self.img_shape[1] + y_idx
        
        if self.clip:
            if self.clip_mask[idx]:
                return -1
        
        return self.map_idx_full[idx]
    
    def latlon_lines(self, l_lines, b_lines,
                           l_spacing=1., b_spacing=1.,
                           clip=True, mode='both'):
        '''
        Project lines of constant Galactic longitude and latitude to
        display space (x, y).
        
        Inputs:
            l_lines    Galactic longitudes at which to place lines.
            b_lines    Galactic latitudes at which to place lines.
            l_spacing  Longitude spacing between dots in lines.
            b_spacing  Latitude spacing between dots in lines.
            clip       If True, clip lines to display-space bounds of
                       the map.
        
        Outputs:
            x          x-coordinates of dots that comprise lines.
            y          y-coordinates of dots that comprise lines.
        '''
        
        x, y = latlon_lines(l_lines, b_lines,
                            l_spacing=l_spacing, b_spacing=b_spacing,
                            proj=self.proj,
                            l_cent=self.l_cent, b_cent=self.b_cent,
                            bounds=self.lb_bounds, xy_bounds=self.xy_bounds,
                            mode=mode)
        
        if clip:
            x_idx = (x >= self.lb_bounds[1]) & (x <= self.lb_bounds[0])
            y_idx = (y >= self.lb_bounds[2]) & (y <= self.lb_bounds[3])
            idx = x_idx & y_idx
            
            return x[idx], y[idx]
        
        return x, y
    
    def label_locs(self, l_locs, b_locs, shift_frac=0.05):
        '''
        Find the locations in display-space (x, y) at which the given
        l and b labels should be placed.
        
        Inputs:
            l_locs      List or array containing Galactic longitudes of labels
            b_locs      List or array containing Galactic latitudes of labels
            shift_frac  Fraction of width/heigh of plot to shift labels
                        away from the edges of the map by.
        
        Outputs:
            l_labels  [l, (x_0, y_0), (x_1, y_1)] for each l in l_locs,
                          where (x, y) are the positions of the labels.
                          There are two label positions given for each
                          longitude (in general, one on the top of the
                          plot and one on the bottom).
            
            b_labels  [b, (x_0, y_0), (x_1, y_1)] for each l in b_locs.
        '''
        
        l_labels = []
        
        std_dist = shift_frac * np.sqrt(
                                          abs(self.lb_bounds[1] - self.lb_bounds[0])
                                        * abs(self.lb_bounds[3] - self.lb_bounds[2])
                                       )
        
        for l in l_locs:
            l_arr = np.array([l])
            b_arr = np.array([0.])
            x, y = self.latlon_lines(l_arr, b_arr, clip=True, mode='meridians')
            
            if x.size != 0:
                dx = np.diff(np.hstack([x[-1], x]))
                dy = np.diff(np.hstack([y[-1], y]))
                
                ds = np.sqrt(dx*dx + dy*dy)
                cut_idx = np.argmax(ds)
                
                # Shift label positions off edge of map
                dx_0, dy_0 = None, None
                try:
                    dx_0 = -dx[cut_idx+1]
                    dy_0 = -dy[cut_idx+1]
                except:
                    dx_0 = -dx[0]
                    dy_0 = -dy[0]
                ds_0 = np.sqrt(dx_0*dx_0 + dy_0*dy_0)
                
                dx_0 *= std_dist / ds_0
                dy_0 *= std_dist / ds_0
                
                dx_1 = dx[cut_idx-1]
                dy_1 = dy[cut_idx-1]
                ds_1 = np.sqrt(dx_1*dx_1 + dy_1*dy_1)
                
                dx_1 *= std_dist / ds_1
                dy_1 *= std_dist / ds_1
                
                x_0, y_0 = x[cut_idx] + dx_0, y[cut_idx] + dy_0
                x_1, y_1 = x[cut_idx-1] + dx_1, y[cut_idx-1] + dy_1
                
                l_labels.append([l, (x_0, y_0),
                                    (x_1, y_1)])
        
        b_labels = []
        
        for b in b_locs:
            b_arr = np.array([b])
            l_arr = np.array([0.])
            x, y = self.latlon_lines(l_arr, b_arr, clip=True, mode='parallels')
            
            if x.size != 0:
                dx = np.diff(np.hstack([x[-1], x]))
                dy = np.diff(np.hstack([y[-1], y]))
                
                ds = np.sqrt(dx*dx + dy*dy)
                cut_idx = np.argmax(ds)
                
                # Shift label positions off edge of map
                dx_0, dy_0 = None, None
                try:
                    dx_0 = -dx[cut_idx+1]
                    dy_0 = -dy[cut_idx+1]
                except:
                    dx_0 = -dx[0]
                    dy_0 = -dy[0]
                ds_0 = np.sqrt(dx_0*dx_0 + dy_0*dy_0)
                
                dx_0 *= std_dist / ds_0
                dy_0 *= std_dist / ds_0
                
                dx_1 = dx[cut_idx-1]
                dy_1 = dy[cut_idx-1]
                ds_1 = np.sqrt(dx_1*dx_1 + dy_1*dy_1)
                
                dx_1 *= std_dist / ds_1
                dy_1 *= std_dist / ds_1
                
                x_0, y_0 = x[cut_idx] + dx_0, y[cut_idx] + dy_0
                x_1, y_1 = x[cut_idx-1] + dx_1, y[cut_idx-1] + dy_1
                
                b_labels.append([b, (x_0, y_0),
                                    (x_1, y_1)])
        
        return l_labels, b_labels
    
    def _scale_disp_coords(self, x, y):
        x_sc = self.lb_bounds[0] + (x - self.xy_bounds[0]) * self.x_scale
        y_sc = self.lb_bounds[2] + (y - self.xy_bounds[2]) * self.y_scale
        
        return x_sc, y_sc
    
    def _unscale_lb_coords(self, x, y):
        l_sc = (x - self.lb_bounds[0]) / self.x_scale + self.xy_bounds[0]
        b_sc = (y - self.lb_bounds[2]) / self.y_scale + self.xy_bounds[2]
        
        return l_sc, b_sc
    
    def get_lb_bounds(self):
        '''
        Return the bounds of the map in Galactic coordinates:
            (l_min, l_max, b_min, b_max)
        '''
        
        return self.lb_bounds
    
    def get_xy_bounds(self):
        '''
        Return the display-space bounds of the map:
            (x_min, x_max, y_min, y_max)
        '''
        
        return self.xy_bounds


def plot_graticules(ax, rasterizer, l_lines, b_lines,
                    l_labels=None, b_labels=None,
                    l_formatter=None, b_formatter=None,
                    meridian_style=70., parallel_style='lh',
                    x_excise=5., y_excise=5.,
                    fontsize=8, txt_c='k', txt_path_effects=None,
                    label_pad=5., label_dist=3.5, label_ang_tol=20.,
                    ls='-', thick_c='k', thin_c='k',
                    thick_alpha=0.2, thin_alpha=0.3,
                    thick_lw=1., thin_lw=0.3,
                    return_bbox=False):
                    
    '''
    Plot meridians and parallels.
    '''
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Auto-generate labels, if not given
    if l_formatter == None:
        l_formatter = lambda ll: r'$\,%s%d^{\circ}\,$' % (r'\,\,' if ll >= 0. else '', ll)
    
    if b_formatter == None:
        b_formatter = lambda bb: r'$\,%s%d^{\circ}\,$' % (r'\,\,' if bb >= 0. else '', bb)
    
    if l_labels == None:
        l_labels = [l_formatter(l) for l in l_lines]
    elif (type(l_labels[0]) in [int, float]) or (type(l_labels) == np.ndarray):
        l_labels = [l_formatter(l) for l in l_labels]
    
    if b_labels == None:
        b_labels = [b_formatter(b) for b in b_lines]
    elif (type(b_labels[0]) in [int, float]) or (type(b_labels) == np.ndarray):
        b_labels = [b_formatter(b) for b in b_labels]
    
    # Project meridians and parallels
    x_merid = []
    y_merid = []
    
    for l in l_lines:
        tmp = rasterizer.latlon_lines([l], 0., mode='meridians',
                                             b_spacing = 0.05)
        x_merid.append(tmp[0])
        y_merid.append(tmp[1])
    
    x_para = []
    y_para = []
    
    for b in b_lines:
        tmp = rasterizer.latlon_lines(0., [b], mode='parallels',
                                             l_spacing = 0.05)
        x_tmp = np.array(tmp[0])
        y_tmp = np.array(tmp[1])
        #idx = np.argsort(x_tmp)
        x_para.append(x_tmp)#[idx])
        y_para.append(y_tmp)#[idx])
    
    txt_list = []
    
    bbox = {
        'facecolor': 'w',
        'edgecolor': 'w',
        'alpha': 0.0,
        'boxstyle': 'round,pad={0}'.format(label_pad)
    }
    
    # Plot meridians and parallels
    for grat_style,lab_grat,x_grat,y_grat in zip([meridian_style, parallel_style],
                                                 [l_labels, b_labels],
                                                 [x_merid, x_para],
                                                 [y_merid, y_para]):
        for lab,x,y in zip(lab_grat, x_grat, y_grat):
            for looped, x, y in split_graticule(x, y):
                idx_plot = np.ones(x.size, dtype=np.bool)
                
                if type(grat_style) == str:
                    k_use = []
                    if 'h' in grat_style:
                        k_use.append(0)
                    if 'l' in grat_style:
                        k_use.append(1)
                    
                    if not looped:
                        x_lab, y_lab, ha, va = segment_label_pos(x, y,
                                                                 dist=label_dist,
                                                                 tol=label_ang_tol)
                        for k in k_use:
                            t = ax.text(x_lab[k], y_lab[k], lab,
                                        ha=ha[k], va=va[k], fontsize=fontsize,
                                        color=txt_c, path_effects=txt_path_effects,
                                        bbox=bbox)
                            txt_list.append(t)
                
                elif type(grat_style) in [float, int, complex]:
                    if (not looped) or (type(grat_style) == complex):
                        pct = grat_style
                        
                        if type(pct) == complex:
                            pct = np.imag(pct)
                        
                        x_lab, y_lab, ha, va = segment_label_pos_middle(x, y,
                                                                        pct=pct)
                        for k in range(len(x_lab)):
                            idx_plot = ((x-x_lab)/x_excise)**2. + ((y-y_lab)/y_excise)**2. > 1.
                            #print np.sum(~idx_plot)
                            
                            t = ax.text(x_lab[k], y_lab[k], lab,
                                        ha=ha[k], va=va[k], fontsize=fontsize,
                                        color=txt_c, path_effects=txt_path_effects,
                                        bbox=bbox)
                            txt_list.append(t)
                
                for idx in np.split(np.arange(idx_plot.size), np.nonzero(~idx_plot)[0]):
                    if len(idx) > 1:
                        ax.plot(x[idx], y[idx], ls=ls, c=thick_c, alpha=thick_alpha, lw=thick_lw)
                        ax.plot(x[idx], y[idx], ls=ls, c=thin_c, alpha=thin_alpha, lw=thin_lw)
    
    if return_bbox:
        bounds = [[xlim[0], ylim[0]], [xlim[1], ylim[1]]]
        #print('Original bounds:', bounds)
        
        renderer = ax.get_figure().canvas.get_renderer()
        transf = ax.transData.inverted()
        
        for t in txt_list:
            bb = t.get_window_extent(renderer).transformed(transf).get_points()
            
            bounds[0][0] = max(bounds[0][0], bb[0,0], bb[1,0])
            bounds[1][0] = min(bounds[1][0], bb[0,0], bb[1,0])
            
            bounds[0][1] = min(bounds[0][1], bb[0,1], bb[1,1])
            bounds[1][1] = max(bounds[1][1], bb[0,1], bb[1,1])
        
        #print('Final bounds:', bounds)
        
        bounds = [bounds[0][0], bounds[1][0], bounds[0][1], bounds[1][1]]
        
        return bounds


class PixelIdentifier:
    def __init__(self, ax, rasterizer, lb_bounds=False,
                                       event_type='button_press_event',
                                       event_key=None):
        self.ax = ax
        self.cid = ax.figure.canvas.mpl_connect(event_type, self)
        
        self.rasterizer = rasterizer
        self.lb_bounds = lb_bounds
        self.event_key = event_key
        
        self.objs = []
    
    def __call__(self, event):
        if event.inaxes != self.ax:
            return
        
        if self.event_key != None:
            #print event.key
            if event.key != self.event_key:
                return
        
        # Determine map index of the raster coordinates
        x, y = event.xdata, event.ydata
        
        map_idx = self.rasterizer.xy2idx(x, y, lb_bounds=self.lb_bounds)
        
        #print '(%.2f, %.2f) -> %d' % (x, y, map_idx)
        
        # Pass map index to attached objects
        for obj in self.objs:
            obj(map_idx)
        
        event.key = None
    
    def attach_obj(self, obj):
        self.objs.append(obj)


def stack_highest_res(*maps):
    '''
    Stack maps of different resolutions, taking from the highest-resolution
    map where available, and using lower resolution maps to fill in the
    gaps. If multiple maps of the same resolution are included, which
    map is prioritized in undefined.
    
    Input:
      *maps  HEALPix maps, in nested ordering, in form of numpy arrays.
    
    Output:
       Numpy array containing nested HEALPix map with nside equal to
       highest input nside.
    '''
    
    # Determine nside of each map
    nside = []
    
    for m in maps:
        nside.append(hp.pixelfunc.npix2nside(m.size))
    
    # Order maps from highest to lowest nside
    idx = np.argsort(nside)[::-1]
    
    # Fill in holes using each map
    stack = np.empty(maps[idx[0]].size, dtype = maps[idx[0]].dtype)
    stack[:] = maps[idx[0]][:]
    
    for i in idx:
        fill_idx = np.nonzero(~np.isfinite(stack))[0]
        
        res_ratio = (nside[idx[0]] / nside[i])**2
        take_idx = fill_idx / res_ratio
        
        stack[fill_idx] = maps[i][take_idx]
    
    return stack


def split_graticule(x, y, threshold=5.):
    '''
    Takes the x- and y-coordinates (on the image) of an arc, and splits
    it up into contiguous segments.
    
    Inputs:
      x          x-coordinates (in projected space) of the arc.
      y          y-coordinates ''
      threshold  The higher the threshold, the larger a gap in the
                 arc must be in order to be considered a split. Should
                 always be greater than unity.
    
    Outputs:
      loop     True if there is only one segment, and it is a loop.
      segment  A list of tuples. Each tuple, (x, y), contains the
               x- and y-coordinates of a contiguous segment of the arc.
    '''
    
    if len(x) < 2:
        return []
    
    # Calculate distance between points in graticule.
    # The zeroeth entry is the distance between the first and last
    # elements of the graticule.
    dx = np.hstack([x[-1] - x[0], np.diff(x)])
    dy = np.hstack([y[-1] - y[0], np.diff(y)])
    ds2 = dx**2. + dy**2.
    
    #print 'x:'
    #print x
    #print 'y:'
    #print y
    #print 'ds:'
    #print ds2
    
    # Locate splits, based on distance between points in graticule.
    ds2_min = threshold**2. * np.percentile(ds2, 98.)
    split_idx = np.where(ds2 > ds2_min)[0]
    
    #print 'split_idx:'
    #print split_idx
    
    # If the first and last points are connected, then append the first
    # point to the end of the list
    if len(split_idx) == 0:
        x = np.hstack([x, x[0]])
        y = np.hstack([y, y[0]])
        
        return [(True, x, y)]
    
    x = np.roll(x, -split_idx[0])
    y = np.roll(y, -split_idx[0])
    
    #print split_idx.dtype
    
    split_idx = np.mod(split_idx - split_idx[0], x.size)
    
    #print split_idx
    
    split_idx = np.hstack([split_idx, x.size])
    
    #print split_idx
    
    segments = []
    
    for s0,s1 in zip(split_idx[:-1], split_idx[1:]):
        xx = x[s0:s1]
        yy = y[s0:s1]
        
        dd2 = (xx[0] - xx[-1])**2. + (yy[0] - yy[-1])**2.
        looped = (dd2 <= ds2_min)
        
        if looped:
            xx = np.hstack([xx, xx[0]])
            yy = np.hstack([yy, yy[0]])
        
        if len(xx) > 5:
            segments.append((looped, xx, yy))
    
    #segments = [(x[s0:s1], y[s0:s1]) for s0,s1 in zip(split_idx[:-1], split_idx[1:])]
    
    return segments


def segment_label_pos(x, y, flip_x=True, dist=5., tol=20.):
    dx = np.array([ np.median(np.diff(x[-5:])), -np.median(np.diff(x[:5])) ])
    dy = np.array([ np.median(np.diff(y[-5:])), -np.median(np.diff(y[:5])) ])
    #dx = np.array([x[-1]-x[-2], x[0]-x[1]])
    #dy = np.array([y[-1]-y[-2], y[0]-y[1]])
    theta = np.mod(np.degrees(np.arctan2(dy, dx)), 360.)
    
    ha = []
    va = []
    
    for t in theta:
        if t < tol:
            ha.append('right')
            va.append('center')
        elif t < 90.-tol:
            ha.append('right')
            va.append('bottom')
        elif t < 90.+tol:
            ha.append('center')
            va.append('bottom')
        elif t < 180.-tol:
            ha.append('left')
            va.append('bottom')
        elif t < 180.+tol:
            ha.append('left')
            va.append('center')
        elif t < 270.-tol:
            ha.append('left')
            va.append('top')
        elif t < 270.+tol:
            ha.append('center')
            va.append('top')
        elif t < 360.-tol:
            ha.append('right')
            va.append('top')
        else:
            ha.append('right')
            va.append('center')
        
        #print t, ha[-1], va[-1]
    
    if not flip_x:
        ha_new = ['right' if a=='left' else 'left' for a in ha]
    
    #print('')
    #print('raw dx:', dx)
    #print('raw dy:', dy)
    
    ds = np.sqrt(dx**2. + dy**2.)
    dx *= dist/ds
    dy *= dist/ds
    
    #print('dx:', dx)
    #print('dy:', dy)
    #print('theta:', theta)
    #print('ha:', ha)
    #print('va:', va)
    #print('')
    
    x = np.array([x[-1], x[0]]) + dx
    y = np.array([y[-1], y[0]]) + dy
    
    return x, y, ha, va

def segment_label_pos_v2(x, y, dist=5.):
    dx = np.array([x[-1]-x[-2], x[0]-x[1]])
    dy = np.array([y[-1]-y[-2], y[0]-y[1]])
    ds = np.sqrt(dx**2. + dy**2.)
    dx *= dist/ds
    dy *= dist/ds
    
    
    ha = ['center', 'center']
    va = ['center', 'center']
    
    x = np.array([x[-1], x[0]]) + dx
    y = np.array([y[-1], y[0]]) + dy
    
    return x, y, ha, va

def segment_label_pos_middle(x, y, pct=50.):
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx**2. + dy**2.)
    s = np.hstack([0., np.cumsum(ds)])
    s /= s[-1]
    
    #print ''
    #print 'pct: %.3f' % pct
    #print ''
    #print 's:'
    #print s[::int(s.size/50.)]
    #print ''
    
    idx = np.sum(s < pct/100.)
    
    #print 'idx: %d of %d' % (idx, s.size)
    #print ''
    
    return [x[idx]], [y[idx]], ['center'], ['center']



#
# Tests
#

def test_Mollweide():
    proj = Mollweide_projection()
    
    phi = np.pi * (np.random.random(10) - 0.5)
    lam = 2. * np.pi * (np.random.random(10) - 0.5)
    
    x, y = proj.proj(phi, lam)
    phi_1, lam_1 = proj.inv(x, y)
    
    print 'lat  lon  x    y'
    
    for i in xrange(len(phi)):
        print '%.2f %.2f %.2f %.2f' % (phi[i]*180./np.pi, lam[i]*180./np.pi, x[i], y[i])
    
    print ''
    print "phi  phi'  lam  lam'"
    for i in xrange(len(phi)):
        print '%.2f %.2f %.2f %.2f' % (phi[i], phi_1[i], lam[i], lam_1[i])


def test_EckertIV():
    proj = EckertIV_projection()
    
    phi = np.pi * (np.random.random(10) - 0.5)
    lam = 2. * np.pi * (np.random.random(10) - 0.5)
    
    x, y = proj.proj(phi, lam)
    phi_1, lam_1 = proj.inv(x, y)
    
    iterations = 10
    theta = proj.EckertIV_theta(phi, iterations)
    lhs = theta + np.sin(theta) * np.cos(theta) + 2. * np.sin(theta)
    rhs = (2. + np.pi / 2.) * np.sin(phi)
    
    print 'lat  lon  x    y'
    
    for i in xrange(len(phi)):
        print '%.2f %.2f %.2f %.2f' % (phi[i]*180./np.pi, lam[i]*180./np.pi, x[i], y[i])
    
    print ''
    print "phi  phi'  lam  lam'"
    for i in xrange(len(phi)):
        print '%.2f %.2f %.2f %.2f' % (phi[i], phi_1[i], lam[i], lam_1[i])
    
    
    print ''
    print 'theta  lhs   rhs'
    for t,l,r in zip(theta, lhs, rhs):
        print '%.3f %.3f %.3f' % (t, l, r)
    
    # Find corners
    phi = np.array([0., 0., np.pi/2., -np.pi/2.])
    lam = np.array([0., 2. * np.pi, 0., 0.])
    x, y = proj.proj(phi, lam)
    
    print ''
    print 'x   y'
    for xx,yy in zip(x, y):
        print xx, yy


def test_Cartesian():
    proj = Cartesian_projection()
    
    phi = np.pi * (np.random.random(10) - 0.5)
    lam = 2. * np.pi * (np.random.random(10) - 0.5)
    
    x, y = proj.proj(phi, lam)
    phi_1, lam_1 = proj.inv(x, y)
    
    print 'lat  lon  x    y'
    
    for i in xrange(len(phi)):
        print '%.2f %.2f %.2f %.2f' % (phi[i]*180./np.pi, lam[i]*180./np.pi, x[i], y[i])
    
    print ''
    print "phi  phi'  lam  lam'"
    for i in xrange(len(phi)):
        print '%.2f %.2f %.2f %.2f' % (phi[i], phi_1[i], lam[i], lam_1[i])


def test_Gnomonic():
    proj = Gnomonic_projection(lam_0=30., phi_0=40., fov=90.)
    
    N = 1000
    phi = 1 * np.pi * (np.random.random(N) - 0.5)
    lam = 1 * 2. * np.pi * (np.random.random(N) - 0.5)
    
    x, y, oob1 = proj.proj(phi, lam, ret_bounds=True)
    phi_1, lam_1, oob2 = proj.inv(x, y)
    
    print 'lat  lon  x    y     o.o.b.'
    
    for i in xrange(len(phi)):
        print '%.4f %.4f %.4f %.4f %d' % (phi[i]*180./np.pi, lam[i]*180./np.pi, x[i], y[i], oob1[i])
    
    lam = np.mod(lam, 2.*np.pi)
    lam_1 = np.mod(lam_1, 2.*np.pi)
    
    print ''
    print "phi  phi'  lam  lam'"
    for i in xrange(len(phi)):
        print '%.4f %.4f %.4f %.4f' % (phi[i], phi_1[i], lam[i], lam_1[i])
        print '  %.4f   %.4f   %.2f' % (phi_1[i]/phi[i], lam_1[i]/lam[i], oob1[i])
    
    import matplotlib.pyplot as plt
    fig = plt.figure()
    
    idx = (np.abs(phi_1/phi - 1.) < 1.e-5) & (np.abs(lam_1/lam - 1.) < 1.e-5)
    c = oob1
    
    ax = fig.add_subplot(2,2,1)
    ax.scatter(c, phi_1/phi, c='k')
    #ax.scatter(c, lam_1/lam, c='r')
    
    ax = fig.add_subplot(2,2,2)
    ax.scatter(lam[idx], phi[idx], c='b')
    ax.scatter(lam[~idx], phi[~idx], c='r')
    
    ax = fig.add_subplot(2,2,3)
    ax.scatter(lam, lam_1/lam, c='k')
    
    ax = fig.add_subplot(2,2,4)
    ax.scatter(lam, phi_1/phi, c='k')
    
    plt.show()



def test_proj():
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as PathEffects
    
    nside = 64
    nest = True
    clip = True
    size = (1000, 500)
    #proj = Cartesian_projection()
    #proj = Hammer_projection()
    #proj = Stereographic_projection(fov=60.5)
    proj = Gnomonic_projection(fov=60.5)
    l_cent = 190.
    b_cent = -10.
    
    n_pix = hp.pixelfunc.nside2npix(nside)
    pix_idx = np.arange(n_pix)#[10000:11000]#[4*n_pix/12:5*n_pix/12]
    nside_arr = nside * np.ones(pix_idx.size, dtype='i4')
    l, b = pix2lb(nside, pix_idx, nest=nest)
    
    idx = lb_in_bounds(l, b, [155., 225., -25., 15.])
    l = l[idx]
    b = b[idx]
    pix_idx = pix_idx[idx]
    nside_arr = nside_arr[idx]
    
    pix_val = pix_idx[:].astype('f8')
    
    
    #idx = np.random.randint(n_pix, size=n_pix/4)
    #pix_val[idx] = np.nan
    
    
    # Plot map
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)#, axisbg=(0.6, 0.8, 0.95, 0.95))
    
    # Generate grid lines
    #ls = np.linspace(-180., 180., 13)
    #bs = np.linspace(-90., 90., 7)[1:-1]
    ls = np.arange(-180., 180.1, 45.)
    bs = np.arange(-90., 90.1, 15.)[1:-1]
    
    # Rasterize map
    print 'Constructing rasterizer ...'
    rasterizer = MapRasterizer(nside_arr, pix_idx, size,
                               proj=proj, l_cent=l_cent, b_cent=b_cent,
                               nest=nest)
    
    print 'Rasterizing map ...'
    img = rasterizer(pix_val)
    bounds = rasterizer.get_lb_bounds()
    
    cimg = ax.imshow(img.T, extent=bounds,
                     origin='lower', interpolation='nearest',
                     aspect='auto')
    
    # Plot meridians and parallels
    stroke = [PathEffects.withStroke(linewidth=0.5, foreground='w')]
    plot_graticules(ax, rasterizer, ls, bs,
                    parallel_style=40.j,
                    txt_path_effects=stroke,
                    fontsize=14,
                    x_excise=12., y_excise=2.)
    
    # Color bar
    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.20, top=0.90)
    cax = fig.add_axes([0.10, 0.10, 0.80, 0.05])
    fig.colorbar(cimg, cax=cax, orientation='horizontal')
    
    # Add pixel identifier
    pix_identifier = PixelIdentifier(ax, rasterizer, lb_bounds=True)
    
    ax.axis('off')
    
    plt.show()


def test_rot():
    p = np.array([[0., 45., 90., 135., 180.],
                  [225., 270., 315., 360., 45.]])
    t = 180. * (np.random.random(size=p.shape) - 0.5)
    
    print t
    
    #t *= np.pi / 180.
    #p *= np.pi / 180.
    
    t, p = Euler_rotation_ang(t, p, 90., 45., 0., degrees=True)
    t, p = Euler_rotation_ang(t, p, 90., 45., 0., degrees=True, inverse=True)
    
    #t *= 180. / np.pi
    #p *= 180. / np.pi
    
    print t
    print p


def main():
    #test_Cartesian()
    #test_EckertIV()
    #test_Mollweide()
    #test_Gnomonic()
    test_proj()
    #test_rot()


if __name__ == '__main__':
    main()
