#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  model.py
#  
#  Copyright 2012 Greg Green <greg@greg-UX31A>
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



import sys, argparse
from os.path import abspath, expanduser

import matplotlib as mplib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator
from matplotlib.patches import Rectangle

import numpy as np

import scipy
from scipy.integrate import quad
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline, RectBivariateSpline



class TGalacticModel:
    rho_0 = None
    R0 = None
    Z0 = None
    H1, L1 = None, None
    f, H2, L2 = None, None, None
    fh, qh, nh, fh_outer, nh_outer, Rbr = None, None, None, None, None, None
    H_mu, Delta_mu, mu_FeH_inf = None, None, None
    
    def __init__(self, R0=8000., Z0=25., L1=2150., H1=245., f=0.13,
                       L2=3261., H2=743., fh=0.0051, qh=0.70, nh=-2.62,
                       nh_outer=-3.8, Rbr=27.8, rho_0=0.0058, Rep=500.,
                       H_mu=500., Delta_mu=0.55, mu_FeH_inf=-0.82,
                       LF_fname=expanduser('~/projects/bayestar/data/PSMrLF.dat')):
        self.R0, self.Z0 = R0, Z0
        self.L1, self.H1 = L1, H1
        self.f, self.L2, self.H2 = f, L2, H2
        self.fh, self.qh, self.nh, self.nh_outer, self.Rbr = fh, qh, nh, nh_outer, Rbr*1000.
        self.Rep = Rep
        self.rho_0 = 1. #rho_0
        self.H_mu, self.Delta_mu, self.mu_FeH_inf = H_mu, Delta_mu, mu_FeH_inf
        self.fh_outer = self.fh * (self.Rbr/self.R0)**(self.nh-self.nh_outer)
        self.L_epsilon = 0.
        
        # Drimmel & Spergel (2001)
        self.H_ISM = 134.4
        self.L_ISM = 2260.
        self.dH_dR_ISM = 0.0148
        self.R_flair_ISM = 4400.
        
        self.data = np.loadtxt(abspath(LF_fname),
                               usecols=(0,1),
                               dtype=[('Mr','f4'), ('LF','f4')],
                               unpack=False)
        self.Mr_min = np.min(self.data['Mr'])
        self.Mr_max = np.max(self.data['Mr'])
        self.LF = interp1d(self.data['Mr'], self.data['LF'], kind='linear')
        #self.LF = InterpolatedUnivariateSpline(LF['Mr'], LF['LF'])
        
        # Normalize dN/dV to value from LF in Solar neighborhood
        LF_integral = quad(self.LF, self.Mr_min, self.Mr_max, epsrel=1.e-5, limit=200)[0]
        
        #print LF_integral
        
        self.rho_0 = LF_integral / self.rho_rz(self.R0, self.Z0)
    
    def Cartesian_coords(self, DM, cos_l, sin_l, cos_b, sin_b):
        d = 10.**(DM/5. + 1.)
        x = self.R0 - cos_l*cos_b*d
        y = -sin_l*cos_b*d
        z = sin_b*d
        
        return x, y, z
    
    def Cartesian_2_cylindrical(self, x, y, z):
        r = np.sqrt(x*x + y*y)
        
        return r, z
    
    def gal_2_cylindrical(self, l, b, DM):
        cos_l, sin_l = np.cos(np.pi/180. * l), np.sin(np.pi/180. * l)
        cos_b, sin_b = np.cos(np.pi/180. * b), np.sin(np.pi/180. * b)
        x,y,z = self.Cartesian_coords(DM, cos_l, sin_l, cos_b, sin_b)
        
        return self.Cartesian_2_cylindrical(x, y, z)
    
    def rho_thin(self, r, z):
        r_eff = np.sqrt(r*r + self.L_epsilon*self.L_epsilon)
        
        return (
                self.rho_0
                * np.exp( - (np.abs(z+self.Z0) - np.abs(self.Z0)) / self.H1
                          - (r_eff-self.R0) / self.L1 )
               )
    
    
    def rho_thick(self, r, z):
        r_eff = np.sqrt(r*r + self.L_epsilon*self.L_epsilon)
        
        return (
                self.rho_0 * self.f
                * np.exp( - (np.abs(z+self.Z0) - np.abs(self.Z0)) / self.H2
                          - (r_eff-self.R0) / self.L2 )
               )
    
    def rho_halo(self, r, z):
        r_eff2 = r*r + (z/self.qh)*(z/self.qh) + self.Rep*self.Rep
        
        if type(r_eff2) == np.ndarray:
            ret = np.empty(r_eff2.size, dtype=np.float64)
            
            idx = (r_eff2 <= self.Rbr*self.Rbr)
            
            ret[idx] = self.rho_0 * self.fh * np.power(r_eff2[idx]/self.R0/self.R0, self.nh/2.)
            ret[~idx] = self.rho_0 * self.fh_outer * np.power(r_eff2[~idx]/self.R0/self.R0, self.nh_outer/2.)
            
            return ret
        else:
            if r_eff2 <= self.Rbr*self.Rbr:
                return self.rho_0 * self.fh * (r_eff2/self.R0/self.R0)**(self.nh/2.)
            else:
                return self.rho_0 * self.fh_outer * (r_eff2/self.R0/self.R0)**(self.nh_outer/2.)
    
    def H_ISM_of_R(self, r):
        return self.H_ISM + self.dH_dR_ISM * np.where(r > self.R_flair_ISM, r - self.R_flair_ISM, 0.)
    
    def rho_ISM(self, r, z):
        r_eff = np.sqrt(r*r + self.L_epsilon*self.L_epsilon)
        H = self.H_ISM_of_R(r)
        
        rad_term = np.exp(- r_eff / self.L_ISM)
        h_term = 1. / np.power(np.cosh((z+self.Z0) / H), 2.)
        
        return rad_term * h_term
    
    def f_halo(self, DM, cos_l, sin_l, cos_b, sin_b):
        x,y,z = self.Cartesian_coords(DM, cos_l, sin_l, cos_b, sin_b)
        r = np.sqrt(x*x + y*y)
        
        return self.rho_rz(r, z, component='halo') / self.rho_rz(r, z, component='disk')
    
    def rho_rz(self, r, z, component=None):
        if component == 'disk':
            return self.rho_thin(r,z) + self.rho_thick(r,z)
        elif component == 'thin':
            return self.rho_thin(r,z)
        elif component == 'thick':
            return self.rho_thick(r,z)
        elif component == 'halo':
            return self.rho_halo(r,z)
        else:
            return self.rho_thin(r,z) + self.rho_thick(r,z) + self.rho_halo(r,z)
    
    def rho(self, DM, cos_l, sin_l, cos_b, sin_b, component=None):
        x,y,z = self.Cartesian_coords(DM, cos_l, sin_l, cos_b, sin_b)
        r = np.sqrt(x*x + y*y)
        
        return self.rho_rz(r, z, component=component)
    
    def dn_dDM(self, DM, cos_l, sin_l, cos_b, sin_b, radius=1.,
               component=None, correct=False):
        
        dV_dDM = np.pi * radius**2. * dV_dDM_dOmega(DM)
        
        dN_dDM_tmp = self.rho(DM, cos_l, sin_l, cos_b, sin_b, component) * dV_dDM
        
        if correct:
            return dN_dDM_tmp * self.dn_dDM_corr(DM, m_max)
        else:
            return dN_dDM_tmp
    
    def dn_dDM_corr(self, DM, m_max=23.):
        Mr_max = m_max - DM
        if Mr_max < self.LF['Mr'][0]:
            return 0.
        i_max = np.argmin(np.abs(self.LF['Mr'] - Mr_max))
        return np.sum(self.LF['LF'][:i_max+1])
    
    def mu_FeH_D(self, z):
        return self.mu_FeH_inf + self.Delta_mu*np.exp(-np.abs(z)/self.H_mu)
    
    def p_FeH(self, FeH, DM, cos_l, sin_l, cos_b, sin_b):
        x,y,z = self.Cartesian_coords(DM, cos_l, sin_l, cos_b, sin_b)
        r = np.sqrt(x*x + y*y)
        rho_halo_tmp = self.rho_halo(r,z)
        f_halo = rho_halo_tmp / (rho_halo_tmp + self.rho_thin(r,z) + self.rho_thick(r,z))
        # Disk metallicity
        a = self.mu_FeH_D(z) - 0.067
        p_D = 0.63*Gaussian(FeH, a, 0.2) + 0.37*Gaussian(FeH, a+0.14, 0.2)
        # Halo metallicity
        p_H = Gaussian(FeH, -1.46, 0.3)
        return (1.-f_halo)*p_D + f_halo*p_H
    
    def p_FeH_los(self, FeH, cos_l, sin_l, cos_b, sin_b, radius=1.,
                                              DM_min=-5., DM_max=30.):
        func = lambda x, Z: self.p_FeH(Z, x, cos_l, sin_l, cos_b, sin_b) * self.dn_dDM(x, cos_l, sin_l, cos_b, sin_b, radius)
        normfunc = lambda x: self.dn_dDM(x, cos_l, sin_l, cos_b, sin_b, radius)
        norm = quad(normfunc, DM_min, DM_max, epsrel=1.e-5, full_output=1)[0]
        ret = np.empty(len(FeH), dtype='f8')
        for i,Z in enumerate(FeH):
            ret[i] = quad(func, DM_min, DM_max, args=Z, epsrel=1.e-2, full_output=1)[0]
        return ret / norm
        #
        #return quad(func, DM_min, DM_max, epsrel=1.e-5)[0] / quad(normfunc, DM_min, DM_max, epsrel=1.e-5)[0]
    
    def tot_num_stars(self, l, b, radius=1., component=None):
        radius = np.pi / 180. * radius
        l = np.pi / 180. * l
        b = np.pi / 180. * b
        cos_l, sin_l = np.cos(l), np.sin(l)
        cos_b, sin_b = np.cos(b), np.sin(b)
        
        dN_dDM_func = lambda DM: self.dn_dDM(DM, cos_l, sin_l, cos_b, sin_b,
                                             component=component)
        
        N_tot = np.pi * radius**2. * quad(dN_dDM_func, -5., 30., epsrel=1.e-5, limit=200)[0]
        
        return N_tot
    
    def dA_dmu(self, l, b, DM):
        r, z = self.gal_2_cylindrical(l, b, DM)
        
        return self.rho_ISM(r, z) * np.power(10., DM / 5.)
    
    def EBV_prior(self, l, b, n_regions=20, EBV_per_kpc=0.04, norm_dist=1.):
        mu_0, mu_1 = 4., 19.
        
        DM = np.linspace(mu_0, mu_1, n_regions+1)
        Delta_DM = DM[1] - DM[0]
        Delta_EBV = np.empty(n_regions+1, dtype='f8')
        
        # Integrate from d = 0 to beginning of first distance bin
        DM_fine = np.linspace(mu_0 - 10., mu_0, 1000)
        Delta_EBV[0] = np.sum(self.dA_dmu(l, b, DM_fine)) * (10. / 1000.)
        
        # Find Delta EBV in each distance bin
        DM_fine = np.linspace(mu_0, mu_1, 64 * n_regions)
        Delta_EBV_tmp = self.dA_dmu(l, b, DM_fine)
        Delta_EBV[1:] = downsample_by_four(downsample_by_four(downsample_by_four(Delta_EBV_tmp))) * Delta_DM
        
        # 1.5 orders of magnitude variance
        #std_dev_coeff = np.array([3.6506, -0.047222, -0.021878, 0.0010066, -7.6386e-06])
        #mean_bias_coeff = np.array([0.57694, 0.037259, -0.001347, -4.6156e-06])
        
        # 1 order of magnitude variance
        std_dev_coeff = np.array([2.4022, -0.040931, -0.012309, 0.00039482, 3.1342e-06])
        mean_bias_coeff = np.array([0.52751, 0.022036, -0.0010742, 7.0748e-06])
        
        # Calculate bias and std. dev. of reddening in each bin
        dist = np.power(10., DM / 5. + 1.) # in pc
        Delta_dist = np.hstack([dist[0], np.diff(dist)])
        DM_equiv = 5. * (np.log10(Delta_dist) - 1.)
        
        bias = (mean_bias_coeff[0] * DM_equiv
              + mean_bias_coeff[1] * DM_equiv * DM_equiv
              + mean_bias_coeff[2] * DM_equiv * DM_equiv * DM_equiv
              + mean_bias_coeff[3] * DM_equiv * DM_equiv * DM_equiv * DM_equiv)
        
        sigma = (std_dev_coeff[0]
               + std_dev_coeff[1] * DM_equiv
               + std_dev_coeff[2] * DM_equiv * DM_equiv
               + std_dev_coeff[3] * DM_equiv * DM_equiv * DM_equiv
               + std_dev_coeff[4] * DM_equiv * DM_equiv * DM_equiv * DM_equiv)
        
        sigma = 2. * np.ones(DM_equiv.size)
        bias = 0.
        
        log_Delta_EBV = np.log(Delta_EBV) + bias
        
        # Calculate mean reddening in each bin
        mean_Delta_EBV = Delta_EBV * np.exp(bias + 0.5 * sigma * sigma)
        mean_EBV = np.cumsum(mean_Delta_EBV)
        
        # Normalize E(B-V) per kpc locally
        DM_norm = 5. * (np.log10(norm_dist) - 1.)
        DM_fine = np.linspace(DM_norm - 10., DM_norm, 100)
        Delta_DM = DM_fine[1] - DM_fine[0]
        EBV_local = np.sum(self.dA_dmu(l, b, DM_fine)) * Delta_DM
        EBV_local *= np.exp(0.5 * std_dev_coeff[0]**2.)
        norm = 0.001 * EBV_per_kpc * norm_dist / EBV_local
        
        #idx = np.max(np.where(dist <= norm_dist, np.arange(dist.size), -1.))
        #print idx
        #norm = 0.001 * EBV_per_kpc * dist[idx] / mean_EBV[idx]
        
        mean_EBV *= norm
        mean_Delta_EBV *= norm
        
        log_Delta_EBV += np.log(norm)
        
        return DM, log_Delta_EBV, sigma, mean_Delta_EBV, norm


def downsample_by_four(x):
    return 0.25 * (x[::4] + x[1::4] + x[2::4] + x[3::4])

def downsample_by_two(x):
    return 0.5 * (x[:-1:2] + x[1::2])


def dV_dDM_dOmega(DM):
    '''
    Volume element of a beam at a given distance, per unit
    distance modulus, per steradian.
    '''
    
    return (1000.*2.30258509/5.) * np.exp(3.*2.30258509/5. * DM)


def Gaussian(x, mu=0., sigma=1.):
    Delta = (x-mu)/sigma
    return np.exp(-Delta*Delta/2.) / 2.50662827 / sigma


class TStellarModel:
    '''
    Loads the given stellar model, and provides access to interpolated
    colors on (Mr, FeH) grid.
    '''
    
    def __init__(self, template_fname):
        self.load_templates(template_fname)
    
    def load_templates(self, template_fname):
        '''
        Load in stellar template colors from an ASCII file. The colors
        should be stored in the following format:
        
        #
        # Arbitrary comments
        #
        # Mr    FeH   gr     ri     iz     zy
        # 
        -1.00 -2.50 0.5132 0.2444 0.1875 0.0298
        -0.99 -2.50 0.5128 0.2442 0.1873 0.0297
        ...
        
        or something similar. A key point is that there be a row
        in the comments that lists the names of the colors. The code
        identifies this row by the presence of both 'Mr' and 'FeH' in
        the row, as above. The file must be whitespace-delimited, and
        any whitespace will do (note that the whitespace is not required
        to be regular).
        '''
        
        f = open(abspath(template_fname), 'r')
        row = []
        self.color_name = ['gr', 'ri', 'iz', 'zy']
        for l in f:
            line = l.rstrip().lstrip()
            if len(line) == 0:    # Empty line
                continue
            if line[0] == '#':    # Comment
                if ('Mr' in line) and ('FeH' in line):
                    try:
                        self.color_name = line.split()[3:]
                    except:
                        pass
                continue
            data = line.split()
            if len(data) < 6:
                print 'Error reading in stellar templates.'
                print 'The following line does not have the correct number of entries (6 expected):'
                print line
                return 0
            row.append([float(s) for s in data])
        f.close()
        template = np.array(row, dtype=np.float64)
        
        # Organize data into record array
        dtype = [('Mr','f4'), ('FeH','f4')]
        for c in self.color_name:
            dtype.append((c, 'f4'))
        self.data = np.empty(len(template), dtype=dtype)
        
        self.data['Mr'] = template[:,0]
        self.data['FeH'] = template[:,1]
        for i,c in enumerate(self.color_name):
            self.data[c] = template[:,i+2]
        
        self.MrFeH_bounds = [[np.min(self.data['Mr']), np.max(self.data['Mr'])],
                             [np.min(self.data['FeH']), np.max(self.data['FeH'])]]
        
        # Produce interpolating class with data
        self.Mr_coords = np.unique(self.data['Mr'])
        self.FeH_coords = np.unique(self.data['FeH'])
        
        self.interp = {}
        for c in self.color_name:
            tmp = self.data[c][:]
            tmp.shape = (len(self.FeH_coords), len(self.Mr_coords))
            self.interp[c] = RectBivariateSpline(self.Mr_coords,
                                                 self.FeH_coords,
                                                 tmp.T,
                                                 kx=3,
                                                 ky=3,
                                                 s=0)
    
    def color(self, Mr, FeH, name=None):
        '''
        Return the colors, evaluated at the given points in
        (Mr, FeH)-space.
        
        Inputs:
            Mr    float or array of floats
            FeH   float or array of floats
            name  string, or list of strings, with names of colors to
                  return. By default, all colors are returned.
        
        Output:
            color  numpy record array of colors
        '''
        
        if name == None:
            name = self.get_color_names()
        elif type(name) == str:
            name = [name]
        
        if type(Mr) == float:
            Mr = np.array([Mr])
        elif type(Mr) == list:
            Mr = np.array(Mr)
        if type(FeH) == float:
            FeH = np.array([FeH])
        elif type(FeH) == list:
            FeH = np.array(FeH)
        
        dtype = []
        for c in name:
            if c not in self.color_name:
                raise ValueError('No such color in model: %s' % c)
            dtype.append((c, 'f4'))
        ret_color = np.empty(Mr.size, dtype=dtype)
        
        for c in name:
            ret_color[c] = self.interp[c].ev(Mr, FeH)
        
        return ret_color
    
    def absmags(self, Mr, FeH):
        '''
        Return the absolute magnitude in each bandpass corresponding to
        (Mr, FeH).
        
        Inputs:
            Mr   r-band absolute magnitude of the star(s) (float or numpy array)
            FeH  Metallicity of the star(s) (float or numpy array)
        
        Output:
            M    Absolute magnitude in each band for each star (numpy record array)
        '''
        
        c = self.color(Mr, FeH)
        
        dtype = [('g','f8'), ('r','f8'), ('i','f8'), ('z','f8'), ('y','f8')]
        M = np.empty(c.shape, dtype=dtype)
        
        M['r'] = Mr
        M['g'] = c['gr'] + Mr
        M['i'] = Mr - c['ri']
        M['z'] = M['i'] - c['iz']
        M['y'] = M['z'] - c['zy']
        
        return M
    
    def get_color_names(self):
        '''
        Return the names of the colors in the templates.
        
        Ex.: For PS1 colors, this would return
             ['gr', 'ri', 'iz', 'zy']
        '''
        
        return self.color_name


def get_SFD_map(fname='~/projects/bayestar/data/SFD_Ebv_512.fits', nside=64):
    import pyfits
    import healpy as hp
    
    fname = expanduser(fname)
    
    f = pyfits.open(fname)
    EBV_ring = f[0].data[:]
    f.close()
    
    EBV_nest = hp.reorder(EBV_ring, r2n=True)
    
    nside2_map = EBV_nest.size / 12
    
    while nside2_map > nside * nside:
        EBV_nest = downsample_by_four(EBV_nest)
        nside2_map = EBV_nest.size / 12
    
    #hp.mollview(np.log10(EBV_nest), nest=True)
    #plt.show()
    
    return EBV_nest

def min_max(x):
    return np.min(x), np.max(x)

def plot_EBV_prior(model, nside=64):
    import healpy as hp
    
    n = np.arange(12 * nside**2)
    EBV = np.empty(n.size)
    norm = np.empty(n.size)
    
    for i in n:
        t, p = hp.pixelfunc.pix2ang(nside, i, nest=True)
        l = 180./np.pi * p
        b = 90. - 180./np.pi * t
        
        DM, log_Delta_EBV, sigma_log_Delta_EBV, mean_Delta_EBV, norm_tmp = model.EBV_prior(l, b)
        
        EBV[i] = np.sum(mean_Delta_EBV)
        norm[i] = norm_tmp
        
        #print '(%.3f, %.3f): %.3f' % (l, b, EBV[i])
    
    print ''
    print np.mean(norm), np.std(norm)
    
    # Compare to SFD
    EBV_SFD = get_SFD_map(nside=nside)
    
    #print np.mean(EBV_SFD / EBV)
    #print np.mean(EBV_SFD) / np.mean(EBV)
    #print np.std(EBV_SFD), np.std(EBV)
    
    # Normalize for b < 10, l > 10
    t, p = hp.pixelfunc.pix2ang(nside, n, nest=True)
    l = 180./np.pi * p
    b = 90. - 180./np.pi * t
    
    idx = (b < 15.) & (l > 10.)
    norm = np.median(EBV_SFD[idx]) / np.median(EBV[idx])
    EBV *= norm
    
    print 'Normalization = %.4g' % norm
    
    vmin, vmax = min_max(np.log10(EBV_SFD))
    
    mplib.rc('text', usetex=True)
    hp.visufunc.mollview(np.log10(EBV), nest=True,
            title=r'$\log_{10} \mathrm{E} \left( B - V \right)$',
            min=vmin, max=vmax)
    hp.visufunc.mollview(np.log10(EBV_SFD), nest=True,
            title=r'$\log_{10} \mathrm{E} \left( B - V \right)_{\mathrm{SFD}}$',
            min=vmin, max=vmax)
    #hp.visufunc.mollview(EBV, nest=True, max=10.)
    hp.visufunc.mollview(np.log10(EBV) - np.log10(EBV_SFD), nest=True)
    
    plt.show()

def test_EBV_prior():
    model = TGalacticModel()
    
    l, b, radius = 0., 0., 0.1
    
    '''
    N_thin = model.tot_num_stars(l, b, radius=radius, component='thin')
    N_thick = model.tot_num_stars(l, b, radius=radius, component='thick')
    N_halo = model.tot_num_stars(l, b, radius=radius, component='halo')
    
    print '# of stars in thin disk: %d' % N_thin
    print '     "     in thick disk: %d' % N_thick
    print '     "     in halo: %d' % N_halo
    '''
    
    DM, log_Delta_EBV, sigma_log_Delta_EBV, mean_Delta_EBV, norm = model.EBV_prior(l, b)
    
    dist = np.power(10., DM / 5. - 2.)  # in kpc
    
    for d, mean, sigma in zip(dist, log_Delta_EBV, sigma_log_Delta_EBV):
        print 'd = %.3f: %.3f +- %.3f --> Delta E(B-V) = %.3f' % (d, mean, sigma, np.exp(mean + 0.5 * sigma * sigma))
    
    print 'E(B-V) = %.3f' % np.sum(mean_Delta_EBV)
    
    plot_EBV_prior(model)

def plot_EBV_prior_profile():
    model = TGalacticModel()
    
    l, b = 0., 0.
    s = np.linspace(0., 16000., 1000)
    
    DM = 5. * np.log10(s/10.)
    r, z = model.gal_2_cylindrical(l, b, DM)
    
    dA_ds = model.rho_ISM(r, z)
    A = np.cumsum(dA_ds)
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(s, A)
    ax.plot(s, dA_ds * np.max(A) / np.max(dA_ds))
    plt.show()

def main():
    plot_EBV_prior_profile()
    test_EBV_prior()
    
    return 0

if __name__ == '__main__':
    main()

