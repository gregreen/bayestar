import numpy
import pdb
import collections

filt = ['g','r','i','z','y']
filtind = dict([(f,i) for i,f in enumerate(filt)])
indfilt = dict([(i,f) for i,f in enumerate(filt)])
filt2sdss = dict(((f+'.0000', i+1 if i < 4 else 4) for i,f in enumerate(filt)))
# canonical colors to draw these filters in
filtcolor = collections.OrderedDict([ ('g','purple'), ('r', 'blue'),
                                      ('i', 'green'),
                                      ('z', 'orange'), ('y', 'red') ])


md_fieldcenters = {
    'md00':( 10.675,  41.267),
    'md01':( 35.875,  -4.250),
    'md02':( 53.100, -27.800),
    'md03':(130.592,  44.317),
    'md04':(150.000,   2.200),
    'md05':(161.917,  58.083),
    'md06':(185.000,  47.117),
    'md07':(213.704,  53.083),
    'md08':(242.787,  54.950),
    'md09':(334.188,   0.283),
    'md10':(352.312,  -0.433),
    'md11':(270.000,  66.561) }

flags = {
'DEFAULT'          	  : 0x00000000, # Initial value: resets all bits
'PSFMODEL'         	  : 0x00000001, # Source fitted with a psf model (linear or non-linear)
'EXTMODEL'         	  : 0x00000002, # Source fitted with an extended-source model
'FITTED'           	  : 0x00000004, # Source fitted with non-linear model (PSF or EXT; good or bad)
'FITFAIL'             	  : 0x00000008, # Fit (non-linear) failed (non-converge, off-edge, run to zero)
'POORFIT'             	  : 0x00000010, # Fit succeeds, but low-SN, high-Chisq, or large (for PSF -- drop?)
'PAIR'             	  : 0x00000020, # Source fitted with a double psf
'PSFSTAR'          	  : 0x00000040, # Source used to define PSF model
'SATSTAR'          	  : 0x00000080, # Source model peak is above saturation
'BLEND'            	  : 0x00000100, # Source is a blend with other sources
'EXTERNALPOS'         	  : 0x00000200, # Source based on supplied input position
'BADPSF'           	  : 0x00000400, # Failed to get good estimate of object's PSF
'DEFECT'           	  : 0x00000800, # Source is thought to be a defect
'SATURATED'        	  : 0x00001000, # Source is thought to be saturated pixels (bleed trail)
'CR_LIMIT'         	  : 0x00002000, # Source has crNsigma above limit
'EXT_LIMIT'        	  : 0x00004000, # Source has extNsigma above limit
'MOMENTS_FAILURE'  	  : 0x00008000, # could not measure the moments
'SKY_FAILURE'      	  : 0x00010000, # could not measure the local sky
'SKYVAR_FAILURE'   	  : 0x00020000, # could not measure the local sky variance
'BELOW_MOMENTS_SN' 	  : 0x00040000, # moments not measured due to low S/N
'BIG_RADIUS'       	  : 0x00100000, # poor moments for small radius, try large radius
'AP_MAGS'          	  : 0x00200000, # source has an aperture magnitude
'BLEND_FIT'        	  : 0x00400000, # source was fitted as a blend
'EXTENDED_FIT'     	  : 0x00800000, # full extended fit was used
'EXTENDED_STATS'   	  : 0x01000000, # extended aperture stats calculated
'LINEAR_FIT'       	  : 0x02000000, # source fitted with the linear fit
'NONLINEAR_FIT'    	  : 0x04000000, # source fitted with the non-linear fit
'RADIAL_FLUX'      	  : 0x08000000, # radial flux measurements calculated
'SIZE_SKIPPED'     	  : 0x10000000, # size could not be determined
'ON_SPIKE'         	  : 0x20000000, # peak lands on diffraction spike
'ON_GHOST'         	  : 0x40000000, # peak lands on ghost or glint
'OFF_CHIP'         	  : 0x80000000, # peak lands off edge of chip
}
for x,y in flags.items():
    flags[y] = x

# definition of flags2 (see http://svn.pan-starrs.ifa.hawaii.edu/trac/ipp/wiki/CMF_PS1_V3)
flags2 = {
'DEFAULT'          	  : 0x00000000, # Initial value: resets all bits
'DIFF_WITH_SINGLE' 	  : 0x00000001, # diff source matched to a single positive detection
'DIFF_WITH_DOUBLE' 	  : 0x00000002, # diff source matched to positive detections in both images
'MATCHED'          	  : 0x00000004, # diff source matched to positive detections in both images
'ON_SPIKE'         	  : 0x00000008, # > 25% of (PSF-weighted) pixels land on diffraction spike
'ON_STARCORE'      	  : 0x00000010, # > 25% of (PSF-weighted) pixels land on starcore
'ON_BURNTOOL'      	  : 0x00000020, # > 25% of (PSF-weighted) pixels land on burntool
'ON_CONVPOOR'      	  : 0x00000040, # > 25% of (PSF-weighted) pixels land on convpoor
'PASS1_SRC'               : 0x00000080, # source detected in first pass analysis
'HAS_BRIGHTER_NEIGHBOR'   : 0x00000100, # peak is not the brightest in its footprint
'BRIGHT_NEIGHBOR_1'       : 0x00000200, # flux_n / (r^2 flux_p) > 1
'BRIGHT_NEIGHBOR_10'      : 0x00000400, # flux_n / (r^2 flux_p) > 10
'DIFF_SELF_MATCH'  	  : 0x00000800, # positive detection match is probably this source 
'SATSTAR_PROFILE'         : 0x00001000, # saturated source is modeled with a radial profile
}
for x,y in flags2.items():
    flags2[y] = x


pssdsstransformdict = {
    ('g-g', 'g-r'): [(-.01417, -.1543), [None, None]],
    ('r-r', 'r-i'):
    {'giant': [(-.001171, -.02429, -.01060), [None, 1.6]],
     'dwarf':[(-.001283, -.01704, .005772), [None, None]]
     }, 
    ('i-i', 'r-i'):   [(.001424, -.03279), [None, None]],
    ('z-z', 'i-z'):   [(-.003029, 0.1174, -.01855, 0.01579), [None, None]],
    ('y-z', 'i-z'):   [(0.01479, -.3777, 0.1078, -.03996), [None, None]],
    ('w-r', 'r-i'):   [(0.05081, 0.1362, -.5670, 0.1384, -.01127),
                              [None, None]]
    }

pssdsstransformdictjt2012 = {
    ('g-g', 'g-r'): [(-.011, -.125, -0.015), [None, None]],
    ('r-r', 'g-r'): [(.001, -.006, -.002), [None, None]],
    ('i-i', 'g-r'): [(.004, -0.014, 0.001), [None, None]],
    ('z-z', 'g-r'): [(-0.013, 0.040, -0.001), [None, None]],
    ('y-z', 'g-r'): [(0.031, -.106, 0.01), [None, None]],
    ('w-r', 'g-r'): [(0.018, 0.118, -0.091), [None, None]]
    }

pssdsstransformdictdpfpreabscal = {
    ('g-g', 'g-i'): [(-0.01710,-0.10915, 0.00540, 0.00126), [None, None]],
    ('r-r', 'g-i'): [( 0.01062,-0.02579, 0.01729,-0.00324), [None, None]],
    ('i-i', 'g-i'): [(-0.00241, 0.00309,-0.00294, 0.00027), [None, None]],
    ('z-z', 'g-i'): [(-0.01810, 0.06830,-0.03026, 0.00696), [None, None]],
    ('y-z', 'g-i'): [( 0.08149,-0.16694, 0.06876,-0.01441), [None, None]]
}

pssdsstransformdictdpf = {
    ('g-g', 'g-i'): [( 0.00128,-0.10699, 0.00392, 0.00152), [None, None]],
    ('r-r', 'g-i'): [(-0.00518,-0.03561, 0.02359,-0.00447), [None, None]],
    ('i-i', 'g-i'): [( 0.00585,-0.01287, 0.00707,-0.00178), [None, None]],
    ('z-z', 'g-i'): [( 0.00144, 0.07379,-0.03366, 0.00765), [None, None]],
    ('y-z', 'g-i'): [( 0.10403,-0.18755, 0.08333,-0.01800), [None, None]]
}

def pssdsstransform(filt, sdsscolor, color, lumclass=None, return_poly=False,
                    mode='dpf'):
    if mode == 'dpf':
        transformdict = pssdsstransformdictdpf
    else:
        transformdict = pssdsstransformdict
    trans = transformdict[(filt, sdsscolor)]
    if type(trans) == type(dict()):
        if lumclass is None:
            trans = trans['dwarf']
        else:
            trans = trans[lumclass]
    bounds = trans[1]
    lb = bounds[0] if bounds[0] is not None else -numpy.inf
    ub = bounds[1] if bounds[1] is not None else numpy.inf
    color2 = numpy.atleast_1d(color).copy()
    m = (color2 > ub) | (color2 < lb)
    color2[m] = numpy.nan
    terms = numpy.atleast_1d(trans[0])
    correction = numpy.polyval(numpy.flipud(terms), color2)
    if not return_poly:
        return correction
    return correction, terms

def pssdsstransformall(sdss, mode='dpf'):
    """ Get PS magnitudes from SDSS magnitudes."""
    psmag = numpy.zeros((len(sdss), 5), dtype='f4')
    if mode=='dpf':
        gi = sdss['g']-sdss['i']
        psmag[:,0] = sdss['g']+pssdsstransform('g-g', 'g-i', gi, mode=mode)
        psmag[:,1] = sdss['r']+pssdsstransform('r-r', 'g-i', gi, mode=mode)
        psmag[:,2] = sdss['i']+pssdsstransform('i-i', 'g-i', gi, mode=mode)
        psmag[:,3] = sdss['z']+pssdsstransform('z-z', 'g-i', gi, mode=mode)
        psmag[:,4] = sdss['z']+pssdsstransform('y-z', 'g-i', gi, mode=mode)
    else:
        psmag[:,0] = sdss['g']+pssdsstransform('g-g','g-r',sdss['g']-sdss['r'],
                                               mode=mode)
        psmag[:,1] = sdss['r']+pssdsstransform('r-r','r-i',sdss['r']-sdss['i'],
                                               mode=mode)
        psmag[:,2] = sdss['i']+pssdsstransform('i-i','r-i',sdss['r']-sdss['i'],
                                               mode=mode)
        psmag[:,3] = sdss['z']+pssdsstransform('z-z','i-z',sdss['i']-sdss['z'],
                                               mode=mode)
        psmag[:,4] = sdss['z']+pssdsstransform('y-z','i-z',sdss['i']-sdss['z'],
                                               mode=mode)
    return psmag

def pssdsstransformerr(sdss, cliperr=.2, mode='dpf'):
    """ Get PS magnitudes from SDSS magnitudes, with errors."""
    psmag = numpy.zeros((len(sdss), 5), dtype='f4')
    psmagerr = numpy.zeros_like(psmag)
    if len(psmag) == 0:
        return psmag, psmagerr
    if mode == 'dpf':
        basecolor, corfun, corcolor = (('g',  'r',  'i',  'z',  'z'),
                                       ('gg', 'rr', 'ii', 'zz', 'yz'),
                                       ('gi', 'gi', 'gi', 'gi', 'gi'))
    else:
        basecolor, corfun, corcolor = (('g',  'r',  'i',  'z',  'z'),
                                       ('gg', 'rr', 'ii', 'zz', 'yz'),
                                       ('gr', 'ri', 'ri', 'iz', 'iz'))
    for i,dat in enumerate(zip(basecolor, corfun, corcolor)):
        f, cc, col = dat
        cn1, cn2 = (cc[0]+'-'+cc[1], col[0]+'-'+col[1])
        tcol = sdss[col[0]]-sdss[col[1]]
        cor, poly = pssdsstransform(cn1, cn2, tcol, return_poly=True,
                                    mode=mode)
        tcol[~numpy.isfinite(cor)] = numpy.nan
        psmag[:,i] = sdss[f]+cor
        dpoly = (numpy.arange(len(poly))*poly)[1:]
        dfpoly = numpy.polyval(numpy.flipud(dpoly), tcol)
        # must compute df, C
        df = numpy.zeros((len(cor), 3), dtype='f4')
        df[:,0] = 1.
        df[:,1] = dfpoly
        df[:,2] = -dfpoly
        c = numpy.zeros((len(df),3,3))
        olderr = numpy.seterr(invalid='ignore')
        filt = [f, col[0], col[1]]
        for ind1, f1 in enumerate(filt):
            err = sdss[f1+'Err']
            err[err > cliperr] = numpy.inf
            for ind2, f2 in enumerate(filt):
                c[:,ind1,ind2] = err**2*(f1 == f2)
        tdot = numpy.tensordot
        from numpy.core.umath_tests import matrix_multiply as mm
        err = numpy.sqrt(mm(mm(df.reshape(df.shape[0], 1, 3), c),
                            df.reshape(df.shape+(1,))))
        numpy.seterr(**olderr)
        err = err.reshape(len(err))
        psmagerr[:,i] = err
    return psmag, psmagerr

jttranscol = ['exp(C)', 'Z', 'A', 'P', 'H_0', 'H_1', 'H_2', 'err']
jttransrow = 'grizywo'
jttransdat = (
    [[ 0.203, 0.983, 0.257, -0.020, 0.001, -0.000, 0.000, 1.1],
     [ 0.123, 0.975, 0.317, -0.012, 0.012, -0.001, 0.005, 1.3],
     [ 0.091, 0.838, 0.339, -0.005, 0.120, -0.010, 0.034, 2.0],
     [ 0.060, 0.881, 0.408,  0.004, 0.312, -0.066, 0.054, 4.1],
     [ 0.154, 0.686, 0.164, -0.014, 0.542, -0.085, 0.028, 3.3],
     [ 0.138, 0.936, 0.292, -0.073, 0.029, -0.002, 0.009, 1.7],
     [ 0.135, 0.897, 0.276, -0.107, 0.094, -0.018, 0.020, 4.5]])

def jttrans(filt, a=1, h=1, p=1, z=1):
    a = numpy.atleast_1d(a)
    h = numpy.atleast_1d(h)
    p = numpy.atleast_1d(p)
    ln = numpy.log
    ind = jttransrow.find(filt[0])
    if ind == -1:
        raise ValueError('Could not find filt = %s', filt)
    dat = jttransdat[ind]
    eC, Z, A, P, H0, H1, H2, err = dat
    lne = ln(eC) + Z*ln(z) + A*ln(a) + P*ln(p) + ln(h)*(H0+H1*ln(z)+H2*ln(h))
    return numpy.exp(lne)

def jttrans_to_kterm(a=.7, h=.7, p=1., z=1.2):
    de = numpy.zeros(5)
    dz = 0.01
    for i,f in enumerate('grizy'):
        de[i] = (jttrans(f, a=a, h=h, p=p, z=z+dz)-
                 jttrans(f, a=a, h=h, p=p, z=z))
    return de/dz

#ri gr iz zy zJ zH yJ wr Or
knotcolors = ['ri', 'gr', 'iz', 'zy', 'zJ', 'zH', 'yJ', 'wr', 'Or']
slknots = numpy.array(
    [[-0.4, -0.50, -0.290, -0.210, 0.12, 0.05, 0.34, -0.085,  0.015],
     [-0.2, -0.19, -0.110, -0.050, 0.48, 0.50, 0.50,  0.000,  0.070],
     [ 0.0,  0.15, -0.030, -0.025, 0.70, 0.87, 0.70,  0.050,  0.060],
     [ 0.2,  0.55,  0.090,  0.035, 0.89, 1.28, 0.86,  0.070, -0.010],
     [ 0.4,  0.97,  0.200,  0.095, 1.14, 1.82, 1.00,  0.045, -0.120],
     [ 0.6,  1.16,  0.295,  0.140, 1.22, 1.96, 1.11, -0.030, -0.280],
     [ 1.0,  1.20,  0.470,  0.195, 1.31, 2.00, 1.10, -0.245, -0.670],
     [ 2.0,  1.26,  0.940,  0.470, 1.23, 2.12, 0.87, -0.940, -1.820]])

def stellar_locus_tonry(ri, outcolor):
    try:
        ind = knotcolors.index(outcolor)
    except:
        raise ValueError('unknown outcolor %s', outcolor)
    riknots = slknots[:,0]
    outknots = slknots[:,ind]
    #from scipy.interpolate import interpolate
    import cubicSpline
    spl = cubicSpline.NaturalCubicSpline(riknots, outknots)
    return spl(ri)
    #return interpolate.spline(riknots, outknots, ri, kind='natural', order=3)
    #spl = interpolate.splmake(riknots, outknots, kind='natural')
    
def clean(tflags):
    badflags = (flags['FITFAIL'] | flags['POORFIT'] |
            flags['SATSTAR'] | flags['BLEND'] |
            flags['BADPSF'] | flags['DEFECT'] | flags['SATURATED'] |
            flags['CR_LIMIT'] | # flags['EXT_LIMIT'] |
            flags['MOMENTS_FAILURE'] | flags['SKY_FAILURE'] |
            flags['SKYVAR_FAILURE'] | flags['BIG_RADIUS'] |
            #flags['SIZE_SKIPPED'] |
            flags['ON_SPIKE'] |
            flags['ON_GHOST'] | flags['OFF_CHIP'])
    return (tflags & badflags) == 0
# EXT_LIMIT commented out

def badflags_eam():
    badflagstr = ('FITFAIL POORFIT PAIR SATSTAR BLEND BADPSF DEFECT SATURATED '+
                  'CR_LIMIT MOMENTS_FAILURE SKY_FAILURE SKYVAR_FAILURE '+
                  'BELOW_MOMENTS_SN BLEND_FIT SIZE_SKIPPED ON_SPIKE ON_GHOST '+
                  'OFF_CHIP')
    badflags = 0
    for s in badflagstr.split():
        badflags |= flags[s]
    badflag2str = ('HAS_BRIGHTER_NEIGHBOR BRIGHT_NEIGHBOR_1 BRIGHT_NEIGHBOR_10')
    badflags2 = 0
    for s in badflag2str.split():
        badflags2 |= flags2[s]
    return badflags, badflags2

def rdm2airmass(ra, dec, mjd_obs):
    import util_efs
    lat = 20.7070999146
    lon = -156.2559127815 # PS1 coordinates
    alt, az = util_efs.rdllmjd2altaz(ra, dec, lat, lon, mjd_obs)
    am = util_efs.alt2airmass(alt)
    m = (am < 0)
    am[m] = numpy.nan
    return am
