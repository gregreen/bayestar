import numpy as np, h5py, tempfile, subprocess

def write_infile(filename, mag, err, maglimit,
                 l=90., b=10., EBV_guess=2., access_mode='a'):
	# Prepare output for pixel
	n_stars = len(mag)
	
	data = np.empty(n_stars, dtype=[('obj_id','u8'),
	                                ('l','f8'), ('b','f8'), 
	                                ('mag','f4',5), ('err','f4',5),
	                                ('maglimit','f4',5),
	                                ('nDet','u4',5),
	                                ('EBV','f4')])
	
	data['obj_id'][:] = np.arange(n_stars)
	data['l'][:] = l
	data['b'][:] = b
	data['mag'][:] = mag[:]
	data['err'][:] = err[:]
	data['maglimit'][:] = maglimit[:]
	data['nDet'][:] = 1
	data['EBV'][:] = EBV_guess
	
	f = h5py.File(filename, access_mode)
	
	pixIdx = 1
	ds_name = '/photometry/pixel %d' % pixIdx
	ds = f.create_dataset(ds_name, data.shape, data.dtype, chunks=True,
	                      compression='gzip', compression_opts=9)
	ds[:] = data[:]
	
	gal_lb = np.array([l, b], dtype='f8')
	
	nside = 512
	nest = True
	EBV = EBV_guess
	att_f8 = np.array([EBV], dtype='f8')
	att_u8 = np.array([pixIdx], dtype='u8')
	att_u4 = np.array([nside], dtype='u4')
	att_u1 = np.array([nest], dtype='u1')
	
	ds.attrs['healpix_index'] = att_u8[0]
	ds.attrs['nested'] = att_u1[0]
	ds.attrs['nside'] = att_u4[0]
	ds.attrs['l'] = gal_lb[0]
	ds.attrs['b'] = gal_lb[1]
	ds.attrs['EBV'] = att_f8[0]
	
	f.close()


def write_true_params(filename, DM, EBV, Mr, FeH,
                      l=90., b=10., access_mode='a'):
	# Prepare output for pixel
	n_stars = len(DM)
	
	data = np.empty(n_stars, dtype=[('obj_id','u8'),
	                                 ('l','f8'), ('b','f8'), 
	                                 ('DM','f4'), ('EBV','f4'),
	                                 ('Mr','f4'), ('FeH','f4')])
	data['obj_id'][:] = 1
	data['l'][:] = l
	data['b'][:] = b
	data['DM'][:] = DM[:]
	data['EBV'][:] = EBV[:]
	data['Mr'][:] = Mr[:]
	data['FeH'][:] = FeH[:]
	
	f = h5py.File(filename, access_mode)
	
	pixIdx = 1
	ds_name = '/parameters/pixel %d' % pixIdx
	ds = f.create_dataset(ds_name, data.shape, data.dtype, chunks=True,
	                      compression='gzip', compression_opts=9)
	ds[:] = data[:]
	
	gal_lb = np.array([l, b], dtype='f8')
	
	nside = 512
	nest = True
	EBV = 0.
	att_f8 = np.array([EBV], dtype='f8')
	att_u8 = np.array([pixIdx], dtype='u8')
	att_u4 = np.array([nside], dtype='u4')
	att_u1 = np.array([nest], dtype='u1')
	
	ds.attrs['healpix_index'] = att_u8[0]
	ds.attrs['nested'] = att_u1[0]
	ds.attrs['nside'] = att_u4[0]
	ds.attrs['l'] = gal_lb[0]
	ds.attrs['b'] = gal_lb[1]
	ds.attrs['EBV'] = att_f8[0]
	
	f.close()


def getProbSurfs(fname):
	f = h5py.File(fname, 'r')
	
	pixIdx = 1
	
	pdf_dset = f['/pixel %d/stellar pdfs' % pixIdx]
	x_min = pdf_dset.attrs['min'][::-1]
	x_max = pdf_dset.attrs['max'][::-1]
	bounds = (x_min[0], x_max[0], x_min[1], x_max[1])
	surfs = np.swapaxes(pdf_dset[:,:,:], 1, 2)
	
	chain_dset = f['/pixel %d/stellar chains' % pixIdx]
	converged = chain_dset.attrs['converged'][:]
	lnZ = chain_dset.attrs['ln(Z)'][:]
	chains = chain_dset[:,1:,1:]
	ln_p = chain_dset[:,1:,0]
	
	#mean = np.mean(chains, axis=2)
	#Delta = chains - mean
	#cov = np.einsum('ijk,ijk->
	
	return surfs, bounds, converged, lnZ, chains, ln_p

def probsurf_bayestar(mag, err, maglimit,
                      l=90., b=10., EBV_guess=0.):
	'''
	Runs bayestar on the given input and returns probability surfaces,
	Markov chains, convergence flags, etc.
	
	Input:
	    mag       Observed magnitudes: shape=(nStars, 5)
	    err       Std. Dev. in mags:   shape=(nStars, 5)
	    maglimit  Limiting mags:       shape=(nStars, 5)
	    l         Galactic longitude (deg)
	    b         Galactic latitude  (deg)
	
	Output:
	    (bounds, surfs), (converged, lnZ), chains, log
	    
	    bounds:
	        [DM_min, DM_max, EBV_min, EBV_max] for surfaces
	    surfs:
	        (DM, EBV) probability surfaces.
	        shape=(nStars,nDM,nEBV)
	    converged:
	        For each star, 1 if converged.
	        shape=(nStars)
	    lnZ:
	        ln(evidence) for each star.
	        shape=(nStars)
	    chains:
	        Markov chain for each star.
	        shape=(nStars,4)
	        order of parameters: EBV, DM, Mr, FeH
	'''
	
	# Temporary files
	infile = tempfile.NamedTemporaryFile()
	write_infile(infile.name, mag, err, maglimit, l, b, EBV_guess)
	outfile = tempfile.NamedTemporaryFile()
	logfile = tempfile.NamedTemporaryFile()
	
	# Run bayestar
	binary = '/home/greg/projects/bayestar/bayestar'
	args = [binary, infile.name, outfile.name,
	        '--save-surfs',
	        #'--star-steps', '1',
	        #'--star-samplers', '100',
	        #'--star-p-replacement', '0.2',
	        '--verbosity', '2',
	        '--clouds', '0',
	        '--regions', '0']
	res = subprocess.call(args, stdout=logfile, stderr=logfile)
	
	# Read output
	logfile.seek(0)
	log = logfile.read()
	
	print log
	
	surfs, bounds, converged, lnZ, chains, ln_p = getProbSurfs(outfile.name)
	
	return (bounds, surfs), (converged, lnZ), chains, ln_p, log


def main():
	'''
	maglimit = np.array([[22.5, 22.5, 22.0, 21.5, 21.0]]) + 5.
	absmag = np.array([[5.3044, 5.03, 4.9499, 4.9267, 4.9624]])
	A = np.array([[3.172, 2.271, 1.682, 1.322, 1.087]])
	mu = 10.
	mag = absmag + mu + 2.*A
	err = np.array([[0.02, 0.02, 0.02, 0.1, 0.1]])
	idx = (mag > maglimit)
	err[idx] = 1.e10
	'''
	
	
	mag = np.array([[22.1953, 0.0000, 21.1190, 20.2057, 19.9909]],
	               dtype='f4')
	err = np.array([[0.239, 10000000100.204, 0.055, 0.045, 0.120]],
	               dtype='f4')
	maglim = np.array([[22.537, 22.195, 22.056, 21.461, 20.544]],
	                  dtype='f4')
	
	'''
	mag = np.array([[ 21.8472,  0.0000,  21.6913,  20.8056,  20.1619]],
	               dtype='f4')
	err = np.array([[ 0.228,  10000000100.204,  0.097,  0.104,  0.270]],
	               dtype='f4')
	maglim = np.array([[22.627,  22.289,  22.195,  21.620,  20.553]],
	                  dtype='f4')
	'''
	
	'''mag = np.array([[ 21.6930, 18.3321, 21.4233, 21.0054, 0.0000],
	                [ 24.3780, 20.9731, 20.3322, 20.0769, 19.7944]],
	               dtype='f4')
	err = np.array([[ 0.196, 0.443, 0.112, 0.132, np.inf],
	                [ 0.276, 0.041, 0.038, 0.066, 0.092]],
	               dtype='f4')
	maglim = np.array([[23., 23., 23., 23., 23.],
	                   [22.448, 22.330, 21.960, 21.278, 20.150]],
	                  dtype='f4')'''
	
	'''mag = np.array([[ 19.15384674, 17.96580505, 17.17712402, 16.8208065 , 16.66966629],
	                [ 19.60111427, 18.40708351, 17.57635117, 17.19038963, 17.02507591],
	                [ 20.14488411, 18.8828907 , 18.08464813, 17.7223835 , 17.54846764]],
	               dtype='f4')
	err = np.array([[ 2.25655623e-02, 2.05883402e-02, 2.04321481e-02, 2.09825877e-02, 2.16421112e-02],
	                [ 2.32891608e-02, 2.07754262e-02, 2.06643697e-02, 2.12468710e-02, 2.20379084e-02],
	                [ 2.67316885e-02, 2.14912761e-02, 2.12796684e-02, 2.34406944e-02, 2.59469077e-02]],
	               dtype='f4')
	maglim = np.array([[24.5, 24.5, 24.5, 24.5, 24.5],
	                   [24.5, 24.5, 24.5, 24.5, 24.5],
	                   [24.5, 24.5, 24.5, 24.5, 24.5]],
	                  dtype='f4')'''
	l = 34.01 #173.3075
	b = -19.0 #89.82
	
	(bounds, surfs), (converged, lnZ), chains, ln_p, log = probsurf_bayestar(mag, err, maglim,
	                                                                         l=l, b=b, EBV_guess=2.)
	#print log
	
	#print 'E(B-V) = %.3f +- %.3f' % (np.mean(chains[0,:,0]), np.std(chains[0,:,0]))
	#print ln_p
	
	import matplotlib.pyplot as plt
	import matplotlib
	
	for i in xrange(len(surfs)):
		fig = plt.figure()
		
		plt.subplot(1,2,1)
		
		plt.imshow(surfs[i].T, origin='lower', extent=bounds,
		aspect='auto', interpolation='nearest', vmax=0.005, cmap='binary', norm=matplotlib.colors.LogNorm())
		
		plt.subplot(1,2,2)
		
		plt.scatter(chains[i,:,1], chains[i,:,0], c=ln_p[i,:], edgecolor='none',
		vmin=np.max(ln_p[i,:])-15)
		plt.colorbar()
	
	'''
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.imshow(surfs[0].T, origin='lower', extent=bounds,
	          aspect='auto', cmap='hot', interpolation='nearest')
	
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	img = np.log(surfs[0].T)
	idx = np.isfinite(img)
	img_min = np.min(img[idx])
	img[~idx] = img_min
	ax.imshow(img, origin='lower', extent=bounds,
	          aspect='auto', cmap='hot', interpolation='nearest')
	'''
	
	plt.show()
	
	return 0

if __name__ == '__main__':
	main()
