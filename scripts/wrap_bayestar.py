import numpy as np, h5py, tempfile, subprocess

def write_infile(filename, mag, err, maglimit, l=90., b=10.):
	# Prepare output for pixel
	data = np.empty(len(mag), dtype=[('obj_id','u8'),
	                                 ('l','f8'), ('b','f8'), 
	                                 ('mag','f4',5), ('err','f4',5),
	                                 ('maglimit','f4',5),
	                                 ('nDet','u4',5),
	                                 ('EBV','f4')])
	data['obj_id'][:] = 1
	data['l'][:] = l
	data['b'][:] = b
	data['mag'][:] = mag[:]
	data['err'][:] = err[:]
	data['maglimit'][:] = maglimit[:]
	data['nDet'][:] = 1
	data['EBV'][:] = 0.
	
	if isinstance(filename, str):
		f = h5py.File(filename, 'w')
	
	pixIdx = 1
	ds_name = '/photometry/pixel %d' % pixIdx
	ds = f.create_dataset(ds_name, data.shape, data.dtype, chunks=True,
	                      compression='gzip', compression_opts=9)
	ds[:] = data[:]
	
	gal_lb = np.array([l, b], dtype='f8')
	
	nside = 512
	nest = True
	EBV = 0.
	att_f4 = np.array([EBV], dtype='f8')
	att_u8 = np.array([pixIdx], dtype='u8')
	att_u4 = np.array([nside], dtype='u4')
	att_u1 = np.array([nest], dtype='u1')
	
	ds.attrs['healpix_index'] = att_u8[0]
	ds.attrs['nested'] = att_u1[0]
	ds.attrs['nside'] = att_u4[0]
	ds.attrs['l'] = gal_lb[0]
	ds.attrs['b'] = gal_lb[1]
	ds.attrs['EBV'] = att_f4[0]
	
	f.close()

def getProbSurfs(fname):
	f = h5py.File(fname, 'r')
	
	pixIdx = 1
	
	pdf_dset = f['/pixel %d/stellar pdfs' % pixIdx]
	x_min = pdf_dset.attrs['min']
	x_max = pdf_dset.attrs['max']
	bounds = (x_min[0], x_max[0], x_min[1], x_max[1])
	surfs = pdf_dset[:,:,:]
	
	chain_dset = f['/pixel %d/stellar chains' % pixIdx]
	converged = chain_dset.attrs['converged'][:]
	lnZ = chain_dset.attrs['ln(Z)'][:]
	chains = chain_dset[:,:,1:]
	#mean = np.mean(chains, axis=2)
	#Delta = chains - mean
	#cov = np.einsum('ijk,ijk->
	
	return surfs, bounds, converged, lnZ, chains

def probsurf_bayestar(mag, err, maglimit, l=90., b=10.):
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
	write_infile(infile.name, mag, err, maglimit, l, b)
	outfile = tempfile.NamedTemporaryFile()
	logfile = tempfile.NamedTemporaryFile()
	
	# Run bayestar
	binary = '/home/greg/projects/bayestar/build/bayestar'
	args = [binary, infile.name, outfile.name, '--save-surfs', '--clouds', '0', '--regions', '0', '--star-steps', '500']
	res = subprocess.call(args, stdout=logfile, stderr=logfile)
	
	# Read output
	logfile.seek(0)
	log = logfile.read()
	surfs, bounds, converged, lnZ, chains = getProbSurfs(outfile.name)
	
	return (bounds, surfs), (converged, lnZ), chains, log


def main():
	maglimit = np.array([[22.5, 22.5, 22.0, 21.5, 21.0]]) + 5.
	absmag = np.array([[5.3044, 5.03, 4.9499, 4.9267, 4.9624]])
	A = np.array([[3.172, 2.271, 1.682, 1.322, 1.087]])
	mu = 10.
	mag = absmag + mu + 5.*A
	err = np.array([[0.02, 0.02, 0.02, 0.1, 0.1]])
	idx = (mag > maglimit)
	err[idx] = 1.e10
	(bounds, surfs), (converged, lnZ), chains, log = probsurf_bayestar(mag, err, maglimit, l=180., b=89.)
	print log
	
	print 'E(B-V) = %.3f +- %.3f' % (np.mean(chains[0,:,0]), np.std(chains[0,:,0]))
	
	import matplotlib.pyplot as plt
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.imshow(surfs[0].T, origin='lower', extent=bounds,
	          aspect='auto', cmap='hot', interpolation='nearest')
	plt.show()
	
	return 0

if __name__ == '__main__':
	main()
