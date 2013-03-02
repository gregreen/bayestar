#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  compareSEGUE.py
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
import scipy.stats
import scipy.special
import h5py
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mplib

import hdf5io

def get2DProbSurfs(fname):
	f = h5py.File(fname, 'r')
	
	# Load in probability surfaces from each pixel
	surfs = []
	pixIdx = []
	minEBV, maxEBV = None, None
	for name,item in f.iteritems():
		if 'pixel' in name:
			dset = str(name + '/stellar pdfs')
			idx = int(name.split()[1])
			stack = hdf5io.TProbSurf(f, dset)
			#tmp = np.sum(stack.p[:,:,:], axis=1)
			#tmp = np.einsum('ij,i->ij', tmp, 1./np.sum(tmp, axis=1))
			surfs.append(stack.p[:,:,:])
			pixIdx.append(idx)
			minEBV = f[dset].attrs['min'][1]
			maxEBV = f[dset].attrs['max'][1]
			break
	
	return surfs, minEBV, maxEBV, pixIdx
	
def get1DProbSurfs(fname):
	f = h5py.File(fname, 'r')
	
	# Load in probability surfaces from each pixel
	surfs = []
	pixIdx = []
	good = []
	minEBV, maxEBV = None, None
	for name,item in f.iteritems():
		if 'pixel' in name:
			dset = str(name + '/stellar pdfs')
			idx = int(name.split()[1])
			stack = hdf5io.TProbSurf(f, dset)
			tmp = np.sum(stack.p[:,:,:], axis=1)
			tmp = np.einsum('ij,i->ij', tmp, 1./np.sum(tmp, axis=1))
			surfs.append(tmp)
			pixIdx.append(idx)
			minEBV = f[dset].attrs['min'][1]
			maxEBV = f[dset].attrs['max'][1]
			
			dset = str(name + '/stellar chains')
			lnZ = f[dset].attrs['ln(Z)'][:]
			conv = f[dset].attrs['converged'][:]
			mask = conv & (lnZ > np.max(lnZ) - 20.)
			good.append(mask.astype(np.bool))
	
	return surfs, good, minEBV, maxEBV, pixIdx

def getSEGUE(fname):
	f = h5py.File(fname, 'r')
	SEGUE = f['SEGUE']
	
	# Load in properties from each pixel
	props = []
	pixIdx = []
	objID = []
	for name,item in SEGUE.iteritems():
		if 'pixel' in name:
			idx = int(name.split()[1])
			dset = str(name)
			prop = SEGUE[dset][:]
			props.append(prop)
			pixIdx.append(idx)
	
	return props, pixIdx

def getSegueEBV(props):
	ACoeff = np.array([4.239, 3.303, 2.285, 1.698, 1.263])
	ADiff = np.diff(ACoeff)
	
	EBVs = []
	sigmaEBVs = []
	for prop in props:
		E = np.diff(prop['ubermag'] - prop['ssppmag'], axis=1)
		#print E.shape
		s1 = prop['ubermagerr'][:,:-1]
		s2 = prop['ubermagerr'][:,1:]
		s3 = prop['ssppmagerr'][:,:-1]
		s4 = prop['ssppmagerr'][:,1:]
		#s1 = np.sqrt(s1*s1+0.02*0.02)
		sigmaE = np.sqrt(s1*s1 + s2*s2 + s3*s3 + s4*s4 + 0.02*0.02*4.)
		EBV = E / ADiff
		sigmaEBV = np.abs(sigmaE / E) * EBV
		
		num = np.sum(EBV * sigmaEBV * sigmaEBV, axis=1)
		den1 = np.sum(sigmaEBV * sigmaEBV, axis=1)
		den2 = np.sum(1. / (sigmaEBV * sigmaEBV), axis=1)
		
		EBV = num / den1
		sigmaEBV = np.sqrt(1. / den2)
		
		EBVs.append(EBV)
		sigmaEBVs.append(sigmaEBV)
	
	return EBVs, sigmaEBVs

def percentile(surf, EBV, minEBV, maxEBV):
	nCells = surf.shape[1]
	nStars = surf.shape[0]
	
	DeltaEBV = (maxEBV - minEBV) / nCells
	cellNo = np.floor((EBV - minEBV) / DeltaEBV).astype('i4')
	maskRemove = (cellNo >= nCells) | (cellNo < 0)
	cellNo[maskRemove] = 0
	
	starNo = np.arange(nStars, dtype='i4')
	
	p_threshold = surf[starNo, cellNo]
	#print p_threshold
	
	p_threshold.shape = (nStars, 1)
	p_threshold = np.repeat(p_threshold, nCells, axis=1)
	
	surfZeroed = surf - p_threshold
	idx = surfZeroed < 0
	surfZeroed[idx] = 0.
	pctiles = 1. - np.sum(surfZeroed, axis=1)
	
	pctiles[maskRemove] = np.nan
	
	return pctiles


def multiply1DSurfs(surfs, EBV, sigmaEBV, minEBV, maxEBV):
	nStars, nCells = surfs.shape
	
	DeltaEBV = (maxEBV - minEBV) / nCells
	muCell = (EBV - minEBV) / DeltaEBV
	sigmaCell = sigmaEBV / DeltaEBV
	
	dist = np.linspace(0.5, nCells - 0.5, nCells)
	dist.shape = (1, nCells)
	dist = np.repeat(dist, nStars, axis=0)
	muCell.shape = (nStars, 1)
	muCell = np.repeat(muCell, nCells, axis=1)
	dist -= muCell
	sigmaCell.shape = (nStars, 1)
	sigmaCell = np.repeat(sigmaCell, nCells, axis=1)
	dist /= sigmaCell
	
	pEBV = np.exp(-0.5 * dist * dist)
	pEBV = np.einsum('ij,i->ij', pEBV, 1./np.sum(pEBV, axis=1))
	
	return np.sum(pEBV * surfs, axis=1)
	
	'''
	projProb = np.sum(pEBV, axis=0)
	maxIdx = np.max( np.where(projProb > 1.e-5*np.max(projProb))[0] )
	plotMaxEBV = minEBV + (maxEBV - minEBV) * (float(maxIdx) / float(nCells))
	plotMaxEBV = max(plotMaxEBV, 1.2 * np.max(EBV))
	
	# Set matplotlib style attributes
	mplib.rc('text', usetex=True)
	mplib.rc('xtick.major', size=6)
	mplib.rc('xtick.minor', size=4)
	mplib.rc('ytick.major', size=6)
	mplib.rc('ytick.minor', size=4)
	mplib.rc('xtick', direction='out')
	mplib.rc('ytick', direction='out')
	mplib.rc('axes', grid=False)
	
	fig = plt.figure(figsize=(4,3), dpi=200)
	ax = fig.add_subplot(1,1,1)
	
	#pEBV /= np.max(pEBV)
	#surfs /= np.max(surfs)
	
	#img = np.ones((nCells, nStars, 3), dtype='f8')
	#img[:,:,0] -= pEBV.T
	#img[:,:,1] -= surfs.T
	
	img = pEBV * surfs
	
	bounds = [0, nStars, minEBV, maxEBV]
	ax.imshow(img.T, extent=bounds, origin='lower', aspect='auto',
	                                            interpolation='nearest')
	
	n = np.linspace(0.5, nStars-0.5, nStars)
	ax.errorbar(n, EBV, yerr=sigmaEBV, fmt=None, ecolor='g', capsize=2,
	                                                          alpha=0.5)
	
	ax.set_xlim(bounds[0:2])
	ax.set_ylim(bounds[2], plotMaxEBV)
	
	ax.set_xlabel(r'$\mathrm{Index \ of \ Star}$', fontsize=16)
	ax.set_ylabel(r'$\mathrm{E} \left( B - V \right)$', fontsize=16)
	fig.subplots_adjust(left=0.18, bottom=0.18)
	'''


def pval1DSurfs(surfs, EBV, sigmaEBV, minEBV, maxEBV):
	nStars, nCells = surfs.shape
	
	DeltaEBV = (maxEBV - minEBV) / nCells
	
	y = np.linspace(0.5, nCells - 0.5, nCells)
	y.shape = (1, nCells)
	y = np.repeat(y, nStars, axis=0)
	yCell = np.sum(y*surfs, axis=1) / np.sum(surfs, axis=1)
	y2Cell = np.sum(y*y*surfs, axis=1) / np.sum(surfs, axis=1)
	
	mu = yCell * DeltaEBV
	sigma2 = (y2Cell - yCell*yCell) * (DeltaEBV*DeltaEBV)
	
	sigma2 = sigmaEBV*sigmaEBV + sigma2
	#print EBV, mu
	nSigma = np.abs(EBV - mu) / np.sqrt(sigma2)
	
	return 1. - scipy.special.erf(nSigma)


def binom_confidence(nbins, ntrials, confidence):
	q = 0.5 * (1. - confidence)
	qprime = (1. - q)**(1./nbins)
	
	rv = scipy.stats.binom(ntrials, 1./float(nbins))
	P = rv.cdf(np.arange(ntrials+1))
	
	lower = np.where((1. - P) >= qprime)[0][-1]
	upper = np.where(P < qprime)[0][-1] + 1
	
	return lower, upper

def plotPercentiles(pctiles):
	# Set matplotlib style attributes
	mplib.rc('text', usetex=True)
	mplib.rc('xtick.major', size=6)
	mplib.rc('xtick.minor', size=4)
	mplib.rc('ytick.major', size=6)
	mplib.rc('ytick.minor', size=4)
	mplib.rc('xtick', direction='out')
	mplib.rc('ytick', direction='out')
	mplib.rc('axes', grid=False)
	
	fig = plt.figure(figsize=(4,3), dpi=200)
	ax = fig.add_subplot(1,1,1)
	
	ax.hist(pctiles, alpha=0.6)
	
	lower, upper = binom_confidence(10, pctiles.shape[0], 0.95)
	ax.fill_between([0., 1.], [lower, lower], [upper, upper], facecolor='g', alpha=0.2)
	
	lower, upper = binom_confidence(10, pctiles.shape[0], 0.50)
	ax.fill_between([0., 1.], [lower, lower], [upper, upper], facecolor='g', alpha=0.2)
	
	ax.set_xlim(0., 1.)
	ax.set_ylim(0., 1.5*upper)
	
	ax.set_xlabel(r'$\% \mathrm{ile}$', fontsize=16)
	ax.set_ylabel(r'$\mathrm{\# \ of \ stars}$', fontsize=16)
	fig.subplots_adjust(left=0.20, bottom=0.18)
	
	return fig

def plot1DSurfs(surfs, good, EBV, sigmaEBV, minEBV, maxEBV):
	surfs = surfs[good]
	EBV = EBV[good]
	sigmaEBV = sigmaEBV[good]
	
	nStars, nCells = surfs.shape
	
	projProb = np.sum(surfs, axis=0)
	maxIdx = np.max( np.where(projProb > 1.e-5*np.max(projProb))[0] )
	plotMaxEBV = minEBV + (maxEBV - minEBV) * (float(maxIdx) / float(nCells))
	plotMaxEBV = max(plotMaxEBV, 1.2 * np.max(EBV))
	
	# Set matplotlib style attributes
	mplib.rc('text', usetex=True)
	mplib.rc('xtick.major', size=6)
	mplib.rc('xtick.minor', size=4)
	mplib.rc('ytick.major', size=6)
	mplib.rc('ytick.minor', size=4)
	mplib.rc('xtick', direction='out')
	mplib.rc('ytick', direction='out')
	mplib.rc('axes', grid=False)
	
	fig = plt.figure(figsize=(4,3), dpi=200)
	ax = fig.add_subplot(1,1,1)
	
	bounds = [0, nStars, minEBV, maxEBV]
	ax.imshow(surfs.T, extent=bounds, origin='lower', aspect='auto',
	                                cmap='hot', interpolation='nearest')
	
	n = np.linspace(0.5, nStars-0.5, nStars)
	ax.errorbar(n, EBV, yerr=sigmaEBV, fmt=None, ecolor='g', capsize=2,
	                                                          alpha=0.5)
	
	ax.set_xlim(bounds[0:2])
	ax.set_ylim(bounds[2], plotMaxEBV)
	
	ax.set_xlabel(r'$\mathrm{Index \ of \ Star}$', fontsize=16)
	ax.set_ylabel(r'$\mathrm{E} \left( B - V \right)$', fontsize=16)
	fig.subplots_adjust(left=0.18, bottom=0.18)

def plot2DSurfs(surfs2D, surfs1D, EBV, sigmaEBV, minEBV, maxEBV):
	nStars, nCells = surfs1D.shape
	
	projProb = np.sum(surfs1D, axis=0)
	maxIdx = np.max( np.where(projProb > 1.e-5*np.max(projProb))[0] )
	plotMaxEBV = minEBV + (maxEBV - minEBV) * (float(maxIdx) / float(nCells))
	plotMaxEBV = max(plotMaxEBV, 1.2 * np.max(EBV))
	
	# Set matplotlib style attributes
	mplib.rc('text', usetex=True)
	mplib.rc('xtick.major', size=6)
	mplib.rc('xtick.minor', size=4)
	mplib.rc('ytick.major', size=6)
	mplib.rc('ytick.minor', size=4)
	mplib.rc('xtick', direction='out')
	mplib.rc('ytick', direction='out')
	mplib.rc('axes', grid=False)
	
	fig = plt.figure(figsize=(4,3), dpi=200)
	ax = fig.add_subplot(1,2,1)
	
	bounds = [0, nStars, minEBV, maxEBV]
	ax.imshow(surfs1D.T, extent=bounds, origin='lower', aspect='auto',
	                                cmap='hot', interpolation='nearest')
	
	n = np.linspace(0.5, nStars-0.5, nStars)
	ax.errorbar(n, EBV, yerr=sigmaEBV, fmt=None, ecolor='g', capsize=2,
	                                                          alpha=0.5)
	
	ax.set_xlim(bounds[0:2])
	ax.set_ylim(bounds[2], plotMaxEBV)
	ax.set_xlabel(r'$\mathrm{Index \ of \ Star}$', fontsize=16)
	ax.set_ylabel(r'$\mathrm{E} \left( B - V \right)$', fontsize=16)
	
	ax = fig.add_subplot(1,2,2)
	bounds = [5., 20., minEBV, maxEBV]
	ax.imshow(surfs2D[2].T, extent=bounds, origin='lower', aspect='auto',
	                                cmap='hot', interpolation='nearest')
	
	ax.set_ylim(minEBV, plotMaxEBV)
	ax.set_xlabel(r'$\mu$', fontsize=16)
	ax.set_ylabel(r'$A_{r}$', fontsize=16)
	
	fig.subplots_adjust(left=0.18, bottom=0.18)

def plotScatter(surfs, EBV, sigmaEBV, minEBV, maxEBV, usePeak=True):
	nStars, nCells = surfs.shape
	DeltaEBV = (maxEBV - minEBV) / nCells
	
	yCell = None
	if usePeak:
		yCell = np.argmax(surfs, axis=1) + 0.5
	else:
		y = np.linspace(0.5, nCells - 0.5, nCells)
		y.shape = (1, nCells)
		y = np.repeat(y, nStars, axis=0)
		yCell = np.sum(y*surfs, axis=1) / np.sum(surfs, axis=1)
		y2Cell = np.sum(y*y*surfs, axis=1) / np.sum(surfs, axis=1)
		sigma2 = (y2Cell - yCell*yCell) * (DeltaEBV*DeltaEBV)
	
	mu = yCell * DeltaEBV
	
	# Set matplotlib style attributes
	mplib.rc('text', usetex=True)
	mplib.rc('xtick.major', size=6)
	mplib.rc('xtick.minor', size=4)
	mplib.rc('ytick.major', size=6)
	mplib.rc('ytick.minor', size=4)
	mplib.rc('xtick', direction='out')
	mplib.rc('ytick', direction='out')
	mplib.rc('axes', grid=False)
	
	fig = plt.figure(figsize=(5,4), dpi=200)
	ax = fig.add_subplot(1,1,1)
	
	print np.mean(EBV)
	print np.mean(mu)
	
	print EBV.shape
	print mu.shape
	
	ax.scatter(EBV, mu, s=1., alpha=0.3)
	x = [0, np.max(EBV)]
	ax.plot(x, x, 'b-', alpha=0.5)
	
	xlim = [np.percentile(EBV, 2.), np.percentile(EBV, 98.)]
	width = xlim[1] - xlim[0]
	xlim[1] += 1. * width
	xlim[0] -= 0.25 * width
	ax.set_xlim(xlim)
	ax.set_ylim(0., 2.*np.percentile(mu, 99.))
	
	ax.set_xlabel(r'$\mathrm{E} \left( B - V \right)_{\mathrm{SEGUE}}$', fontsize=16)
	ax.set_ylabel(r'$\mathrm{E} \left( B - V \right)_{\mathrm{Bayes}}$', fontsize=16)
	
	fig.subplots_adjust(left=0.20, bottom=0.20)
	
	return fig


def tests():
	inFName = '../input/SEGUE.00005.h5'
	outFName = '../output/SEGUE.00005.h5'
	
	print 'Loading probability surfaces...'
	surfs, good, minEBV, maxEBV, pixIdx1 = get1DProbSurfs(outFName)
	print 'Loading SEGUE properties...'
	props, pixIdx2 = getSEGUE(inFName)
	print 'Calculating E(B-V)...'
	SegueEBVs, SegueSigmaEBVs = getSegueEBV(props)
	
	print '# of pixels: %d' % (len(surfs))
	
	#print 'Loading 2D probability surfaces...'
	#surfs2D, minEBV, maxEBV, pixIdx3 = get2DProbSurfs(outFName)
	#plot2DSurfs(surfs2D[0], surfs[0], SegueEBVs[0], SegueSigmaEBVs[0], minEBV, maxEBV)
	#plt.show()
	
	print 'Calculating percentiles...'
	pctiles = []
	overlaps = []
	pvals = []
	for surf, SegueEBV, sigmaEBV, mask in zip(surfs, SegueEBVs, SegueSigmaEBVs, good):
		surf = surf[mask]
		SegueEBV = SegueEBV[mask]
		sigmaEBV = sigmaEBV[mask]
		
		#np.random.shuffle(SegueEBV)
		overlaps.append(multiply1DSurfs(surf, SegueEBV, sigmaEBV, minEBV, maxEBV))
		pvals.append(pval1DSurfs(surf, SegueEBV, sigmaEBV, minEBV, maxEBV))
		
		for i in range(1):
			norm = 0.#np.random.normal(size=len(SegueEBV))
			EBV = SegueEBV + sigmaEBV * norm
			pctiles.append(percentile(surf, EBV, minEBV, maxEBV))
	
	pctiles = np.hstack(pctiles)
	overlaps = np.hstack(overlaps)
	idx = ~np.isnan(pctiles)
	pctiles = pctiles[idx]
	idx = ~np.isnan(overlaps)
	overlaps = overlaps[idx]
	pvals = np.hstack(pvals)
	
	print 'Plotting percentiles...'
	plotPercentiles(pctiles)
	plotPercentiles(overlaps)
	plotPercentiles(pvals)
	
	print 'Plotting 1D surfaces...'
	for i in xrange(300,310):
		#multiply1DSurfs(surfs[i], SegueEBVs[i], SegueSigmaEBVs[i], minEBV, maxEBV)
		plot1DSurfs(surfs[i], good[i], SegueEBVs[i], SegueSigmaEBVs[i],
		                                                 minEBV, maxEBV)
		print ''
		print pval1DSurfs(surfs[i], SegueEBVs[i], SegueSigmaEBVs[i],
		                                                 minEBV, maxEBV)
	
	plt.show()

def main():
	inFNames = ['/n/wise/ggreen/bayestar/input/SEGUE.0000%d.h5' % i for i in range(6)]
	outFNames = ['/n/wise/ggreen/bayestar/output/SEGUE.0000%d.h5' % i for i in range(6)]
	
	surfs, SegueEBVs, SegueSigmaEBVs, minEBV, maxEBV = [], [], [], None, None
	
	for inFName, outFName in zip(inFNames, outFNames):
		print inFName
		print outFName
		print 'Loading probability surfaces...'
		surfs_tmp, good_tmp, minEBV, maxEBV, pixIdx1 = get1DProbSurfs(outFName)
		print 'Loading SEGUE properties...'
		props, pixIdx2 = getSEGUE(inFName)
		print 'Calculating E(B-V)...'
		SegueEBVs_tmp, SegueSigmaEBVs_tmp = getSegueEBV(props)
		
		print 'Combining pixels...'
		surfs_tmp = np.vstack(surfs_tmp)
		good_tmp = np.hstack(good_tmp)
		SegueEBVs_tmp = np.hstack(SegueEBVs_tmp)
		SegueSigmaEBVs_tmp = np.hstack(SegueSigmaEBVs_tmp)
		
		print good_tmp
		nanMask = ~np.isnan(SegueEBVs_tmp)
		good_tmp &= nanMask
		
		print good_tmp
		
		surfs.append(surfs_tmp[good_tmp])
		SegueEBVs.append(SegueEBVs_tmp[good_tmp])
		SegueSigmaEBVs.append(SegueSigmaEBVs_tmp[good_tmp])
	
	surfs = np.vstack(surfs)
	SegueEBVs = np.hstack(SegueEBVs)
	SegueSigmaEBVs = np.hstack(SegueSigmaEBVs)
	
	fig1 = plotScatter(surfs, SegueEBVs, SegueSigmaEBVs, minEBV, maxEBV)
	fig2 = plotScatter(surfs, SegueEBVs, SegueSigmaEBVs, minEBV, maxEBV, False)
	
	print '# of pixels: %d' % (len(surfs))
	
	print 'Calculating p-values...'
	pvals = pval1DSurfs(surfs, SegueEBVs, SegueSigmaEBVs, minEBV, maxEBV)
	idx = np.arange(len(SegueEBVs))
	np.random.shuffle(idx)
	SegueEBVs = SegueEBVs[idx]
	SegueSigmaEBVs = SegueSigmaEBVs[idx]
	pvals_shuffled = pval1DSurfs(surfs, SegueEBVs, SegueSigmaEBVs, minEBV, maxEBV)
	
	fig3 = plotPercentiles(pvals)
	fig4 = plotPercentiles(pvals_shuffled)
	
	fig1.savefig('plots/SEGUE-scatter-maxprob.png', dpi=300)
	fig2.savefig('plots/SEGUE-scatter-mean.png', dpi=300)
	fig3.savefig('plots/pvals.png', dpi=300)
	fig4.savefig('plots/pvals-shuffled.png', dpi=300)
	
	plt.show()
	
	return 0

if __name__ == '__main__':
	main()

