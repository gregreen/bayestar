#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  model-colors.py
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
import numpy as np

import matplotlib as mplib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

def load_empirical(fname):
	# Load in templates
	f = open(templatefn, 'r')
	row = []
	for l in f:
		line = l.rstrip().lstrip()
		if len(line) == 0:	# Empty line
			continue
		if line[0] == '#':	# Comment
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
	
	template = template[template[:,0] <= max_r]
	
	

def main():
	outfn = '../plots/color-vs-Mr.png'
	templatefn = '../data/pscolors_efs_v0.txt'
	max_r = 15.
	
	# Load in templates
	f = open(templatefn, 'r')
	row = []
	for l in f:
		line = l.rstrip().lstrip()
		if len(line) == 0:	# Empty line
			continue
		if line[0] == '#':	# Comment
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
	template = template[template[:,0] <= max_r]
	
	# Set matplotlib defaults
	mplib.rc('text', usetex=True)
	mplib.rc('xtick.major', size=6)
	mplib.rc('xtick.minor', size=4)
	mplib.rc('axes', grid=True)
	
	fig = plt.figure(figsize=(5,5.5), dpi=200)
	
	# Color - Magnitude
	image = None
	for i,c in enumerate(['g-r', 'r-i', 'i-z', 'z-y']):
		ax = fig.add_subplot(2,2,i+1)
		image = ax.scatter(template[:,i+2],
						   template[:,0],
						   c=template[:,1],
						   s=1, edgecolors='none')
		#ax.set_xlim(-0.2, 2.1)
		ylim = ax.get_ylim()
		ax.set_ylim(ylim[1], ylim[0])
		ax.label_outer()
		
		ax.grid(which='major', alpha=0.7)
		ax.grid(which='minor', alpha=0.1)
		ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
		ax.xaxis.set_minor_locator(AutoMinorLocator())
		ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
		ax.yaxis.set_minor_locator(AutoMinorLocator())
		
		if i+1 in [1,2]:
			ax2 = ax.twiny()
			ax2.set_xlabel(r'$%s$' % c, fontsize=16, labelpad=7.)
			xlim = ax.get_xlim()
			ax2.set_xlim(xlim[0], xlim[1])
			ax2.xaxis.set_major_locator(MaxNLocator(nbins=5))
			ax2.xaxis.set_minor_locator(AutoMinorLocator())
			ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))
			ax2.yaxis.set_minor_locator(AutoMinorLocator())
			ax2.grid(which='minor', alpha=0.)
			ax2.grid(which='major', alpha=0.)
		else:
			ax.set_xlabel(r'$%s$' % c, fontsize=16, labelpad=4.)
	
	fig.text(0.05, 0.575, r'$M_{r}$', fontsize=16, rotation='vertical',
	                                            multialignment='center')
	
	# Create colorbar
	fig.subplots_adjust(bottom=0.25, top=0.9,
	                    left=0.15, right=0.9,
	                    wspace=0., hspace=0.)
	cax = fig.add_axes([0.15, 0.1, 0.75, 0.035])
	cb = fig.colorbar(image, cax=cax, orientation='horizontal')
	cax.set_xlabel(r'$\left[ \mathrm{Fe} / \mathrm{H} \right]$', fontsize=16)
	cb.set_ticks([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0])
	
	fig.savefig(outfn, dpi=500)
	plt.show()
	
	return 0

if __name__ == '__main__':
	main()

