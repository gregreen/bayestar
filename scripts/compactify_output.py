#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  compactify_output.py
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

import argparse, sys, glob

import maptools


def main():
	parser = argparse.ArgumentParser(prog='compactify_output.py',
	                                 description='Store line-of-sight output to one file.',
	                                 add_help=True)
	parser.add_argument('input', type=str, help='Bayestar output files.')
	parser.add_argument('--unified', '-u', type=str, default=None,
	                                     help='Filename for unified output.')
	parser.add_argument('--stacks', action='store_true',
	                                     help='Save stacked pdf surfaces.')
	parser.add_argument('--summary', '-s', type=str, default=None,
	                                     help='Filename for summary output.')
	parser.add_argument('--bounds', '-b', type=float, nargs=4, default=None,
	                                     help='Bounds of pixels to include (l_min, l_max, b_min, b_max).')
	parser.add_argument('--processes', '-proc', type=int, default=1,
	                                     help='# of processes to spawn.')
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	args = parser.parse_args(sys.argv[offset:])
	
	# Load in line-of-sight data
	print 'Loading Bayestar output files ...'
	fnames = glob.glob(args.input)
	mapper = maptools.LOSMapper(fnames, bounds=args.bounds,
	                                    processes=args.processes,
	                                    load_stacked_pdfs=args.stacks)
	
	# Save to unified output file
	if args.unified != None:
		print 'Saving to unified output file ...'
		mapper.data.save_unified(args.unified,
		                         save_stacks=args.stacks)
	
	# Save to summary output file
	#if args.summary != None:
	#	print 'Saving to summary output file ...'
	#	los_coll.save_summary(args.summary)
	
	print 'Done.'
	
	return 0

if __name__ == '__main__':
	main()

