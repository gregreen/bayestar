#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       astroutils.py
#       
#       Copyright 2011 Gregory <greg@greg-G53JW>
#       
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later version.
#       
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#       
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.

import sys
from math import log10, acos, asin, cos, sin, pi, atan2, log, sqrt

debug = False	# Turn on debugging messages?

def deg2rad(x):
	return x*pi/180.

def rad2deg(x):
	return x*180./pi

# Parse string representing RA in one of several forms
def parse_RA(ra_str):
	if ':' in ra_str:
		tmp = ra_str.split(':')
		if len(tmp) == 3:		# hh:mm:ss
			return (float(tmp[0]) + (float(tmp[1]) + float(tmp[2])/60.)/60.)/24. * 360.
		elif len(tmp) == 2:		# hh:mm
			return (float(tmp[0]) + float(tmp[1])/60.)/24. * 360.
		else:					# Invalid input format for RA
			return None
	else:						# deg
		return float(ra_str)

# Parse string representing DEC in one of several forms
def parse_DEC(dec_str):
	if ':' in dec_str:
		tmp = dec_str.split(':')
		if len(tmp) == 3:		# deg:mm:ss
			return float(tmp[0]) + (float(tmp[1]) + float(tmp[2])/60.)/60.
		elif len(tmp) == 2:		# deg:mm
			return float(tmp[0]) + float(tmp[1])/60.
		else:					# Invalid input format for DEC
			return None
	else:
		return float(dec_str)	# deg

def parse_Radius(radius_str):
	# Parse Radius
	if ':' in radius_str:
		tmp = radius_str.split(':')
		if len(tmp) == 2:		# mm:ss
			return(float(tmp[0]) + float(tmp[1])/60.)/60.
		else:
			return 0
	elif 'm' in radius_str:
		tmp = radius_str[:-1]
		return float(tmp)/60.
	else:
		return float(radius_str)

# Parse string containing angle in some combination of degrees, hours, minutes and seconds.
# 	Ex. 20 hours, 5 minutes and 23 seconds would be '20h5m23s'
# 	Ex. 5 degrees, 3 seconds would be '5d3s'
def parse_dhms(dhms_str):
	d_pos = dhms_str.find('d')
	h_pos = dhms_str.find('h')
	m_pos = dhms_str.find('m')
	s_pos = dhms_str.find('s')
	
	# Check that the qualifier 'd', 'h', 'm' and 's' appear in order
	order, mult, val = [-1], [], []
	if d_pos != -1: order.append(d_pos); mult.append(1.)
	if h_pos != -1: order.append(h_pos); mult.append(15.)
	if m_pos != -1: order.append(m_pos); mult.append(0.25)
	if s_pos != -1: order.append(s_pos); mult.append(0.004166666666666667)
	for i in range(1, len(order)):
		if order[i-1] + 1 >= order[i]:
			if debug: print 'Unordered input', order
			return None
		try:
			val.append(float(dhms_str[(order[i-1]+1):order[i]]))
		except:
			if debug: print 'Cannot cast %s to float' % dhms_str[(order[i-1]+1):order[i]]
			return None
	deg = 0.
	for i in range(len(val)):
		deg += val[i]*mult[i]
	return deg % 360.

# Convert from RA and DEC (both in degrees) to galactic (l, b) (likewise in degrees)
def equatorial2galactic(alpha, delta):
	a = deg2rad(alpha)
	d = deg2rad(delta)
	a_NGP = deg2rad(192.8595)	# RA of NGP
	d_NGP = deg2rad(27.12825)	# DEC of NGP
	l_NCP = deg2rad(123.932)	# l of NCP
	
	sinb = sin(d_NGP)*sin(d) + cos(d_NGP)*cos(d)*cos(a-a_NGP)
	b = asin(sinb)
	if (b < -pi/2.) or (b > pi/2.):	# Ensure that b is in the correct quadrant
		b = pi - b
	cosb = cos(b)
	y = cos(d)*sin(a-a_NGP)/cosb
	x = (cos(d_NGP)*sin(d) - sin(d_NGP)*cos(d)*cos(a-a_NGP))/cosb
	l = l_NCP - atan2(y,x)
	if debug:	# Test if everything has worked out
		if (abs(cosb*sin(l_NCP-l) - y*cosb) > 0.01) or (abs(cosb*cos(l_NCP-l) - x*cosb) > 0.01): print "ERROR!!!"
	return rad2deg(l), rad2deg(b)

def main():
	print parse_dhms(sys.argv[-1])
	
	return 0

if __name__ == '__main__':
	main()

