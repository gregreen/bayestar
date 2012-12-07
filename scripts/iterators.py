#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#
#       iterators.py
#       
#       Iterators for common data-processing tasks.
#       
#       
#       Copyright 2012 Greg <greg@greg-G53JW>
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
#       
#       

import numpy as np


class data_by_key(object):
	'''Returns blocks of data having the same key, sorted by key.'''
	
	def __init__(self, key, data):
		self.data = data
		
		self.indices = key.argsort()
		self.key = key[self.indices]
		self.newblock = np.concatenate((np.where(np.diff(self.key))[0] + 1, [data.size]))
		
		self.start_index = 0
		self.end_index = self.newblock[0]
		self.block_num = 0
	
	def __iter__(self):
		return self
	
	def next(self):
		if self.block_num == self.newblock.size:
			raise StopIteration
		else:
			block_indices = self.indices[self.start_index:self.end_index]
			block_key = self.key[self.start_index]
			
			self.block_num += 1
			if self.block_num < self.newblock.size:
				self.start_index = self.end_index
				self.end_index = self.newblock[self.block_num]
			
			return block_key, self.data[block_indices]


class index_by_key(object):
	'''Returns sets of indices referring to each value of the key,
	in ascending order of key value.'''
	
	def __init__(self, key):
		self.indices = key.argsort()
		self.key = key[self.indices]
		self.newblock = np.concatenate((np.where(np.diff(self.key))[0] + 1, [key.size]))
		
		self.start_index = 0
		self.end_index = self.newblock[0]
		self.block_num = 0
	
	def __iter__(self):
		return self
	
	def next(self):
		if self.block_num == self.newblock.size:
			raise StopIteration
		else:
			block_indices = self.indices[self.start_index:self.end_index]
			block_key = self.key[self.start_index]
			
			self.block_num += 1
			if self.block_num < self.newblock.size:
				self.start_index = self.end_index
				self.end_index = self.newblock[self.block_num]
			
			return block_key, block_indices


class index_by_unsortable_key(object):
	'''Returns sets of indices referring to each value of the key,
	where the key is not sortable (does not admit < or > operators).'''
	
	def __init__(self, key):
		self.key = key
		self.unused = np.ones(len(key), dtype=np.bool)
	
	def __iter__(self):
		return self
	
	def next(self):
		if np.all(~self.unused):
			raise StopIteration
		else:
			block_indices = []
			unused_indices = np.where(self.unused)[0]
			for i in unused_indices:
				if np.all(self.key[i] == self.key[unused_indices[0]]):
					block_indices.append(i)
					self.unused[i] = False
			return self.key[unused_indices[0]], block_indices



def main():
	x = np.array([0, 1, 1, -3, 5, 5, 5, -3, 0, 0])
	y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
	
	for key, block_indices in index_by_key(x):
		print key, y[block_indices]
	
	print ''
	
	x = [np.array([1,2,3]), np.array([2,3,4]), np.array([1,2,3]), np.array([4,6,-1]), np.array([4,6,-1]), np.array([4,6,-2]), np.array([1,2,3])]
	y = np.array([1, 2, 3, 4, 5, 6, 7])
	for key, block_indices in index_by_unsortable_key(x):
		print key, y[block_indices]
	
	print ''
	
	return 0

if __name__ == '__main__':
	main()

