#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np


#def calc_digit_multiplier(n_levels):
#    #n_levels = int(np.log2(nside))
#    g = [12]
#    for n in range(2,n_levels):
#        g.append(4)
#    
#    mult = [1]
#    for gg in g[::-1]:
#        mult.append(mult[-1], 
#
#
#digit_multiplier = {
#    n: [
#}


def loc2digits(nside, idx):
    """
    Converts a healpix (nside, index) specification
    to a set of digits representing the location of
    the pixel in the nested scheme.
    """
    
    n_levels = int(np.log2(nside)) + 1
    radix = (n_levels-1)*[4] + [12]
    remainder = idx
    digits = []
    
    for n in range(n_levels):
        d = remainder % radix[n]
        remainder = (remainder - d) // radix[n]
        digits.append(d)
    
    return digits[::-1]


def digits2loc(digits):
    """
    Converts digits representing the nested location of
    a HEALPix pixel to a nested (nside, index) specification.
    """
    
    idx = 0
    nside = 1
    mult = 1
    for d in digits[::-1]:
        idx += mult * d
        nside *= 2
        mult *= 4
    
    nside //= 2
    
    return (nside, idx)


def set_node(node, digits, value):
    # Arrived at insertion point
    if len(digits) == 1:
        if len(node) == 0:
            node += [None] * 4
        node[digits[0]] = value
    else: # Not at insertion point yet
        if len(node) == 0:
            node += [None] * 4
        if not isinstance(node[digits[0]], list):
            node[digits[0]] = []
        set_node(node[digits[0]], digits[1:], value)


def healtree_map_flat(node, f):
    ret = []
    for subnode in node:
        if isinstance(subnode, list):
            ret += healtree_map_flat(subnode, f)
        elif subnode is not None:
            ret.append(f(node))
    return ret

def healtree_set(tree, nside, idx, value):
    # Determine pixel location in tree
    digits = loc2digits(nside, idx)
    
    # Insert value
    set_node(tree, digits, value)


def healtree_init():
    return [None] * 12


def get_node(node, digits):
    if node is None:
        return (None, 0)
    if not isinstance(node, list):
        return ([node], len(digits))
    if len(digits) == 1:
        return (node[digits[0]], 0)
    return get_node(node[digits[0]], digits[1:])


def flatten_node(node):
    if isinstance(node, list):
        res = []
        for child in node:
            res += flatten_node(child)
        return res
    if node is None:
        return []
    return [node]


def healtree_get(tree, nside, idx, flatten=False):
    digits = loc2digits(nside, idx)
    node, remainder = get_node(tree, digits)
    if remainder != 0:
        nside, idx = digits2loc(digits[:len(digits)-remainder])
    if flatten:
        node = flatten_node(node)
    return node, (nside, idx)


def healtree_get_container(tree, nside, idx, flatten=False):
    """
    Retrieve the given pixel in the tree. If it doesn't exist, retrieve
    the smallest parent pixel containing the requested pixel.
    """
    digits = loc2digits(nside, idx)
    res = get_node(tree, digits)
    while (len(res) == 0) and (len(digits) != 0):
        digits = digits[:-1]
        res = get_node(tree, digits)
    if flatten:
        res = flatten_node(res)
    return res


def healtree_navigate(tree, nside, idx, flatten=False):
    pass


def main():
    #print(loc2digits(8, 97))
    
    #tree = []
    #set_node(tree, [1,2,0,1], 'a')
    #print(tree)
    #set_node(tree, [1,2,0,2], 'b')
    #set_node(tree, [1,2,0,0,0], 'c')
    #print(tree)
    
    tree = healtree_init()
    
    healtree_set(tree, 8, 97, 'a')
    print(tree)
    
    print(healtree_get(tree, 8, 97))
    print(healtree_get(tree, 8, 95))
    print(healtree_get(tree, 4, 24))
    print(healtree_get(tree, 1, 1, flatten=True))
    print('')
    
    set_node(tree, [0,2], 5)
    print(tree)
    
    print(get_node(tree, [0,2]))
    print(get_node(tree, [0,2,1]))
    print(get_node(tree, [0,2,1,1]))
    
    print('')
    digits = [5,1,3,0,0,3,2]
    print(digits)
    nside, idx = digits2loc(digits)
    print(nside, idx)
    digits = loc2digits(nside, idx)
    print(digits)
    
    import gzip, json
    
    with gzip.open('bmk_tree.json.gz', 'r') as f:
        tree = json.load(f)
    
    print(healtree_get(tree, 1024, 0))
    
    return 0


if __name__ == '__main__':
    main()
