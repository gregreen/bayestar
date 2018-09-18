#!/usr/bin/env bash
#
# Lists the pixels in an input file, formatted as
# "nside-index".
# 
# $ bash list_pixels.sh foo.h5
# 512-0
# 1024-5
# 1024-8
# 512-100
# ...

fname="$1"

h5ls "${fname}/photometry" | awk '{ print $2 }'
