#!/usr/bin/env bash
#
# Filters a list of pixels in a file, removing pixels
# that pass the convergence criteria. What remain are
# non-converged pixels.
# 
# $ bash filter_converged_pixels.sh foo.h5 pixlist.txt
# 512-0
# 512-5
# ...

fname="$1"
pixlist="$2"
n_min=${3:-20}

bayestar_dir=${HOME}/projects/bayestar

for pix_descr in `cat ${pixlist}`; do
    n_tau=`python ${bayestar_dir}/scripts/chain_convergence.py -i "${fname}" -d "/pixel ${pix_descr}/discrete-los" --add-attribute`
    n_tau=`printf "%.0f" ${n_tau}`
    #echo "    * ${pix_descr} -> ${n_tau}"
    if [[ ${n_tau} -lt ${n_min} ]]; then
        echo ${pix_descr}
    fi
done
