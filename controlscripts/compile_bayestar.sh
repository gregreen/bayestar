#!/usr/bin/env bash
# 
# Looks for the correct version of bayestar in a cache directory,
# and if that version is not present, compiles bayestar and
# stores a copy in the cache directory.
# 
# Expects the variable "bayestar_ver" to be set. Will look for
# ${bayestar_dir}/bin_cache/bayestar_XXXXXXXX, where the Xs are a hash
# calculated from the CentOS version, CPU model and Bayestar version.
# If the binary does not exist, will compile the contents of the
# tarball ${bayestar_dir}/deploy/bayestar_${bayestar_ver}.tar.gz,
# and store the results in the bin_cache directory.
# 
# If "bayestar_dir" is not set, it will default to
# "${HOME}/projects/bayestar".
# 
# This script is best run inside a temporary working directory, as it
# will possibly pull in bayestar source files and leave various
# unneeded files behind (e.g., *.o files from compilation).

# Check that bayestar_ver has been set
if [[ -z ${bayestar_ver} ]]; then
    echo "bayestar_ver not set! Quitting."
    exit 1
fi

# Detect CentOS version number and CPU model
centos_ver=7
six_str=`grep "release 6" /etc/centos-release`
if [[ ! -z ${six_str} ]]; then
    centos_ver=6
fi

cpu_spec=`grep "model name" /proc/cpuinfo | head -n 1 `
cpu_flags=`grep "flags" /proc/cpuinfo | head -n 1 `

full_spec="${centos_ver} ${cpu_info} ${cpu_flags} ${bayestar_ver}"
full_hash=`echo ${full_spec} | md5sum | awk '{print $1}' | cut -c -8`

echo "CentOS ${centos_ver}"
echo "CPU spec: ${cpu_spec}"
echo "CPU flags: ${cpu_flags}"
grep "cache size" /proc/cpuinfo | head -n 1
grep "cpu MHz" /proc/cpuinfo | head -n 1
grep "cpu cores" /proc/cpuinfo | head -n 1
echo "Bayestar version: ${bayestar_ver}"
echo "Hash: ${full_hash}"

# Set up environment
if [[ ${centos_ver} -eq 6 ]]; then
    echo "Sourcing CentOS 6 environment ..."
    source ~/environments/activate-bayestar-centos6.sh
    source activate pyenv6
else
    echo "Sourcing CentOS 7 environment ..."
    source ~/environments/activate-bayestar-centos7.sh
    source activate /n/fink2/ggreen/environments/pyenv7
fi

# Grab cached bayestar, or compile and cache new copy
#bayestar_ver="2018y08m28d"
if [[ -z ${bayestar_dir} ]]; then
    bayestar_dir="${HOME}/projects/bayestar"
fi
cache_dir="${bayestar_dir}/bin_cache"
bayestar_bin="${cache_dir}/bayestar_${full_hash}"
bayestar_source="bayestar_${bayestar_ver}.tar.gz"

if [[ -f ${bayestar_bin} ]]; then
    echo "Grabbing cached binary ..."
    
    # Grab binary
    cp ${bayestar_bin} ./bayestar
else
    echo "Compiling bayestar ..."
    
    # Grab bayestar source code
    cp ${bayestar_dir}/deploy/${bayestar_source} .
    tar -xzf ${bayestar_source}
    export BAYESTARCOMMIT=`cat commit.txt`
    
    # Compile new binary
    bash recompile.sh

    # Copy binary over to cache directory
    cp bayestar ${bayestar_bin}
fi

echo "Testing whether binary works:"
ver_response=`./bayestar --version`
echo "${ver_response}"
ver_response=`echo ${ver_response} | grep "git commit"`
if [[ -z ${ver_response} ]]; then
    echo "Binary does not appear to work. Quitting."
    exit 1
fi

echo "Done deploying."
