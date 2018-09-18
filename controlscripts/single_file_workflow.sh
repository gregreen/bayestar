#!/usr/bin/env bash
# 
# Fetches or compiles a bayestar binary; generates a temporary working
# directory; copies in the input file; runs bayestar, ensuring
# convergence; copies the output to the destination directory; and
# deletes the working directory.
# 
# Invocation:
#   $ exec single_file_workflow.sh \
#         /full/path/to/input.h5 \
#         /full/path/to/output.h5 \
#         /config/base/filename.cfg \
#         /full/path/to/tarball/containing/extras.tar.gz (optional)
# 
# Configuration files of the format /config/base/filename.cfg.X
# should exist, with X=0,1,2,.... Pixels that do not converge using
# the options provided in the X=0 configuration file will be
# reprocessed using the X=1 configuration file, and then with
# the X=2 configuration file, etc., if necessary.
# 
# The "extras" tarball may contain data files needed to run
# bayestar, such as the luminosity function, SED template
# library and extinction coefficients.
# 
# The output file is first generated locally, and then copied
# to the destination directory at the end of the script.
# 
# If "bayestar_dir" is not set, it will default to
# "${HOME}/projects/bayestar".
# 
# Expects the variable "bayestar_ver" to be set. Will look for
# ${bayestar_dir}/bin_cache/bayestar_XXXXXXXX, where the Xs are a hash
# calculated from the CentOS version, CPU model and Bayestar version.
# If the binary does not exist, will compile the contents of the
# tarball ${bayestar_dir}/deploy/bayestar_${bayestar_ver}.tar.gz,
# and store the results in the bin_cache directory.

# Full path must be given for all filenames
in_fname="$1"       # Input filename
out_fname="$2"      # Output filename
config_fname="$3"   # Configuration filename.
                    # ${config_fname}.X should exist, where X = 0,1,2.
extras_tarball="$4" # Tarball containing extra files (e.g., LF)

# Exit if any part fails
#set -e

# Timestamp at beginning of workflow
date

# Default bayestar directory
if [[ -z ${bayestar_dir} ]]; then
    bayestar_dir="${HOME}/projects/bayestar"
fi
echo "# bayestar directory: \"${bayestar_dir}\""

# Temporary working directory
work_dir=`env TMPDIR="/scratch" mktemp -d -p /scratch -t bayestar.XXXXXXX`
echo "# Working directory: ${work_dir}"
cd ${work_dir}

# Copy the input file to the working directory
echo "# Copying input file to working directory ..."
in_fname_base=`basename "${in_fname}"`
cp "${in_fname}" "${in_fname_base}"

# Copy the configuration files to the working directory
echo "# Copying configuration files to working directory ..."
config_fname_base=`basename "${config_fname}"`
cp ${config_fname}.* .

# Copy the extras into the working directory
if [[ ! -z "${extras_tarball}" ]]; then
    echo "# Copying tarball of extra files (LF, SEDs, etc.) into working dir ..."
    extras_tarball_base=`basename "${extras_tarball}"`
    cp "${extras_tarball}" .
    tar -xzf "${extras_tarball_base}"
fi

# Compile bayestar and set the environment
echo "# Compiling bayestar and setting the environment ..."
source "${bayestar_dir}/controlscripts/compile_bayestar.sh"

# Timestamp at beginning of bayestar 
date

# Process the input file
echo "# Processing the input file ..."
exec "${bayestar_dir}/controlscripts/process_file.sh" \
    "./bayestar" \
    "${in_fname_base}" \
    "${out_fname}" \
    "${config_fname_base}"

# Timestamp at end of bayestar
date

# Delete working directory
echo "# Deleting the working directory ..."
rm -rf "${work_dir}"

echo "# Done."
