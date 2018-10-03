#!/usr/bin/env bash
# 
# Processes one bayestar input file, ensuring that all the
# pixels converge.
# 
# Invocation:
#   $ exec process_file.sh \
#         /full/path/to/bayestar_binary \
#         /full/path/to/input.h5 \
#         /full/path/to/output.h5 \
#         /config/base/filename.cfg
# 
# Configuration files of the format /config/base/filename.cfg.X
# should exist, with X=0,1,2,.... Pixels that do not converge using
# the options provided in the X=0 configuration file will be
# reprocessed using the X=1 configuration file, and then with
# the X=2 configuration file, etc., if necessary.
# 
# The output file is first generated locally, and then copied
# to the destination directory at the end of the script.
# 
# If "bayestar_dir" is not set, it will default to
# "${HOME}/projects/bayestar".

bayestar_bin="$1"
in_fname="$2"
out_fname="$3"
config_fname="$4"

echo "## bayestar binary: ${bayestar_bin}"
echo "## input filename: ${in_fname}"
echo "## output filename: ${out_fname}"
echo "## configuration file: ${config_fname}"

# Default bayestar directory
if [[ -z ${bayestar_dir} ]]; then
    bayestar_dir="${HOME}/projects/bayestar"
fi

# Temporary output filename
out_fname_tmp="out.h5"
out_fname_tmp_p="out_repacked.h5"

# On termination, copy partial output file to destination
termination_handler()
{
    echo "~ Termination handler:"
    echo "~   * Copying output file to ${out_fname} ..."
    cp "${out_fname_tmp}" "${out_fname}"
    echo "~   * Done."
    exit 17
}

trap 'termination_handler' USR1 # Job script should specify --signal=USR1@120

# Get list of pixels in input file
echo "## Getting list of pixels in input file ..."
bash ${bayestar_dir}/controlscripts/list_pixels.sh ${in_fname} > pixels.0.txt
n_pixels=`wc -l < pixels.0.txt`

# Get number of config files
max_iter=`ls -1 ${config_fname}.? | wc -l`
echo "## Found ${max_iter} config files."
max_iter=$(( ${max_iter} - 1 ))
echo "## max_iter = ${max_iter}"

# If output already exists, copy it into temporary directory
if [[ -f "${out_fname}" ]]; then
    echo "## Copying existing output file into working directory ..."
    cp "${out_fname}" "${out_fname_tmp}"
fi

# Timestamp
date

# First pass through file
echo "## Running bayestar ..."
${bayestar_bin} ${in_fname} ${out_fname_tmp} --config "${config_fname}.0" &

# Wait for bayestar to finish in background
pid=$!
wait ${pid}

# Check for non-converged files
must_repack=0
for (( k=0; k<${max_iter}; k++ )); do
    kp=$(( $k + 1 ))
    
    echo "## Determining which pixels are non-converged ..."
    bash ${bayestar_dir}/controlscripts/filter_converged_pixels.sh ${out_fname_tmp} pixels.${k}.txt > pixels.${kp}.txt
    
    n_redo=`wc -l < pixels.${kp}.txt`
    if [[ ${n_redo} -eq 0 ]]; then
        break
    fi

    must_repack=1
    
    pix_descr=`cat pixels.${kp}.txt`
    pix_descr="${pix_descr//$'\n'/ }"
    echo "## ${n_redo} of ${n_pixels} pixels non-converged after round ${k}."
    echo "## pix_descr: ${pix_descr}"
    
    # Copy temporary output file to permanent location.
    # This is not strictly necessary, but it feels safer.
    echo "## Copying output to ${out_fname} ..."
    cp "${out_fname_tmp}" "${out_fname}"
    
    echo "## Re-running bayestar (${kp}) ..."
    ${bayestar_bin} ${in_fname} ${out_fname_tmp} --config "${config_fname}.${kp}" --force-pix ${pix_descr} &
    
    # Wait for bayestar to finish in background
    pid=$!
    wait ${pid}
done

echo "## Non-converged pixels:"
if [[ -f pixels.${max_iter}.txt ]]; then
    bash ${bayestar_dir}/controlscripts/filter_converged_pixels.sh ${out_fname_tmp} pixels.${max_iter}.txt > pixels_final.txt
    if [[ `wc -l < pixels_final.txt` -eq 0 ]]; then
        echo "## All pixels appear converged."
    else
        n_redo=`wc -l < pixels_final.txt`
        echo "## ${n_redo} of ${n_pixels} pixels non-converged after final round."
        pix_descr=`cat pixels_final.txt`
        pix_descr="${pix_descr//$'\n'/ }"
        echo "## pix_descr: ${pix_descr}"
    fi
else
    echo "## All pixels appear converged."
fi

# Repack output file
if [[ ${must_repack} -eq 1 ]]; then
    echo "## Repacking output file ..."
    h5repack "${out_fname_tmp}" "${out_fname_tmp_p}"
    mv "${out_fname_tmp_p}" "${out_fname_tmp}"
#else
#    mv "${out_fname_tmp}" "${out_fname_tmp_p}"
fi

# Copy temporary output file to permanent location
echo "## Copying output to ${out_fname} ..."
cp "${out_fname_tmp}" "${out_fname}"
rm "${out_fname_tmp}"

echo "## Done."
