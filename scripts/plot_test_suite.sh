#!/usr/bin/env bash

#cloudPattern="(?<=cl)([0-9]+\.?[0-9]*\-?)*(?=\.h5)"
cloudPattern="cl([0-9]+\.?[0-9]*\-?)*_n"

gitHash=$(git rev-parse --short HEAD)

plotDir="/n/fink2/www/ggreen/mock-tests/test-suite/plots/${gitHash}"

for f in test-suite/output/${gitHash}/*.h5; do
    if [[ $f =~ $cloudPattern ]]; then
        cloudSpec=${BASH_REMATCH#cl}
        cloudSpec=${cloudSpec%_n}
        cloudSpec=${cloudSpec//-/ }
        echo ${cloudSpec}
    fi
    baseFname="${f/output/plots}"
    baseFname="${baseFname%.h5}"
    catFname="${f/output\/${gitHash}/input}"
    
    outFname="${baseFname}_sightline.png"
    if [ ! -f "${outFname}" ]; then
        echo "$f -> $outFname"
        python scripts/plotpdf.py "$f" 512 1 -o "${outFname}" -pdfs -dsc -ovplt $cloudSpec -y 2.0 -cat "${catFname}"
        cp "${outFname}" "${plotDir}"
    fi
    
    outFname="${baseFname}_lineint.png"
    if [ ! -f "${outFname}" ]; then
        echo "$f -> $outFname"
        python scripts/calculate_line_integrals.py -i "$f" -l 512 1 -o "${outFname}" -cl ${cloudSpec} -cat "${catFname}"
        cp "${outfname}" "${plotDir}"
    fi
    
    outFname="${baseFname}_pca.png"
    if [ ! -f "${outFname}" ]; then
        echo "$f -> $outFname"
        python scripts/chain_convergence.py -i "$f" -d "pixel 512-1/discrete-los" -o "${outFname}" 
        cp "${outFname}" "${plotDir}"
    fi
done
