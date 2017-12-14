#!/usr/bin/env bash

for f in input/test-suite/*.h5; do
    outFname="output${f#input}"
    echo "$f -> $outFname"
    ./bayestar --input "$f" --output "$outFname" --clobber --save-surfs --verbosity 2 --config input/test-suite/config.cfg
done
