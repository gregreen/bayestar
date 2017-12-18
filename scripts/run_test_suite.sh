#!/usr/bin/env bash

for f in test-suite/input/*.h5; do
    #outFname="output${f#input}"
    outFname="${f/input/output}"
    echo "$f -> $outFname"
    ./bayestar --input "$f" --output "$outFname" --clobber --save-surfs --verbosity 2 --config test-suite/input/config.cfg
done
