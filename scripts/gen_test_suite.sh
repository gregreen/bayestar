#!/usr/bin/env bash

lValues=(0 90 180)
bValues=(0 15 45)
nValues=(25 100 1000)
cloudDistsSingle=(7 10 14)
cloudESingle=(1.0 0.5 0.35)
cloudDistsDouble=(11 14)
cloudEDouble=(0.5 0.25)

for l in ${lValues[@]}; do
    for b in ${bValues[@]}; do
        for n in ${nValues[@]}; do
            k=0
            for da in ${cloudDistsSingle[@]}; do
                Ea=${cloudESingle[$k]}

                echo "(l,b,n,d0,E0) = $l, $b, $n, $da, $Ea"

                fname="test_l${l}_b${b}_n${n}_cl${da}-${Ea}.h5"

                python scripts/gen_test_input.py \
                    -N $n \
                    -o "input/test-suite/${fname}" \
                    -lb $l $b \
                    -EBV 0.00001 \
                    -cl $da $Ea \
                    -nb 4 \
                    -LF data/PSMrLF.dat \
                    -t data/PS1_qz_2MASS_colors.dat
                
                if [ $b -lt 20 ] && [ $da -lt 9 ]; then
                    j=0

                    for db in ${cloudDistsDouble[@]}; do
                        Eb=${cloudEDouble[$j]}

                        echo "(l,b,n,d0,E0,d1,E1) = $l, $b, $n, $da, $Ea, $db, $Eb"

                        fname="test_l${l}_b${b}_n${n}_cl${da}-${Ea}-${db}-${Eb}.h5"

                        python scripts/gen_test_input.py \
                            -N $n \
                            -o "input/test-suite/${fname}" \
                            -lb $l $b \
                            -EBV 0.00001 \
                            -cl $da $Ea $db $Eb \
                            -nb 4 \
                            -LF data/PSMrLF.dat \
                            -t data/PS1_qz_2MASS_colors.dat

                        j=$(( $j + 1 ))
                    done
                fi

                k=$(( $k + 1 ))
            done
        done
    done
done
