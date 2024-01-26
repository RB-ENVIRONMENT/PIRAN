#!/usr/bin/env bash
rke=1.0  # MeV
alpha=($(seq 1.0 1.0 89.0))  # degrees
resonance=($(seq -5 1 5))  # degrees

for r in ${resonance[*]}
do
    for a in ${alpha[*]}
    do
        python -u landau_singularities.py --rke "$rke" --alpha "$a" --resonance "$r" --save 2>&1 | tee -a landau_singularities.log
    done
done
