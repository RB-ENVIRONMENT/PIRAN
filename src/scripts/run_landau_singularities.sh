#!/usr/bin/env bash
rke=1.0  # MeV
alpha=($(seq 65.0 0.1 68.0))  # degrees

for a in ${alpha[*]}
do
    python landau_singularities.py --rke "$rke" --alpha "$a" --save
done
