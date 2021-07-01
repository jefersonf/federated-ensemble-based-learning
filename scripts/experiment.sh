#!/bin/bash

A=(0.01 0.1 1 100)
C=(3 5 7 10 20 25 50)

for k in ${C[*]}; do
    for a in ${A[*]}; do
        logpath="./reports/shelter-experiment"
    	cmd=(main.py \
            --data-path ./datasets/shelter.csv \
    	    -v \
    	    --dirichlet-alpha $a \
    	    --clients $k \
            --log-dir $logpath)

        # To run you need to remove echo commands
        echo python3 ${cmd[@]}
        echo python3 ${cmd[@]} --fedavg

    done
done
