#!/bin/bash

ALPHA=(0.01 0.1 1 10 100)
NUM_NODES=(3 5 7 10 20 25 50)

for k in ${NUM_NODES[*]}; do
    for a in ${ALPHA[*]}; do
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