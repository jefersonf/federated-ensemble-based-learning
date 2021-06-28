#!/bin/bash

A=(0.01 0.1 1 100)
C=(3 5 7 10 20 25 50)
for a in ${A[*]}; do
    for c in ${C[*]}; do
    	al="a"
    	cl="_c"
    	tag="$al$a$cl$c$al"
        logpath="./logs/covid19"
    	cmd=(main.py \
            --data-path ./datasets/covid19/covid19.csv \
    	    --target label \
    	    -v \
    	    --dirichlet-alpha $a \
    	    --clients $c \
            --tag $tag \
            --log-dir $logpath \
            --fedavg)

         python3.7 ${cmd[@]}
    done
done
