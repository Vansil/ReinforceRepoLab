#!/bin/bash

NPROC=12
NLINES=$(wc -l < $FILE)
JOBS=$((($NLINES + $NPROC -1)/$NPROC))

line=0
for i in `seq 1 $JOBS`; do
    for j in `seq 1 $NPROC-1`; do
        line=$((line+1))
        if [ -f RESULTS_JOB_$i_$j.csv ]; then
            continue
        else
            sed '$lineq;d' exp_settings
        fi
    done
done`
