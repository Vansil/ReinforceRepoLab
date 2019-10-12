#!/bin/bash

FILE='exp_settings'

bash create_settings.sh > $FILE

# NOTE: I am not sure whether NPROC is actually the number of cores of the node we submit the job to

#NPROC=`nproc --all`
NPROC=12
NLINES=$(wc -l < $FILE)
JOBS=$((($NLINES + $NPROC -1)/$NPROC))

echo "Creating $JOBS jobs"

# Create results_experiments directory if not exist yet
mkdir -p results_experiments

for i in `seq 1 $JOBS`; do
    head -$(($i * $NPROC)) $FILE | tail -$NPROC > "$FILE.job$i"
    # args: $1: where to read params from, $2: job number
    # sbatch -p gpu run_experiment.sh "$FILE.job$i" $i
    sbatch run_experiment.sh "$FILE.job$i" $i
    #bash run_experiment.sh "$FILE.job$i" $i
done
rm $FILE
