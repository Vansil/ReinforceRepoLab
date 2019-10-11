#!/bin/bash

FILE='exp_settings.txt'

bash create_settings.sh > $FILE

NPROC=`nproc --all`
NLINES=$(wc -l < $FILE)
JOBS=$((($NLINES + $NPROC -1)/$NPROC))

echo "Creating $JOBS jobs"

for i in `seq 1 $JOBS`; do
    head -$(($i * $NPROC)) $FILE | tail -$NPROC > "$FILE.job$i"
    sbatch -p gpu run_experiment.sh "$FILE.job$i"
    #bash run_experiment.sh "$FILE.job$i"
done
rm $FILE
