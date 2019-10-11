#!/bin/bash

#SBATCH --job-name=repolab
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=oskar.vanderwal@gmail.com

module purge
module load eb

module load Python/3.6.3-foss-2017b
module load cuDNN/7.0.5-CUDA-9.0.176
module load NCCL/2.0.5-CUDA-9.0.176
module load matplotlib/2.1.1-foss-2017b-Python-3.6.3

export LD_LIBRARY_PATH=/hpc/eb/Debian9/cuDNN/7.1-CUDA-8.0.44-GCCcore-5.4.0/lib64:$LD_LIBRARY_PATH
export PYTHONIOENCODING=utf8

# Copy all files required for the tasks to SCRATCH
DIRECTORY='ReinforceRepoLab/code'
cp -r $HOME/"$DIRECTORY" "$TMPDIR"/ReinforceRepoLab

INPUT_FILE=$1

# For each core available on this node, run the program in parallel
NPROC=`nproc --all`
for i in `seq 1 $NPROC`; do
    # read first line
    read -r argsstring<"$INPUT_FILE"

    # if string is empty, skip
    if [ -z "$argsstring" ]; then
        break
    fi

    # remove first line
    tail -n +2 "$INPUT_FILE" > "$INPUT_FILE.tmp" && mv "$INPUT_FILE.tmp" "$INPUT_FILE"

	  (
	      cd "$TMPDIR/$DIRECTORY"
              ls
	      # run program ...
        #echo "$argsstring"
	      python main.py "$argsstring" --output "RESULTS_JOB_$2_$i"
    	  # copy the results back
    	  cp -r results $HOME/"$DIRECTORY"/results_experiments
    ) &
done
rm $INPUT_FILE
wait
