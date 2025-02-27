#!/bin/bash

#SBATCH --job-name=repo_lab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=60000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:2

# Code for running the experiments

module purge
module load 2019
module load eb
module load Python/3.6.6-foss-2018b
module load cuDNN/7.6.3-CUDA-10.0.130
export LD_LIBRARY_PATH=/hpc/eb/Debian9/cuDNN/7.1-CUDA-8.0.44-GCCcore-5.4.0/lib64:$LD_LIBRARY_PATH


pip3 install --user torch torchvision matplotlib gym box2d-py wandb
# Copy all files required for the tasks to SCRATCH
DIRECTORY='ReinforceRepoLab/code'
mkdir -p "$TMPDIR/ReinforceRepoLab"
cp -r $HOME/"$DIRECTORY" "$TMPDIR/ReinforceRepoLab"

INPUT_FILE=$1

# For each core available on this node, run the program in parallel
NPROC=`nproc --all`
echo $NPROC 'cores available'
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
	      # run program ...
        #echo "$argsstring"
	      python3 main.py $argsstring --output "RESULTS_JOB__$argsstring"
    	  # copy the results back
    	  cp -r results $HOME/"$DIRECTORY"/results_experiments
    	  cp -r wandb $HOME/"$DIRECTORY"
    ) &
done
#rm $INPUT_FILE
wait
