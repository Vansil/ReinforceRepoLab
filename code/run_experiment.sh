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
cp -r $HOME/.../DIRECTORY "$TMPDIR"

# For each core available on this node, run the program in parallel
NPROC=`nproc --all`
#for i in `seq 1 $NPROC`; do

node=1

# For each environment
for j in `seq 0 3`; do

# For without and with experience replay
for k in `seq 0 1`; do

# Without fixed target policy or with
for l in `seq 0 1`; do

# For each setting for reward clipping
for m in `seq 0 2`; do
	if (("$node" > $NPROC)); then
	   break
	fi
	(
	cd "$TMPDIR"/DIRECTORY
	# run program ...
	python SCRIPT --env $j --replay $k --fixed_T_policy $l --reward_clip $m
    	# copy the results back
    	cp -r results $HOME/.../RESULTS/
    ) &
	let "node=node++" # increment counter	

done
done
done
done
wait
