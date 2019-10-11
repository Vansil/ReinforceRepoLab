#!/bin/bash

# For each environment
for env in `seq 1 4`; do

    # For without and with experience replay
    for replay in `seq 0 1`; do

        # Without fixed target policy or with
        for fixed_policy in `seq 0 1`; do

            # For each setting for reward clipping
            reward_array=( 0 1 100)
            for reward_clip in ${reward_array[@]}; do

                # Decide on architecture
                for arch in `seq 0 2`; do

                    # Write setting on line
	                  echo --env $env --replay $replay --fixed_T_policy $fixed_policy --reward_clip $reward_clip --seed 42 --architecture $arch

                done
            done
        done
    done
done
