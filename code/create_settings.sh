#!/bin/bash

# For each environment
env=2

seeds_array=( 5925 3778 9685 8723 4267 7200 5413 7887 582 5476 )
for seed in ${seeds_array[@]}; do

    #lr_array=( 0.0001 0.001 0.01 0.1 0.5 )
    lr_array=( 0.001 0.01 0.1 )
    for lr in ${lr_array[@]}; do


        settings_array=( 0 1 2 3 4 )
        for setting in ${settings_array[@]}; do
            case "$setting" in
                0)
                    replay=1
                    fixed_policy=1
                    reward_clip=1
                    ;;
                1)
                    replay=0
                    fixed_policy=1
                    reward_clip=1
                    ;;
                2)
                    replay=1
                    fixed_policy=0
                    reward_clip=1
                    ;;
                3)
                    replay=1
                    fixed_policy=1
                    reward_clip=0
                    ;;
                4)
                    replay=1
                    fixed_policy=1
                    reward_clip=100
                    ;;
            esac
                # Write setting on line
	              echo --env $env --replay $replay --fixed_T_policy $fixed_policy --reward_clip $reward_clip --seed $seed --lr $lr

        done
    done
done
