#!/bin/bash

for gamma in 0 1.0
do    
    # run varying diffusion
    for diff in 5 10 20 30 40 50
    do
        echo ...running gamma = $gamma, diff = $diff
        python circulatory_flow_model.py --config figure3_config.ini \
        --overrides General:outdir=flow_results/figure3/K2_Diff${diff}_gamma_${gamma} \
        General:mixing_gamma=$gamma \
        Goods:diffusion=$diff \
        Toxins:diffusion=$diff
    done

    # run varying velocity
    for vel in 0 25 50 75 100 
    do    
        echo ...running gamma = $gamma, vel = $vel
        python circulatory_flow_model.py --config figure3_config.ini \
        --overrides General:outdir=flow_results/figure3/K2_Vel${vel}_gamma_${gamma} \
        General:mixing_gamma=$gamma \
        General:velocity=$vel
    done
done