#!/bin/bash
# cd circulatory_systems_aging/

# python circulatory_flow_model.py --config figure2_config.ini --overrides General:outdir=flow_results/gamma_
for k in 1
#  2 3
do
    for gamma in 0 
    # 0.25 0.5 0.75 1.0
    do    
        echo ...running gamma = $gamma, k = $k
        python circulatory_flow_model.py --config figure2_config.ini --overrides General:outdir=flow_results/K${k}_gamma_${gamma} General:mixing_gamma=$gamma Goods:hill_coefficient=$k Toxins:hill_coefficient=$k 
    done
done
