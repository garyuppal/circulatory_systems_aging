#!/bin/bash -ue
echo "...running gamma = 0, k = 1"
python circulatory_flow_model.py         --config figure2_config.ini         --overrides General:outdir=flow_results/K1_gamma_0                     General:mixing_gamma=0                     Goods:hill_coefficient=1                     Toxins:hill_coefficient=1
