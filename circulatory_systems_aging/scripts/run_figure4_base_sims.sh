#!/bin/bash

runtimes=(40 40 40 40 40 40 10 10 10 10 10)
gammas=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# Get the length of the array (assuming both arrays have the same length)
length=${#gammas[@]}

echo "running figure 4 base simulations"
for k in 1 2 3
do
    for ((i=0; i<length; i++))
    do    
        echo "Index $i: runtimes[i]=${runtimes[i]}, gammas[i]=${gammas[i]}"
        gamma=${gammas[i]}
        runtime=${runtimes[i]}
        echo ...running gamma = $gamma, k = $k, runtime = $runtime
        # Run the Python script with the specified parameters
        python circulatory_flow_model.py --config figure4_config.ini \
        --overrides General:outdir=results/fig4_base/K${k}_gamma_${gamma} \
        General:mixing_gamma=$gamma \
        General:runtime=$runtime \
        Goods:hill_coefficient=$k \
        Toxins:hill_coefficient=$k 
    done
done
