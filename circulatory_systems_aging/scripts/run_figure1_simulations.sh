#!/bin/bash
set -e
source settings.sh

cd ${PROJECT_DIR}

networks=("star" "line" "branching" "directed_SFN" "directed_SFN" "erdos_reyni" "erdos_reyni" "full")
sf_m=(0 0 0 2 50 0 0 0)
er_p=(0 0 0 0 0 0.1 0.5 0)

for i in "${!networks[@]}"; do
    network=${networks[$i]}
    sf_m_value=${sf_m[$i]}
    er_p_value=${er_p[$i]}

    echo "Running simulation for network: $network with sf_m: $sf_m_value and er_p: $er_p_value"

    # loop over repair probabilities and initial conditions
    for repair_prob in 0.0001 0.001; do
        for initial_condition in 0 0.2; do
            echo "Repair probability: $repair_prob, Initial non-functional: $initial_condition"

            # run the python script with the specified parameters
            python network_model.py --config figure1_config.ini \
                --overrides General:outdir=${OUTPUT_DIR}/network_results/${network}_SFm_${sf_m_value}_ERp_${er_p_value}_repair_${repair_prob}_eta_${initial_condition} \
                Network:type=$network \
                Network:scale_free_m=$sf_m_value \
                Network:random_network_p=$er_p_value \
                General:prob_repair=$repair_prob \
                General:init_nonfunctional=$initial_condition
        done
    done
done