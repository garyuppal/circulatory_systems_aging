#!/bin/bash
set -e
source settings.sh

cd ${PROJECT_DIR}

while IFS=',' read -r K Species gamma; do
    # Skip header
    [[ "$K" == "K" ]] && continue
    # Skip cases with no gamma fit
    [[ "$gamma" == "-1" ]] && continue
    echo "Running with K=$K, gamma=$gamma, for species $Species"

    python circulatory_flow_model.py --config figure4_config.cfg \
    --overrides General:outdir="${OUTPUT_DIR}/figure4_results/${Species}_K${K}_gamma_${gamma}" \
    General:mixing_gamma=$gamma \
    General:runtime=40 \
    Goods:hill_coefficient=$K \
    Toxins:hill_coefficient=$K
done < ${OUTPUT_DIR}/empirical_simulation_parameters/empirical_fitgammas.csv