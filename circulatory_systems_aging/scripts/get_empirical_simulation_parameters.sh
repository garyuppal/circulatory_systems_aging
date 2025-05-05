#!/bin/bash
set -e
source settings.sh

cd ${PROJECT_DIR}

python get_sim_params_for_empirical.py \
--empirical_fit_path ${OUTPUT_DIR}/empirical_ramp_fits \
--base_simulation_path ${OUTPUT_DIR}/figure4_base \
--outpath ${OUTPUT_DIR}/empirical_simulation_parameters