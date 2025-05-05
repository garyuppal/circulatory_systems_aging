#!/bin/bash
set -e
source settings.sh

cd ${PROJECT_DIR}

python fit_ramp_to_empirical.py \
--datapath ${PROJECT_DIR}/41586_2014_BFnature12789_MOESM42_ESM.xls \
--outpath ${OUTPUT_DIR}/empirical_ramp_fits