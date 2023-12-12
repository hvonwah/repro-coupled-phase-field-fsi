#!/bin/bash

source /mnt/local/Applications/ngs-venv/bin/activate
export MKL_THREADING_LAYER=GNU
mkdir -p results

python3 example_1_convergence_study_interface_formulation_sneddon.py -h0 2 -Lx 6