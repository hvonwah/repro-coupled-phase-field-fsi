#!/bin/bash

source /mnt/local/Applications/ngs-venv/bin/activate
export MKL_THREADING_LAYER=GNU
mkdir -p results/vtk

python3 example_2_convergence_sneddon_coupled_stationary.py -h0 2 -Lx 6 -vtk 1