#!/bin/bash

source /mnt/local/Applications/ngs-venv/bin/activate
export MKL_THREADING_LAYER=GNU
mkdir -p results/vtk

python3 example_3_convergence_inflow.py -h0 1 -Lx 5 -vtk 1