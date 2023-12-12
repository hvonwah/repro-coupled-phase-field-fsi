#!/bin/bash

source /mnt/local/Applications/ngs-venv/bin/activate
export MKL_THREADING_LAYER=GNU
mkdir -p results/vtk

python3 example_4_coupled_time_dependent_phase_field.py -hmax 0.4  -dt 0.1 -vtk 1 -eps 0.04 -f 5 -cp 10
python3 example_4_coupled_time_dependent_phase_field.py -hmax 0.2  -dt 0.1 -vtk 1 -eps 0.04 -f 5 -cp 10
python3 example_4_coupled_time_dependent_phase_field.py -hmax 0.1  -dt 0.1 -vtk 1 -eps 0.04 -f 5 -cp 10
