[![DOI](https://zenodo.org/badge/684649214.svg)](https://zenodo.org/badge/latestdoi/684649214)

This repository contains the reproduction scripts and resulting data for the work "A coupled high-accuracy phase-field fluid-structure interaction framework for Stokes fluid-filled fracture surrounded by an elastic medium" by H. v. Wahl and T. Wick. The code is based in parts on our previous work [1].

**References**

[1] H. von Wahl and T. Wick. A high-accuracy framework for phase-field fracture interface reconstructions with application to Stokes fluid-filled fracture surrounded by an elastic medium. Comput. Methods Appl. Mech. Engrg., 415:116202, October 2023. [Code repository](https://github.com/hvonwah/stationary_phase_field_stokes_fsi/tree/v1).



# Files
```
+- README.md                                   // This file
+- LICENSE                                     // The licence file
+- install.txt                                 // Installation help
+- example_1_convergence_study_interface_formulation_sneddon.py  // Example 1 main file
+- example_1_run.bash						   // Run example 1 
+- example_2_convergence_sneddon_coupled_stationary.py  // Example 2 main file
+- example_2_run.bash                          // Run example 2
+- example_3_convergence_inflow.py             // Example 3 main file
+- example_3_run.bash                          // Run example 3
+- example_4_coupled_time_dependent_phase_field.py  // Example 4 main file
+- example_4_run.bash                          // Run example 4
+- meshes.py                                   // Functions to construct meshes
+- phase_field.py                              // Implementation of the phase-field problem
+- sneddon.py                                  // Functions to compute exact Sneddon COD and TCV
+- stokes_fluid_structure_interaction.py       // Implementation of the fluid-structure-interaction solver
+- results/*                                   // The raw text files produced by the computations 
```

# Installation

See the instructions in `install.txt`

# How to reproduce
The scripts to reproduce the computational results are located in the base folder. The resulting data is located in the `results` directory.

The individual examples are implemented the `example_*.py` scripts. These can be executed ith the parameters as presented in the manuscript using the bash scripts `example_*.bash.`

By default, the direct solver `pardiso` is used to solve the linear systems resulting from the discretisation. If this is not available, this may be replaced with `umfpack` in the `DATA` block of each example python study script.
