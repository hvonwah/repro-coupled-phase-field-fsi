[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXXX)

This repository contains the reproduction scripts and resulting data for the work "A coupled high-accuracy phase-field fluid-structure interaction framework for Stokes fluid-filled fracture surrounded by an elastic medium" by H. v. Wahl and T. Wick. The code is based in parts on our previous work [1].

**References**

[1] H. von Wahl and T. Wick. A high-accuracy framework for phase-field fracture interface reconstructions with application to Stokes fluid-filled fracture surrounded by an elastic medium. Comput. Methods Appl. Mech. Engrg., 415:116202, October 2023. [Code repository](https://github.com/hvonwah/stationary_phase_field_stokes_fsi/tree/v1).



# Files
```
+- README.md                                   // This file
+- LICENSE                                     // The licence file
+- install.txt                                 // Installation help
+- convergence_study_formulation_interface.py  // Main file for Example 1
+- convergence_sneddon_coupled_stationary.py   // Main file for Example 2 
+- time_dep_phase_field.py                     // Main file for Example 3
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

The individual convergence studies presented (Examples 1 and 2) are computed using the `convergence_*.py` scripts. Example 3 can be computed using 
`time_dep_phase_field.py ` and setting the mesh parameter to reproduce our results.

By default, the direct solver `pardiso` is used to solve the linear systems resulting from the discretisation. If this is not available, this may be replaced with `umfpack` in the `DATA` block of each convergence study script.
