# ICEP-exact-implementation
An exact implementation of different variants of the Isolated Community Evacuation Problem using the Gurobi solver.
Branches exist for:
- deterministic ICEP (D-ICEP) (deterministic-icep branch)
- stochastic ICEP for planning purposes (S-ICEP) (main branch)
- robust ICEP for response purposes (R-ICEP) (robust-icep branch)
- rolling-horizon ICEP for response purposes (RH-ICEP) (rolling-horizon-icep branch)

The package also includes some work in progress branches on column generation and a step wise solver. These are not functional yet. 

The corresponding research paper published for these algorithms can be found under:
Krutein, K. F. & Goochild, A. The Isolated Community Evacuation Problem with Mixed Integer Programming. Transportation Research Part E: Logistics & Transportation Review. (2022) 102710. https://doi.org/10.1016/j.tre.2022.102710

Furthermore, a case study has been published using the S-ICEP, implemented in the main branch, in:
Krutein, K. F., McGowan, J. & Goodchild, A. Evacuating isolated islands with marine resources: A Bowen Island case study. International Journal of Disaster Risk Reduction. (2022) 102865. https://doi.org/10.1016/j.ijdrr.2022.102865

The functionality of all files in each branch are described in the README file in each branch. The remainder of this file describes the files in the main branch. The main run file for this branch is "pyomo_ICEP_model_run.py", which implements all other files, given the input parameters. 

# File description

## pyomo_ICEP_model_generator.py
This file generates an instance of a S-ICEP model, given input data that can be solved using the Pyomo environment.

## pyomo_ICEP_model_run.py
This file runs an instance of the S-ICEP model and prints route plan outputs and performance stats to data frames for each scenario.
This data set takes all the input data required to describe the S-ICEP as inputs. It can be run from the command line,
and takes the following inputs:
- the path to the dataset
- a penalty for every evacuee that is not evacuated in any scenario
- a route time limit setting the maximum duration of the route plan in any scenario
- a run time limit for the optimization procedure
- an objective function, that is to be selected from the objectives presented in the corresponding paper.  

## visualization_evac_within_config.py
This file visualizes the progress of evacuation for a given route plan within a single configurations in a single plot. 

## visualization_evac_cross_config.py
This file visualizes the progress of evacuation for a given route plan across all configurations in a single plot. 
