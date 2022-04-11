# RH-ICEP Documentation
An exact implementation of the Rolling Horizon Isolated Community Evacuation Problem using the Gurobi solver and the Pyomo modeling package.

# pyomo_ICEP_model_generator.py
This file generates an instance of a D-ICEP model, given input data that can be solved using the Pyomo environment.

# pyomo_ICEP_model_run.py
This file runs an instance of the D-ICEP model and prints route plan outputs and performance stats to data frames.
This data set takes all the input data required to describe the D-ICEP as inputs. It can be run from the command line,
and takes the following inputs:
- the path to the dataset
- a run time limit for the optimization procedure
- update time, which describes the time that has passed since the first solve of the problem. Default should be 0.*
- iteration, which describes the demand column in the data set. Default here should be 0.*
This variation of the model run, always updates the most recent model result with the new information, for all parts of the previous plan that were not executed yet. 

\* These features allow to re-solve the problem after an initial solve; the feature is useful to compare the performance of the algorithm to other algorithms, when better information is revealed at a later stage

# main_rolling_horizon.py
This file implements the entire rolling horizon procedure, updating the existing solution at every new information retrieval interval. 
Its inputs are:
- the problem path
- the run time limit for the underlying D-ICEP model
- a list of update times
- the number of iterations
The problem structure makes sure that all updates are performed

# run_experiments.py (main run file)
This file starts the rolling horizon procedure for a given data set. 

# experiment_generator.py
This file takes a test instance and generates a full factorial experiment for different ranges of some key parameters, given a random seed:
1. Demand Variance Factor
2. Demand Capacity Ratio
3. Time interval of information updates
   More can be added based on the modeler's interest.

# pull_experiment_results.py
This file pulls the results for each run of each instance from an experiment and condenses them into a single file with summary statistics.

# visualizations.py
This file generates plots illustrating the differences in peformance between different algorithms.
Generally, this works best, if the results from the same experiment files, run by competing algorithms in the other branches of this package
(e.g. RH-ICEP, R-ICEP), are imported into the file produced by "pull_experiment_results.py", such that more data is available. 


