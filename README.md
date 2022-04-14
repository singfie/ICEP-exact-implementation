# D-ICEP Documentation
An exact implementation of the Deterministic Isolated Community Evacuation Problem using the Gurobi solver and the Pyomo modeling package. 

The main run file for this branch is "pyomo_ICEP_model_run.py" to run a single instance, which implements all other files, given the input parameters. 
To run experiments, the main run file is "run_experiments.py"

## pyomo_ICEP_model_generator.py 
This file generates an instance of a D-ICEP model, given input data that can be solved using the Pyomo environment.

## pyomo_ICEP_model_run.py
This file runs an instance of the D-ICEP model and prints route plan outputs and performance stats to data frames. 
This data set takes all the input data required to describe the D-ICEP as inputs. It can be run from the command line, 
and takes the following inputs:
- the path to the dataset
- a run time limit for the optimization procedure
- update time, which describes the time that has passed since the first solve of the problem. Default should be 0.*
- iteration, which describes the demand column in the data set. Default here should be 0.*

\* These features allow to re-solve the problem after an initial solve; the feature is useful to compare the performance of the algorithm to other algorithms, when better information is revealed at a later stage

## run_experiments.py
This file runs a benchmark experiment for an input data set, where initially only a first estimate over the evacuation 
data is known, and the true demand is revealed later. It runs in three steps for every instance generated for the experiment:
1. A benchmark is set, running at time zero, but with the true demand (this is to act as if there was no uncertainty in the data). The results are saved as the benchmark route plan. 
2. The problem is solved at time zero with the initial estimate. Results are saved.
3. Using the outputs from 2., the problem is updated at the time of true demand reveal. This means, that if the true demand is higher at any location than initially estimated, the D-ICEP is re-solved just for this excess demand, considering the last known location of every resource from step 2. 

## experiment_generator.py
This file takes a test instance and generates a full factorial experiment for different ranges of some key parameters, given a random seed:
1. Demand Variance Factor
2. Demand Capacity Ratio
3. Time interval of information updates
More can be added based on the modeler's interest. 

## pull_experiment_results.py
This file pulls the results for each run of each instance from an experiment and condenses them into a single file with summary statistics. 

## analysis.py 
This file can run ANOVA on an experiment output to identify statistically significant parameters that influence which algorithm performs best

## visualizations.py 
This file generates plots illustrating the differences in peformance between different algorithms. 
Generally, this works best, if the results from the same experiment files, run by competing algorithms in the other branches of this package
(e.g. RH-ICEP, R-ICEP), are imported into the file produced by "pull_experiment_results.py", such that more data is available. 


