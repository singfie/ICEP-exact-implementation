# R-ICEP Documentation
An exact implementation of the Robust Isolated Community Evacuation Problem using the Gurobi solver and the Pyomo modeling package.

# pyomo_ICEP_model_generator.py
This file generates an instance of a R-ICEP model, given input data that can be solved using the Pyomo environment.

# pyomo_ICEP_model_run.py
This file runs an instance of the R-ICEP model and prints route plan outputs and performance stats to data frames.
This data set takes all the input data required to describe the R-ICEP as inputs, including solving the sub-problem. It can be run from the command line,
and takes the following inputs:
- the path to the dataset
- a run time limit for the optimization procedure
- level of robustness (gamma value, which determines for how many locations the most conservative estimate from the uncertainty set will be used)

# robust_sub_problem.py
Builds an instance of the subproblem of R-ICEP to identify the locations that make the dataset most conservative for a given gamma value. 

# run_robust_sub_problem.py
Runs the subproblem of R-ICEP for a given Gamma value and a given dataset including the uncertainty set. 

# D_ICEP_model_generator.py
This file generates an instance of a D-ICEP model, given input data that can be solved using the Pyomo environment.

# D_ICEP_model_run.py
This file runs an instance of the D-ICEP model and prints route plan outputs and performance stats to data frames.
This data set takes all the input data required to describe the D-ICEP as inputs. It can be run from the command line,
and takes the following inputs:
- the path to the dataset
- a run time limit for the optimization procedure
- update time, which describes the time that has passed since the first solve of the problem. Default should be 0.*
- iteration, which describes the demand column in the data set. Default here should be 0.*

\* These features allow to re-solve the problem after an initial solve; the feature is useful to compare the performance of the algorithm to other algorithms, when better information is revealed at a later stage

# run_experiments.py
This file runs a benchmark experiment for an input data set, where initially an uncertainty set over the evacuation demand is known, and the true demand is revealed later. 
It runs in two steps for every instance generated for the experiment:
1. We solve the robust optimization problem at time zero. The results are saved.
2. Using the outputs from 1., the problem is updated at the time of true demand reveal. This means, that if the true demand is higher at any location than initially estimated, the R-ICEP is re-solved using the D-ICEP just for this excess demand, considering the last known location of every resource from step 1.

# experiment_generator.py
This file takes a test instance and generates a full factorial experiment for different ranges of some key parameters, given a random seed:
1. Demand Variance Factor
2. Demand Capacity Ratio
3. Time interval of information updates
   More can be added based on the modeler's interest.

# pull_experiment_results.py
This file pulls the results for each run of each instance from an experiment and condenses them into a single file with summary statistics.

# anova_test.py
This file can run ANOVA on an experiment output to identify statistically significant parameters that influence which algorithm performs best

# visualizations.py
This file generates plots illustrating the differences in peformance between different algorithms.
Generally, this works best, if the results from the same experiment files, run by competing algorithms in the other branches of this package
(e.g. RH-ICEP, D-ICEP), are imported into the file produced by "pull_experiment_results.py", such that more data is available. 


