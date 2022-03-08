"""
@fiete
March 3, 2022
"""

import pandas as pd
import argparse
import shutil
import os
import random
import numpy as np

def diversify_uncertain_experiments():

    """
    A method to diversify a dataset. The parameters modified are:
    - ratio between total demand and total available resource capacity
    - variance of information gain over times
    - time interval of updates
    - number of updates / data points in uncertainty set
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="the path of the original ICEP instance files")
    parser.add_argument("-s", "--seed", help="random seed to control experiment generation")

    args = parser.parse_args()

    # get directory path
    dirname = os.getcwd()

    rel_path = args.path
    path = os.path.join(dirname, rel_path)

    seed = args.seed
    random.seed(seed) # setting the seed

    # check if a directory for experiment files exists
    if not os.path.exists(os.path.join(dirname, 'experiment_files')):
        os.makedirs(os.path.join(dirname, 'experiment_files'))

    # make variations
    for ratio in [2, 5, 7]: # ratio = (total demand)/(total capacity of resources)
        for variance_factor in [0.2, 0.4, 0.6]: # variance_factor = (std_deviation over time)/(total demand)
            for time_interval in [15, 30, 60]: # time in between updates
                for number_updates in [3, 5, 10]: # number of information updates
                    # create a file copy
                    new_file_path = os.path.join(dirname, 'experiment_files', 'experiment_' + rel_path.split('/')[1] +
                                                 '_ratio_' + str(ratio) +
                                                 '_var_factor_' + str(variance_factor) +
                                                 '_update_interval_' + str(time_interval) +
                                                 '_number_updates_' + str(number_updates) +
                                                 '_seed_' + str(seed))
                    shutil.copytree(path, new_file_path)

                    # read in demand
                    demand = pd.read_csv(os.path.join(path, new_file_path, 'input', 'scenarios.csv'))
                    # read in resources
                    resources = pd.read_csv(os.path.join(path, new_file_path, 'input', 'vessels.csv'))

                    # calculating metrics
                    total_capacity = resources['max_cap'].sum()

                    # resulting demand figure
                    total_demand = ratio * total_capacity

                    ### DEMAND RATIO ###
                    demand = demand[['Scenario','Location','private_evac','Demand_0']]
                    # randomly distribute the initial demand estimate across locations
                    distribution = np.random.dirichlet(np.ones(len(demand)),size=1)
                    for i in range(len(demand)):
                        demand['Demand_0'].iloc[i] = max(int(np.round(distribution[0][i] * total_demand)),0)

                    ### NUMBER OF UPDATES AND VARIANCE FACTOR ###
                    for i in range(1, number_updates + 1):
                        demand['Demand_' + str(i)] = 0
                        for j in range(len(demand)):
                            demand['Demand_' + str(i)].iloc[j] = max(int(np.round(np.random.normal(loc=demand['Demand_0'].iloc[j],
                                                                                                   scale=variance_factor * demand['Demand_0'].iloc[j]))),0)

                    # update the actual demand
                    demand['Robust_demand'] = 0
                    demand['Actual_demand'] = 0
                    for j in range(len(demand)):
                        demand_columns = demand.loc[:, demand.columns.str.startswith('Demand_')]
                        demand['Robust_demand'].iloc[j] = max(demand_columns.iloc[j])
                        demand['Actual_demand'].iloc[j] = demand['Demand_' + str(number_updates)].iloc[j]
                    # update the robust demand
                    demand.to_csv(os.path.join(path, new_file_path, 'input', 'scenarios.csv'), index = False)

    return(-1)

if __name__ == '__main__':
    diversify_uncertain_experiments()