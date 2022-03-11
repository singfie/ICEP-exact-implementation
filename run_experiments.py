"""
@fiete
March 8, 2022
"""

import pandas as pd
import argparse
import os
import shutil

def main():
    """
    Main experiment run file for dissertation experiments
    - We are trying to evaluate the effectiveness of the algorithm for
      different levels of data knowledge.
    - This runs an experiment for a set of experiment files
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="the path of the experiment folder")

    args = parser.parse_args()

    # get directory path
    dirname = os.getcwd()

    rel_path = args.path
    path = os.path.join(dirname, rel_path)

    for dataset in os.listdir(path):

        # shutil.rmtree(os.path.join(path, dataset, 'Solutions'))

        # read in number of locations
        demand = pd.read_csv(os.path.join(path, dataset, 'input', 'scenarios.csv'))

        information = dataset.split('_')
        iteration = int(information[-3])
        update_interval = int(information[-6])
        reveal_time = iteration * update_interval

        for gamma in range(len(demand) + 1):

            # optimize using R-ICEP
            os.system('python pyomo_ICEP_model_run.py -path ' + rel_path + '/' + dataset + ' -run_time_limit 3600' + ' -gamma ' + str(gamma))

            # update with D-ICEP once true information is revealed
            os.system('python DICEP_model_run.py -path ' + rel_path + '/' + dataset + ' -run_time_limit 3600' + ' -gamma ' + str(gamma) + ' -update_time ' + str(reveal_time) + ' -iteration ' + str(iteration))
            # # update route plan with true information revealed
            # os.system('python simulate_usage_in_execution.py -path ' + rel_path + '/' + dataset + ' -gamma ' + str(gamma))

    return(-1)

if __name__ == '__main__':
    main()