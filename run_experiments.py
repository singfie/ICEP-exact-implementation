"""
@fiete
March 8, 2022
"""

import pandas as pd
import argparse
import os

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

        # read in number of locations
        demand = pd.read_csv(os.path.join(path, dataset, 'input', 'scenarios.csv'))
        for gamma in range(len(demand) + 1):

            # optimize using D-ICEP
            os.system('python pyomo_ICEP_model_run.py -path ' + rel_path + '/' + dataset + ' -run_time_limit 3600' + ' -gamma ' + str(gamma))

            # update route plan with true information revealed
            os.system('python simulate_usage_in_execution.py -path ' + rel_path + '/' + dataset + ' -gamma ' + str(gamma))

    return(-1)

if __name__ == '__main__':
    main()