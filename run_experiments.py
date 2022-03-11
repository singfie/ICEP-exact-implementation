"""
@fiete
March 8, 2022
"""

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

        # if os.path.exists(os.path.join(path, dataset, 'Solutions')):
        #     shutil.rmtree(os.path.join(path, dataset, 'Solutions'))

        # optimize using D-ICEP based on initial estimate
        os.system('python pyomo_ICEP_model_run.py -path ' + rel_path + '/' + dataset + ' -run_time_limit 3600 -update_time 0 -iteration 0')

        # time of information reveal
        time_of_reveal = int(dataset.split('_')[-3]) * int(dataset.split('_')[-6])

        # optimize using D-ICEP actual demand once revealed
        os.system('python pyomo_ICEP_model_run.py -path ' + rel_path + '/' + dataset + ' -run_time_limit 3600 -update_time ' + str(time_of_reveal) + ' -iteration 1')

        # update route plan with true information revealed
        # os.system('python simulate_usage_in_execution.py -path ' + rel_path + '/' + dataset)

    return(-1)

if __name__ == '__main__':
    main()