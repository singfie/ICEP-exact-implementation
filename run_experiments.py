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

        # if not os.path.exists(os.path.join(path, dataset, 'Solutions')):
        #     shutil.rmtree(os.path.join(path, dataset, 'Solutions'))

        try:

            # time of information reveal
            iteration = int(dataset.split('_')[-3])
            # print(iteration)
            time_of_reveal = iteration * int(dataset.split('_')[-6])

            ### RUN BENCHMARK ###

            # optimize using D-ICEP actual demand once revealed
            os.system('python pyomo_ICEP_model_run.py -path ' + rel_path + '/' + dataset + ' -run_time_limit 3600 -update_time 0 -iteration ' + str(iteration))

            ### RUN AS IF ONLY D-ICEP AVAILABLE ###

            # optimize using D-ICEP based on initial estimate
            os.system('python pyomo_ICEP_model_run.py -path ' + rel_path + '/' + dataset + ' -run_time_limit 3600 -update_time 0 -iteration 0')

            # optimize using D-ICEP actual demand once revealed
            os.system('python pyomo_ICEP_model_run.py -path ' + rel_path + '/' + dataset + ' -run_time_limit 3600 -update_time ' + str(time_of_reveal) + ' -iteration ' + str(iteration))

            # update route plan with true information revealed
            # os.system('python simulate_usage_in_execution.py -path ' + rel_path + '/' + dataset)
        except:
            print("insufficient data for ", dataset)

    return(-1)

if __name__ == '__main__':
    main()