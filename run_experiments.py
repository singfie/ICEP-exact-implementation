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

        # read in number of updates
        number_updates = int(dataset.split('_')[-3])
        # read in update interval
        update_interval = int(dataset.split('_')[-6])

        updates = [0]
        for i in range(number_updates):
            updates.append(updates[i] + update_interval)

        update_string = ""
        for i in updates:
            update_string += " " + str(i)

        os.system("python main_rolling_horizon.py -p " + path + '/' + dataset +
                  " -r 3600 -u" + update_string + " -i " + str(number_updates))

    return(-1)

if __name__ == '__main__':
    main()