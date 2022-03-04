"""
@fiete
March 3, 2022
"""

import pandas as pd
import argparse
import shutil
import os

def diversify_uncertain_experiments():

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="the path of the original ICEP instance files")

    args = parser.parse_args()

    # get directory path
    dirname = os.getcwd()

    rel_path = args.path
    path = os.path.join(dirname, rel_path)

    source = os.path.join(path, 'input/')
    inc_source = os.path.join(path, 'incidences/')

    # check if a directory for experiment files exists
    if not os.path.exists(os.path.join(dirname, 'experiment_files')):
        os.makedirs(os.path.join(dirname, 'experiment_files'))

    # parameters to vary
    shutil.copytree(path, os.path.join(dirname, 'experiment_files', 'experiment_' + rel_path.split('/')[1] + '_percentage'))

    # do parametric tests here. TODO

    return(-1)

if __name__ == '__main__':
    diversify_uncertain_experiments()