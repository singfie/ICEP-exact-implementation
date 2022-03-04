"""
@fiete
March 3, 2022
"""

import pandas as pd
import argparse
import os

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-path", help="the path of the ICEP instance files")
    parser.add_argument("-gamma", type = int, help="the parameter determining how many evacuation locations are allowed to go to the highest deviation.")

    args = parser.parse_args()

    gamma = args.gamma

    # get directory path
    dirname = os.getcwd()

    rel_path = args.path
    path = os.path.join(dirname, rel_path)

    source = os.path.join(path, 'input/')
    inc_source = os.path.join(path, 'incidences/')

    # read in calculated route plan
    existing_route_plan = pd.read_csv(os.path.join(path, 'Solutions', 'route_plan_scenario_GUROBI_gamma_' + str(gamma)))

    # check if all evacuees have been evacuated

    # if not, come up with remainder of route plan


if __name__ == '__main__':
    main()