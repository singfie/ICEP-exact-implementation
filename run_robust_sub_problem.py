"""
@fiete
February 18, 2022
"""

from pyomo.environ import *
import pandas as pd
import time
import os
import argparse

# import modules
import robust_sub_problem

def run_R_ICEP_sub_model(m, runtime_limit = 3600):

    start_time = time.time()

    opt = SolverFactory('gurobi')
    opt.options['IntFeasTol']= 10e-10
    opt.options['MIPGap'] = 0.1 #1e-4
    opt.options['TimeLimit'] = runtime_limit
    results = opt.solve(m, tee=True)
    m.solutions.load_from(results)

    end_time = time.time()
    run_time = end_time - start_time
    print ('Time to optimal solution:'+ str(run_time))

    print('Optimal solution for selected input data')
    print('')

    print('Selected locations:')
    for i in m.a:
        print('Decision for ', i, ':', value(m.l[i]))

    print(round(value(m.objective),2))

    return(m, round(value(m.objective),2), run_time)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-path", help="the path of the ICEP instance files")
    parser.add_argument("-gamma", type = int, help="the parameter determining how many evacuation locations are allowed to go to the highest deviation.")
    parser.add_argument("-run_time_limit", type = float, help="the upper time limit for the algorithm run time")

    args = parser.parse_args()

    # get directory path
    dirname = os.getcwd()

    rel_path = args.path
    path = os.path.join(dirname, rel_path)

    source = os.path.join(path, 'input/')

    run_time_limit = args.run_time_limit
    Gamma_parameter = args.gamma
    # print(run_time_limit)

    #print(trips_source)
    demand_source = pd.read_csv(source + 'scenarios.csv', index_col = False,
                                header=0, delimiter = ',', skipinitialspace=True)

    #print(src_node_source)
    is_locs_source = pd.read_csv(source + 'island locations.csv', index_col=False,
                                 header=0, delimiter = ',', skipinitialspace=True)

    print("Starting GUROBI solver to solve sub problem of R-ICEP...")
    print("")

    start_time = time.time()

    m = robust_sub_problem.main(is_locs_source, demand_source, Gamma_parameter)

    m, optimal_solution, run_time = run_R_ICEP_sub_model(m, runtime_limit = run_time_limit)

    end_time = time.time()
    total_time = end_time - start_time

    print('Time to solution:', total_time)

    return(m)

if __name__ == "__main__":
    main()