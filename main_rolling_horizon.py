"""
@fiete
February 25, 2022
"""

import argparse
import os
import envoy
import subprocess

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="the path of the ICEP instance files")
    parser.add_argument("-r", "--run_time_limit", type = float, help="the upper time limit for the algorithm run time in every iteration")
    parser.add_argument("-u", "--update_times", type = float, nargs = '+', help="sequence of times passed provided in a list")
    parser.add_argument("-i", "--iterations", type = int, help="number of iterations")

    args = parser.parse_args()

    # get directory path
    dirname = os.getcwd()
    rel_path = args.path

    # parse remaining arguments
    run_time_limit = args.run_time_limit
    list_times = args.update_times
    iterations = args.iterations

    iters = range(iterations)

    # try:
    executable_frame = list(zip(iters, list_times))
    print(executable_frame)
    # execute all iterations
    my_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    print(my_path)
    for i,t in executable_frame:
        print("executing for:", rel_path, run_time_limit, t, i)
        os.system('python3 ./pyomo_ICEP_model_run.py -path ' + str(rel_path) + ' -run_time_limit ' + str(run_time_limit) + ' -update_time ' + str(t) + ' -iteration ' + str(i))
# except:
    #     print("Number of iterations and number of updates do not match.")

if __name__ == "__main__":
    main()