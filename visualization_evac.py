"""
@fiete
November 22, 2021
"""

import pandas as pd
import os
import argparse
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# auxiliary function for reading and plotting data
def plot_progress(route_plan):
    """A method that returns a plot of the progress speed in an evacuation."""

    route_plan = route_plan[route_plan["evacuated_location"] != "None"]

    lm = sns.lmplot('trip_end_time', 'location_evacuees', 'evacuated_location', route_plan,
                    size = 7, truncate = True, legend = True)

    plt.show()


def add_evacuees_remaining_col(route_plan, scenario_file):
    """A method that adds columns that track the remaining evacuees at every time step"""

    # calculate the total number of evacuees
    # np.unique(route_plan["scenario"])[0].split("_")[0]
    scenario_file_sub = scenario_file[scenario_file["Scenario"].str.contains(np.unique(route_plan["scenario"])[0].split("_")[0])]
    total_evacuees = scenario_file_sub["Demand"].sum() - scenario_file_sub["private_evac"].sum()

    # first calculate the column for the overall progress
    route_plan = route_plan.sort_values(by="load_end_time", ascending = True)
    route_plan["total_evacuees"] = 0
    route_plan["total_evacuees"].iloc[0] = total_evacuees
    for i in range(1,len(route_plan)):
        if route_plan["evacuees"].iloc[i] == 0:
            route_plan["total_evacuees"].iloc[i] = route_plan["total_evacuees"].iloc[i-1]
        else:
            route_plan["total_evacuees"].iloc[i] = route_plan["total_evacuees"].iloc[i-1] - route_plan["evacuees"].iloc[i]

    # second calculate the column for location wise progress
    new_route_plan = pd.DataFrame()
    for location in np.unique(route_plan["evacuated_location"]):
        sub_route_plan = route_plan[route_plan["evacuated_location"] == location]
        if location != "None":
            # find total evacuations from scenario file
            evacuations_loc = scenario_file_sub["Demand"][scenario_file_sub["Location"] == location] - scenario_file_sub["private_evac"][scenario_file_sub["Location"] == location].values[0]
            sub_route_plan["location_evacuees"] = evacuations_loc
            sub_route_plan = sub_route_plan.sort_values(by="load_end_time", ascending = True)
            sub_route_plan["location_evacuees"].iloc[0] = evacuations_loc - sub_route_plan["evacuees"].iloc[0]
            for j in range(1, len(sub_route_plan)):
                sub_route_plan["location_evacuees"].iloc[j] = sub_route_plan["location_evacuees"].iloc[j-1] - sub_route_plan["evacuees"].iloc[j]
        else:
            pass
        new_route_plan = new_route_plan.append(sub_route_plan, ignore_index = True)
    new_route_plan = new_route_plan.sort_values(by="load_end_time", ascending = True)
    print(new_route_plan)

    return(new_route_plan)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-path", help="the path of the experiment")

    args = parser.parse_args()

    # get directory path
    dirname = os.getcwd()

    rel_path = args.path
    path = os.path.join(dirname, 'case_study_instances', rel_path)

    solutions = os.path.join(path, 'Solutions/')

    scenario_file = pd.read_csv(os.path.join(path, 'input', 'scenarios.csv'))

    # check if a solution directory exists
    if not os.path.exists(os.path.join(path, 'Solutions')):
        print("No solution directory exists yet, run experiment first")

    else:
        route_plans = pd.DataFrame()
        for file in os.listdir(solutions):
            if (os.path.isfile(os.path.join(solutions, file))) and ("route_plan" in file):
                scenario_name = re.sub(" ", "_", file).split(":_")[1].split("_GUROBI")[0]
                route_plan_sc = pd.read_csv(os.path.join(solutions, file))
                route_plan_sc["scenario"] = scenario_name
                route_plan_sc = add_evacuees_remaining_col(route_plan_sc, scenario_file)
                plot_progress(route_plan_sc)
                route_plans = route_plans.append(route_plan_sc.iloc[:,1:], ignore_index = True)
        print(route_plans)





if __name__ == "__main__":
    main()