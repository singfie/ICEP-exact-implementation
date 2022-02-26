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
def plot_progress_scenario(route_plan, filepath):
    """A method that returns a plot of the progress speed in an evacuation."""

    route_plan = route_plan.append({'total_evacuees': max(route_plan['total_evacuees']),
                                    'load_end_time': 0,
                                    'load_start_time': 0,
                                    'scenario': route_plan['scenario'].iloc[0]}, ignore_index = True)

    route_plan = route_plan.sort_values(by=['total_evacuees','load_end_time'], ascending = [False, True])

    # for better display
    for i in range(1,len(route_plan)):
        if route_plan['load_end_time'].iloc[i] == route_plan['load_end_time'].iloc[i-1]:
            route_plan['load_end_time'].iloc[i] += 0.01
    # print(route_plan)

    lm = sns.lineplot(
        x='load_end_time', y='total_evacuees', data=route_plan, hue='scenario', drawstyle='steps-post',
        legend = False, markers = True, estimator=None)
    lm.set_title('Remaining evacuee evolution for Scenario: ' + re.sub("_", " ", route_plan['scenario'].iloc[0]))
    lm.set(xlabel='time in min', ylabel='remaining evacuees')

    pic = lm.get_figure()
    pic.savefig(os.path.join(filepath))
    plt.close()

    return(lm)

def plot_progress_all(route_plan, filepath, scenario_file):
    """A method that returns a plot of the progress speed in an evacuation for all locations together."""

    scenario_file = scenario_file[scenario_file["Scenario"].str.contains(route_plan['scenario'].iloc[0].split("_")[0])] # subsetting

    for i in np.unique(route_plan['evacuated_location']):
        if i != "None":
            evac_no_loc = scenario_file["Demand"][scenario_file["Location"] == i].values[0] - scenario_file["private_evac"][scenario_file["Location"] == i].values[0]
            route_plan = route_plan.append({'location_evacuees': evac_no_loc,
                                            'load_end_time': 0,
                                            'load_start_time': 0,
                                            'evacuated_location': i,
                                            'scenario': route_plan['scenario'].iloc[0]}, ignore_index = True)
    route_plan = route_plan.sort_values(by=['load_end_time', 'location_evacuees'], ascending = [True, False])
    # print(route_plan)

    # for better display
    imp_route_plan = pd.DataFrame()
    for j in np.unique(route_plan['evacuated_location']):
        sub_route_plan = route_plan[route_plan['evacuated_location'] == j]
        for i in range(1,len(sub_route_plan)):
            if sub_route_plan['load_end_time'].iloc[i] == sub_route_plan['load_end_time'].iloc[i-1]:
                sub_route_plan['load_end_time'].iloc[i] += 0.01
        imp_route_plan = imp_route_plan.append(sub_route_plan, ignore_index = True)

    imp_route_plan = imp_route_plan[imp_route_plan["evacuated_location"] != "None"]

    imp_route_plan = imp_route_plan.sort_values(by=['location_evacuees','load_end_time'], ascending = [False, True])

    # imp_route_plan.to_csv('route_plan' + route_plan['scenario'].iloc[0] + '.csv', index = False) # for testing

    lm = sns.lineplot(x='load_end_time', y='location_evacuees', data=imp_route_plan, hue='evacuated_location', drawstyle='steps-post',
                      legend = True, markers = True, estimator=None
                      )
    lm.set_title('Remaining evacuee evolution per location for Scenario: ' + re.sub("_", " ", route_plan['scenario'].iloc[-1]))
    lm.set(xlabel='time in min', ylabel='remaining evacuees')
    plt.legend(title='Affected location')

    pic = lm.get_figure()
    pic.savefig(os.path.join(filepath))
    plt.close()

    return(lm)

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
    if route_plan["total_evacuees"].iloc[-1] < 0:
        route_plan["total_evacuees"] += -route_plan["total_evacuees"].iloc[-1]

    # second calculate the column for location wise progress
    new_route_plan = pd.DataFrame()
    for location in np.unique(route_plan["evacuated_location"]):
        sub_route_plan = route_plan[route_plan["evacuated_location"] == location]
        if location != "None":
            # find total evacuations from scenario file
            print(location)
            print(scenario_file_sub)
            print(sub_route_plan)
            evacuations_loc = scenario_file_sub["Demand"][scenario_file_sub["Location"] == location] - scenario_file_sub["private_evac"][scenario_file_sub["Location"] == location]
            sub_route_plan["location_evacuees"] = evacuations_loc
            sub_route_plan = sub_route_plan.sort_values(by="load_end_time", ascending = True)
            sub_route_plan["location_evacuees"].iloc[0] = int(evacuations_loc - sub_route_plan["evacuees"].iloc[0])
            for j in range(1, len(sub_route_plan)):
                sub_route_plan["location_evacuees"].iloc[j] = int(sub_route_plan["location_evacuees"].iloc[j-1] - sub_route_plan["evacuees"].iloc[j])
            if sub_route_plan["location_evacuees"].iloc[-1] < 0:
                sub_route_plan["location_evacuees"] += -sub_route_plan["location_evacuees"].iloc[-1]
        else:
            pass
        new_route_plan = new_route_plan.append(sub_route_plan, ignore_index = True)
    new_route_plan = new_route_plan.sort_values(by="load_end_time", ascending = True)
    # print(new_route_plan)

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

    scenario_file = pd.read_csv(os.path.join(path, 'input', 'scenarios_old.csv'))

    # check if a solution directory exists
    if not os.path.exists(os.path.join(path, 'Solutions')):
        print("No solution directory exists yet, run experiment first")

    else:
        route_plans = pd.DataFrame()
        for file in os.listdir(solutions):
            if (os.path.isfile(os.path.join(solutions, file))) and ("route_plan" in file) and ("balanced" not in file):
                scenario_name = re.sub(" ", "_", file).split(":_")[1].split("_GUROBI")[0]
                route_plan_sc = pd.read_csv(os.path.join(solutions, file))
                route_plan_sc["scenario"] = scenario_name
                route_plan_sc = add_evacuees_remaining_col(route_plan_sc, scenario_file)
                plot_progress_scenario(route_plan_sc, os.path.join(solutions, scenario_name + '_progress.jpg'))
                plot_progress_all(route_plan_sc, os.path.join(solutions, scenario_name + '_progress_all.jpg'), scenario_file)
                route_plans = route_plans.append(route_plan_sc.iloc[:,1:], ignore_index = True)

if __name__ == "__main__":
    main()