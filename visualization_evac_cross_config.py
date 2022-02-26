"""
@fiete
November 23, 2021
"""

import pandas as pd
import os
import argparse
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# auxiliary function for reading and plotting data

def plot_scenario_config_variation(route_plan, filepath, scenario_file):
    """A method to plot the variation across different configurations for a given scenario"""

    new_route_plan = pd.DataFrame()

    for i in np.unique(route_plan['fleet size']):
        for j in np.unique(route_plan['staging choice']):
            sub_route_plan = route_plan[(route_plan['fleet size'] == i) & (route_plan['staging choice'] == j)]
            for k in np.unique(sub_route_plan['evacuated_location']):
                if k != "None":
                    evac_no_loc = scenario_file["Demand"][(scenario_file["Location"] == k) & (scenario_file["Scenario"] == route_plan['scenario'].iloc[0])].values[0] - \
                                  scenario_file["private_evac"][(scenario_file["Location"] == k) & (scenario_file["Scenario"] == route_plan['scenario'].iloc[0])].values[0]
                    # print(k)
                    # print(evac_no_loc)
                    sub_route_plan = sub_route_plan.append({'location_evacuees': evac_no_loc,
                                            'load_end_time': 0,
                                            'load_start_time': 0,
                                            'evacuated_location': k,
                                            'scenario': sub_route_plan['scenario'].iloc[0],
                                            'fleet size': i,
                                            'staging choice': j}, ignore_index = True)
            # print(sub_route_plan)
            new_route_plan = new_route_plan.append(sub_route_plan, ignore_index = True)

    new_route_plan = new_route_plan.sort_values(by=['load_end_time', 'location_evacuees'], ascending = [True, False])
    # print(route_plan)

    # for better display
    imp_route_plan = pd.DataFrame()
    for j in np.unique(new_route_plan['evacuated_location']):
        for k in np.unique(new_route_plan['staging choice']):
            for l in np.unique(new_route_plan['fleet size']):
                sub_route_plan = new_route_plan[(new_route_plan['evacuated_location'] == j) &
                                                (new_route_plan['staging choice'] == k) &
                                                (new_route_plan['fleet size'] == l)]
                for i in range(1,len(sub_route_plan)):
                    if sub_route_plan['load_end_time'].iloc[i] == sub_route_plan['load_end_time'].iloc[i-1]:
                        sub_route_plan['load_end_time'].iloc[i] += 0.01
                imp_route_plan = imp_route_plan.append(sub_route_plan, ignore_index = True)

    imp_route_plan = imp_route_plan[imp_route_plan["evacuated_location"] != "None"]

    imp_route_plan = imp_route_plan.sort_values(by=['location_evacuees','load_end_time'], ascending = [False, True])

    for t in np.unique(imp_route_plan['evacuated_location']):
        rp = imp_route_plan[imp_route_plan['evacuated_location'] == t]
        lm = sns.lineplot(x='load_end_time', y='location_evacuees', data=rp,
                          hue='staging choice', style = 'fleet size', #'staging_choice',
                          drawstyle='steps-post', legend = True, markers = True, estimator=None
                          )
        lm.set_title('Evacuation progress for ' + str(t) + ' in ' + re.sub("_", " ", route_plan['scenario'].iloc[-1]))
        lm.set(xlabel='time in min', ylabel='remaining evacuees')
        plt.legend(title='Affected location')

        pic = lm.get_figure()
        pic.savefig(filepath + '_' + str(t) + '.jpg')
        plt.close()


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
            evacuations_loc = scenario_file_sub["Demand"][scenario_file_sub["Location"] == location] - scenario_file_sub["private_evac"][scenario_file_sub["Location"] == location].values[0]
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
    kn_path = os.path.join(dirname, rel_path)

    scenario_file = pd.read_csv(os.path.join(kn_path, 'small_fleet_default', 'input', 'scenarios_old.csv'))

    for scenario in np.unique(scenario_file['Scenario']):
        print(re.sub("Scenario ", "_", scenario))
        config_plans = pd.DataFrame()
        for file in os.listdir(kn_path):
            # print(file)
            if (not file.startswith('.')) and (not file.startswith('Scenario')):
                route_plan = pd.read_csv(os.path.join(kn_path, file, 'Solutions', 'route_plan_scenario' + re.sub("Scenario ", "_", scenario) + '_GUROBI.csv'))
                route_plan["scenario"] = scenario
                if file.startswith('large'):
                    route_plan["fleet size"] = 'entire fleet'
                if file.startswith('small'):
                    route_plan["fleet size"] = 'primary fleet'
                if file.endswith('default'):
                    route_plan["staging choice"] = 'nominal'
                if file.endswith('staging'):
                    route_plan["staging choice"] = 'staging on Keats and Pasley Island'
                if file.endswith('keats'):
                    route_plan["staging choice"] = 'staging on Keats Island'
                # print(route_plan)
                route_plan = add_evacuees_remaining_col(route_plan, scenario_file)
                config_plans = config_plans.append(route_plan, ignore_index = True)
        config_plans['scenario'] = scenario
        # plotting
        plot_scenario_config_variation(config_plans,
                                       os.path.join(kn_path, scenario + '_progress_comparison'),
                                       scenario_file)


if __name__ == "__main__":
    main()