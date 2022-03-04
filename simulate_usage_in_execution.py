"""
@fiete
March 3, 2022
"""

import pandas as pd
import argparse
import os
import numpy as np

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-path", help="the path of the ICEP instance files")

    args = parser.parse_args()

    # get directory path
    dirname = os.getcwd()

    rel_path = args.path
    path = os.path.join(dirname, rel_path)

    source = os.path.join(path, 'input/')
    inc_source = os.path.join(path, 'incidences/')

    # read in calculated route plan
    existing_route_plan = pd.read_csv(os.path.join(path, 'Solutions', 'route_plan_scenario_GUROBI.csv'))

    # read in scenario demand
    demand_plan = pd.read_csv(os.path.join(path, 'input', 'scenarios.csv'))

    # read in vessels
    vessels = pd.read_csv(os.path.join(path, 'input', 'vessels.csv'))

    # read in delta arcs
    deltas = pd.read_csv(os.path.join(path, 'incidences', 'delta.csv'))

    # read in gammas
    gammas = pd.read_csv(os.path.join(path, 'incidences', 'gamma.csv'))

    # vessel compatibility
    comp = pd.read_csv(os.path.join(path, 'input', 'vessel compatibility.csv'))

    # check if all evacuees have been evacuated
    for location in np.unique(demand_plan['Location']):
        sub_route_plan = existing_route_plan[existing_route_plan['evacuated_location'] == location]
        # print(sub_route_plan['evacuees'].sum())
        # print(float(demand_plan['Actual_demand'][demand_plan['Location'] == location]))
        if sub_route_plan['evacuees'].sum() < float(demand_plan['Actual_demand'][demand_plan['Location'] == location]):
            remaining_evacuees = float(demand_plan['Actual_demand'][demand_plan['Location'] == location]) - sub_route_plan['evacuees'].sum()

            # select last resource that visited to ensure feasibility
            sub_route_plan = sub_route_plan.sort_values(by='route_start_time', ascending = True)
            resource_used = sub_route_plan['resource_id'].iloc[-1]

            resource_route_plan = existing_route_plan[existing_route_plan['resource_id'] == resource_used]
            resource_route_plan = resource_route_plan.sort_values(by='route_start_time', ascending = True)

            # organize more trips by the last resource that visited the location until all evacuees are left
            while remaining_evacuees > 0:
                # print(remaining_evacuees)

                route_start = float(resource_route_plan['load_end_time'].iloc[-1])
                route_end = route_start + float(float(deltas['Distance'].loc[(deltas['Origin'] == resource_route_plan['destination'].iloc[-1]) & (deltas['Destination'] == sub_route_plan['origin'].iloc[-1])]) / float(vessels['vmax'].loc[vessels['Vessel_name'] == resource_used])) * 60
                load_start = float(route_end)
                load_end = load_start + float(vessels['loading time'].loc[vessels['Vessel_name'] == resource_used])

                # add trip back to evacuation location
                existing_route_plan = existing_route_plan.append({'segment_id': existing_route_plan['segment_id'].iloc[-1] + 1,
                                                                  'resource_id': resource_used,
                                                                  'route_segment_id': resource_route_plan['route_segment_id'].iloc[-1] + 1,
                                                                  'origin': resource_route_plan['destination'].iloc[-1],
                                                                  'destination': sub_route_plan['origin'].iloc[-1],
                                                                  'route_start_time': route_start,
                                                                  'route_end_time': route_end,
                                                                  'load_start_time': load_start,
                                                                  'load_end_time': load_end,
                                                                  'resource_speed': float(vessels['vmax'].loc[vessels['Vessel_name'] == resource_used]),
                                                                  'evacuees': 0,
                                                                  'evacuated_location': "None",
                                                                  'post_run_correction': "yes"
                                                                  }, ignore_index = True)

                candidate_return_docks = comp['Dock'][(comp['Resource'] == resource_used) & (comp['Compatibility'] == 1)]
                distances = gammas[gammas['Origin'] == existing_route_plan['destination'].iloc[-1]]
                distances = pd.merge(distances, candidate_return_docks, left_on = 'Origin', right_on = 'Dock', how = 'inner')
                distances = distances.sort_values(by='Distance', ascending = True)

                route_start = float(existing_route_plan['load_end_time'].iloc[-1])
                route_end = route_start + (float(distances['Distance'].iloc[0]) / float(vessels['v_loaded'].loc[vessels['Vessel_name'] == resource_used])) * 60
                load_start = float(route_end)
                load_end = float(load_start) + float(vessels['loading time'].loc[vessels['Vessel_name'] == resource_used])

                # add return trip to closest location
                existing_route_plan = existing_route_plan.append({'segment_id': existing_route_plan['segment_id'].iloc[-1] + 1,
                                                                  'resource_id': resource_used,
                                                                  'route_segment_id': resource_route_plan['route_segment_id'].iloc[-1] + 2,
                                                                  'origin': existing_route_plan['destination'].iloc[-1],
                                                                  'destination': distances['Destination'].iloc[0],
                                                                  'route_start_time': route_start,
                                                                  'route_end_time': route_end,
                                                                  'load_start_time': load_start,
                                                                  'load_end_time': load_end,
                                                                  'resource_speed': float(vessels['v_loaded'].loc[vessels['Vessel_name'] == resource_used]),
                                                                  'evacuees': float(max(float(vessels['max_cap'].loc[vessels['Vessel_name'] == resource_used]), remaining_evacuees)),
                                                                  'evacuated_location': location,
                                                                  'post_run_correction': "yes"
                                                                  }, ignore_index = True)

                # update evacuees
                remaining_evacuees = max(remaining_evacuees - max(float(vessels['max_cap'].loc[vessels['Vessel_name'] == resource_used]), remaining_evacuees), 0)

                # update route plans
                sub_route_plan = existing_route_plan[existing_route_plan['evacuated_location'] == location]
                # print(sub_route_plan)
                sub_route_plan = sub_route_plan.sort_values(by='route_start_time', ascending = True)
                resource_route_plan = existing_route_plan[existing_route_plan['resource_id'] == resource_used]
                resource_route_plan = resource_route_plan.sort_values(by='route_start_time', ascending = True)
                # print(resource_route_plan)

    existing_route_plan.to_csv(os.path.join(path, 'Solutions', 'route_plan_scenario_GUROBI_after_TRUE_DEMAND_REVEAL.csv'), index = False)

    return(-1)

if __name__ == '__main__':
    main()