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

    # read in zetas
    zetas = pd.read_csv(os.path.join(path, 'incidences', 'zeta.csv'))

    # vessel compatibility
    comp = pd.read_csv(os.path.join(path, 'input', 'vessel compatibility.csv'))

    # island docks
    island_docks = pd.read_csv(os.path.join(path, 'input', 'island docks.csv'))

    # check if all evacuees have been evacuated
    for location in np.unique(demand_plan['Location']):

        sub_route_plan = existing_route_plan[existing_route_plan['evacuated_location'] == location]
        # print(sub_route_plan['evacuees'].sum())
        # print(float(demand_plan['Actual_demand'][demand_plan['Location'] == location]))
        if sub_route_plan['evacuees'].sum() < float(demand_plan['Actual_demand'][demand_plan['Location'] == location]):
            remaining_evacuees = float(demand_plan['Actual_demand'][demand_plan['Location'] == location]) - sub_route_plan['evacuees'].sum()

            # select last resource that visited to ensure feasibility, if previous visit exists, otherwise choose next available that is compatible
            sub_route_plan = sub_route_plan.sort_values(by='route_start_time', ascending = True)
            if not sub_route_plan.empty:
                resource_used = sub_route_plan['resource_id'].iloc[-1]
            else:
                related_docks = island_docks[island_docks['Location'] == location]
                possible_resources = []
                for dock in related_docks['Dock']:
                    compatible_vessels = comp['Resource'][(comp['Dock'] == dock) & (comp['Compatibility'] == 1)].tolist()
                    possible_resources.extend(compatible_vessels)
                usable_resources = np.unique(possible_resources)
                # check with resource is closest to emergency dock
                current_distances = pd.DataFrame()
                for resource in usable_resources:
                    res_plan = existing_route_plan[existing_route_plan['resource_id'] == resource]
                    res_plan.sort_values(by='route_start_time', ascending = True)
                    if not res_plan.empty:
                        current_location = res_plan['destination'].iloc[-1]
                    else:
                        current_location = str(vessels['Regular_origin'][vessels['Vessel_name'] == resource])
                    distances_to_evac = []
                    for dock in related_docks['Dock']:
                        distance_to_evac_location = deltas['Distance'][(deltas['Origin'] == current_location) & (deltas['Destination'] == dock)]
                        distances_to_evac.append(distance_to_evac_location)
                    # choose mean
                    distance_to_evac = distances_to_evac[0]
                    current_distances = current_distances.append({'Resource': resource,
                                                                  'Distance': str(distance_to_evac),
                                                                  'Current_location': current_location,
                                                                  'Evac_dock': related_docks['Dock'].iloc[0]},
                                                                 ignore_index = True)

                resource_used = current_distances['Resource'][current_distances['Distance'] == current_distances['Distance'].min()].values[0]
                current_location = current_distances['Current_location'][current_distances['Resource'] == resource_used].values[0]
                targeted_evac_dock = current_distances['Evac_dock'][current_distances['Resource'] == resource_used].values[0]
                # print(resource_used)

            resource_route_plan = existing_route_plan[existing_route_plan['resource_id'] == resource_used]
            resource_route_plan = resource_route_plan.sort_values(by='route_start_time', ascending = True)

            # print(resource_used)
            # print(resource_route_plan)

            # organize more trips by the last resource that visited the location until all evacuees are left
            while remaining_evacuees > 0:
                # print(remaining_evacuees)

                if resource_route_plan.empty:
                    route_start = float(vessels['time to availability'][vessels['Vessel_name'] == resource_used])
                    route_end = route_start + float(float(deltas['Distance'].loc[(deltas['Origin'] == current_location) & (deltas['Destination'] == sub_route_plan['origin'].iloc[-1])]) / float(vessels['vmax'].loc[vessels['Vessel_name'] == resource_used])) * 60
                elif sub_route_plan.empty:
                    route_start = float(resource_route_plan['load_end_time'].iloc[-1])
                    route_end = route_start + float(float(zetas['Distance'].loc[(zetas['Origin'] == resource_route_plan['destination'].iloc[-1]) & (zetas['Destination'] == targeted_evac_dock)]) / float(vessels['vmax'].loc[vessels['Vessel_name'] == resource_used])) * 60
                elif resource_route_plan.empty and sub_route_plan.empty:
                    route_start = float(vessels['time to availability'][vessels['Vessel_name'] == resource_used])
                    route_end = route_start + float(float(zetas['Distance'].loc[(zetas['Origin'] == current_location) & (zetas['Destination'] == targeted_evac_dock)]) / float(vessels['vmax'].loc[vessels['Vessel_name'] == resource_used])) * 60
                else:
                    route_start = float(resource_route_plan['load_end_time'].iloc[-1])
                    route_end = route_start + float(float(zetas['Distance'].loc[(zetas['Origin'] == resource_route_plan['destination'].iloc[-1]) & (zetas['Destination'] == sub_route_plan['origin'].iloc[-1])]) / float(vessels['vmax'].loc[vessels['Vessel_name'] == resource_used])) * 60
                load_start = float(route_end)
                load_end = load_start + float(vessels['loading time'].loc[vessels['Vessel_name'] == resource_used])

                # add trip back to evacuation location

                if resource_route_plan.empty:
                    route_segment_id = 1
                    origin = current_location
                    destination = sub_route_plan['origin'].iloc[-1]
                elif sub_route_plan.empty:
                    route_segment_id = resource_route_plan['route_segment_id'].iloc[-1] + 1
                    origin = resource_route_plan['destination'].iloc[-1]
                    destination = targeted_evac_dock
                elif resource_route_plan.empty and sub_route_plan.empty:
                    route_segment_id = 1
                    origin = current_location
                    destination = targeted_evac_dock
                else:
                    route_segment_id = resource_route_plan['route_segment_id'].iloc[-1] + 1
                    origin = resource_route_plan['destination'].iloc[-1]
                    destination = sub_route_plan['origin'].iloc[-1]

                existing_route_plan = existing_route_plan.append({'segment_id': existing_route_plan['segment_id'].iloc[-1] + 1,
                                                                  'resource_id': resource_used,
                                                                  'route_segment_id': route_segment_id,
                                                                  'origin': origin,
                                                                  'destination': destination,
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
                                                                  'route_segment_id': route_segment_id + 1,
                                                                  'origin': existing_route_plan['destination'].iloc[-1],
                                                                  'destination': distances['Destination'].iloc[0],
                                                                  'route_start_time': route_start,
                                                                  'route_end_time': route_end,
                                                                  'load_start_time': load_start,
                                                                  'load_end_time': load_end,
                                                                  'resource_speed': float(vessels['v_loaded'].loc[vessels['Vessel_name'] == resource_used]),
                                                                  'evacuees': float(min(float(vessels['max_cap'].loc[vessels['Vessel_name'] == resource_used]), remaining_evacuees)),
                                                                  'evacuated_location': location,
                                                                  'post_run_correction': "yes"
                                                                  }, ignore_index = True)

                # update evacuees
                remaining_evacuees = max(remaining_evacuees - min(float(vessels['max_cap'].loc[vessels['Vessel_name'] == resource_used]), remaining_evacuees), 0)

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