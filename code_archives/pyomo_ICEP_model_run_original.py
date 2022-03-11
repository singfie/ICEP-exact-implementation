"""
@fiete
December 9, 2020
"""


from pyomo.environ import *
import pandas as pd
import time
import os
import numpy as np
import argparse
from os.path import basename

# import modules
from code_archives import pyomo_ICEP_model_generator_original


def reformat_compatibility(matrix):
    """An auxiliary function to reformat the compatibility file"""

    newframe = pd.DataFrame()
    j = 1
    while j < len(matrix.columns):
        i = 0
        while i < matrix.shape[0]:
            newframe = newframe.append({"Dock": matrix.iloc[i,0],
                                        "Resource": matrix.columns[j],
                                        "Compatibility": int(matrix.iloc[i,j])}, ignore_index = True)
            i += 1
        j += 1
    newframe = newframe.drop_duplicates()

    return newframe

def run_S_ICEP_model(m, dirname, vessel_source, is_docks_source, runtime_limit = 3600):

    start_time = time.time()

    # if __name__ == '__main__':
    # from pyomo.opt import SolverFactory
    # import pyomo.environ
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
    print('')
    print('Optimal objective value:')
    print(round(value(m.objective),2))
    print('')
    print('Optimal shipment plans')

    print('    ','Total evacuation time with selected vessel fleet:', value(m.comp), 'minutes')
    print('    ','Private evacuations:')
    for a in m.flat:
        print('        ', 'From ', a[0], ':', int(m.flat[a].value), 'people')
    for j in m.i:
        print('    ',j, 'completion time:', value(m.time_record[j]))
        print('    ',j, 'routing plan/shipping plan:')
        for t in m.k:#round_trips:
            if t == min(m.k):#min(round_trips):
                for a in m.w:
                    if j in a and t in a:
                        if np.round(m.w[a].value) == 1:
                            print('            ', '(', a[0],')','(A) from', a[1], 'to',a[2], 'on trip', a[0])
                for a in m.x:#m.flab:
                    if j in a and t in a:
                        if np.round(m.x[a].value) == 1:
                        #if m.flbc[a].value != 0:
                            print('            ', '(', a[2],')','(T) from', a[0].replace('load', ''), 'to',a[3].replace('dock', ''), 'on trip', a[2],':', round(m.flbc[a].value), 'passengers')
                for a in m.y:
                    if j in a and t == a[2]:
                        if np.round(m.y[a].value) == 1:
                            print('            ', '(', a[2],')','(R) from', a[0], 'to',a[3], 'on trip', a[2])
            else:
                for a in m.x:#m.flab:
                    if j in a and t in a:
                        if np.round(m.x[a].value) == 1:
                        #if m.flbc[a].value != 0:
                            print('            ', '(', a[2],')','(T) from', a[0].replace('load', ''), 'to',a[3].replace('dock', ''), 'on trip', a[2],':', round(m.flbc[a].value), 'passengers')
                for a in m.y:
                    if j in a and t == a[2]:
                        if np.round(m.y[a].value) == 1:
                            print('            ', '(', a[2],')','(R) from', a[0], 'to',a[3], 'on trip', a[2])

    print('')     
    print('Vessels should follow this shipment plan given their initial locations as per input data.')
    print('This schedule respects distances in between different locations, vessel capabilities')
    print('inital vessel locations and compatibility between vessels and docks and local demand patterns')
    print('as per input data.')
    print('This model does not consider the constraints of on-land transportation.')

    # create target directories
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    DATA_DIR = os.path.join(BASE_DIR, "ICEP-exact-implementation", dirname)
    # print(DATA_DIR)
    SOL_DIR = os.path.join(DATA_DIR, "Solutions")
    # print(SOL_DIR)


    # create new folders if they do not exist
    if os.path.exists(SOL_DIR):
        pass
    else:
        os.makedirs(SOL_DIR)

    # change working directory and set seed
    os.chdir(SOL_DIR)

    with open(os.path.join(SOL_DIR, 'performance_statistics_Gurobi.txt'), 'w') as f:
        f.write("\n\nInstance,NumResources,NumMaxTrips,TotalTime,Cost\n")
        f.write(f"{basename(str(dirname))},"
          f"{len(m.i)},{len(m.k)},{run_time},"
          f"{float(value(m.objective)):.0f}")
        f.close()

    #### MAKE ROUTE DETAILS

    # create an empty template
    route_details = pd.DataFrame({
                   'segment_id': pd.Series([], dtype='int'),
                   'resource_id': pd.Series([], dtype='str'),
                   'route_segment_id': pd.Series([], dtype='int'),
                   'origin': pd.Series([], dtype='str'),
                   'destination': pd.Series([], dtype='str'),
                   'route_start_time': pd.Series([], dtype='float'),
                   'route_end_time': pd.Series([], dtype='float'),
                   'load_start_time': pd.Series([], dtype='float'),
                   'load_end_time': pd.Series([], dtype='float'),
                   'resource_speed': pd.Series([], dtype='float'),
                   'evacuees': pd.Series([], dtype='int'),
                   'evacuated_location': pd.Series([], dtype='str')
    })

    # assign the data to each entry
    for j in m.i:
        segment_id = 1
        for t in m.k:
            if t == min(m.k):
                for a in m.w:
                    if j in a and t in a:
                        if np.round(m.w[a].value) == 1:
                            route_start_time = m.h_c[a[1],j]
                            route_end_time = route_start_time + m.zeta_c[t,a[1],a[2],j]
                            load_start_time = route_end_time
                            load_end_time = load_start_time + m.b_c[a[2],j,t]
                            route_details = route_details.append({'segment_id': segment_id,
                                                                  'resource_id': j,
                                                                  'route_segment_id': segment_id,
                                                                  'origin': a[1],
                                                                  'destination': a[2],
                                                                  'route_start_time': route_start_time,
                                                                  'route_end_time': route_end_time,
                                                                  'load_start_time': load_start_time,
                                                                  'load_end_time': load_end_time,
                                                                  'resource_speed': vessel_source['vmax'].loc[vessel_source['Vessel_name'] == j].values[0],
                                                                  'evacuees': 0,
                                                                  'evacuated_location': "None"}, ignore_index = True)
                            segment_id += 1
                for a in m.x:
                    if j in a and t in a:
                        if np.round(m.x[a].value) == 1:
                            # print('            ', '(', a[3],')','(T) from', a[1].replace('load', ''), 'to',a[4].replace('dock', ''), 'on trip', a[3],':', round(m.flbc[a].value), 'passengers')
                            route_start_time = load_end_time
                            route_end_time = route_start_time + m.gamma_c[a[0],j,a[2],a[3],a[4]]
                            load_start_time = route_end_time
                            load_end_time = load_start_time + m.c_c[a[3],j,t]
                            route_details = route_details.append({'segment_id': segment_id,
                                                                  'resource_id': j,
                                                                  'route_segment_id': segment_id,
                                                                  'origin': a[0],
                                                                  'destination': a[3],
                                                                  'route_start_time': route_start_time,
                                                                  'route_end_time': route_end_time,
                                                                  'load_start_time': load_start_time,
                                                                  'load_end_time': load_end_time,
                                                                  'resource_speed': vessel_source['v_loaded'].loc[vessel_source['Vessel_name'] == j].values[0],
                                                                  'evacuees': round(m.flbc[a].value),
                                                                  'evacuated_location': is_docks_source["Location"][is_docks_source["Dock"] == a[0]].values[0]}, ignore_index = True)
                            segment_id += 1
                for a in m.y:
                    if j in a and t == a[2]:
                        if np.round(m.y[a].value) == 1:
                            # print('            ', '(', a[3],')','(R) from', a[1], 'to',a[4], 'on trip', a[3])
                            route_start_time = load_end_time
                            route_end_time = route_start_time + m.delta_c[a[0],j,a[2],a[3],a[4]]#m.delta_c[xi,c,ves,k1,b,k2]
                            load_start_time = route_end_time
                            load_end_time = load_start_time + m.b_c[a[3],j,t]
                            route_details = route_details.append({'segment_id': segment_id,
                                                                  'resource_id': j,
                                                                  'route_segment_id': segment_id,
                                                                  'origin': a[0],
                                                                  'destination': a[3],
                                                                  'route_start_time': route_start_time,
                                                                  'route_end_time': route_end_time,
                                                                  'load_start_time': load_start_time,
                                                                  'load_end_time': load_end_time,
                                                                  'resource_speed': vessel_source['v_loaded'].loc[vessel_source['Vessel_name'] == j].values[0],
                                                                  'evacuees': 0,
                                                                  'evacuated_location': "None"}, ignore_index = True)
                            segment_id += 1

            else:
                for a in m.x:
                    if j in a and t in a:
                        if np.round(m.x[a].value) == 1:
                            # print('            ', '(', a[3],')','(T) from', a[1].replace('load', ''), 'to',a[4].replace('dock', ''), 'on trip', a[3],':', round(m.flbc[a].value), 'passengers')
                            route_start_time = load_end_time
                            route_end_time = route_start_time + m.gamma_c[a[0],j,a[2],a[3],a[4]]
                            load_start_time = route_end_time
                            load_end_time = load_start_time + m.c_c[a[3],j,t]
                            route_details = route_details.append({'segment_id': segment_id,
                                                                  'resource_id': j,
                                                                  'route_segment_id': segment_id,
                                                                  'origin': a[0],
                                                                  'destination': a[3],
                                                                  'route_start_time': route_start_time,
                                                                  'route_end_time': route_end_time,
                                                                  'load_start_time': load_start_time,
                                                                  'load_end_time': load_end_time,
                                                                  'resource_speed': vessel_source['v_loaded'].loc[vessel_source['Vessel_name'] == j].values[0],
                                                                  'evacuees': round(m.flbc[a].value),
                                                                  'evacuated_location': is_docks_source["Location"][is_docks_source["Dock"] == a[0]].values[0]}, ignore_index = True)
                            segment_id += 1
                for a in m.y:
                    if j in a and t == a[2]:
                        if np.round(m.y[a].value) == 1:
                            # print('            ', '(', a[3],')','(R) from', a[1], 'to',a[4], 'on trip', a[3])
                            route_start_time = load_end_time
                            route_end_time = route_start_time + m.delta_c[a[0],j,a[2],a[3],a[4]]#m.delta_c[xi,c,ves,k1,b,k2]
                            load_start_time = route_end_time
                            load_end_time = load_start_time + m.b_c[a[3],j,t]
                            route_details = route_details.append({'segment_id': segment_id,
                                                                  'resource_id': j,
                                                                  'route_segment_id': segment_id,
                                                                  'origin': a[0],
                                                                  'destination': a[3],
                                                                  'route_start_time': route_start_time,
                                                                  'route_end_time': route_end_time,
                                                                  'load_start_time': load_start_time,
                                                                  'load_end_time': load_end_time,
                                                                  'resource_speed': vessel_source['v_loaded'].loc[vessel_source['Vessel_name'] == j].values[0],
                                                                  'evacuees': 0,
                                                                  'evacuated_location': "None"}, ignore_index = True)
                            segment_id += 1

        route_details.to_csv(os.path.join(SOL_DIR, 'route_plan_scenario_GUROBI.csv'), index = False)
        

    #### END ROUTE DETAILS

    # calculate total expected cost 
    # totcost = sum(value(m.cfix[i]) * value(m.z[i]) for i in m.i) + sum(value(m.ps[xi]) * sum(value(m.var_cost[i]) * value(m.u[i, xi]) for i in m.i) for xi in m.xi)

    print(round(value(m.objective),2))

    return(round(value(m.objective),2), run_time)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-path", help="the path of the ICEP instance files")
    parser.add_argument("-run_time_limit", type = float, help="the upper time limit for the algorithm run time")

    args = parser.parse_args()

    # get directory path
    dirname = os.getcwd()

    rel_path = args.path
    path = os.path.join(dirname, rel_path)

    source = os.path.join(path, 'input/')
    inc_source = os.path.join(path, 'incidences/')

    # check if a solution directory exists
    if not os.path.exists(os.path.join(path, 'Solutions')):
        os.makedirs(os.path.join(path, "Solutions"))

    run_time_limit = args.run_time_limit
    # print(run_time_limit)

    # read in data source files for nodes
    vessel_source = pd.read_csv(source + 'vessels.csv', index_col=False,
                                header=0, delimiter = ',', skipinitialspace=True)
    #print(vessel_source)
    trips_source = pd.read_csv(source + 'roundtrips.csv', index_col=False,
                               header=0, delimiter = ',', skipinitialspace=True)
    #print(trips_source)
    demand_source = pd.read_csv(source + 'scenarios.csv', index_col = False,
                                  header=0, delimiter = ',', skipinitialspace=True)
    #print(scenarios_src)
    vessel_pos_source = pd.read_csv(source + 'initial vessel docks.csv', index_col = False,
                                  header=0, delimiter = ',', skipinitialspace=True)
    #print(vessel_pos_source)
    src_node_source = pd.read_csv(source + 'island source.csv', index_col=False,
                                 header=0, delimiter = ',', skipinitialspace=True)
    #print(src_node_source)
    is_locs_source = pd.read_csv(source + 'island locations.csv', index_col=False,
                                 header=0, delimiter = ',', skipinitialspace=True)
    #print(is_locs_source)
    is_docks_source = pd.read_csv(source + 'island docks.csv', index_col=False,
                                  header=0, delimiter = ',', skipinitialspace=True)
    #print(is_docks_source)
    mn_locs_source = pd.read_csv(source + 'mainland locations.csv', index_col=False,
                                 header=0, delimiter = ',', skipinitialspace=True)
    #print(mn_locs_source)
    mn_docks_source = pd.read_csv(source + 'mainland docks.csv', index_col=False,
                                  header=0, delimiter = ',', skipinitialspace=True)
    #print(mn_docks_source)
    # vessel compatibility
    compat_source = pd.read_csv(source + 'vessel compatibility.csv', index_col=False,
                        header=0, delimiter = ',', skipinitialspace=True)
    #print(compat_source)

    # check the forma tof the compat_source. If it is wrong, change the format:
    if compat_source.shape[1] > 3:
        compat_source = reformat_compatibility(compat_source)
    else:
        pass

    alpha_source = pd.read_csv(inc_source + 'alpha.csv', index_col=False,
                    header=0, delimiter = ',', skipinitialspace=True)
    #print(beta_source)
    beta_source = pd.read_csv(inc_source + 'beta.csv', index_col=False,
                        header=0, delimiter = ',', skipinitialspace=True)
    #print(beta_source)
    gamma_source = pd.read_csv(inc_source + 'gamma.csv', index_col=False,
                        header=0, delimiter = ',', skipinitialspace=True)
    #print(gamma_source)
    delta_source = pd.read_csv(inc_source + 'delta.csv', index_col=False,
                        header=0, delimiter = ',', skipinitialspace=True)
    #print(delta_source)
    epsilon_source = pd.read_csv(inc_source + 'epsilon.csv', index_col=False,
                        header=0, delimiter = ',', skipinitialspace=True)
    #print(epsilon_source)
    zeta_source = pd.read_csv(inc_source + 'zeta.csv', index_col=False,
                        header=0, delimiter = ',', skipinitialspace=True)
    #print(zeta_source)
    lambda_source = pd.read_csv(inc_source + 'lambda.csv', index_col=False,
                        header=0, delimiter = ',', skipinitialspace=True)

    # distances and compatibility
    distance_data = pd.read_csv(inc_source + 'distance matrix.csv', index_col=0,
                        header=0, delimiter = ',', skipinitialspace=True)
    #print(distance_source)

    print("Starting GUROBI solver to S-ICEP...")
    print("")

    start_time = time.time()

    m = pyomo_ICEP_model_generator_original.main(vessel_source, vessel_pos_source,
                                                 is_locs_source, is_docks_source, mn_locs_source, mn_docks_source, compat_source,
                                                 distance_data, trips_source, demand_source, src_node_source,
                                                 alpha_source, beta_source, gamma_source, delta_source, epsilon_source, zeta_source,
                                                 lambda_source)

    optimal_solution, run_time = run_S_ICEP_model(m, rel_path, vessel_source, is_docks_source, runtime_limit = run_time_limit)

    end_time = time.time()
    total_time = end_time - start_time

    print('Time to solution:', total_time)

    # print("************************************")
    # print("The best objective value obtained:", best_cost[0])
    # print("The best set of route plans obtained:")
    # for i in range(len(best_route_set)):
    #     print("Scenario", i+1, ":")
    #     print("Population not evacuated:")
    #     print(not_evacuated[i])
    #     print("Evacuation time:")
    #     print(best_evacuation_times[i])
    #     print(best_route_set[i])
    #     best_route_set[i].to_csv(os.path.join(path, 'solution/Greedy_S_ICEP_best_route_plan_scenario_') + str(i+1) + ".csv")

    # # write a performance file
    # performance_metrics = open(os.path.join(path, "solution/Greedy_S_ICEP_solution_metrics.txt"),"w+")
    # for i in range(len(best_route_set)):
    #     performance_metrics.write("Input parameters:\n")
    #     performance_metrics.write("Penalty: " + str(penalty) + "\n")
    #     performance_metrics.write("Upper time limit: " + str(time_limit) + "\n")
    #     performance_metrics.write("")
    #     performance_metrics.write("Results: \n")
    #     performance_metrics.write("Scenario " + str(i+1) + ": \n")
    #     performance_metrics.write("Population not evacuated: " + str(not_evacuated[i][0]) + "\n")
    #     performance_metrics.write("Evacuation time: " + str(best_evacuation_times[i]) + "\n")
    #     performance_metrics.write("Algorithm run time: " + str(run_time))
    # performance_metrics.close()

    # generate the evolution plots
    # create_best_cost_plot(best_cost_evo, path)
    # create_total_cost_plot(all_cost_evo, path)

if __name__ == "__main__":
    main()