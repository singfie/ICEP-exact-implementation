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
import pyomo_ICEP_model_generator

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

def run_S_ICEP_model(m, dirname, vessel_source, is_docks_source, objective_function, runtime_limit = 3600):

    import gurobipy

    start_time = time.time()

    # if __name__ == '__main__':
    # from pyomo.opt import SolverFactory
    # import pyomo.environ
    opt = SolverFactory('gurobi')
    opt.options['IntFeasTol']= 10e-10
    opt.options['MIPGap'] = 0.1 #1e-4
    opt.options['TimeLimit'] = runtime_limit
    results = opt.solve(m, tee=False)
    m.solutions.load_from(results)

    end_time = time.time()
    run_time = end_time - start_time

    # print('Optimal solution for selected input data')
    # print('')
    # print('Optimal vessel fleet:')
    # for l in m.z:#vessels:
    #     if value(m.z[l]) == 1:
    #         print(l)
    # print('')
    # print('Optimal objective value:')
    # print(round(value(m.objective),2))
    # print('')
    # # print('Optimal evacuation time')
    # # for r in m.xi:
    # #     print(r)
    # #     for q in m.i:
    # #         print(m.u[q,r].value)
    # print('')
    totcost = sum(value(m.cfix[i]) * value(m.z[i]) for i in m.i) + sum(value(m.ps[xi]) * sum(value(m.var_cost[i]) * value(m.u[i, xi]) for i in m.i) for xi in m.xi)
    best_cost = value(m.objective)
    # print('Optimal cost:')
    # print(round(totcost,2))
    # print('')
    # print('Optimal shipment plans per scenario')
    # for k in m.xi:#scenarios:
    #     print(k)
    #     print('    ','Total evacuation time with selected vessel fleet:', value(m.comp[k]), 'minutes')
    #     print('    ','Private evacuations:')
    #     for a in m.flat:
    #         if k in a:
    #             print('        ', 'From ', a[1], ':', int(m.flat[a].value), 'people')
    #     print('    ','Evacuees left behind:')
    #     for n in m.flan:
    #         if k in n:
    #             print('        ', n[0], ':', int(m.flan[n].value), 'people')
    #     for j in m.z:#vessels:
    #         print('    ',j, 'completion time:', value(m.time_record[j,k]))
    #         print('    ',j, 'routing plan/shipping plan:')
    #         for t in m.k:#round_trips:
    #             if t == min(m.k):#min(round_trips):
    #                 for a in m.w:
    #                     if j in a and k in a and t in a:
    #                         if np.round(m.w[a].value) == 1:
    #                             print('            ', '(', a[1],')','(A) from', a[2], 'to',a[3], 'on trip', a[1])
    #                 for a in m.x:#m.flab:
    #                     if j in a and k in a and t in a:
    #                         if np.round(m.x[a].value) == 1:
    #                         #if m.flbc[a].value != 0:
    #                             print('            ', '(', a[3],')','(T) from', a[1].replace('load', ''), 'to',a[4].replace('dock', ''), 'on trip', a[3],':', round(m.flbc[a].value), 'passengers')
    #                 for a in m.y:
    #                     if j in a and k in a and t == a[3]:
    #                         if np.round(m.y[a].value) == 1:
    #                             print('            ', '(', a[3],')','(R) from', a[1], 'to',a[4], 'on trip', a[3])
    #             else:
    #                 for a in m.x:#m.flab:
    #                     if j in a and k in a and t in a:
    #                         if np.round(m.x[a].value) == 1:
    #                         #if m.flbc[a].value != 0:
    #                             print('            ', '(', a[3],')','(T) from', a[1].replace('load', ''), 'to',a[4].replace('dock', ''), 'on trip', a[3],':', round(m.flbc[a].value), 'passengers')
    #                 for a in m.y:
    #                     if j in a and k in a and t == a[3]:
    #                         if np.round(m.y[a].value) == 1:
    #                             print('            ', '(', a[3],')','(R) from', a[1], 'to',a[4], 'on trip', a[3])
    #
    # print('')
    # print('Vessels should follow this shipment plan given their initial locations as per input data.')
    # print('This schedule respects distances in between different locations, vessel capabilities')
    # print('inital vessel locations and compatibility between vessels and docks and local demand patterns')
    # print('as per input data.')
    # print('This model does not consider the constraints of on-land transportation.')

    # create target directories
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    DATA_DIR = os.path.join(BASE_DIR, "ICEP-exact-implementation/case_study_instances", dirname)
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

    with open(os.path.join(SOL_DIR, str(objective_function) + '_performance_statistics_Gurobi.txt'), 'w') as f:
        f.write("\n\nInstance,NumScenarios,NumResources,NumMaxTrips,TotalIterations,TotalTime,Cost")
        f.write(f"{basename(str(dirname))},"
          f"{len(m.xi)},{len(m.i)},{len(m.k)},{run_time},"
          f"{run_time},{best_cost:.0f}")
        f.close()

    #### MAKE ROUTE DETAILS

    for k in m.xi:

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
        for j in m.z:
            segment_id = 1
            for t in m.k:
                if t == min(m.k):
                    for a in m.w:
                        if j in a and k in a and t in a:
                            if np.round(m.w[a].value) == 1:
                                route_start_time = m.h_c[a[2],j,k]
                                route_end_time = route_start_time + m.zeta_c[k,t,a[2],a[3],j]
                                load_start_time = route_end_time
                                load_end_time = load_start_time + m.b_c[a[3],j,t,k]
                                route_details = route_details.append({'segment_id': segment_id,
                                                                      'resource_id': j,
                                                                      'route_segment_id': segment_id,
                                                                      'origin': a[2],
                                                                      'destination': a[3],
                                                                      'route_start_time': route_start_time,
                                                                      'route_end_time': route_end_time,
                                                                      'load_start_time': load_start_time,
                                                                      'load_end_time': load_end_time,
                                                                      'resource_speed': vessel_source['vmax'].loc[vessel_source['Vessel_name'] == j].values[0],
                                                                      'evacuees': 0,
                                                                      'evacuated_location': "None"}, ignore_index = True)
                                segment_id += 1
                    for a in m.x:
                        if j in a and k in a and t in a:
                            if np.round(m.x[a].value) == 1:
                                # print('            ', '(', a[3],')','(T) from', a[1].replace('load', ''), 'to',a[4].replace('dock', ''), 'on trip', a[3],':', round(m.flbc[a].value), 'passengers')
                                route_start_time = load_end_time
                                route_end_time = route_start_time + m.gamma_c[k,a[1],j,a[3],a[4],a[5]]
                                load_start_time = route_end_time
                                load_end_time = load_start_time + m.c_c[a[4],j,t,k]
                                route_details = route_details.append({'segment_id': segment_id,
                                                                      'resource_id': j,
                                                                      'route_segment_id': segment_id,
                                                                      'origin': a[1],
                                                                      'destination': a[4],
                                                                      'route_start_time': route_start_time,
                                                                      'route_end_time': route_end_time,
                                                                      'load_start_time': load_start_time,
                                                                      'load_end_time': load_end_time,
                                                                      'resource_speed': vessel_source['v_loaded'].loc[vessel_source['Vessel_name'] == j].values[0],
                                                                      'evacuees': round(m.flbc[a].value),
                                                                      'evacuated_location': is_docks_source["Location"][is_docks_source["Dock"] == a[1]].values[0]}, ignore_index = True)
                                segment_id += 1
                    for a in m.y:
                        if j in a and k in a and t == a[3]:
                            if np.round(m.y[a].value) == 1:
                                # print('            ', '(', a[3],')','(R) from', a[1], 'to',a[4], 'on trip', a[3])
                                route_start_time = load_end_time
                                route_end_time = route_start_time + m.delta_c[k,a[1],j,a[3],a[4],a[5]]#m.delta_c[xi,c,ves,k1,b,k2]
                                load_start_time = route_end_time
                                load_end_time = load_start_time + m.b_c[a[4],j,t,k]
                                route_details = route_details.append({'segment_id': segment_id,
                                                                      'resource_id': j,
                                                                      'route_segment_id': segment_id,
                                                                      'origin': a[1],
                                                                      'destination': a[4],
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
                        if j in a and k in a and t in a:
                            if np.round(m.x[a].value) == 1:
                                # print('            ', '(', a[3],')','(T) from', a[1].replace('load', ''), 'to',a[4].replace('dock', ''), 'on trip', a[3],':', round(m.flbc[a].value), 'passengers')
                                route_start_time = load_end_time
                                route_end_time = route_start_time + m.gamma_c[k,a[1],j,a[3],a[4],a[5]]
                                load_start_time = route_end_time
                                load_end_time = load_start_time + m.c_c[a[4],j,t,k]
                                route_details = route_details.append({'segment_id': segment_id,
                                                                      'resource_id': j,
                                                                      'route_segment_id': segment_id,
                                                                      'origin': a[1],
                                                                      'destination': a[4],
                                                                      'route_start_time': route_start_time,
                                                                      'route_end_time': route_end_time,
                                                                      'load_start_time': load_start_time,
                                                                      'load_end_time': load_end_time,
                                                                      'resource_speed': vessel_source['v_loaded'].loc[vessel_source['Vessel_name'] == j].values[0],
                                                                      'evacuees': round(m.flbc[a].value),
                                                                      'evacuated_location': is_docks_source["Location"][is_docks_source["Dock"] == a[1]].values[0]}, ignore_index = True)
                                segment_id += 1
                    for a in m.y:
                        if j in a and k in a and t == a[3]:
                            if np.round(m.y[a].value) == 1:
                                # print('            ', '(', a[3],')','(R) from', a[1], 'to',a[4], 'on trip', a[3])
                                route_start_time = load_end_time
                                route_end_time = route_start_time + m.delta_c[k,a[1],j,a[3],a[4],a[5]]#m.delta_c[xi,c,ves,k1,b,k2]
                                load_start_time = route_end_time
                                load_end_time = load_start_time + m.b_c[a[4],j,t,k]
                                route_details = route_details.append({'segment_id': segment_id,
                                                                      'resource_id': j,
                                                                      'route_segment_id': segment_id,
                                                                      'origin': a[1],
                                                                      'destination': a[4],
                                                                      'route_start_time': route_start_time,
                                                                      'route_end_time': route_end_time,
                                                                      'load_start_time': load_start_time,
                                                                      'load_end_time': load_end_time,
                                                                      'resource_speed': vessel_source['v_loaded'].loc[vessel_source['Vessel_name'] == j].values[0],
                                                                      'evacuees': 0,
                                                                      'evacuated_location': "None"}, ignore_index = True)
                                segment_id += 1

        route_details.to_csv(os.path.join(SOL_DIR, 'route_plan_scenario_' + k.replace('Scenario ', '') + '_GUROBI.csv'))
        

    #### END ROUTE DETAILS

    # calculate total expected cost 
    # totcost = sum(value(m.cfix[i]) * value(m.z[i]) for i in m.i) + sum(value(m.ps[xi]) * sum(value(m.var_cost[i]) * value(m.u[i, xi]) for i in m.i) for xi in m.xi)

    # print(totcost)

    print(best_cost)

    # for i in m.xi:
    #     print("Scenario", i, ":", m.comp[i].value)

    return(best_cost, run_time)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-path", help="the path of the ICEP instance files")
    parser.add_argument("-penalty", type = int, help="the penalty value applied to every evacuee not evacuated.")
    parser.add_argument("-route_time_limits", type = int, nargs="*", help="the upper time limits for the evacuation plan.")
    parser.add_argument("-run_time_limit", type = float, help="the upper time limit for the algorithm run time")
    parser.add_argument("-max_iterations", type = int, help="maximum number of iterations in approximation")
    parser.add_argument("-objective", type = str, default="balanced4", help="maximum number of iterations in approximation")

    args = parser.parse_args()

    # get directory path
    dirname = os.getcwd()

    rel_path = args.path
    path = os.path.join(dirname, 'test_instances', rel_path)

    source = os.path.join(path, 'input/')
    inc_source = os.path.join(path, 'incidences/')

    # check if a solution directory exists
    if not os.path.exists(os.path.join(path, 'Solutions')):
        os.makedirs(os.path.join(path, "Solutions"))

    # parse remaining arguments
    penalty = args.penalty
    # print(penalty)
    tls = args.route_time_limits

    run_time_limit = args.run_time_limit
    # print(run_time_limit)

    max_iterations = args.max_iterations

    # read in objective
    objective = args.objective
    if objective in (['conservative1', 'conservative2', 'balanced1', 'balanced2',
                          'balanced3', 'balanced4', 'economic1', 'economic2']):

        # read in data source files for nodes
        vessel_source = pd.read_csv(source + 'vessels.csv', index_col=False,
                                    header=0, delimiter = ',', skipinitialspace=True)
        #print(vessel_source)
        trips_source = pd.read_csv(source + 'roundtrips.csv', index_col=False,
                                   header=0, delimiter = ',', skipinitialspace=True)
        #print(trips_source)
        scenarios_source = pd.read_csv(source + 'scenarios.csv', index_col = False,
                                      header=0, delimiter = ',', skipinitialspace=True)

        if len(tls) != len(np.unique(scenarios_source["Scenario"])):
            raise Exception("Number of max route times provided does not match number of scenarios provided.")

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

        route_time_limits = dict.fromkeys(np.unique(scenarios_source["Scenario"]))
        i = 0
        for p in route_time_limits:
            route_time_limits[p] = int(tls[i])
            i += 1
        # print(route_time_limits)

        print("Starting GUROBI solver to S-ICEP...")
        print("")

        print("Building model...")

        m = pyomo_ICEP_model_generator.main(vessel_source, vessel_pos_source,
            is_locs_source, is_docks_source, mn_locs_source, mn_docks_source,compat_source,
            distance_data, route_time_limits, penalty, trips_source, scenarios_source, src_node_source,
            alpha_source, beta_source, gamma_source, delta_source, epsilon_source, zeta_source,
            lambda_source, 'balanced4')

        print("Solving model...")

        start_time = time.time()

        current_optimal_solution = 999999999

        iteration = 0
        # previous_evacuation_times =
        reduce = True

        current_evacuation_times = dict.fromkeys(np.unique(scenarios_source["Scenario"]))
        current_step_sizes = dict.fromkeys(np.unique(scenarios_source["Scenario"]))
        current_non_evacuated = dict.fromkeys(np.unique(scenarios_source["Scenario"]))

        for xi in current_evacuation_times:
            current_evacuation_times[xi] = m.T[xi]
            current_step_sizes[xi] = 0.1 * m.T[xi]

        while iteration < max_iterations:

            start_iter_time = time.time()

            print("###############################")
            print("running iteration:", iteration)

            current_optimal_solution, run_time = run_S_ICEP_model(m, rel_path, vessel_source, is_docks_source, objective, runtime_limit = run_time_limit)
            print("Current best solution for tsums:", current_optimal_solution)

            # calculate comp-based objective function
            current_best_solution_comp = (sum(m.cfix[i] * sum(1/m.K[xi] for xi in m.xi) * m.z[i].value for i in m.i) +
                                          sum(m.ps[xi] * m.comp[xi].value for xi in m.xi) +
                                          sum(m.ps[xi] * 1/m.K[xi] * sum(m.var_cost[i] * m.u[i, xi].value for i in m.i) for xi in m.xi) +
                                          m.P * sum(m.ps[xi] * m.flan[a,xi].value for a,xi in m.a))
            print("Current best solution for comp:", current_best_solution_comp)

            # current_evacuation_times = [m.comp[i].value for i in m.comp]
            for xi in current_evacuation_times:
                current_evacuation_times[xi] = m.comp[xi].value
            for xi in current_step_sizes:
                current_step_sizes[xi] = 0.5 * current_evacuation_times[xi]
            for xi in current_non_evacuated:
                scenario_value = 0
                for a,x in m.a:
                    if xi == x:
                        scenario_value += m.flan[a,xi].value
                current_non_evacuated[xi] = scenario_value
                # current_non_evacuated[xi] = sum(m.flan[a,xi].value for a,xi in m.a)
            # current_step_sizes = [0.1 * i for i in current_evacuation_times]
            print("Current evacuation times per scenario:")
            print(current_evacuation_times)
            print("Current non-evacuated evacuees per scenario:")
            print(current_non_evacuated)

            for xi in m.xi:
                # reduce the route time limit if the current solution has shorter completion than imposed by limits
                # print("achieved longest route time:", m.comp[xi].value)
                # print(m.tsum_control[xi].value)
                # print("limit route time:", m.T[xi])
                if m.comp[xi].value < m.T[xi]:
                    route_time_limits[xi] = m.comp[xi].value
                # count the number of people not evacuated
                number_not_evacuated = 0
                for n in m.flan:
                    if xi in n:
                        # print('        ', n[0], ':', int(m.flan[n].value), 'people')
                        number_not_evacuated += int(m.flan[n].value)
                # if someone is not evacuated, increase route time limit
                if number_not_evacuated > 0:
                    # if the last step has increased the route time, decrease step size and increase the route time by that
                    if reduce == True:
                        current_step_sizes[xi] = 0.5 * current_step_sizes[xi]
                    else:
                        pass
                    route_time_limits[xi] += current_step_sizes[xi]
                    route_time_limits[xi] = max(0,route_time_limits[xi])
                    reduce = False # indicate that this iteration did not reduce the route time
                # if everyone is evacuated, decrease route time limit
                else:
                    if reduce == False:
                        current_step_sizes[xi] = 0.5 * current_step_sizes[xi]
                    else:
                        pass
                    route_time_limits[xi] -= current_step_sizes[xi]
                    route_time_limits[xi] = max(0,route_time_limits[xi])
                    reduce = True

            iteration += 1

            m.del_component(m.T)
            m.T = Param(m.xi, initialize = route_time_limits)

            cost_relativizers = dict.fromkeys(np.unique(scenarios_source["Scenario"]))
            for xi in cost_relativizers:
                cost_relativizers[xi] = sum(m.cfix[i] for i in m.i) + sum(m.var_cost[i] * m.T[xi] for i in m.i)
            m.del_component(m.K)
            m.K = Param(m.xi, initialize = cost_relativizers)
            m.K.pprint()
            # m.K = Param(initialize = (sum(m.cfix[i] for i in m.i) + sum(m.var_cost[i] * m.T[xi] for i in m.i))) #len(vessel_source) * time_limit)/10)
            m.del_component(m.max_time)
            m.max_time = Constraint(m.xi, rule = pyomo_ICEP_model_generator.max_time)

            end_iter_time = time.time()
            total_iter_time = end_iter_time - start_iter_time

            print('Time for iteration', iteration-1, ':', total_iter_time)

        current_optimal_solution, run_time = run_S_ICEP_model(m, rel_path, vessel_source, is_docks_source, objective, runtime_limit = run_time_limit)

        end_time = time.time()
        total_time = end_time - start_time

        print('Best solution found for tsums:', current_optimal_solution)

        # calculate comp-based objective function
        current_best_solution_comp = (sum(m.cfix[i] * sum(1/m.K[xi] for xi in m.xi) * m.z[i].value for i in m.i) +
                                      sum(m.ps[xi] * m.comp[xi].value for xi in m.xi) +
                                      sum(m.ps[xi] * 1/m.K[xi] * sum(m.var_cost[i] * m.u[i, xi].value for i in m.i) for xi in m.xi) +
                                      m.P * sum(m.ps[xi] * m.flan[a,xi].value for a,xi in m.a))
        print("Current best solution for comp:", current_best_solution_comp)

        # current_evacuation_times = [m.comp[i].value for i in m.comp]
        for xi in current_evacuation_times:
            current_evacuation_times[xi] = m.comp[xi].value
        for xi in current_step_sizes:
            current_step_sizes[xi] = 0.8 * current_evacuation_times[xi]
        for xi in current_non_evacuated:
            current_non_evacuated[xi] = sum(m.flan[a,xi].value for a,xi in m.a)
        # current_step_sizes = [0.1 * i for i in current_evacuation_times]
        print(current_evacuation_times)
        print(current_non_evacuated)

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
    else:
        print("Objective function does not exist.")

if __name__ == "__main__":
    main()