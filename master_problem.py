#!/usr/bin/env python
# coding: utf-8
# @fiete

from pyomo.environ import *
import itertools as it
import numpy as np

def generate_comb_3keys(a_keys, b_keys, c_keys):
    combinations = list(it.product(a_keys, b_keys, c_keys))
    return combinations


if __name__ == "__main__":
    main()


def main(vessel_source, vessel_pos_source,
         is_locs_source, is_docks_source, mn_locs_source, mn_docks_source,
         compat_source, distance_data, time_limit, penalty, trips_source, scenarios_source, src_node_source,
         alpha_source, beta_source, gamma_source, delta_source, epsilon_source, zeta_source, lambda_source, objective_name, routes):
    """
    A function that returns a pyomo model implementation of the S-ICEP model.
    """

    # creation of model frame
    m = ConcreteModel()

    ############################# SETS AND NODES ##############################

    # initialize sets
    vessels = vessel_source['Vessel_name'].tolist()
    m.i = Set(initialize = vessels, ordered = True)
    m.i.pprint()

    scenarios = np.unique(scenarios_source['Scenario'].tolist())
    m.xi = Set(initialize = scenarios, ordered = True)
    m.xi.pprint()

    routes = np.unique(routes['route_id'].tolist()) # provide inputs in form of route list
    m.o = Set(initialize = routes)
    m.o.pprint()

    # read in lists of real nodes
    src_node = src_node_source['Location'].tolist()
    # vessel_locs = vessel_pos_source['Dock'].tolist()
    is_loc = is_locs_source['Location'].tolist()
    is_docks = is_docks_source['Dock'].tolist()
    # mn_docks = mn_docks_source['Dock'].tolist()
    # mn_loc = mn_locs_source['Location'].tolist()

    ## total slots
    m.tot_docks = Set(initialize = is_docks, ordered = True)
    # m.tot_docks.pprint()


    ## source node
    source_node = list(it.product(src_node, scenarios))
    m.s = Set(initialize = source_node, ordered = True)
    # m.s.pprint()

    ## island locations
    is_locs = list(it.product(is_loc, scenarios))
    m.a = Set(initialize = is_locs, ordered = True)
    # m.a.pprint()

    ## island docks
    is_schedule = []
    for i in range(0,len(is_docks_source)):
        for xi in scenarios:
            is_schedule.append((is_docks_source['Dock'].iloc[i], xi))
    m.b = Set(initialize = is_schedule, ordered = True)
    # m.b.pprint()

    ## mainland location
    mn_locs = list(it.product(mn_loc, scenarios))
    m.t = Set(initialize = mn_locs, ordered = True)
    # m.t.pprint()

    ############################# ARCS ##############################

    # Arc definitions based on compatibility
    # alphas
    alpha = []
    for i in range(0, len(alpha_source)):
        for xi in scenarios:
            alpha.append((xi, alpha_source['Source'].iloc[i],
                          alpha_source['Island location'].iloc[i]))
    m.alpha = Set(initialize = alpha, ordered = True)
    # m.alpha.pprint()

    # betas
    beta = []
    for i in range(0,len(beta_source)):
        for xi in scenarios:
            beta.append((xi, beta_source['Island location'].iloc[i],
                         beta_source['Island dock'].iloc[i]))
    m.beta = Set(initialize = beta, ordered = True)
    m.beta.pprint()

    # lambdas
    lambdas = []
    for i in range(0,len(lambda_source)):
        for xi in scenarios:
            lambdas.append((xi, lambda_source['Origin'].iloc[i],
                            lambda_source['Destination'].iloc[i]))
    m.lambdas = Set(initialize = lambdas, ordered = True)
    #m.lambdas.pprint()

    ############################# ARC PARAMETERS ##############################

    lambda_caps = dict.fromkeys(lambdas)
    for i in lambda_caps:
        if scenarios_source['private_evac'].loc[(scenarios_source['Location'] == i[1]) &
                                                (scenarios_source['Scenario'] == i[0])].empty:
            lambda_caps[i] = 0.0
        else:
            lambda_caps[i] = float(scenarios_source['private_evac'].loc[(scenarios_source['Location'] == i[1]) &
                                                                        (scenarios_source['Scenario'] == i[0])])
    m.lambda_cap = Param(m.lambdas, initialize = lambda_caps)
    #m.lambda_cap.pprint()

    # Add arc capacities
    vessel_caps = dict.fromkeys(vessels)
    for i in vessels:
        vessel_caps[i] = float(vessel_source['max_cap'].loc[vessel_source['Vessel_name'] == i])
    m.vessel_cap = Param(m.i, initialize = vessel_caps)
    m.vessel_cap.pprint()

    stops = dict.fromkeys(generate_comb_3keys(scenarios, vessels, routes, is_docks))
    for stop in stops:
        for i in vessels:
            for o in routes:
                for b in is_docks:
                    for xi in scenarios:
                        if (i in stop) and (o in stop) and (b in stop) and (xi in stop):
                            stops[stop] = int(routes['route_id'].loc[(routes['Vessel_name'] == i) & (routes['scenario'] == xi) & (routes['route_id'] == o) & (routes['is_dock'] == b)])
    m.stops = Param(m.xi, m.i, m.o, m.b, initialize = stops)
    m.stops.pprint()
    # all other arcs carry infinite capacity

    ############################# DEMAND PARAMETERS ##############################

    # demand parameters
    Evac_demand = generate_comb_3keys(scenarios, src_node, is_loc)
    Evac_demand = dict.fromkeys(Evac_demand)
    for i in Evac_demand:
        if scenarios_source['Demand'].loc[(scenarios_source['Scenario'] == i[0]) &
                                          (scenarios_source['Location'] == i[2])].empty:
            Evac_demand[i] = 0.0
        else:
            Evac_demand[i] = float(scenarios_source['Demand'].loc[(scenarios_source['Scenario'] == i[0]) &
                                                                  (scenarios_source['Location'] == i[2])])
    m.demand = Param(m.xi, src_node, is_loc, initialize = Evac_demand) # equivalent to fl_sa
    # m.demand.pprint()

    ############################# REMAINING PARAMETERS ##############################

    # fixed cost per vessel selection
    var_cost = dict.fromkeys(vessels)
    for i in var_cost:
        var_cost[i] = float(vessel_source['operating_cost'].loc[vessel_source['Vessel_name'] == i])/60 # cost per minute
    m.var_cost = Param(m.i, initialize = var_cost)
    # m.var_cost.pprint()

    # max time of evacuation
    time_limits = dict.fromkeys(scenarios)
    # i = 0
    for xi in time_limits:
        time_limits[xi] = int(time_limit[xi])
        # i += 1
    m.T = Param(m.xi, initialize = time_limits)
    # m.T.pprint()

    # Penalty cost for leaving a person behind
    m.P = Param(initialize = penalty)
    # m.P.pprint()




    # probabilities per scenario
    probs = dict.fromkeys(scenarios)
    for i in probs:
        probs[i] = float(np.unique(scenarios_source['Probability'].loc[(scenarios_source['Scenario'] == i)]))
    m.ps = Param(m.xi, initialize = probs)
    # m.ps.pprint()

    # fixed cost per vessel selection
    fixed_cost = dict.fromkeys(vessels)
    for i in fixed_cost:
        fixed_cost[i] = float(vessel_source['contract_cost'].loc[vessel_source['Vessel_name'] == i])
    m.cfix = Param(m.i, initialize = fixed_cost)
    #m.cfix.pprint()

    ############################# DECISION VARIABLES ##############################

    ## Define completion variables
    m.comp = Var(m.xi, within = NonNegativeReals, bounds = (0, None), initialize = 0)
    m.tsums = Var(m.xi, within = NonNegativeReals, bounds = (0, None), initialize = 0)
    # m.tsums_control = Var(m.xi, within = NonNegativeReals, bounds = (0, None), initialize = 0)

    ## Define vessel selection variable
    m.z = Var(m.i, within = Binary, initialize = 0)

    ## Define the flow variables
    m.flab = Var(m.beta, within =  NonNegativeReals, bounds = (0, None), initialize = 0)
    m.flan = Var(m.a, within = NonNegativeReals, bounds = (0, None), initialize = 0)
    m.flat = Var(m.lambdas, within = NonNegativeReals, bounds = (0, None), initialize = 0)

    # Define binary variables

    m.q = Var(m.i, m.o, within = Binary, initialize = 0)
    m.q.pprint()

    # Define usage length of each vessel
    m.u = Var(m.i, m.xi, within = NonNegativeReals, bounds = (0, None), initialize = 0)
    m.time_record = Var(m.i, m.xi, within = NonNegativeReals, bounds = (0, None), initialize = 0)

    con_vars = len(m.comp) + len(m.tsums) + len(m.flab) + len(m.flan) + len(m.flat)
    bin_vars = len(m.q)
    all_vars = con_vars + bin_vars
    print("Continuous variables:", con_vars)
    print("binary variables:", bin_vars)

    ############################# CONSTRAINTS ##############################

    # Constraint definition

    m.time_const = Constraint(m.i, m.xi, rule = times)
    # m.time_const.pprint()

    m.max_time = Constraint(m.xi, rule = max_time)
    # m.max_time.pprint()

    m.time_per_vess = Constraint(m.i, m.xi, rule = times_vess)
    # m.time_per_vess.pprint()

    m.time_counts = Constraint(m.i, m.xi, rule = times_vess_count)
    # m.time_counts.pprint()

    ## Capacity vessel choice dependent constraints

    ### gamma arcs (island dock to mainland dock)
    m.capa_bc = Constraint(m.gamma, rule = cap_bc)
    # m.capa_bc.pprint()

    ### gamma arcs (island dock to mainland dock)
    m.capa_bc_rt = Constraint(m.gamma, rule = cap_bc_rt)
    # m.capa_bc_rt.pprint()


    ### lambda arcs (island to mainland through private evacuation)
    m.capa_at = Constraint(m.lambdas, rule = cap_at)
    #m.capa_at.pprint()

    ## Flow conservation constraints
    # flow through island locations
    m.fl_a = Constraint(m.a, rule = flow_a)
    # m.fl_a.pprint()

    m.fl_b = Constraint(m.b, rule = flow_b)
    # m.fl_b.pprint()

    m.fl_c = Constraint(m.c, rule = flow_c)
    #m.fl_c.pprint()

    m.select_zeta = Constraint(m.i, m.k, m.xi, rule = select_arrive)
    # m.select_zeta.pprint()

    m.select_gamma = Constraint(m.i, m.k, m.xi, rule = select_save)
    #m.select_gamma.pprint()

    m.select_delta = Constraint(m.i, m.k, m.xi, rule = select_return)
    #m.select_delta.pprint()

    m.ad_b = Constraint(m.b, rule = adj_b)
    # m.ad_b.pprint()

    m.ad_c = Constraint(m.c, rule = adj_c)
    # m.ad_c.pprint()

    m.sum_time = Constraint(m.xi, rule = tsum_calc)
    # m.sum_time.pprint()

    # m.control_time = Constraint(m.xi, rule = tsum_control)
    # m.control_time.pprint()

    # Variable Cost high enough

    cost_relativizers = dict.fromkeys(scenarios)
    print(cost_relativizers)
    for xi in cost_relativizers:
        cost_relativizers[xi] = sum(m.cfix[i] for i in m.i) + sum(m.var_cost[i] * m.T[xi] for i in m.i)
    m.K = Param(m.xi, initialize = cost_relativizers) #len(vessel_source) * time_limit)/10)
    # m.K.pprint()

    # fixed cost high enough
    # m.J = Param(initialize = sum(m.cfix[i] for i in m.i)) # DELETE THIS
    #m.J.pprint()

    # Objective function definitions

    if objective_name == 'conservative1':
        m.objective = Objective(rule=conservative1, sense=minimize, doc='Define stochastic objective function')
        # m.objective.pprint()
    elif objective_name == 'conservative2':
        m.objective = Objective(rule=conservative2, sense=minimize, doc='Define stochastic objective function')
        # m.objective.pprint()
    elif objective_name == 'balanced1':
        m.objective = Objective(rule=balanced1, sense=minimize, doc='Define stochastic objective function')
        # m.objective.pprint()
    elif objective_name == 'balanced2':
        m.objective = Objective(rule=balanced2, sense=minimize, doc='Define stochastic objective function')
        # m.objective.pprint()
    elif objective_name == 'balanced3':
        m.objective = Objective(rule=balanced3, sense=minimize, doc='Define stochastic objective function')
        # m.objective.pprint()
    elif objective_name == 'balanced4':
        m.objective = Objective(rule=balanced4, sense=minimize, doc='Define stochastic objective function')
        # m.objective.pprint()
    elif objective_name == 'economic1':
        m.objective = Objective(rule=economic1, sense=minimize, doc='Define stochastic objective function')
        # m.objective.pprint()
    elif objective_name == 'economic2':
        m.objective = Objective(rule=economic2, sense=minimize, doc='Define stochastic objective function')
        # m.objective.pprint()
    else:
        print('Passed objective function does not exist.')

    return(m)

