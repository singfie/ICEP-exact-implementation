"""
@fiete
February 18, 2022
"""

from pyomo.environ import *
import itertools as it

####### AUXILIARY FUNCTIONS ########

def generate_comb_2keys(a_keys, b_keys):
    combinations = list(it.product(a_keys, b_keys))
    return combinations

def max_number_worst_case(m):
    return(sum(m.l[a] for a in m.a) <= m.GammaParam)

def largest(m):
    return(sum(m.robust_demand[a] * m.l[a] for a in m.a))

####### THE MAIN MODEL INSTANCE GENERATOR ########

def main(is_locs_source, demand_source, Gamma_parameter):
    """
    A function that returns a pyomo model implementation for the sub-problem of the R-ICEP.
    """

    # creation of model frame
    m = ConcreteModel()

    ############################# SETS AND NODES ##############################

    # read in lists of real nodes
    is_loc = is_locs_source['Location'].tolist()

    # Main nodes in the graph

    ## island locations
    m.a = Set(initialize = is_loc, ordered = True)
    # m.a.pprint()

    ############################# PARAMETERS ##############################

    # Gamma parameter determining the maximum number of locations that are allowed to go to a very high level
    m.GammaParam = Param(initialize = Gamma_parameter)

    # robust design parameters
    Robust_demand = dict.fromkeys(is_loc)
    for i in Robust_demand:
        if demand_source['Robust_demand'].loc[(demand_source['Location'] == i)].empty:
            Robust_demand[i] = 0.0
        else:
            Robust_demand[i] = float(demand_source['Robust_demand'].loc[(demand_source['Location'] == i)])
    m.robust_demand = Param(is_loc, initialize = Robust_demand)
    m.robust_demand.pprint()

    ############################# DECISION VARIABLES ##############################

    # Define binary variables
    m.l = Var(m.a, within = Binary, initialize = 0)
    bin_vars = len(m.l)
    print("binary variables:", bin_vars)

    ############################# CONSTRAINTS ##############################

    # Constraint definition
    m.max_size = Constraint(rule = max_number_worst_case)
    m.max_size.pprint()

    # Objective function definitions
    m.objective = Objective(rule=largest, sense=maximize, doc='Define stochastic objective function')
    # m.objective.pprint()

    return(m)

if __name__ == "__main__":
    main()
