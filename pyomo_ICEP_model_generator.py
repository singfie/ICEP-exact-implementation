#!/usr/bin/env python
# coding: utf-8
# @fiete

from pyomo.environ import *
import glob
import pandas as pd
import time
import itertools as it
import numpy as np
# from gurobipy import *
# import argparse

# auxiliary functions

def generate_comb_2keys(a_keys, b_keys):
    combinations = list(it.product(a_keys, b_keys))
    return combinations

def generate_comb_3keys(a_keys, b_keys, c_keys):
    combinations = list(it.product(a_keys, b_keys, c_keys))
    return combinations

# auxiliary functions for constraint definitions

## Time constraint
def times(m, ves, xi):
    m.sub_zeta = Set(dimen = 5, ordered = True)
    for s, k, h, b, i in m.zeta:
        if ves == i and xi == s:
            m.sub_zeta.add((s,k,h,b,i))
    #m.sub_zeta.pprint()
    m.sub_gamma = Set(dimen = 6, ordered = True)
    for s,b,i,k1,c,k2 in m.gamma:
        if ves == i and xi == s:
            m.sub_gamma.add((s,b,i,k1,c,k2))
    #m.sub_gamma.pprint()
    m.sub_delta = Set(dimen = 6, ordered = True)
    for s,c,i,k1,b,k2 in m.delta:
        if ves == i and xi == s:
            m.sub_delta.add((s,c,i,k1,b,k2))
    #m.sub_delta.pprint()
    m.sub_h = Set(dimen = 3, ordered = True)
    for h,i,s in m.h:
        if ves == i and xi == s:
            m.sub_h.add((h,i,s))
    #m.sub_h.pprint()
    m.sub_b = Set(dimen = 4, ordered = True)
    for b,i,k,s in m.b:
        if ves == i and xi == s:
            m.sub_b.add((b,i,k,s))
    #m.sub_b.pprint()
    m.sub_c = Set(dimen = 4, ordered = True)
    for c,i,k,s in m.c:
        if ves == i and xi == s:
            m.sub_c.add((c,i,k,s))
    #m.sub_c.pprint()
    constr = (sum(m.zeta_c[xi,k,h,b,ves] * m.w[xi,k,h,b,ves] for xi,k,h,b,ves in m.sub_zeta) +
           sum(m.gamma_c[xi,b,ves,k1,c,k2] * m.x[xi,b,ves,k1,c,k2] for xi,b,ves,k1,c,k2 in m.sub_gamma) + 
           sum(m.delta_c[xi,c,ves,k1,b,k2] * m.y[xi,c,ves,k1,b,k2] for xi,c,ves,k1,b,k2 in m.sub_delta) + 
           (sum(m.h_c[h,ves,xi] * m.w[xi,k,h,b,ves] for xi,k,h,b,ves in m.sub_zeta)) +
           (sum(m.b_c[b,ves,k,xi] * m.w[xi,k,h,b,ves] for xi,k,h,b,ves in m.sub_zeta)) + 
           (sum(m.b_c[b,ves,k,xi] * m.y[xi,c,ves,k1,b,k2] for xi,c,ves,k1,b,k2 in m.sub_delta)) + 
           (sum(m.c_c[c,ves,k,xi] * m.x[xi,b,ves,k1,c,k2] for xi,b,ves,k1,c,k2 in m.sub_gamma)) <= m.comp[xi])#+
           #(sum(m.rt_c[k1] * m.flbc[xi,b,ves,k1,c,k2] for xi,b,ves,k1,c,k2 in m.sub_gamma))
    m.del_component(m.sub_zeta)
    m.del_component(m.sub_gamma)
    m.del_component(m.sub_delta)
    m.del_component(m.sub_h)
    m.del_component(m.sub_b)
    m.del_component(m.sub_c)
    return(constr)

# a constraint to deliver an upper bound to the time
def max_time(m, xi):
    return(m.comp[xi] <= m.T)

## Support constraint to save time consumption of every resource
def times_vess(m, ves, xi):
    m.sub_zeta = Set(dimen = 5, ordered = True)
    for s, k, h, b, i in m.zeta:
        if ves == i and xi == s:
            m.sub_zeta.add((s,k,h,b,i))
    #m.sub_zeta.pprint()
    m.sub_gamma = Set(dimen = 6, ordered = True)
    for s,b,i,k1,c,k2 in m.gamma:
        if ves == i and xi == s:
            m.sub_gamma.add((s,b,i,k1,c,k2))
    #m.sub_gamma.pprint()
    m.sub_delta = Set(dimen = 6, ordered = True)
    for s,c,i,k1,b,k2 in m.delta:
        if ves == i and xi == s:
            m.sub_delta.add((s,c,i,k1,b,k2))
    #m.sub_delta.pprint()
    m.sub_h = Set(dimen = 3, ordered = True)
    for h,i,s in m.h:
        if ves == i and xi == s:
            m.sub_h.add((h,i,s))
    #m.sub_h.pprint()
    m.sub_b = Set(dimen = 4, ordered = True)
    for b,i,k,s in m.b:
        if ves == i and xi == s:
            m.sub_b.add((b,i,k,s))
    #m.sub_b.pprint()
    m.sub_c = Set(dimen = 4, ordered = True)
    for c,i,k,s in m.c:
        if ves == i and xi == s:
            m.sub_c.add((c,i,k,s))
    #m.sub_c.pprint()
    constr = (sum(m.zeta_c[xi,k,h,b,ves] * m.w[xi,k,h,b,ves] for xi,k,h,b,ves in m.sub_zeta) +
           sum(m.gamma_c[xi,b,ves,k1,c,k2] * m.x[xi,b,ves,k1,c,k2] for xi,b,ves,k1,c,k2 in m.sub_gamma) + 
           sum(m.delta_c[xi,c,ves,k1,b,k2] * m.y[xi,c,ves,k1,b,k2] for xi,c,ves,k1,b,k2 in m.sub_delta) + 
           (sum(m.h_c[h,ves,xi] * m.w[xi,k,h,b,ves] for xi,k,h,b,ves in m.sub_zeta)) +
           (sum(m.b_c[b,ves,k,xi] * m.w[xi,k,h,b,ves] for xi,k,h,b,ves in m.sub_zeta)) + 
           (sum(m.b_c[b,ves,k,xi] * m.y[xi,c,ves,k1,b,k2] for xi,c,ves,k1,b,k2 in m.sub_delta)) + 
           (sum(m.c_c[c,ves,k,xi] * m.x[xi,b,ves,k1,c,k2] for xi,b,ves,k1,c,k2 in m.sub_gamma)) +
           (sum(m.rt_c[k1] * m.flbc[xi,b,ves,k1,c,k2] for xi,b,ves,k1,c,k2 in m.sub_gamma))
           == m.u[ves, xi])
    m.del_component(m.sub_zeta)
    m.del_component(m.sub_gamma)
    m.del_component(m.sub_delta)
    m.del_component(m.sub_h)
    m.del_component(m.sub_b)
    m.del_component(m.sub_c)
    return(constr)

    ## Support constraint to save time consumption of every resource
def times_vess_count(m, ves, xi):
    m.sub_zeta = Set(dimen = 5, ordered = True)
    for s, k, h, b, i in m.zeta:
        if ves == i and xi == s:
            m.sub_zeta.add((s,k,h,b,i))
    #m.sub_zeta.pprint()
    m.sub_gamma = Set(dimen = 6, ordered = True)
    for s,b,i,k1,c,k2 in m.gamma:
        if ves == i and xi == s:
            m.sub_gamma.add((s,b,i,k1,c,k2))
    #m.sub_gamma.pprint()
    m.sub_delta = Set(dimen = 6, ordered = True)
    for s,c,i,k1,b,k2 in m.delta:
        if ves == i and xi == s:
            m.sub_delta.add((s,c,i,k1,b,k2))
    #m.sub_delta.pprint()
    m.sub_h = Set(dimen = 3, ordered = True)
    for h,i,s in m.h:
        if ves == i and xi == s:
            m.sub_h.add((h,i,s))
    #m.sub_h.pprint()
    m.sub_b = Set(dimen = 4, ordered = True)
    for b,i,k,s in m.b:
        if ves == i and xi == s:
            m.sub_b.add((b,i,k,s))
    #m.sub_b.pprint()
    m.sub_c = Set(dimen = 4, ordered = True)
    for c,i,k,s in m.c:
        if ves == i and xi == s:
            m.sub_c.add((c,i,k,s))
    #m.sub_c.pprint()
    constr = (sum(m.zeta_c[xi,k,h,b,ves] * m.w[xi,k,h,b,ves] for xi,k,h,b,ves in m.sub_zeta) +
           sum(m.gamma_c[xi,b,ves,k1,c,k2] * m.x[xi,b,ves,k1,c,k2] for xi,b,ves,k1,c,k2 in m.sub_gamma) + 
           sum(m.delta_c[xi,c,ves,k1,b,k2] * m.y[xi,c,ves,k1,b,k2] for xi,c,ves,k1,b,k2 in m.sub_delta) + 
           (sum(m.h_c[h,ves,xi] * m.w[xi,k,h,b,ves] for xi,k,h,b,ves in m.sub_zeta)) +
           (sum(m.b_c[b,ves,k,xi] * m.w[xi,k,h,b,ves] for xi,k,h,b,ves in m.sub_zeta)) + 
           (sum(m.b_c[b,ves,k,xi] * m.y[xi,c,ves,k1,b,k2] for xi,c,ves,k1,b,k2 in m.sub_delta)) + 
           (sum(m.c_c[c,ves,k,xi] * m.x[xi,b,ves,k1,c,k2] for xi,b,ves,k1,c,k2 in m.sub_gamma))
           == m.time_record[ves, xi])
    m.del_component(m.sub_zeta)
    m.del_component(m.sub_gamma)
    m.del_component(m.sub_delta)
    m.del_component(m.sub_h)
    m.del_component(m.sub_b)
    m.del_component(m.sub_c)
    return(constr)

### gamma arcs (island dock to mainland dock)
def cap_bc(m,s,b,i,k1,c,k2):
    return(m.flbc[s,b,i,k1,c,k2] <= m.gamma_cap[s,b,i,k1,c,k2] * m.z[i])

### gamma arcs (island dock to mainland dock)
def cap_bc_rt(m,s,b,i,k1,c,k2):
    return(m.flbc[s,b,i,k1,c,k2] <= m.gamma_cap[s,b,i,k1,c,k2] * m.x[s,b,i,k1,c,k2])

### lambda arcs (island to mainland through private evacuation)
def cap_at(m,s,a,t):
    return(m.flat[s,a,t] <= m.lambda_cap[s,a,t])

## Flow conservation constraints
# flow through island locations
def flow_a(m, a, x):
    m.sub_beta = Set(dimen = 6, ordered = True)
    for xi, a1, i, k1, b, k2 in m.beta:
        if a == a1 and x == xi:
            m.sub_beta.add((xi, a1, i, k1, b, k2))
    constr = (sum(m.flab[x,a,i,k1,b,k2] for x,a,i,k1,b,k2 in m.sub_beta) + m.flat[x,a,'Mainland'] + m.flan[a,x] == m.demand[x,'Island',a])
    m.del_component(m.sub_beta)
    return(constr)

# flow through island docks
def flow_b(m, b, ves, trip, xi):
    m.sub_beta = Set(dimen = 6, ordered = True)
    for h,i,j,k,l,n in m.beta:
        if b == l and ves == j and trip == k and xi == h:
            m.sub_beta.add((h,i,j,k,l,n))
    #m.sub_beta.pprint()
    m.sub_gamma = Set(dimen = 6, ordered = True)
    for h,i,j,k,l,n in m.gamma:
        if b == i and ves == j and trip == k and xi == h:
            m.sub_gamma.add((h,i,j,k,l,n))
    #m.sub_gamma.pprint()
    constr = (sum(m.flab[xi,a,ves,trip,b,trip] for xi,a,ves,trip,b,trip in m.sub_beta) == sum(m.flbc[xi,b,ves,trip,c,trip] for xi,b,ves,trip,c,trip in m.sub_gamma))
    m.del_component(m.sub_beta)
    m.del_component(m.sub_gamma)
    return(constr)

# flow through mainland docks
def flow_c(m, c, ves, trip, xi):
    m.sub_gamma = Set(dimen = 6, ordered = True)
    for h,i,j,k,l,n in m.gamma:
        if c == l and ves == j and trip == k and xi == h:
            m.sub_gamma.add((h,i,j,k,l,n))
    #m.sub_gamma.pprint()
    m.sub_epsilon = Set(dimen = 5, ordered = True)
    for h,i,j,k,l in m.epsilon:
        if c == i and ves == j and trip == k and xi == h:
            m.sub_epsilon.add((h,i,j,k,l))
    #m.sub_epsilon.pprint()
    constr = (sum(m.flbc[xi,b,ves,trip,c,trip] for xi,b,ves,trip,c,trip in m.sub_gamma) == sum(m.flct[xi,c,ves,trip,t] for xi,c,ves,trip,t in m.sub_epsilon))
    m.del_component(m.sub_gamma)
    m.del_component(m.sub_epsilon)
    return(constr)


## Selection of max one route per shipping segment
### zeta arcs
def select_arrive(m, ves, trip, s):
    m.sub_zeta = Set(dimen = 5, ordered = True)
    if trip == min(m.k):
        for h, l, j, d, i in m.zeta:
            if ves == i and trip == l and s == h:
                m.sub_zeta.add((h, l, j, d, i))
        const = (sum(m.w[s,trip,j,d,ves] for s,trip,j,d,ves in m.sub_zeta) <= m.z[ves])
    else:
        const = Constraint.Skip
    m.del_component(m.sub_zeta)
    return(const)

### gamma arcs
def select_save(m, ves, trip, s):
    m.sub_gamma = Set(dimen = 6, ordered = True)
    for xi,b,i,k1,c,k2 in m.gamma:
        if ves == i and trip == k1 and trip == k2 and s == xi:
            m.sub_gamma.add((xi,b,i,k1,c,k2))
    constr = (sum(m.x[s,b,ves,trip,c,trip] for s,b,ves,trip,c,trip in m.sub_gamma) <= m.z[ves])
    m.del_component(m.sub_gamma)
    return(constr)

### delta arcs
def select_return(m, ves, trip, s):
    if trip != max(m.k):
        m.sub_delta = Set(dimen = 6, ordered = True)
        for xi,c,i,k1,b,k2 in m.delta:
            if ves == i and trip == k1 and s == xi:
                m.sub_delta.add((xi,c,i,k1,b,k2))
        constr = (sum(m.y[s,c,ves,trip,b,k2] for s,c,ves,trip,b,k2 in m.sub_delta) <= m.z[ves])
        m.del_component(m.sub_delta)
    elif trip == max(m.k):
        constr = Constraint.Skip
    return(constr)

## Route adjacency constraints
### D node adjacency
def adj_b(m, is_dock, ves, trip, s):
    if trip != min(m.k):
        m.sub_delta = Set(dimen = 6, ordered = True)
        for xi, c, i, k, b, k2 in m.delta:
            if is_dock == b and ves == i and trip == k2 and s == xi:
                m.sub_delta.add((xi, c, i, k, b, k2))
                #m.sub_delta.pprint()
        m.sub_gamma = Set(dimen = 6, ordered = True)
        for xi, b, i, k, c, k2 in m.gamma:
            if is_dock == b and ves == i and trip == k and s == xi:
                m.sub_gamma.add((xi, b, i, k, c, k2))
                #m.sub_gamma.pprint()
        constr = (sum(m.y[s, c, ves, k, is_dock, trip] for s, c, ves, k, is_dock, trip in m.sub_delta) == 
                  sum(m.x[s, is_dock, ves, trip, c, k2] for s, is_dock, ves, trip, c, k2 in m.sub_gamma))
        m.del_component(m.sub_delta)
        m.del_component(m.sub_gamma)
    elif trip == min(m.k):
        m.sub_zeta = Set(dimen = 5, ordered = True)
        for xi, k, h, b, i in m.zeta:
            if is_dock == b and ves == i and trip == k and s == xi:
                m.sub_zeta.add((xi, k, h, b, i))
                #m.sub_zeta.pprint()
        m.sub_gamma = Set(dimen = 6, ordered = True)
        for xi, b, i, k, c, k2 in m.gamma:
            if is_dock == b and ves == i and trip == k and s == xi:
                m.sub_gamma.add((xi, b, i, k, c, k2))
                #m.sub_gamma.pprint()
        constr = (sum(m.w[s, trip, h, b, ves] for s, trip, h, b, ves in m.sub_zeta) == 
                  sum(m.x[s, is_dock, ves, trip, c, k2] for s, is_dock, ves, trip, c, k2 in m.sub_gamma))
        m.del_component(m.sub_zeta)
        m.del_component(m.sub_gamma)
    return(constr)

### C node adjacency
def adj_c(m, mn_dock, ves, trip, s):
    if trip != max(m.k):
        m.sub_gamma = Set(dimen = 6, ordered = True)
        for xi, b, i, k, c, k2 in m.gamma:
            if mn_dock == c and ves == i and trip == k2 and s == xi:
                m.sub_gamma.add((xi, b, i, k, c, k2))
                #m.sub_gamma.pprint()for h,i,j,k,l,o,p in m.zeta:
        m.sub_delta = Set(dimen = 6, ordered = True)
        for xi, c, i, k, b, k2 in m.delta:
            if mn_dock == c and ves == i and trip == k and s == xi:
                m.sub_delta.add((xi, c, i, k, b, k2))
                #m.sub_delta.pprint()
        constr = (sum(m.x[s, b, ves, k, mn_dock, trip] for s, b, ves, k, mn_dock, trip in m.sub_gamma) >= 
                  sum(m.y[s, mn_dock, ves, trip, b, k2] for s, mn_dock, ves, trip, b, k2 in m.sub_delta))
        m.del_component(m.sub_gamma)
        m.del_component(m.sub_delta)
    elif trip == max(m.k):
        constr = Constraint.Skip
    return(constr)

# a constraint to deliver an upper bound to the time
def tsum_calc(m, xi):
    return(sum(m.u[ves, xi] for ves in m.i) == m.tsums[xi])

######## OBJECTIVE FUNCTION GENERATORS ########

def conservative1(m):
    return(sum(m.ps[xi] * m.comp[xi] for xi in m.xi) + 
           m.P * sum(m.ps[xi] * m.flan[a,xi] for a,xi in m.a))

def conservative2(m):
    return(sum(m.ps[xi] * m.tsums[xi] for xi in m.xi) + 
           m.P * sum(m.ps[xi] * m.flan[a,xi] for a,xi in m.a))

def balanced1(m):
    return(sum(m.cfix[i] * m.z[i] for i in m.i) + 
           sum(m.ps[xi] * m.comp[xi] for xi in m.xi) +
           sum(m.ps[xi] * 1/m.K * sum(m.var_cost[i] * m.u[i, xi] for i in m.i) for xi in m.xi) + 
           m.P * sum(m.ps[xi] * m.flan[a,xi] for a,xi in m.a))

def balanced2(m):
    return(sum(m.cfix[i] * m.z[i] for i in m.i) + 
           sum(m.ps[xi] * m.tsums[xi] for xi in m.xi) +
           sum(m.ps[xi] * 1/m.K * sum(m.var_cost[i] * m.u[i, xi] for i in m.i) for xi in m.xi) + 
           m.P * sum(m.ps[xi] * m.flan[a,xi] for a,xi in m.a))

def balanced3(m):
    return(sum(m.cfix[i] * 1/m.K * m.z[i] for i in m.i) + 
           sum(m.ps[xi] * m.comp[xi] for xi in m.xi) +
           sum(m.ps[xi] * 1/m.K * sum(m.var_cost[i] * m.u[i, xi] for i in m.i) for xi in m.xi) + 
           m.P * sum(m.ps[xi] * m.flan[a,xi] for a,xi in m.a))

def balanced4(m):
    return(sum(m.cfix[i] * 1/m.K * m.z[i] for i in m.i) + 
           sum(m.ps[xi] * m.tsums[xi] for xi in m.xi) +
           sum(m.ps[xi] * 1/m.K * sum(m.var_cost[i] * m.u[i, xi] for i in m.i) for xi in m.xi) + 
           m.P * sum(m.ps[xi] * m.flan[a,xi] for a,xi in m.a))

def economic1(m):
    return(sum(m.cfix[i] * m.z[i] for i in m.i) + 
           sum(m.ps[xi] * sum(m.var_cost[i] * m.u[i, xi] for i in m.i) for xi in m.xi) + 
           m.P * sum(m.ps[xi] * m.flan[a,xi] for a,xi in m.a))

def economic2(m):
    return(sum(m.cfix[i] * m.z[i] for i in m.i) + 
           sum(m.ps[xi] * sum(m.var_cost[i] * m.u[i, xi] for i in m.i) for xi in m.xi) + 
           sum(m.ps[xi] * 1/m.T * m.comp[xi] for xi in m.xi) +
           m.P * sum(m.ps[xi] * m.flan[a,xi] for a,xi in m.a))

def old_objective_rule(m): 
    return(sum(m.cfix[i] * m.z[i] for i in m.i) + m.cvar * sum(m.ps[xi] * m.comp[xi] for xi in m.xi) + m.P * sum(m.ps[xi] * m.flan[a,xi] for a,xi in m.a))

####### THE MAIN MODEL INSTANCE GENERATOR ########

def main(vessel_source, vessel_pos_source,
    is_locs_source, is_docks_source, mn_locs_source, mn_docks_source,
    compat_source, distance_data, time_limit, penalty, trips_source, scenarios_source, src_node_source,
    alpha_source, beta_source, gamma_source, delta_source, epsilon_source, zeta_source, lambda_source):
    """
    A function that returns a pyomo model implementation of the S-ICEP model.
    """

    # creation of model frame
    m = ConcreteModel()

    ############################# SETS AND NODES ##############################

    # initialize sets
    vessels = vessel_source['Vessel_name'].tolist()
    # print(vessels)
    m.i = Set(initialize = vessels, ordered = True)
    # m.i.pprint()

    round_trips = trips_source['Round trip'].tolist()
    # print(round_trips)
    m.k = Set(initialize = round_trips, ordered = True)
    # m.k.pprint()

    # scenarios = np.unique(scenarios_source['Scenario'].tolist())
    scenarios = ['Scenario 1']
    #print(scenarios)
    m.xi = Set(initialize = scenarios, ordered = True)
    # m.xi.pprint()

    # read in lists of real nodes
    src_node = src_node_source['Location'].tolist()
    vessel_locs = vessel_pos_source['Dock'].tolist()
    is_loc = is_locs_source['Location'].tolist()
    is_docks = is_docks_source['Dock'].tolist()
    mn_docks = mn_docks_source['Dock'].tolist()
    mn_loc = mn_locs_source['Location'].tolist()

    ## total slots
    m.tot_docks = Set(initialize = is_docks + mn_docks + vessel_locs, ordered = True)
    # m.tot_docks.pprint()

    # compatibility vessels to vehicles
    compat_keys = generate_comb_2keys(vessels, is_docks + mn_docks)# + vessel_locs)
    compat_keys = dict.fromkeys(compat_keys)
    for i in compat_keys:
        # print(compat_source['Compatibility'][(compat_source['Dock'] == i[1]) & (compat_source['Resource'] == i[0])])
        compat_keys[i] = int(compat_source['Compatibility'][(compat_source['Dock'] == i[1]) & (compat_source['Resource'] == i[0])].values)
    m.compat = Param(m.i, m.tot_docks, initialize = compat_keys)
    # m.compat.pprint()

    # Main nodes in the graph

    ## source node
    source_node = list(it.product(src_node, scenarios))
    m.s = Set(initialize = source_node, ordered = True)
    #m.s.pprint()

    ## initial vessel locations
    h = []
    for i in range(0,len(vessel_pos_source)):
        for s in scenarios:
            h.append((vessel_pos_source['Dock'].iloc[i],
                         vessel_pos_source['Vessel'].iloc[i], s))
    m.h = Set(initialize = h, ordered = True)
    #m.h.pprint()

    ## island locations
    is_locs = list(it.product(is_loc, scenarios))
    m.a = Set(initialize = is_locs, ordered = True)
    #m.a.pprint()

    ## island docks
    #is_schedule = list(it.product(is_docks, vessels, round_trips, scenarios))
    is_schedule = []
    for i in range(0,len(is_docks_source)):
        for xi in scenarios:
            for j in vessels:
                for k in round_trips:
                    if m.compat[(j, is_docks_source['Dock'].iloc[i])] == 1:
                        is_schedule.append((is_docks_source['Dock'].iloc[i], j, k, xi))
    m.b = Set(initialize = is_schedule, ordered = True)
    #m.b.pprint()

    ## mainland docks
    #mn_schedule = list(it.product(mn_docks, vessels, round_trips, scenarios))
    mn_schedule = []
    for i in range(0,len(mn_docks_source)):
        for xi in scenarios:
            for j in vessels:
                for k in round_trips:
                    if m.compat[(j, mn_docks_source['Dock'].iloc[i])] == 1:
                        mn_schedule.append((mn_docks_source['Dock'].iloc[i], j, k, xi))
    m.c = Set(initialize = mn_schedule, ordered = True)
    # m.c.pprint()

    ## mainland location
    mn_locs = list(it.product(mn_loc, scenarios))
    m.t = Set(initialize = mn_locs, ordered = True)
    #m.t.pprint()

    ############################# ARCS ##############################

    # Arc definitions based on compatibility
    # alphas
    alpha = []
    for i in range(0, len(alpha_source)):
        for xi in scenarios:
            alpha.append((xi, alpha_source['Source'].iloc[i],
                         alpha_source['Island location'].iloc[i]))
    m.alpha = Set(initialize = alpha, ordered = True)
    #m.alpha.pprint()

    # betas
    beta = []
    for i in range(0,len(beta_source)):
        for xi in scenarios:
            for j in vessels:
                for k in round_trips:
                    if m.compat[(j, gamma_source['Origin'].iloc[i])] == 1 and m.compat[(j, gamma_source['Destination'].iloc[i])] == 1:
                        beta.append((xi, beta_source['Island location'].iloc[i], j, k,
                                     beta_source['Island dock'].iloc[i], k))
    m.beta = Set(initialize = beta, ordered = True)
    #m.beta.pprint()

    # gammas
    gamma = []
    for i in range(0, len(gamma_source)):
        for xi in scenarios:
            for j in vessels:
                for k in round_trips:
                    if m.compat[(j, gamma_source['Origin'].iloc[i])] == 1 and m.compat[(j, gamma_source['Destination'].iloc[i])] == 1:
                        gamma.append((xi, gamma_source['Origin'].iloc[i], j, k, 
                                      gamma_source['Destination'].iloc[i], k))
    m.gamma = Set(initialize = gamma, ordered = True)
    # m.gamma.pprint()

    # deltas
    delta = []
    for i in range(0, len(delta_source)):
        for xi in scenarios:
            for j in vessels:
                for k in range(0, len(round_trips)-1):
                    if m.compat[(j, gamma_source['Origin'].iloc[i])] == 1 and m.compat[(j, gamma_source['Destination'].iloc[i])] == 1:
                        delta.append((xi, delta_source['Origin'].iloc[i], j, round_trips[k],
                                      delta_source['Destination'].iloc[i], round_trips[k+1]))
    m.delta = Set(initialize = delta, ordered = True)
    #m.delta.pprint()

    # epsilons
    epsilon = []
    for i in range(0, len(epsilon_source)):
        for xi in scenarios:
            for j in vessels:
                for k in round_trips:
                    epsilon.append((xi, epsilon_source['Origin'].iloc[i], j, k,
                                    epsilon_source['Destination'].iloc[i]))
    m.epsilon = Set(initialize = epsilon, ordered = True)
    #m.epsilon.pprint()

    # mus
    zeta = []
    for i in range(0,len(zeta_source)):
        for xi in scenarios:
            for k in vessels:
                for t in range(0, len(round_trips)-1):
                    if round_trips[t] == min(round_trips):
                        if m.compat[(k, zeta_source['Origin'].iloc[i])] == 1 and m.compat[(k, zeta_source['Destination'].iloc[i])] == 1:
                            #print((k, zeta_source['Origin'].iloc[i]), (k, zeta_source['Destination'].iloc[i]))
                            #print(vessel_source['Regular_origin'].loc[vessel_source['Vessel_name'] == k].to_string(index=False))#.to_string()[5:])
                            #print(zeta_source['Origin'].iloc[i])
                            if vessel_source['Regular_origin'].loc[vessel_source['Vessel_name'] == k].to_string(index=False)[1:] == zeta_source['Origin'].iloc[i]:
                                #print(vessel_source['Regular_origin'].loc[vessel_source['Vessel_name'] == k].to_string()[5:], zeta_source['Origin'].iloc[i])
                                zeta.append((xi, round_trips[t], zeta_source['Origin'].iloc[i],
                                             zeta_source['Destination'].iloc[i],k))
    m.zeta = Set(initialize = zeta, ordered = True)
    #m.zeta.pprint()

    # .to_string()[5:]
    # lambdas
    lambdas = []
    for i in range(0,len(lambda_source)):
        for xi in scenarios:
            lambdas.append((xi, lambda_source['Origin'].iloc[i],
                            lambda_source['Destination'].iloc[i]))
    m.lambdas = Set(initialize = lambdas, ordered = True)
    #m.lambdas.pprint()

    ############################# ARC PARAMETERS ##############################

    # Add arc capacities
    gamma_caps = dict.fromkeys(gamma)
    for i in gamma_caps:
        for k in vessels:
            if k in i:
                gamma_caps[i] = float(vessel_source['max_cap'].loc[vessel_source['Vessel_name'] == k])
    m.gamma_cap = Param(m.gamma, initialize = gamma_caps)
    #m.gamma_cap.pprint()

    delta_caps = dict.fromkeys(delta)
    for i in delta_caps:
        delta_caps[i] = 0
    m.delta_cap = Param(m.delta, initialize = delta_caps)
    #m.delta_cap.pprint()

    zeta_caps = dict.fromkeys(zeta)
    for i in zeta_caps:
        zeta_caps[i] = 0
    m.zeta_cap = Param(m.zeta, initialize = zeta_caps)
    #m.zeta_cap.pprint()

           
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

    # all other arcs carry infinite capacity

    # Add arc cost that represent transit times
    beta_cost = dict.fromkeys(beta)
    for i in beta_cost:
        for k in round_trips:
            if k in i:
                beta_cost[i] = float(trips_source['Delay cost'].loc[trips_source['Round trip'] == k])
                # this is a modeling trick used to ensure that even if a later shipment does not change the cost
                # an earlier shipment is to be prefered.
    m.beta_c = Param(m.beta, initialize = beta_cost)
    #m.beta_c.pprint()

    # time to go from a b to a c
    gamma_cost = dict.fromkeys(gamma)
    for i in gamma_cost:
        for k in vessels:
            for j in is_docks:
                for l in mn_docks:
                    if k in i and j in i and l in i:
                        gamma_cost[i] = (float(gamma_source['Distance'][(gamma_source['Origin'] == j) 
                                                                          & (gamma_source['Destination'] == l)])/
                        float(vessel_source['v_loaded'].loc[vessel_source['Vessel_name'] == k])) * 60
    m.gamma_c = Param(m.gamma, initialize = gamma_cost)
    #m.gamma_c.pprint()

    delta_cost = dict.fromkeys(delta)
    for i in delta_cost:
        for k in vessels:
            for j in mn_docks:
                for l in is_docks:
                    if k in i and j in i and l in i:
                        delta_cost[i] = (float(delta_source['Distance'][(delta_source['Origin'] == j)
                                                                  & (delta_source['Destination'] == l)])/
                        float(vessel_source['vmax'].loc[vessel_source['Vessel_name'] == k])) * 60
    m.delta_c = Param(m.delta, initialize = delta_cost)
    #m.delta_c.pprint()

    zeta_cost = dict.fromkeys(zeta)
    for i in zeta_cost:
        for k in vessels:
            for j in vessel_locs:
                for l in is_docks:
                    if k in i and j in i and l in i:
                        zeta_cost[i] = (float(zeta_source['Distance'][(zeta_source['Origin'] == j)
                                                                  & (zeta_source['Destination'] == l)])/
                        float(vessel_source['vmax'].loc[vessel_source['Vessel_name'] == k])) * 60
    m.zeta_c = Param(m.zeta, initialize = zeta_cost)
    #m.zeta_c.pprint()

    # Add node costs that represent loading / unloading times
    # time to availability
    h_cost = dict.fromkeys(m.h)
    # print(h_cost)
    for i in h_cost:
        for k in vessels:
            if k in i:
                h_cost[i] = float(vessel_source['time to availability'].loc[vessel_source['Vessel_name'] == k])
    m.h_c = Param(m.h, initialize = h_cost)
    # m.h_c.pprint()

    # time to go from a b to a c
    b_cost = dict.fromkeys(m.b)
    for i in b_cost:
        for k in vessels:
            if k in i:
                b_cost[i] = float(vessel_source['loading time'].loc[vessel_source['Vessel_name'] == k])
    m.b_c = Param(m.b, initialize = b_cost)
    #m.b_c.pprint()

    c_cost = dict.fromkeys(m.c)
    for i in c_cost:
        for k in vessels:
            if k in i:
                c_cost[i] = float(vessel_source['loading time'].loc[vessel_source['Vessel_name'] == k]) 
    m.c_c = Param(m.c, initialize = c_cost)
    #m.c_c.pprint()

    ############################# ROUND TRIP PARAMETERS ##############################

    # Add cost parameter for round trips
    # this is an artificial variable to cause earlier shipments to carry more people
    rt_cost = dict.fromkeys(m.k)
    for i in rt_cost:
        for k in round_trips:
            if k == i:
                rt_cost[i] = float(trips_source['Delay cost'].loc[trips_source['Round trip'] == i])
    m.rt_c = Param(m.k, initialize = rt_cost)
    #m.rt_c.pprint()

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

    # variable cost per time
    m.cvar = Param(initialize = 100) # cost per minute elapsed

    # max time of evacuation
    m.T = Param(initialize = time_limit)
    # m.T.pprint()

    # Penalty cost for leaving a person behind
    m.P = Param(initialize = penalty)
    # m.P.pprint()


    # probabilities per scenario
    probs = dict.fromkeys(scenarios)
    for i in probs:
        probs[i] = float(np.unique(scenarios_source['Probability'].loc[(scenarios_source['Scenario'] == i)])) 
    m.ps = Param(m.xi, initialize = probs)
    #m.ps.pprint()

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

    ## Define vessel selection variable
    m.z = Var(m.i, within = Binary, initialize = 0)

    ## Define the flow variables
    m.flab = Var(m.beta, within =  NonNegativeReals, bounds = (0, None), initialize = 0)
    m.flan = Var(m.a, within = NonNegativeReals, bounds = (0, None), initialize = 0)
    m.flat = Var(m.lambdas, within = NonNegativeReals, bounds = (0, None), initialize = 0)
    m.flbc = Var(m.gamma, within =  NonNegativeReals, bounds = (0, None), initialize = 0)
    m.flct = Var(m.epsilon, within =  NonNegativeReals, bounds = (0, None), initialize = 0)

    # Define binary variables
    m.w = Var(m.zeta, within = Binary, initialize = 0)
    m.x = Var(m.gamma, within = Binary, initialize = 0)
    m.y = Var(m.delta, within = Binary, initialize = 0)

    # Define usage length of each vessel
    m.u = Var(m.i, m.xi, within = NonNegativeReals, bounds = (0, None), initialize = 0)
    m.time_record = Var(m.i, m.xi, within = NonNegativeReals, bounds = (0, None), initialize = 0)

    con_vars = len(m.comp) + len(m.tsums) + len(m.flab) + len(m.flan) + len(m.flat) + len(m.flbc) + len(m.flct)
    bin_vars = len(m.z) + len(m.w) + len(m.x) + len(m.y)
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
    m.time_counts.pprint()

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
    #m.ad_b.pprint()

    m.ad_c = Constraint(m.c, rule = adj_c)
    # m.ad_c.pprint()

    m.sum_time = Constraint(m.xi, rule = tsum_calc)
    # m.sum_time.pprint()

    # Variable Cost high enough
    m.K = Param(initialize = sum(m.var_cost[i] * m.T for i in m.i)) #len(vessel_source) * time_limit)/10)
    # m.K.pprint()

    # fixed cost high enough
    # m.J = Param(initialize = sum(m.cfix[i] for i in m.i))
    #m.J.pprint()

    # Objective function definitions

    m.objective = Objective(rule=balanced3, sense=minimize, doc='Define stochastic objective function')
    # m.objective.pprint()

    return(m)


if __name__ == "__main__":
    main()

