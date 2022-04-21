import gurobipy as gp
from gurobipy import GRB
import sys

import model_utils
from write_output import write_output
from create_graph import Graph

SETTINGS = {
    'min_edge_length': 1, 
    'max_edge_length': 5,
    'min_distance': 1,
    'callback_selector': 1,
        # 0 for no edge spacing constraints
        # 1 for edge spacing constraints implemented in a callback
        # 2 for all edge spacing contraints 
    'MIPFocus': 0,
        # 0 for standard settings 
        # 1 to emphasize finding a feasible solution quickly
        # 2 to focus on proving optimality
        # 3 to focus on shrinking best bound
    'obj_weights': [1, 1, 1],
        # [bend cost, relative position, minimize length]
    'early_stop': 0.04
        # 0 for standard settings
        # n>0 for stoping at optimality gap of n%
}


def main(folder, name):
#***************************************
#Problem Setup
#***************************************  
    
    graph = Graph(folder, name)

#****************************************
#Model Setup
#****************************************
    m=gp.Model('METRO_MAPS')
    m.modelSense = GRB.MINIMIZE
    m._settings = SETTINGS

    node_id_list = graph.nodes.keys()
    x = m.addVars(node_id_list, lb=0, vtype=GRB.CONTINUOUS, name='x')
    y = m.addVars(node_id_list, lb=0, vtype=GRB.CONTINUOUS, name='y')
    z1 = m.addVars(node_id_list, lb=0, vtype=GRB.CONTINUOUS, name='z1')
    z2 = m.addVars(node_id_list, lb=0, vtype=GRB.CONTINUOUS, name='z2')
    m._x = x
    m._y = y
    m._z1 = z1
    m._z2 = z2
    m._graph = graph #needed for gurobi callback

    m.addConstrs((z1[node] == (x[node]+y[node])/2 for node in node_id_list), 'z1_coord')
    m.addConstrs((z2[node] == (x[node]-y[node])/2 for node in node_id_list), 'z2_coord')

    model_utils.add_octolinear_constrs(m, graph)
    model_utils.add_max_edge_length_constrs(m, graph)
    model_utils.add_ordering_constrs(m, graph)
    model_utils.add_bend_costs(m, graph)
    model_utils.add_relative_pos_cost(m, graph)
    model_utils.add_edge_length_cost(m, graph)

    if m._settings['MIPFocus']:
        m.Params.MIPFocus = m._settings['MIPFocus']
    if m._settings['early_stop']:
        m.Params.MIPGap = m._settings['early_stop']
    
    m.update()

    if m._settings['callback_selector'] == 0:
        model_utils.add_edge_spacing_constrs(m, graph)
        m.write('output.lp')
        m.optimize()

    elif m._settings['callback_selector'] == 1:        
        model_utils.add_gammas_only(m, graph)
        m.Params.lazyConstraints = 1
        m.update()
        m.write('output.lp')
        m.optimize(model_utils.edge_spacing_callback)

    elif m._settings['callback_selector'] == 2:
        model_utils.add_edge_spacing_constrs(m, graph)
        m.update()
        m.write('output.lp')
        m.optimize()

    else:
        raise ValueError('No call_back_selector variable specified in Model Settings') #FIXME might be incorrect implementation

    write_output(m, graph)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(str(sys.argv[1]))
    else:
        main('./graphs/', 'montpellier.input.json')
        