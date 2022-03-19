import numpy as np
import gurobipy as gp
from gurobipy import GRB
import sys
import math

import model_utils
from create_graph import Graph

SETTINGS = {
    'min_edge_length': 2, 
    'max_edge_length': 4
}


def main(graph_path):
#***************************************
#Problem Setup
#***************************************
    try:
        graph = Graph(graph_path)
    except:
        print('Path' + graph_path + 'not found')
        sys.exit(1)
    
    # sections = [0,1,2,3,4,5,6,7]
    graph.calc_sections()

#****************************************
#Model Setup
#****************************************
    m=gp.Model('METRO_MAPS')
    m.modelSense = GRB.MINIMIZE
    m._settings = SETTINGS

    node_id_list = graph.nodes.keys()
    x = m.addvars(node_id_list, lb=0, vtype=GRB.CONTINUOUS, name='x')
    y = m.addvars(node_id_list, lb=0, vtype=GRB.CONTINUOUS, name='y')
    z1 = m.addvars(node_id_list, lb=0, vtype=GRB.CONTINUOUS, name='z1')
    z2 = m.addvars(node_id_list, lb=0, vtype=GRB.CONTINUOUS, name='z2')

    m.addConstrs((z1[node] == (x[node]+y[node])/2 for node in node_id_list), 'z1_coord')
    m.addConstrs((z2[node] == (x[node]-y[node])/2 for node in node_id_list), 'z2_coord')

    #Get feasible sectors for all edges
    model_utils.add_octolinear_constrs(m, graph)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(str(sys.argv[1]))
    else:
        main('./graphs/test.input.json')
        