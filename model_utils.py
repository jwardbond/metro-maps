import numpy as np
import gurobipy as gp
from gurobipy import GRB

from create_graph import Graph

# Adds sectioning constraints. When there are 8 sections, this means it


def add_octolinear_constrs(model, graph):
    '''
    Constrains sections to octolinearity

    :params model: a gurobi model
    :params graph: a Graph object
    '''

    num_feas_dirs = len(graph.edges[0].source_directions)

    # add binary selection variables
    alphas = model.addVars(len(graph.edges), num_feas_dirs,
                           lb=0, ub=1, vtype=GRB.BINARY, name="alpha")
    
    # add edge direction variables
    dirs = model.addVars(len(graph.edges), lb=0, ub=1,
                         vtype=GRB.INTEGER, name="dir")  # FIXME do I need to add other vars for other direction?

    # disjunct modeling constraints, confine dir to one of the three feasible sections in the graph
    model.addConstrs((gp.quicksum(alphas[e, j] for j in range(
        num_feas_dirs)) == 1 for e in graph.edges), 'octolinear_disjunct_binaries')

    model.addConstrs((gp.quicksum(graph.edges[e].source_directions[j] * alphas[e, j]
                     for j in range(num_feas_dirs)) == 1 for e in graph.edges), "octolinear_disjunct")  # FIXME currently alphas can be 0, which might mess with this


if __name__ == '__main__':
    graph = Graph('./graphs/test.input.json')
    graph.calc_sections()

    m = gp.Model('METRO_MAPS')
    m.modelSense = GRB.MINIMIZE

    node_id_list = graph.nodes.keys()
    x = m.addVars(node_id_list, lb=0, vtype=GRB.CONTINUOUS, name='x')
    y = m.addVars(node_id_list, lb=0, vtype=GRB.CONTINUOUS, name='y')
    z1 = m.addVars(node_id_list, lb=0, vtype=GRB.CONTINUOUS, name='z1')
    z2 = m.addVars(node_id_list, lb=0, vtype=GRB.CONTINUOUS, name='z2')

    m.addConstrs((z1[node] == (x[node]+y[node])/2 for node in node_id_list), 'z1_coord')
    m.addConstrs((z2[node] == (x[node]-y[node])/2 for node in node_id_list), 'z2_coord')

    # Get feasible sectors for all edges
    add_octolinear_constrs(m, graph)
    m.write('oct_output.lp')
