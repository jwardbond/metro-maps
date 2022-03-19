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

    # TODO might be something tighter here
    bigM = len(graph.fwd_edges) * model._settings['max_edge_length']
    min_length = model._settings['min_edge_length']
    num_feas_dirs = len(graph.fwd_edges[0].feas_sections)

    # VARS: disjunct selector vars. one for every fwd/reverse pair of edges
    alphas = model.addVars(len(graph.fwd_edges), num_feas_dirs,
                           lb=0, ub=1, vtype=GRB.BINARY, name="alphas")
    model._alphas = alphas

    # VARS: edge directions
    fwd_dirs = model.addVars(len(graph.fwd_edges), lb=0,
                         vtype=GRB.INTEGER, name="fwd_dirs")  # FIXME do I need to add other vars for other direction?
    rev_dirs = model.addVars(len(graph.rev_edges), lb=0,
                         vtype=GRB.INTEGER, name="rev_dirs")
    model._fwd_dirs = fwd_dirs
    model._rev_dirs = rev_dirs

    # CONSTRAINTS: disjunct modeling constraints, confine dir to one of the three feasible sections in the graph
    model.addConstrs((gp.quicksum(alphas[e, j] for j in range(
        num_feas_dirs)) == 1 for e in graph.fwd_edges), 'octolinear_disjunct_binaries')

    model.addConstrs((gp.quicksum(graph.fwd_edges[e].feas_sections[j] * alphas[e, j]
                     for j in range(num_feas_dirs)) == fwd_dirs[e] for e in graph.fwd_edges), "octolinear_disjunct_fwd")  # FIXME currently alphas can be 0, which might mess with this
    
    model.addConstrs((gp.quicksum(graph.rev_edges[e].feas_sections[j] * alphas[e, j]
                     for j in range(num_feas_dirs)) == rev_dirs[e] for e in graph.rev_edges), "octolinear_disjunct_rev") # forces section of reverse edge to be the exact opposite of fwd

    # CONSTRAINTS: octolinear
    x = model._x
    y = model._y
    z1 = model._z1
    z2 = model._z2

    for i, edgeset in enumerate((graph.fwd_edges, graph.rev_edges)):
        if i==1:
            name = 'rev'
        else:
            name = 'fwd'
        for edge_id, edge in edgeset.items():
            source = edge.source
            target = edge.target
            for i, section in enumerate(edge.feas_sections):
                if section == 0:
                    model.addConstr(y[source] - y[target] <= bigM*(1-alphas[edge_id, i]), name='{}_edge{}-sec{}-1'.format(name, edge_id, section))
                    model.addConstr(-y[source] + y[target] <= bigM*(1-alphas[edge_id, i]), name='{}_edge{}-sec{}-2'.format(name, edge_id, section))
                    model.addConstr(-x[source] + x[target] >= -1*bigM*(1-alphas[edge_id, i]) + min_length, name='{}_edge{}-sec{}-3'.format(name, edge_id, section))
                elif section == 1:
                    model.addConstr(z2[source] - z2[target] <= bigM*(1-alphas[edge_id, i]), name='{}_edge{}-sec{}-1'.format(name, edge_id, section))
                    model.addConstr(-z2[source] + z2[target] <= bigM*(1-alphas[edge_id, i]), name='{}_edge{}-sec{}-2'.format(name, edge_id, section))
                    model.addConstr(-z1[source] + z1[target] >= -1*bigM*(1-alphas[edge_id, i]) + min_length, name='{}_edge{}-sec{}-3'.format(name, edge_id, section))
                elif section == 2:
                    model.addConstr(x[source] - x[target] <= bigM*(1-alphas[edge_id, i]), name='{}_edge{}-sec{}-1'.format(name, edge_id, section))
                    model.addConstr(-x[source] + x[target] <= bigM*(1-alphas[edge_id, i]), name='{}_edge{}-sec{}-2'.format(name, edge_id, section))
                    model.addConstr(-y[source] + y[target] >= -1*bigM*(1-alphas[edge_id, i]) + min_length, name='{}_edge{}-sec{}-3'.format(name, edge_id, section))
                elif section == 3:
                    model.addConstr(z1[source] - z1[target] <= bigM*(1-alphas[edge_id, i]), name='{}_edge{}-sec{}-1'.format(name, edge_id, section))
                    model.addConstr(-z1[source] + z1[target] <= bigM*(1-alphas[edge_id, i]), name='{}_edge{}-sec{}-2'.format(name, edge_id, section))
                    model.addConstr(z2[source] - z2[target] >= -1*bigM*(1-alphas[edge_id, i]) + min_length, name='{}_edge{}-sec{}-3'.format(name, edge_id, section))
                elif section == 4:
                    model.addConstr(y[source] - y[target] <= bigM*(1-alphas[edge_id, i]), name='{}_edge{}-sec{}-1'.format(name, edge_id, section))
                    model.addConstr(-y[source] + y[target] <= bigM*(1-alphas[edge_id, i]), name='{}_edge{}-sec{}-2'.format(name, edge_id, section))
                    model.addConstr(x[source] - x[target] >= -1*bigM*(1-alphas[edge_id, i]) + min_length, name='{}_edge{}-sec{}-3'.format(name, edge_id, section))
                elif section == 5:
                    model.addConstr(z2[source] - z2[target] <= bigM*(1-alphas[edge_id, i]), name='{}_edge{}-sec{}-1'.format(name, edge_id, section))
                    model.addConstr(-z2[source] + z2[target] <= bigM*(1-alphas[edge_id, i]), name='{}_edge{}-sec{}-2'.format(name, edge_id, section))
                    model.addConstr(z1[source] - z1[target] >= -1*bigM*(1-alphas[edge_id, i]) + min_length, name='{}_edge{}-sec{}-3'.format(name, edge_id, section))
                elif section == 6:
                    model.addConstr(x[source] - x[target] <= bigM*(1-alphas[edge_id, i]), name='{}_edge{}-sec{}-1'.format(name, edge_id, section))
                    model.addConstr(-x[source] + x[target] <= bigM*(1-alphas[edge_id, i]), name='{}_edge{}-sec{}-2'.format(name, edge_id, section))
                    model.addConstr(y[source] - y[target] >= -1*bigM*(1-alphas[edge_id, i]) + min_length, name='{}_edge{}-sec{}-3'.format(name, edge_id, section))
                elif section == 7:
                    model.addConstr(z1[source] - z1[target] <= bigM*(1-alphas[edge_id, i]), name='{}_edge{}-sec{}-1'.format(name, edge_id, section))
                    model.addConstr(-z1[source] + z1[target] <= bigM*(1-alphas[edge_id, i]), name='{}_edge{}-sec{}-2'.format(name, edge_id, section))
                    model.addConstr(-z2[source] + z2[target] >= -1*bigM*(1-alphas[edge_id, i]) + min_length, name='{}_edge{}-sec{}-3'.format(name, edge_id, section))

# def add_ordering_constrs(model, graph):
#     '''
#     Conserves counterclockwise ordering around all nodes

#     :params model: a gurobi model
#     :params graph: a Graph object   
#     '''

#     betas = {}
#     for node_id, node in graph.nodes:
#         if node.degree >= 2:
#             betas[node_id] = model.addVars(node.degree, lb=0, ub=1, vtype=GRB.BINARY, name='beta_{}_'.format(node_id))


if __name__ == '__main__':
    graph = Graph('./graphs/test.input.json')

    m = gp.Model('METRO_MAPS')
    m.modelSense = GRB.MINIMIZE

    node_id_list = graph.nodes.keys()
    x = m.addVars(node_id_list, lb=0, vtype=GRB.CONTINUOUS, name='x')
    y = m.addVars(node_id_list, lb=0, vtype=GRB.CONTINUOUS, name='y')
    z1 = m.addVars(node_id_list, lb=0, vtype=GRB.CONTINUOUS, name='z1')
    z2 = m.addVars(node_id_list, lb=0, vtype=GRB.CONTINUOUS, name='z2')

    m.addConstrs((z1[node] == (x[node]+y[node]) /
                 2 for node in node_id_list), 'z1_coord')
    m.addConstrs((z2[node] == (x[node]-y[node]) /
                 2 for node in node_id_list), 'z2_coord')

    # Get feasible sectors for all edges
    add_octolinear_constrs(m, graph)
    m.write('oct_output.lp')
