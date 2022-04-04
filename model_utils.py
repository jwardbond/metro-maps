import numpy as np
import gurobipy as gp
from gurobipy import GRB
from itertools import combinations, compress

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

def add_ordering_constrs(model, graph):
    '''
    Conserves counterclockwise ordering around all nodes

    :params model: a gurobi model
    :params graph: a Graph object   
    '''
    fwd_dirs = model._fwd_dirs
    rev_dirs = model._rev_dirs

    fwd_edges_index = {(edge.source, edge.target): id for id, edge in graph.fwd_edges.items()} #id: (source, target) is a hack to help with indexing
    rev_edges_index = {(edge.source, edge.target): id for id, edge in graph.rev_edges.items()}
    
    betas = {}
    for node_id, node in graph.nodes.items():
        if node.degree >= 2:
            betas[node_id] = model.addVars(node.degree, lb=0, ub=1, vtype=GRB.BINARY, name='beta_{}_'.format(node_id))
            model.addConstr(betas[node_id].sum() == 1, name='node{}_circ_order_binaries'.format(node_id))
        
            neighbour_ids = node.neighbours
            for i in range(len(neighbour_ids)):
                next_i = (i+1)%len(neighbour_ids) # hack for looping back to beginning of neighbour array

                # Get dir of first (node, neighbour) edge
                if (node_id, neighbour_ids[i]) in fwd_edges_index:
                    first = fwd_dirs[fwd_edges_index[(node_id, neighbour_ids[i])]]
                elif (node_id, neighbour_ids[i]) in rev_edges_index:
                    first = rev_dirs[rev_edges_index[(node_id, neighbour_ids[i])]]
                
                # Get dir of next (node, neighbour) edge
                if (node_id, neighbour_ids[next_i]) in fwd_edges_index:
                    second = fwd_dirs[fwd_edges_index[(node_id, neighbour_ids[next_i])]]
                elif (node_id, neighbour_ids[next_i]) in rev_edges_index:
                    second = rev_dirs[rev_edges_index[(node_id, neighbour_ids[next_i])]]
                
                model.addConstr(first <= second - 1 + 8*betas[node_id][i], 'circular_order_node{}_{}'.format(node_id, i))
            
    model._betas = betas
            
def add_edge_spacing_constrs(model, graph):

    x = model._x
    y = model._y
    z1 = model._z1
    z2 = model._z2
    bigM = 10000 #TODO calculate a better bound
    d_min = model._settings['min_distance']

    edge_combinations = list(combinations(graph.fwd_edges, 2))
    list_filter_booleans = []

    # Find which edge sets contain incident edges (shared node)
    for edge_combination in edge_combinations:
        node_set = set()
        for edge_id in edge_combination:
            node_set.add(graph.fwd_edges[edge_id].source)
            node_set.add(graph.fwd_edges[edge_id].target)
        list_filter_booleans.append(len(node_set) == 4) # checks for duplicated edges

    non_incident_edges = list(compress(edge_combinations, list_filter_booleans))
    
    for edge_pair in non_incident_edges:
        e1_id = edge_pair[0]
        e2_id = edge_pair[1]
        s1 = graph.fwd_edges[e1_id].source
        t1 = graph.fwd_edges[e1_id].target
        s2 = graph.fwd_edges[e2_id].source
        t2 = graph.fwd_edges[e2_id].target

        gamma = model.addVars(8, lb=0, ub=1, vtype=GRB.BINARY, name='gamma_(e{},e{})'.format(e1_id, e2_id))
        model.addConstr(gp.quicksum(gamma[i] for i in range(8)) == 1, 'gamma_sum(e{},e{})'.format(e1_id, e2_id))

        #section 0
        model.addConstr(x[s2]-x[s1] <= bigM*(1-gamma[0])-d_min)
        model.addConstr(x[s2]-x[t1] <= bigM*(1-gamma[0])-d_min)
        model.addConstr(x[t2]-x[s1] <= bigM*(1-gamma[0])-d_min)
        model.addConstr(x[t2]-x[t1] <= bigM*(1-gamma[0])-d_min)

        #section 1
        model.addConstr(z1[s2]-z1[s1] <= bigM*(1-gamma[1])-d_min)
        model.addConstr(z1[s2]-z1[t1] <= bigM*(1-gamma[1])-d_min)
        model.addConstr(z1[t2]-z1[s1] <= bigM*(1-gamma[1])-d_min)
        model.addConstr(z1[t2]-z1[t1] <= bigM*(1-gamma[1])-d_min)

        #section 2
        model.addConstr(y[s2]-y[s1] <= bigM*(1-gamma[2])-d_min)
        model.addConstr(y[s2]-y[t1] <= bigM*(1-gamma[2])-d_min)
        model.addConstr(y[t2]-y[s1] <= bigM*(1-gamma[2])-d_min)
        model.addConstr(y[t2]-y[t1] <= bigM*(1-gamma[2])-d_min)

        # #section 3
        model.addConstr(-z2[s2]+z2[s1] <= bigM*(1-gamma[3])-d_min)
        model.addConstr(-z2[s2]+z2[t1] <= bigM*(1-gamma[3])-d_min)
        model.addConstr(-z2[t2]+z2[s1] <= bigM*(1-gamma[3])-d_min)
        model.addConstr(-z2[t2]+z2[t1] <= bigM*(1-gamma[3])-d_min)

        #section 4
        model.addConstr(-x[s2]+x[s1] <= bigM*(1-gamma[4])-d_min)
        model.addConstr(-x[s2]+x[t1] <= bigM*(1-gamma[4])-d_min)
        model.addConstr(-x[t2]+x[s1] <= bigM*(1-gamma[4])-d_min)
        model.addConstr(-x[t2]+x[t1] <= bigM*(1-gamma[4])-d_min)

        #section 5
        model.addConstr(-z1[s2]+z1[s1] <= bigM*(1-gamma[5])-d_min)
        model.addConstr(-z1[s2]+z1[t1] <= bigM*(1-gamma[5])-d_min)
        model.addConstr(-z1[t2]+z1[s1] <= bigM*(1-gamma[5])-d_min)
        model.addConstr(-z1[t2]+z1[t1] <= bigM*(1-gamma[5])-d_min)

        #section 6
        model.addConstr(-y[s2]+y[s1] <= bigM*(1-gamma[6])-d_min)
        model.addConstr(-y[s2]+y[t1] <= bigM*(1-gamma[6])-d_min)
        model.addConstr(-y[t2]+y[s1] <= bigM*(1-gamma[6])-d_min)
        model.addConstr(-y[t2]+y[t1] <= bigM*(1-gamma[6])-d_min)

        #section 7
        model.addConstr(z2[s2]-z2[s1] <= bigM*(1-gamma[7])-d_min)
        model.addConstr(z2[s2]-z2[t1] <= bigM*(1-gamma[7])-d_min)
        model.addConstr(z2[t2]-z2[s1] <= bigM*(1-gamma[7])-d_min)
        model.addConstr(z2[t2]-z2[t1] <= bigM*(1-gamma[7])-d_min)                      

def add_bend_costs(model, graph):

    fwd_dirs = model._fwd_dirs

    # Get all metro lines
    lines = set()
    for edge in graph.fwd_edges.values():
        for line in edge.lines:
            lines.add(str(line))
    lines = {i: [] for i in lines} #convert to dict

    #TODO combine into the above block
    for line, line_edges in lines.items():
        for edge_id, edge in graph.fwd_edges.items():
            if line in edge.lines:
                line_edges.append(edge_id)

    # Sort edges in order
    for line,  line_edges in lines.items():
        q = line_edges
        sorted = []
        i = 0
        while q:

            if not sorted:
                sorted.append(q.pop(0))
            
            if graph.fwd_edges[sorted[0]].source == graph.fwd_edges[i].target:
                sorted.insert(0, q.pop(i))
                i = 0
            elif graph.fwd_edges[sorted[-1]].target == graph.fwd_edges[i].source:
                sorted.append(q.pop(i))
                i = 0
            else:
                i += 1
            
        line_edges = sorted
    
    print(lines)

    


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
    add_edge_spacing_constrs(m, graph)
    m.write('oct_output.lp')
