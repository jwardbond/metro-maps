import numpy as np
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from itertools import combinations, compress
from types import SimpleNamespace

from geometry_utils import do_intersect, too_close

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
                         vtype=GRB.INTEGER, name="fwd_dirs") 
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

def add_max_edge_length_constrs(model, graph):
    fwd_dirs = model._fwd_dirs
    fwd_edges = graph.fwd_edges
    nodes = graph.nodes
    x = model._x
    y = model._y
    lmax = model._settings['max_edge_length']

    model.addConstrs((x[fwd_edges[e].source] - x[fwd_edges[e].target] <= lmax for e in fwd_edges), 'L_max_x_pos')
    model.addConstrs((-x[fwd_edges[e].source] + x[fwd_edges[e].target] <= lmax for e in fwd_edges), 'L_max_x_neg')
    model.addConstrs((y[fwd_edges[e].source] - y[fwd_edges[e].target] <= lmax for e in fwd_edges), 'L_max_y_pos')
    model.addConstrs((-y[fwd_edges[e].source] + y[fwd_edges[e].target] <= lmax for e in fwd_edges), 'L_max_y_neg')

    model.update()

def add_edge_spacing_constrs(model, graph):
    '''
    Ensures planarity in the resulting solution. i.e. edges can't cross over eachother
    '''

    x = model._x
    y = model._y
    z1 = model._z1
    z2 = model._z2
    bigM = len(graph.fwd_edges) * model._settings['max_edge_length']
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
    rev_dirs = model._rev_dirs

    # Create dictionaries for looking up edge_id by source and target nodes
    fwd_edges = {(edge.source, edge.target): edge_id for edge_id, edge in graph.fwd_edges.items()}
    rev_edges = {(edge.source, edge.target): edge_id for edge_id, edge in graph.rev_edges.items()}

    # Get all metro lines
    # TODO this can be changed to use the line attribute of graph
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

    bend_costs = []
    # Sort edges in order - essentially finding a path   
    for line,  line_edges in lines.items():
        
        # Get start and end nodes for each item
        line_dict = {}
        for edge_id in line_edges:
            source = graph.fwd_edges[edge_id].source
            target = graph.fwd_edges[edge_id].target
            line_dict[(source, target)] = edge_id
        
        #Order edges in line according to path
        G = nx.Graph()
        G.add_edges_from(list(line_dict.keys()))
        leaf_nodes = [x for x in G.nodes if G.degree(x) == 1]
        if leaf_nodes: #no cycle
            path = nx.all_simple_edge_paths(G, leaf_nodes[0], leaf_nodes[1]) 
        else: 
            to_remove = [i for i in list(G.edges())[0]]
            G.remove_edge(to_remove[0], to_remove[1]) #TODO add bend cost of last arc ~~~ Fixes cycle
            path = nx.all_simple_edge_paths(G, to_remove[0], to_remove[1])
    
        #Add bend constr for each edge along the path
        path = list(path)[0]
        for i, _ in enumerate(path[:-1]): 
            
            edge1 = path[i]
            if (edge1[0], edge1[1]) in fwd_edges: 
                dir1 = fwd_dirs[fwd_edges[edge1]]
            else: 
                dir1 = rev_dirs[rev_edges[edge1]]
            
            edge2 = path[i+1]
            if (edge2[0], edge2[1]) in fwd_edges: 
                dir2 = fwd_dirs[fwd_edges[edge2]]
            else: 
                dir2 = rev_dirs[rev_edges[edge2]]

            d = model.addVars(2, lb=0, ub=1, vtype = GRB.BINARY, name='d({},{},{})'.format(edge1[0], edge1[1], edge2[1]))
            bend_cost = model.addVar(lb=0, vtype = GRB.CONTINUOUS, name='bend_cost({},{},{})'.format(edge1[0], edge1[1], edge2[1]))
            model.addConstr(-bend_cost <= dir1 - dir2 - 8*d[0] + 8*d[1])
            model.addConstr(bend_cost >= dir1 - dir2 - 8*d[0] + 8*d[1])
            
            bend_costs.append(bend_cost)


    model.setObjective(gp.LinExpr(np.ones(len(bend_costs)), bend_costs))
    model.update()

def add_relative_pos_cost(model, graph):

    M = 7

    fwd_dirs = model._fwd_dirs
    fwd_edges = graph.fwd_edges

    rpos = model.addVars(len(fwd_edges), lb=0, ub=1, vtype=GRB.BINARY, name='pos_binary')
    
    model.addConstrs((fwd_dirs[e] - fwd_edges[e].feas_sections[1] <= M*rpos[e] for e in fwd_edges), 'pos_upper')
    model.addConstrs((fwd_dirs[e] - fwd_edges[e].feas_sections[1] >= -1*M*rpos[e] for e in fwd_edges), 'pos_upper')

    
    old_objective = model.getObjective()
    model.setObjective(old_objective + gp.quicksum(rpos))
    model.update()

def add_edge_length_cost(model, graph): 
    fwd_edges = graph.fwd_edges
    x = model._x
    y = model._y

    l = model.addVars(len(fwd_edges), lb=0, vtype=GRB.CONTINUOUS, name='edge_length')
    model.addConstrs((x[fwd_edges[e].source] - x[fwd_edges[e].target] <= l[e] for e in fwd_edges), 'L_x_pos')
    model.addConstrs((-x[fwd_edges[e].source] + x[fwd_edges[e].target] <= l[e] for e in fwd_edges), 'L_x_neg')
    model.addConstrs((y[fwd_edges[e].source] - y[fwd_edges[e].target] <= l[e] for e in fwd_edges), 'L_y_pos')
    model.addConstrs((-y[fwd_edges[e].source] + y[fwd_edges[e].target] <= l[e] for e in fwd_edges), 'L_y_neg')

    old_objective = model.getObjective()
    model.setObjective(old_objective + gp.quicksum(l))
    model.update()


def add_gammas_only(model, graph):
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
    
    gamma_dict = {}
    for edge_pair in non_incident_edges:
        e1_id = edge_pair[0]
        e2_id = edge_pair[1]

        gammas = model.addVars(8, lb=0, ub=1, vtype=GRB.BINARY, name='gamma_(e{},e{})'.format(e1_id, e2_id))
        gamma_dict[edge_pair]= gammas
    
    model._gamma_dict = gamma_dict
    model._non_incident_edges = non_incident_edges
    model.update()


def edge_spacing_callback(model, where):
    if where == GRB.Callback.MIPSOL:
        
        graph = model._graph
        bigM = len(graph.fwd_edges) * model._settings['max_edge_length']
        d_min = model._settings['min_distance']

        gamma_dict = model._gamma_dict
        non_incident_edges = model._non_incident_edges

        x_val = model.cbGetSolution(model._x)
        y_val = model.cbGetSolution(model._y)
        z1_val = model.cbGetSolution(model._z1)
        z2_val = model.cbGetSolution(model._z2)
        x = model._x
        y = model._y
        z1 = model._z1
        z2 = model._z2

        for edge_pair in non_incident_edges:
            e1_id = edge_pair[0]
            e2_id = edge_pair[1]

            s1 = graph.fwd_edges[e1_id].source
            t1 = graph.fwd_edges[e1_id].target
            s2 = graph.fwd_edges[e2_id].source
            t2 = graph.fwd_edges[e2_id].target

            p1 = SimpleNamespace()
            p1.x = x_val[s1]
            p1.y = y_val[s1]
            p1.z1 = z1_val[s1]
            p1.z2 = z2_val[s1]

            q1 = SimpleNamespace()
            q1.x = x_val[t1]
            q1.y = y_val[t1]
            q1.z1 = z1_val[t1]
            q1.z2 = z2_val[t1]

            p2 = SimpleNamespace()
            p2.x = x_val[s2]
            p2.y = y_val[s2]
            p2.z1 = z1_val[s2]
            p2.z2 = z2_val[s2]

            q2 = SimpleNamespace()
            q2.x = x_val[t2]
            q2.y = y_val[t2]
            q2.z1 = z1_val[t2]
            q2.z2 = z2_val[t2]
            
            if too_close(p1, q1, p2, q2, d_min) or do_intersect(p1, q1, p2, q2):
                gamma = gamma_dict[edge_pair]
                # ggg = [g for g in gamma.values()]
                model.cbLazy(gp.quicksum(gamma) == 1, 'gamma_sum(e{},e{})'.format(e1_id, e2_id))

                #section 0
                model.cbLazy(x[s2]-x[s1] <= bigM*(1-gamma[0])-d_min)
                model.cbLazy(x[s2]-x[t1] <= bigM*(1-gamma[0])-d_min)
                model.cbLazy(x[t2]-x[s1] <= bigM*(1-gamma[0])-d_min)
                model.cbLazy(x[t2]-x[t1] <= bigM*(1-gamma[0])-d_min)

                #section 1
                model.cbLazy(z1[s2]-z1[s1] <= bigM*(1-gamma[1])-d_min)
                model.cbLazy(z1[s2]-z1[t1] <= bigM*(1-gamma[1])-d_min)
                model.cbLazy(z1[t2]-z1[s1] <= bigM*(1-gamma[1])-d_min)
                model.cbLazy(z1[t2]-z1[t1] <= bigM*(1-gamma[1])-d_min)

                #section 2
                model.cbLazy(y[s2]-y[s1] <= bigM*(1-gamma[2])-d_min)
                model.cbLazy(y[s2]-y[t1] <= bigM*(1-gamma[2])-d_min)
                model.cbLazy(y[t2]-y[s1] <= bigM*(1-gamma[2])-d_min)
                model.cbLazy(y[t2]-y[t1] <= bigM*(1-gamma[2])-d_min)

                # #section 3
                model.cbLazy(-z2[s2]+z2[s1] <= bigM*(1-gamma[3])-d_min)
                model.cbLazy(-z2[s2]+z2[t1] <= bigM*(1-gamma[3])-d_min)
                model.cbLazy(-z2[t2]+z2[s1] <= bigM*(1-gamma[3])-d_min)
                model.cbLazy(-z2[t2]+z2[t1] <= bigM*(1-gamma[3])-d_min)

                #section 4
                model.cbLazy(-x[s2]+x[s1] <= bigM*(1-gamma[4])-d_min)
                model.cbLazy(-x[s2]+x[t1] <= bigM*(1-gamma[4])-d_min)
                model.cbLazy(-x[t2]+x[s1] <= bigM*(1-gamma[4])-d_min)
                model.cbLazy(-x[t2]+x[t1] <= bigM*(1-gamma[4])-d_min)

                #section 5
                model.cbLazy(-z1[s2]+z1[s1] <= bigM*(1-gamma[5])-d_min)
                model.cbLazy(-z1[s2]+z1[t1] <= bigM*(1-gamma[5])-d_min)
                model.cbLazy(-z1[t2]+z1[s1] <= bigM*(1-gamma[5])-d_min)
                model.cbLazy(-z1[t2]+z1[t1] <= bigM*(1-gamma[5])-d_min)

                #section 6
                model.cbLazy(-y[s2]+y[s1] <= bigM*(1-gamma[6])-d_min)
                model.cbLazy(-y[s2]+y[t1] <= bigM*(1-gamma[6])-d_min)
                model.cbLazy(-y[t2]+y[s1] <= bigM*(1-gamma[6])-d_min)
                model.cbLazy(-y[t2]+y[t1] <= bigM*(1-gamma[6])-d_min)

                #section 7
                model.cbLazy(z2[s2]-z2[s1] <= bigM*(1-gamma[7])-d_min)
                model.cbLazy(z2[s2]-z2[t1] <= bigM*(1-gamma[7])-d_min)
                model.cbLazy(z2[t2]-z2[s1] <= bigM*(1-gamma[7])-d_min)
                model.cbLazy(z2[t2]-z2[t1] <= bigM*(1-gamma[7])-d_min)   



def edge_spacing_callback_no_vars(model, where):

    if where == GRB.Callback.MIPSOL:
        graph = model._graph
        x_val = model.cbGetSolution(model._x)
        y_val = model.cbGetSolution(model._y)
        x = model._x
        y = model._y
        z1 = model._z1
        z2 = model._z2
        bigM = len(graph.fwd_edges) * model._settings['max_edge_length']
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

            p1 = SimpleNamespace()
            p1.x = x_val[s1]
            p1.y = y_val[s1]

            q1 = SimpleNamespace()
            q1.x = x_val[t1]
            q1.y = y_val[t1]

            p2 = SimpleNamespace()
            p2.x = x_val[s2]
            p2.y = y_val[s2]

            q2 = SimpleNamespace()
            q2.x = x_val[t2]
            q2.y = y_val[t2]
            
            if too_close(p1, q1, p2, q2, d_min) or do_intersect(p1, q1, p2, q2):
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

