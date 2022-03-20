import json
import math
from types import SimpleNamespace
from copy import deepcopy

class Node:
    def __init__(self, node_as_dict): 
        self.id = node_as_dict['id']
        self.label = node_as_dict['label']
        self.x = node_as_dict['metadata']['x']
        self.y = node_as_dict['metadata']['y']
        self.degree = 0
        self.neighbours = []

class Edge:
    def __init__(self, edge_as_dict, reverse=False):
        
        if reverse:
            self.source = edge_as_dict['target']
            self.target = edge_as_dict['source']
            self.relation = edge_as_dict['relation']
            # self.time = edge_as_dict['metadata']['time']
            self.line = edge_as_dict['metadata']['lines']
            self.feas_sections = []
        else:        
            self.source = edge_as_dict['source']
            self.target = edge_as_dict['target']
            self.relation = edge_as_dict['relation']
            # self.time = edge_as_dict['metadata']['time']
            self.line = edge_as_dict['metadata']['lines']
            self.feas_sections = []


class Graph: 
   
    def __init__(self, JSON_graph_path):
        with open(JSON_graph_path) as f:
            x = json.load(f)
        
        self.nodes = {node['id']: Node(node) for node in x['nodes']}
        self.fwd_edges = {i: Edge(edge) for i, edge in enumerate(x['edges'])} #edges as they are given in JSON
        self.rev_edges = {i: Edge(edge, reverse=True) for i, edge in enumerate(x['edges'])} #


        self.__find_neighbours()
        self.__calc_sections()

    def __find_neighbours(self):
        fwd_edges = self.fwd_edges
        rev_edges = self.rev_edges
        nodes = self.nodes
        for node_id, node in nodes.items():
            for edgeset in (fwd_edges, rev_edges):
                for edge in edgeset.values():
                    if edge.source == node_id:
                        node.neighbours.append(edge.target)
                        node.degree += 1
          
    def __calc_sections(self):
        '''
        Determines the "sections" of all edges in the graph. 

        :param sections: a list of section numbers, starting from 0. Sections are assumed to go counterclockwise from 3:00 [0,1,2...]
        '''
        sections = [0,1,2,3,4,5,6,7]
        nodes = self.nodes
        fwd_edges = self.fwd_edges
        rev_edges = self.rev_edges
        num_sections = len(sections)

        get_angle = lambda vector: (math.atan2(vector.y, vector.x) + 2*math.pi) % (2*math.pi) #angle between 0 and 2pi
        get_opposite_section = lambda section: (section+4)%8

        for edgeset in (fwd_edges, rev_edges):
            for edge in edgeset.values():

                #determine angle of edge
                source_node = nodes[edge.source]
                target_node = nodes[edge.target]
                vector = SimpleNamespace()
                vector.x = target_node.x - source_node.x
                vector.y = target_node.y - source_node.y
                a = get_angle(vector)
                del vector

                #determine feasible sections for a given angle
                section = round((a / (2*math.pi)) * num_sections)
                next_section = sections[(section+1)%len(sections)]
                prev_section = sections[(section-1)%len(sections)]
                edge.feas_sections = [prev_section, section, next_section]
                # edge.target_directions = list(map(get_opposite_section, edge.source_directions))
        
    def sort_neighbours(self):
        '''
        sorts a list of neighbours counterclockwise from positive x direction
        '''
        nodes = self.nodes
        get_angle = lambda vector: (math.atan2(vector.y, vector.x) + 2*math.pi) % (2*math.pi) #angle between 0 and 2p

        for node_id, node in nodes.items():
            to_sort = []
            for neighbour_id in node.neighbours:
                vector = SimpleNamespace()
                vector.x = nodes[neighbour_id].x - node.x
                vector.y = nodes[neighbour_id].y - node.y
                a = get_angle(vector)
                del vector
                
                to_sort.append([neighbour_id, a])

            sorted_neighbours = sorted(to_sort, key=lambda l:l[1], reverse=False) 
            node.neighbours = [sorted_neighbours[i][0] for i,_ in enumerate(sorted_neighbours)]

if __name__ == '__main__':

    #test code
    graph = Graph('./graphs/bvg.input.json')
    for node in graph.nodes.values():
        print(node.id)
        print(node.degree)
        print(node.neighbours, '\n')\
    
    # graph.sort_neighbours()
    # for node in graph.nodes.values():
    #     print(node.id)
    #     print(node.degree)
    #     print(node.neighbours, '\n')\
    
    # for edgeset in (graph.fwd_edges, graph.rev_edges):
    #     for id, edge in edgeset.items():
    #         print(id)
    #         print(edge.feas_sections)
