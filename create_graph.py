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
    def __init__(self, edge_as_dict):
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
        
        #create dictionaries of node and edge OBJECTS from the JSON
        self.nodes = {node['id']: Node(node) for node in x['nodes']}
        self.edges = {i: Edge(edge) for i, edge in enumerate(x['edges'])}

        
        self.__find_neighbours()
        self.__add_reverse_edges()
        self.__calc_sections()

    def __find_neighbours(self):
        for node_id, node in self.nodes.items():
            for edge_id, edge in self.edges.items():
                if edge.source == node_id:
                    node.neighbours.append(edge.target)
                    node.degree += 1
                elif edge.target == node_id:
                    node.neighbours.append(edge.source)
                    node.degree += 1

    def __add_reverse_edges(self): 
        edges = self.edges
        num_edges = len(edges)

        for edge_id in range(num_edges):
            to_reverse = edges[edge_id]
            edges[num_edges + edge_id] = deepcopy(to_reverse)
            edges[num_edges + edge_id].source = to_reverse.target
            edges[num_edges + edge_id].target = to_reverse.source

            
    def __calc_sections(self):
        '''
        Determines the "sections" of all edges in the graph. 

        :param sections: a list of section numbers, starting from 0. Sections are assumed to go counterclockwise from 3:00 [0,1,2...]
        '''
        sections = [0,1,2,3,4,5,6,7]
        nodes = self.nodes
        edges = self.edges
        num_sections = len(sections)

        get_angle = lambda vector: (math.atan2(vector.y, vector.x) + 2*math.pi) % (2*math.pi) #angle between 0 and 2pi
        get_opposite_section = lambda section: (section+4)%8

        for edge in edges.values():

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
        

if __name__ == '__main__':

    #test code
    graph = Graph('./graphs/test.input.json')
    for node in graph.nodes.values():
        print(node.id)
        print(node.degree)
        print(node.neighbours, '\n')\

    for id, edge in graph.edges.items():
        print(id)
        print(edge.feas_sections)
