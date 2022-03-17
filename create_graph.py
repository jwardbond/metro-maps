import json


class Node:
    def __init__(self, node_as_dict): 
        self.id = node_as_dict['id']
        self.label = node_as_dict['label']
        self.x = node_as_dict['metadata']['x']
        self.y = node_as_dict['metadata']['y']

class Edge:
    def __init__(self, edge_as_dict):
        self.source = edge_as_dict['source']
        self.target = edge_as_dict['target']
        self.relation = edge_as_dict['relation']
        self.time = edge_as_dict['metadata']['time']
        self.line = edge_as_dict['metadata']['lines']
        self.source_direction = []
        self.target_direction = []


class Graph: 
   
    def __init__(self, JSON_graph_path):
        with open(JSON_graph_path) as f:
            x = json.load(f)
        
        #create dictionaries of node and edge OBJECTS from the JSON
        self.nodes = {node['id']: Node(node) for node in x['nodes']}
        self.edges = {i: Edge(edge) for i, edge in enumerate(x['edges'])}


if __name__ == '__main__':
    graph = Graph('./graphs/test.input.json')
    print(graph.nodes)
    print(graph.edges)
