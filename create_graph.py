import json

class Graph: 
   
    def __init__(self, JSON_graph_path):
        with open(JSON_graph_path) as f:
            x = json.load(f)
        self.nodes = self.get_nodes_from_JSON(x['nodes'])
        self.edges = self.get_edges_from_JSON(x['edges'])
    
    
    def get_nodes_from_JSON(self, node_list):
        """
        Converts a list of nodes to a dictionary where each node is indexed by its id
        i.e. [{id:'', label:'', metadata:''},...] to {id: {label:'', metadata:''},...}
        """
        node_dict = {}
        for node in node_list: 
            id = node.pop('id')
            node_dict[id] = node
        return node_dict

    def get_edges_from_JSON(self, edge_list):
        edge_dict = {i: edge for i, edge in enumerate(edge_list)}
        return edge_dict

if __name__ == '__main__':
    graph = Graph('./graphs/bvg.input.json')
