import json
import re

def write_output(model, graph):
    
    data = {
        'nodes': [],
        'edges': [],
        'lines': [line for line in graph.lines]
        }
    
    for id, node in graph.nodes.items():
        data['nodes'].append({
            'id': node.id,
            'label': node.label,
            'metadata': {
                'x': model._x[id].x,
                'y': model._y[id].x
            }
        })
    
    for id, edge in graph.fwd_edges.items():
        data['edges'].append({
            'source': edge.source,
            'target': edge.target,
            'metadata': {
                'lines': edge.lines
            }
        })

    file_name = re.sub( 'input', 'output', graph.file_name)
    with open(graph.folder+file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
