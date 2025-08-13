import json
import os

from graph_structure import Graph, Node


def parse_graph_from_json():
    try:
        with open(os.path.join("..", "output", "graph.json"), 'r') as f:
            graph_content = json.load(f)
            return graph_content
    except Exception as e:
        print(e)


if __name__ == "__main__":
    content = parse_graph_from_json()

    graph = Graph(**content)

    root = graph.get_root()
    reverse_graph = graph.graph_reverse()

    print(reverse_graph)
