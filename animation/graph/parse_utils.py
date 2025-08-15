import json
import os

from graph.graph_raw_representations import Graph


def parse_graph_from_json() -> Graph:
    try:
        with open(os.path.join("..", "output", "graph.json"), 'r') as f:
            graph_content = json.load(f)
            return Graph(**graph_content)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    graph = parse_graph_from_json()

    root = graph.get_root()
    forward_nodes = graph.get_forward_nodes()

    print(root)
    print(forward_nodes)
