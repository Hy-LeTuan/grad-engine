import json
import os

from graph.graph_raw_representations import Graph, TensorRepr, Node, AcyclicGraph


def parse_graph_from_json() -> Graph:
    try:
        with open(os.path.join("..", "output", "graph.json"), 'r') as f:
            graph_content = json.load(f)
            return Graph(**graph_content)
    except Exception as e:
        print(e)


def parse_tensors_for_acylic_graph() -> dict[str, TensorRepr]:
    root = os.path.join("..", "output", "tensors")
    tensor_map = {}

    for tensor_file in os.listdir(root):
        full_path = os.path.join(root, tensor_file)

        try:
            with open(full_path, 'r') as f:
                tensor_content = json.load(f)
                tensor = TensorRepr(**tensor_content)
                tensor_id = os.path.splitext(tensor_file)[0]

                tensor.set_id(tensor_id)
                tensor_map[tensor_id] = tensor
        except Exception as e:
            print(e)

    return tensor_map


def parse_nodes_for_acyclic_graph(tensor_map) -> dict[str, Node]:
    root = os.path.join("..", "output", "nodes")
    node_map = {}

    for tensor_file in os.listdir(root):
        full_path = os.path.join(root, tensor_file)

        try:
            with open(full_path, 'r') as f:
                node_content = json.load(f)
                origin = tensor_map[node_content["origin"]]
                gradient = tensor_map[node_content["gradient"]]

                node = Node(**node_content)
                node.set_gradient(gradient)
                node.set_origin(origin)

                node_id = os.path.splitext(tensor_file)[0]
                node.set_id(id)
                node_map[node_id] = node
        except Exception as e:
            print(e)

    return node_map


def parse_acyclic_graph():
    try:
        with open(os.path.join("..", "output", "graph_acyclic.json"), 'r') as f:
            graph_content = json.load(f)
            return graph_content
    except Exception as e:
        print(e)


def create_acyclic_graph() -> AcyclicGraph:
    tensor_map = parse_tensors_for_acylic_graph()
    node_map = parse_nodes_for_acyclic_graph(tensor_map)

    edges = parse_acyclic_graph()

    acyclic_graph = AcyclicGraph(
        tensor_map=tensor_map, node_map=node_map, edges=edges)

    return acyclic_graph


if __name__ == "__main__":
    acyclic_graph = create_acyclic_graph()
    print(acyclic_graph)
