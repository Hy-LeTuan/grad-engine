import numpy as np
from collections import deque
from manim import *
import re


class TensorRepr:
    def __init__(self, data=[], shape=(), offset=0, id=""):
        self.shape = shape
        self.offset = offset
        self.data = TensorRepr.create_array_from_raw(data, shape)
        self.id = id

    def __repr__(self):
        return f"Tensor(data={self.get_data()}, shape={self.get_shape()}, offset={self.get_offset()})"

    def create_array_from_raw(data, shape):
        data_np = np.array(data)
        data_np_reshape = np.reshape(data_np, shape)

        return data_np_reshape

    def get_data(self) -> np.ndarray:
        return self.data

    def get_shape(self):
        return self.shape

    def get_offset(self):
        return self.offset

    def get_id(self) -> str:
        return self.id

    def set_id(self, id: str):
        self.id = id


class BackwardNode:
    def __init__(self, name: str, origin, gradient, children, preset: bool = False):
        self.name = name

        if preset:
            self.origin = origin
            self.gradient = gradient
            self.children = children
        else:
            self.origin = TensorRepr(**origin)
            self.gradient = TensorRepr(**gradient)
            self.children = [BackwardNode(**child) for child in children]

    def __repr__(self):
        return f"BackwardNode(name={self.get_name()}, origin={self.get_origin()}, origin_gradient={self.get_gradient()})"

    def get_name(self) -> str:
        return self.name

    def get_origin(self) -> TensorRepr:
        return self.origin

    def get_gradient(self) -> TensorRepr:
        return self.gradient

    def get_children(self):
        return self.children

    def get_children_len(self):
        return len(self.children)

    def clone(node):
        return BackwardNode(
            node.get_name(),
            node.get_origin(),
            node.get_gradient(),
            node.get_children(),
            preset=True
        )

    def clear_children(self):
        self.children = []

    def append_child(self, child):
        self.children.append(child)


class ForwardNode:
    def __init__(self, backward_node: BackwardNode):
        self.backward_node = backward_node
        self.name = ForwardNode.format_name(backward_node.get_name())
        self.children = []

    def __repr__(self):
        return f"ForwardNode(name={self.get_name()}, num_contributes_to={len(self.get_children())})"

    def format_name(backward_node_name: str):
        if backward_node_name == "GradAccum":
            return "LeafCreation"
        else:
            backward_node_name_split = backward_node_name.split("Backward")
            return backward_node_name_split[0] + "Forward"

    def get_backward(self):
        return self.backward_node

    def get_children(self):
        return self.children

    def get_name(self):
        return self.name

    def clear_children(self):
        self.children = []

    def append_child(self, child):
        self.children.append(child)


class Node:
    def __init__(self, name="", origin="", gradient="", id=""):
        self.name = name
        self.origin = origin
        self.gradient = gradient
        self.id = id

    def __repr__(self):
        return f"Node(name={self.get_name()}, origin={self.origin}, gradient={self.gradient})"

    def get_name(self) -> str:
        return self.name

    def get_origin(self) -> str:
        return self.origin

    def get_gradient(self) -> str:
        return self.gradient

    def get_origin_tensor(self) -> TensorRepr:
        return self.origin_tensor

    def get_gradient_tensor(self) -> TensorRepr:
        return self.gradient_tensor

    def get_id(self) -> str:
        return self.id

    def set_origin(self, tensor: TensorRepr):
        self.origin_tensor = tensor

    def set_gradient(self, tensor: TensorRepr):
        self.gradient_tensor = tensor

    def set_id(self, id: str):
        self.id = id


class Graph:
    def __init__(self, root: BackwardNode):
        self.root = BackwardNode(**root)
        self.forward_nodes = self.graph_reverse()

    def get_root(self) -> BackwardNode:
        return self.root

    def get_forward_nodes(self) -> [ForwardNode]:
        return self.forward_nodes

    def graph_reverse(self) -> [BackwardNode]:
        root = self.get_root()
        ending_list = []

        q = deque()
        q.append((root, None))

        while len(q) > 0:
            parent_parent_clone_pair = q.popleft()
            parent: BackwardNode = parent_parent_clone_pair[0]
            parent_forward: ForwardNode | None = parent_parent_clone_pair[1]

            if parent_forward is None:
                parent_forward = ForwardNode(parent)

            if parent.get_children_len() > 0:
                for child in parent.get_children():
                    child_forward = ForwardNode(child)
                    child_forward.append_child(parent_forward)

                    q.append((child, child_forward))
            else:
                ending_list.append(parent_forward)

        return ending_list


def create_anim_tensor_from_tensor(tensor: TensorRepr) -> MobjectMatrix:
    tensor: np.ndarray = tensor.get_data()

    if tensor.ndim == 0:
        name = Text("Sca")
    elif tensor.ndim == 1:
        name = Text("Vec")
    else:
        name = Text("Mat")

    shape = Text("(" + ", ".join([str(d) for d in tensor.shape]) + ")")

    cell_content = VGroup(name, shape).arrange(
        DOWN, buff=0.5)

    matrix = MobjectMatrix([[cell_content]])

    return matrix


def create_anim_node_from_acyclic_node(node: Node) -> VGroup:
    accum = False
    if node.get_name() == "GradAccum":
        node_text = Text("GradAccum", color=WHITE).scale(0.7)
        accum = True
    else:
        split_string = re.findall('[A-Z][a-z]*', node.get_name())
        start_text = Text(split_string[0], color=WHITE).scale(0.7)
        end_text = Text(split_string[1], color=WHITE).scale(0.7)
        node_text = VGroup(start_text, end_text).arrange(DOWN, buff=0.15)

    node_circle = Circle(
        radius=1.6,
        color=PURPLE_E if accum else BLUE,
        stroke_width=2.5,
        stroke_color=ORANGE if accum else WHITE,
        fill_color=PURPLE_E if accum else BLUE,
        fill_opacity=1.0      # fully opaque, hides arrows behind
    )

    node_text.move_to(node_circle.get_center())

    new_node = VGroup(node_circle, node_text)
    return new_node


class AcyclicGraph:
    def __init__(self, tensor_map, node_map, edges):
        self.tensor_map = tensor_map
        self.node_map = node_map

        self.anim_tensor_map = self.create_anim_tensor_map(tensor_map)
        self.anim_node_map = self.create_anim_node_map(node_map)

        self.edges = edges
        self.reversed_edges = self.reverse_edge(edges)

    def __repr__(self):
        return f"AcyclicGraph(nodes={len(self.node_map)}, edges={len(self.edges)})"

    def reverse_edge(self, edges) -> list[(str, str)]:
        reversed_edges = []
        for edge in edges:
            reversed_edge = (edge[1], edge[0])

            reversed_edges.append(reversed_edge)

        return reversed_edges

    def create_anim_node_map(self, node_map: dict[str, Node]):
        anim_node_map = {}
        for (id, node) in node_map.items():
            anim_node = create_anim_node_from_acyclic_node(node)

            anim_node_map[id] = anim_node

        return anim_node_map

    def create_anim_tensor_map(self, tensor_map: dict[str, TensorRepr]):
        anim_tensor_map = {}

        for (id, tensor) in tensor_map.items():
            anim_tensor = create_anim_tensor_from_tensor(tensor)
            anim_tensor_map[id] = anim_tensor

        return anim_tensor_map

    def sort_edge(self, rank_map: dict[str, int]):
        self.edges = sorted(self.edges, key=lambda x: rank_map[x[0]])

    def get_tensor_map(self) -> dict[str, TensorRepr]:
        return self.tensor_map

    def get_node_map(self) -> dict[str, Node]:
        return self.node_map

    def get_edges(self) -> list[(str, str)]:
        return self.edges

    def get_anim_node_map(self) -> dict[str, VGroup]:
        return self.anim_node_map

    def get_anim_tensor_map(self) -> dict[str, MobjectMatrix]:
        return self.anim_tensor_map

    def query_tensor(self, tensor_id: str) -> TensorRepr:
        return self.tensor_map[tensor_id]

    def query_node(self, node_id: str) -> Node:
        return self.node_map[node_id]

    def query_anim_tensor(self, tensor_id: str) -> MobjectMatrix:
        return self.anim_tensor_map[tensor_id]

    def query_anim_node(self, node_id: str) -> VGroup:
        return self.anim_node_map[node_id]
