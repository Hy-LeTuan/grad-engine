import numpy as np
from collections import deque


class TensorRepr:
    def __init__(self, data, shape, offset):
        self.shape = shape
        self.offset = offset
        self.data = TensorRepr.create_array_from_raw(data, shape)

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
