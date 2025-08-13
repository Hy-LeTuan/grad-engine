import numpy as np
from collections import deque


class Node:
    def __init__(self, name: str, origin, gradient, children, preset: bool = False):
        self.name = name

        if preset:
            self.origin = origin
            self.gradient = gradient
            self.children = children
        else:
            self.origin = TensorRepr(**origin)
            self.gradient = TensorRepr(**gradient)
            self.children = [Node(**child) for child in children]

    def __repr__(self):
        return f"Node(name={self.get_name()}, origin={self.get_origin()}, gradient={self.get_gradient()})"

    def get_name(self) -> str:
        return self.name

    def get_origin(self):
        return self.origin

    def get_gradient(self):
        return self.gradient

    def get_children(self):
        return self.children

    def get_children_len(self):
        return len(self.children)

    def clone(node):
        return Node(
            node.get_name(),
            node.get_origin(),
            node.get_gradient(),
            node.get_children(),
            preset=True
        )

    def clear_children(self):
        self.children = []

    def append_child(self, child: None):
        self.children.append(child)


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

    def get_data(self):
        return self.data

    def get_shape(self):
        return self.shape

    def get_offset(self):
        return self.offset


class Graph:
    def __init__(self, root: Node):
        self.root = Node(**root)

    def get_root(self) -> Node:
        return self.root

    def graph_reverse(self) -> [Node]:
        root = self.get_root()
        curr_clone = Node.clone(root)
        ending_list = []

        q = deque()
        q.append((root, None))

        while len(q) > 0:
            parent_parent_clone_pair = q.popleft()
            parent: Node = parent_parent_clone_pair[0]
            parent_clone: Node | None = parent_parent_clone_pair[1]

            if parent_clone is None:
                parent_clone = Node.clone(parent)

            if parent.get_children_len() > 0:
                for child in parent.get_children():
                    child_clone = Node.clone(child)
                    child_clone.clear_children()
                    child_clone.append_child(parent_clone)

                    q.append((child, child_clone))
            else:
                ending_list.append(parent_clone)

        return ending_list
