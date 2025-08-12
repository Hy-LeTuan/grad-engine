import numpy as np


class Node:
    def __init__(self, name: str, origin, gradient, children):
        self.name = name
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
