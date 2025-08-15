from graph.graph_structures import ForwardNode, BackwardNode
from manim import *
import numpy as np


def create_anim_computation_node(node: BackwardNode | ForwardNode, forward=False):
    return create_anim_node_from_forward_node(node) if forward else create_anim_node_from_backward_node(node)


def create_anim_node_from_backward_node(node: BackwardNode) -> VGroup:
    node_text = Text(node.get_name(), color=WHITE).scale(0.5)

    node_circle = Circle(color=BLUE, stroke_width=4, stroke_color=BLUE,
                         fill_color=BLUE, fill_opacity=0.8, radius=1.8)

    node_text.move_to(node_circle.get_center())
    node = VGroup(node_circle, node_text)

    return node


def create_anim_node_from_forward_node(node: ForwardNode) -> VGroup:
    node_text = Text(node.get_name(), color=WHITE).scale(0.8)

    node_circle = Circle(color=GREEN, stroke_width=4)
    node_circle.surround(node_text, buffer_factor=1.2)
    node = VGroup(node_circle, node_text)

    return node


def create_origin_anim_tensor_from_backward_node(node: BackwardNode) -> MobjectMatrix:
    origin: np.ndarray = node.get_origin().get_data()
    shape = Text("Shape=(" + ", ".join([str(d) for d in origin.shape]) + ")")

    if origin.ndim == 0:
        name = Text("Sca. Tensor")
    elif origin.ndim == 1:
        name = Text("Vec. Tensor")
    else:
        name = Text("Mat. Tensor")

    cell_content = VGroup(name, shape).arrange(
        DOWN, buff=0.5)

    matrix = MobjectMatrix([[cell_content]])

    return matrix


def create_arrow_to_connect_node(start_node, end_node, color=WHITE, stroke_width=0.8, tip_length=0.2, buff=1.5):
    arrow = Arrow(start=start_node.get_center(), end=end_node.get_center(
    ), color=color, stroke_width=stroke_width, tip_length=tip_length, buff=buff)

    return arrow
