from graph.graph_structures import ForwardNode, BackwardNode
from manim import *


def create_anim_computation_node(node: BackwardNode | ForwardNode, forward=False):
    return create_anim_node_from_forward_node(node) if forward else create_anim_node_from_backward_node(node)


def create_anim_node_from_backward_node(node: BackwardNode):
    node_text = Text(node.get_name(), color=WHITE).scale(0.5)

    node_circle = Circle(color=BLUE, stroke_width=4, stroke_color=BLUE,
                         fill_color=BLUE, fill_opacity=0.8, radius=1.8)

    node_text.move_to(node_circle.get_center())
    node = VGroup(node_circle, node_text)

    return node


def create_anim_node_from_forward_node(node: ForwardNode):
    node_text = Text(node.get_name(), color=WHITE).scale(0.8)

    node_circle = Circle(color=GREEN, stroke_width=4)
    node_circle.surround(node_text, buffer_factor=1.2)
    node = VGroup(node_circle, node_text)

    return node


def create_arrow_to_connect_node(start_node, end_node, color=WHITE, stroke_width=0.8, tip_length=0.2, buff=1.5):
    arrow = Arrow(start=start_node.get_center(), end=end_node.get_center(
    ), color=color, stroke_width=stroke_width, tip_length=tip_length, buff=buff)

    return arrow
