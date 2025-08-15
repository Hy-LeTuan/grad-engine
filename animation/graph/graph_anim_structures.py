from graph.graph_raw_representations import ForwardNode, BackwardNode
from graph.graph_control_structure import NodeLayer, TensorLayer, EdgeBetweenLayerMembers

from manim import *
import numpy as np
import re


def create_anim_computation_node(node: BackwardNode | ForwardNode, forward=False):
    return create_anim_node_from_forward_node(node) if forward else create_anim_node_from_backward_node(node)


def create_anim_node_from_backward_node(node: BackwardNode) -> VGroup:
    if node.get_name() == "GradAccum":
        node_text = Text("Accum", color=WHITE).scale(0.8)
    else:
        split_string = re.findall('[A-Z][a-z]*', node.get_name())

        start_text = Text(split_string[0], color=WHITE).scale(0.8)

        end_text = Text(split_string[1], color=WHITE).scale(0.8)

        node_text = VGroup(start_text, end_text).arrange(DOWN, buff=0.2)

    node_circle = Circle(color=BLUE, stroke_width=4, stroke_color=BLUE,
                         fill_color=BLUE, fill_opacity=0.8, radius=1.8)

    node_text.move_to(node_circle.get_center())
    new_node = VGroup(node_circle, node_text)

    return new_node


def create_anim_node_from_forward_node(node: ForwardNode) -> VGroup:
    node_text = Text(node.get_name(), color=WHITE).scale(0.8)

    node_circle = Circle(color=GREEN, stroke_width=4)
    node_circle.surround(node_text, buffer_factor=1.2)
    node = VGroup(node_circle, node_text)

    return node


def create_origin_anim_tensor_from_backward_node(node: BackwardNode) -> MobjectMatrix:
    origin: np.ndarray = node.get_origin().get_data()
    shape = Text("(" + ", ".join([str(d) for d in origin.shape]) + ")")

    if origin.ndim == 0:
        name = Text("Sca")
    elif origin.ndim == 1:
        name = Text("Vec")
    else:
        name = Text("Mat")

    cell_content = VGroup(name, shape).arrange(
        DOWN, buff=0.5)

    matrix = MobjectMatrix([[cell_content]])

    return matrix


def create_origin_anim_tensor_from_forward_node(node: ForwardNode) -> MobjectMatrix:
    return create_origin_anim_tensor_from_backward_node(node.get_backward())


def create_arrow_to_connect_node(start_node, end_node, color=WHITE, stroke_width=0.8, tip_length=0.2, buff=1.5):
    arrow = Arrow(start=start_node.get_center(), end=end_node.get_center(
    ), color=color, stroke_width=stroke_width, tip_length=tip_length, buff=buff)

    return arrow


def build_anim_backward_graph(starting_nodes, parent_node_layer, layer_list: list):
    tensor_layer = TensorLayer(
        convert=create_origin_anim_tensor_from_backward_node)

    if tensor_layer not in layer_list:
        layer_list.append(tensor_layer)

    if parent_node_layer not in layer_list:
        layer_list.append(parent_node_layer)

    next_node_layer = NodeLayer(
        convert=create_anim_computation_node, forward=False)

    next_starting_nodes = []

    for node in starting_nodes:
        (parent_anim_node_id, parent_anim_node) = parent_node_layer.safe_append_and_return(
            node, with_id=True)

        (tensor_anim_id, tensor_anim) = tensor_layer.safe_append_and_return(
            node, with_id=True)

        parent_node_layer.append_edge(EdgeBetweenLayerMembers(
            tensor_layer, tensor_anim_id, parent_node_layer, parent_anim_node_id))

        if len(node.get_children()) > 0:

            for child in node.get_children():
                (child_anim_node_id, child_anim_node) = next_node_layer.safe_append_and_return(
                    child, with_id=True)

                tensor_layer.append_edge(EdgeBetweenLayerMembers(
                    tensor_layer, tensor_anim_id, next_node_layer, child_anim_node_id))

                next_starting_nodes.append(child)

    if len(next_starting_nodes) > 0:
        build_anim_backward_graph(
            next_starting_nodes, next_node_layer, layer_list)


def position_layer_list_horizontal(layer_list: list[NodeLayer | TensorLayer], total_width: float, spacing: float):
    max_obj_width = 0

    max_obj_width = 0

    for layer in layer_list:
        if not layer.get_is_setup():
            layer.setup()
        max_obj_width = max(max_obj_width, layer.get_display_group().width)

    slot_width = spacing

    if max_obj_width > slot_width:
        scale_factor = slot_width / max_obj_width
        for layer in layer_list:
            layer.get_display_group().scale(scale_factor)
