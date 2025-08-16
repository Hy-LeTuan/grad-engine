from graph.graph_raw_representations import ForwardNode, BackwardNode, Node, TensorRepr
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


def create_anim_tensor(node: ForwardNode | BackwardNode, forward=False, tensor_type="gradient") -> MobjectMatrix:
    if tensor_type == "gradient":
        tensor: np.ndarray = node.get_backward().get_gradient(
        ).get_data() if forward else node.get_gradient().get_data()
    else:
        tensor: np.ndarray = node.get_backward().get_origin(
        ).get_data() if forward else node.get_origin().get_data()

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


def create_arrow_to_connect_node(start_node, end_node, color=WHITE, stroke_width=0.8, tip_length=0.2, buff=1.5):
    arrow = Arrow(start=start_node.get_center(), end=end_node.get_center(
    ), color=color, stroke_width=stroke_width, tip_length=tip_length, buff=buff)

    return arrow


def build_anim_backward_graph(starting_nodes, parent_tensor_layer, layer_list: list):
    node_layer = NodeLayer(
        convert=create_anim_computation_node, forward=False)

    if parent_tensor_layer not in layer_list:
        layer_list.append(parent_tensor_layer)

    if node_layer not in layer_list:
        layer_list.append(node_layer)

    next_tensor_layer = TensorLayer(
        convert=create_anim_tensor, forward=False, tensor_type="gradient")

    next_starting_nodes = []

    for node in starting_nodes:
        (tensor_anim_id, tensor_anim) = parent_tensor_layer.safe_append_and_return(
            node, with_id=True)

        (node_anim_id, node_anim) = node_layer.safe_append_and_return(
            node, with_id=True)

        # upstream gradient -> backward node
        parent_tensor_layer.append_edge(EdgeBetweenLayerMembers(
            parent_tensor_layer, tensor_anim_id, node_layer, node_anim_id))

        if len(node.get_children()) > 0:
            for child in node.get_children():
                (child_tensor_id, child_tensor_anim) = next_tensor_layer.safe_append_and_return(
                    child, with_id=True)

                parent_tensor_layer.append_edge(EdgeBetweenLayerMembers(
                    node_layer, node_anim_id, next_tensor_layer, child_tensor_id))

                next_starting_nodes.append(child)

    if len(next_starting_nodes) > 0:
        build_anim_backward_graph(
            next_starting_nodes, next_tensor_layer, layer_list)


def position_layer_list_horizontal(layer_list: list[NodeLayer | TensorLayer], total_width: float, spacing: float):
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
