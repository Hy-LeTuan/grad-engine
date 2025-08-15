from manim import *
from graph.graph_raw_representations import Graph
from graph.parse_utils import parse_graph_from_json
from graph.graph_anim_structures import create_anim_computation_node, create_arrow_to_connect_node, create_origin_anim_tensor_from_forward_node, create_origin_anim_tensor_from_backward_node, build_anim_backward_graph, position_layer_list_horizontal
from graph.graph_control_structure import NodeLayer, TensorLayer, EdgeBetweenLayerMembers


def build_anim_forward_graph(starting_nodes, parent_node_layer, layer_list: list):
    tensor_layer = TensorLayer(
        convert=create_origin_anim_tensor_from_forward_node)

    layer_list.append(parent_node_layer)
    layer_list.append(tensor_layer)

    for node in starting_nodes:
        (parent_anim_node_id, parent_anim_node) = parent_node_layer.safe_append_and_return(
            node, with_id=True)

        (tensor_anim_id, tensor_anim) = tensor_layer.safe_append_and_return(
            node, with_id=True)

        parent_node_layer.append_edge(EdgeBetweenLayerMembers(
            parent_node_layer, parent_anim_node_id, tensor_layer, tensor_anim_id))

        # parent node -> tensor (result of that node) -> compute function that the tensor contributes to

        if len(node.get_children()) > 0:
            next_node_layer = NodeLayer(
                convert=create_anim_computation_node, forward=True)

            for child in node.get_children():
                (child_anim_node_id, child_anim_node) = next_node_layer.safe_append_and_return(
                    child, with_id=True)

                tensor_layer.append_edge(EdgeBetweenLayerMembers(
                    tensor_layer, tensor_anim_id, next_node_layer, child_anim_node_id))

            build_anim_forward_graph(
                node.get_children(), next_node_layer, layer_list)


class CreateGraph(Scene):
    def setup(self):
        self.graph: Graph = parse_graph_from_json()

    def get_graph(self) -> Graph:
        return self.graph

    def animate_forward_graph(self):
        starting_nodes = self.graph.get_forward_nodes()
        layer_list: list[NodeLayer] = []
        parent_node_layer = NodeLayer(
            convert=create_anim_computation_node, forward=True)

        build_anim_forward_graph(
            starting_nodes, parent_node_layer, layer_list)

        total_width = 1.0 * config.frame_width  # 90% of screen width
        spacing = (total_width / (len(layer_list) - 1)) * 1.05

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

        for i, layer in enumerate(layer_list):

            layer.get_display_group().move_to(
                LEFT * (total_width / 2) + RIGHT * (i * spacing)
            )

            layer.display(self)

    def animate_backward_graph(self):
        root = [self.get_graph().get_root()]
        layer_list: list[NodeLayer | TensorLayer] = []
        parent_node_layer = NodeLayer(
            convert=create_anim_computation_node, forward=False)

        # create layers and populate layer_list
        build_anim_backward_graph(
            root, parent_node_layer, layer_list)

        # position all layers
        total_width = 0.9 * config.frame_width
        spacing = total_width / (len(layer_list) - 1)
        position_layer_list_horizontal(
            layer_list=layer_list, total_width=total_width, spacing=spacing)

        for i, layer in enumerate(layer_list):
            layer.get_display_group().move_to(
                LEFT * (total_width / 2) + RIGHT * (i * spacing)
            )
            layer.display(self)

    def construct(self):
        self.animate_backward_graph()
