from manim import *
from graph.graph_raw_representations import Graph
from graph.parse_utils import parse_graph_from_json
from graph.graph_anim_structures import create_anim_computation_node, create_arrow_to_connect_node, position_layer_list_horizontal, build_anim_backward_graph
from graph.graph_control_structure import NodeLayer, TensorLayer, EdgeBetweenLayerMembers


class CreateGraph(Scene):
    def setup(self):
        self.graph: Graph = parse_graph_from_json()

    def get_graph(self) -> Graph:
        return self.graph

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
