from manim import *
from graph.graph_raw_representations import Graph
from graph.parse_utils import parse_graph_from_json
from graph.graph_anim_structures import create_anim_computation_node, create_arrow_to_connect_node, position_layer_list_horizontal, create_arrow_to_connect_node, create_anim_tensor, build_anim_backward_graph
from graph.graph_control_structure import NodeLayer, TensorLayer, EdgeBetweenLayerMembers


class CreateGraph(Scene):
    def setup(self):
        self.graph: Graph = parse_graph_from_json()

    def get_graph(self) -> Graph:
        return self.graph

    def animate_backward_graph(self):
        root = [self.get_graph().get_root()]
        layer_list: list[NodeLayer | TensorLayer] = []
        parent_tensor_layer = TensorLayer(
            convert=create_anim_tensor, forward=False, tensor_type="gradient")

        # create layers and populate layer_list
        build_anim_backward_graph(
            root, parent_tensor_layer, layer_list)

        # position all layers
        total_width = 0.9 * config.frame_width
        spacing = total_width / (len(layer_list) - 1)
        position_layer_list_horizontal(
            layer_list=layer_list, total_width=total_width, spacing=spacing)

        for i, layer in enumerate(layer_list):
            print("layer: ", layer)
            layer.get_display_group().move_to(
                LEFT * (total_width / 2) + RIGHT * (i * spacing)
            )
            layer.display(self)

        self.animate_arrow(layer_list)

    def animate_arrow(self, layer_list: list[NodeLayer | TensorLayer]):
        for layer in layer_list:
            edge_list = layer.get_edges()

            for edge in edge_list:
                origin = edge.get_origin()
                destination = edge.get_destination()

                origin_mem = origin.get_conneted_mem()
                destination_mem = destination.get_conneted_mem()

                arrow = create_arrow_to_connect_node(
                    origin_mem, destination_mem)

                self.add(arrow)
                self.play(FadeIn(arrow))

    def construct(self):
        self.animate_backward_graph()
