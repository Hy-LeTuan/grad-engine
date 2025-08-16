from manim import *
from graph.graph_raw_representations import Graph, AcyclicGraph
from graph.parse_utils import parse_graph_from_json, create_acyclic_graph
from graph.graph_anim_structures import create_anim_computation_node, create_arrow_to_connect_node, position_layer_list_horizontal, create_arrow_to_connect_node, create_anim_tensor, build_anim_backward_graph
from graph.graph_control_structure import NodeLayer, TensorLayer, EdgeBetweenLayerMembers

from graph.acyclic_graph_utils import rank_nodes_in_acyclic_graph, scale_nodes_horizontally, create_rank_first_anim_nod_map, scale_vgroup


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


class CreateAcyclicGraph(Scene):
    def setup(self):
        self.acyclic_graph: AcyclicGraph = create_acyclic_graph()

    def draw_arrows(self):
        for edge in self.acyclic_graph.get_edges():
            origin_node = self.acyclic_graph.query_anim_node(edge[0])
            destination_node = self.acyclic_graph.query_anim_node(edge[1])

            arrow = create_arrow_to_connect_node(
                start_node=origin_node, end_node=destination_node, buff=0)

            self.add(arrow)
            self.play(FadeIn(arrow))

    def construct(self):
        (max_rank, node_rank) = rank_nodes_in_acyclic_graph(self.acyclic_graph)

        # rank base representation
        rank_first_anim_node_map = create_rank_first_anim_nod_map(
            node_rank, acylic_graph=self.acyclic_graph)

        # scale
        max_height = config.frame_height * 0.95
        total_width = 0.85 * config.frame_width
        spacing = total_width / (max_rank)

        scale_nodes_horizontally(
            total_width, max_rank, self.acyclic_graph.get_anim_node_map())

        for (rank, node_list) in rank_first_anim_node_map.items():
            layer_position = LEFT * (total_width / 2) + \
                RIGHT * rank * spacing

            anim_node_list = list(map(lambda x: x[1], node_list))

            self.add(*anim_node_list)

            # create vgroup
            layer = VGroup(*anim_node_list).arrange(DOWN,
                                                    buff=1.5).move_to(ORIGIN).move_to(layer_position)

            scale_vgroup(num_elem=len(anim_node_list),
                         max_height=max_height, layer=layer)

            self.play(FadeIn(layer))

        # draw arrows
        self.draw_arrows()
