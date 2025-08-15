from manim import *
from graph.graph_structures import Graph
from graph.parse_utils import parse_graph_from_json
from graph.graph_anim_structures import create_anim_computation_node, create_arrow_to_connect_node, create_origin_anim_tensor_from_backward_node
from graph.graph_control_structure import NodeLayer


class CreateGraph(Scene):
    def setup(self):
        self.graph: Graph = parse_graph_from_json()

    def get_graph(self) -> Graph:
        return self.graph

    def animate_forward_graph(self):
        starting_nodes = self.graph.get_forward_nodes()

        for node in starting_nodes:
            tensor = create_origin_anim_tensor_from_backward_node(
                node.get_backward())

            print(tensor.get_center())
            self.play(FadeIn(tensor))

            node_layer = NodeLayer(
                convert=create_anim_computation_node, forward=True)

            for child in node.get_children():
                child_anim_node = node_layer.safe_append_and_return(child)

                self.play(FadeIn(child_anim_node))
                self.add(child_anim_node)

                arrow = create_arrow_to_connect_node(
                    tensor, child_anim_node)

                self.play(FadeIn(arrow))
                self.add(arrow)

            break

        print(next_layer_nodes)
        print(next_layer_anim_nodes)

    def animate_backward_graph(self):
        root = self.graph.get_root()

        root_node = create_anim_computation_node(root, forward=False)
        self.play(FadeIn(root_node), root_node.animate.shift(LEFT * 5))

        anchor = root_node

        for i, child in enumerate(root.get_children()):
            child_anim_node = create_anim_computation_node(
                child, forward=False)
            child_anim_node.next_to(
                anchor, direction=RIGHT, buff=(0.4))

            self.play(FadeIn(child_anim_node), )
            self.add(child_anim_node)

            arrow = create_arrow_to_connect_node(root_node, child_anim_node)

            self.play(FadeIn(arrow))
            self.add(arrow)

            anchor = child_anim_node

    def construct(self):
        self.animate_forward_graph()
