from manim import *
from graph.graph_structures import Graph
from graph.parse_utils import parse_graph_from_json
from graph.graph_anim_structures import create_anim_computation_node, create_arrow_to_connect_node


class CreateGraph(Scene):
    def setup(self):
        self.graph: Graph = parse_graph_from_json()

    def get_graph(self) -> Graph:
        return self.graph

    def construct(self):
        root = self.graph.get_root()

        root_node = create_anim_computation_node(root, forward=False)
        self.play(FadeIn(root_node), root_node.animate.shift(LEFT * 5))

        anchor = root_node

        for i, child in enumerate(root.get_children()):
            child_node = create_anim_computation_node(child, forward=False)
            child_node.next_to(
                anchor, direction=RIGHT, buff=(0.4))

            self.play(FadeIn(child_node), )
            self.add(child_node)

            arrow = create_arrow_to_connect_node(root, child_node)

            self.play(FadeIn(arrow))
            self.add(arrow)

            anchor = child_node
            self.wait(0.5)
