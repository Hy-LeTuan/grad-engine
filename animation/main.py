from manim import *
from graph.acyclic_graph_utils import rank_nodes_in_acyclic_graph, scale_nodes_horizontally, create_rank_first_anim_nod_map, scale_vgroup
from graph.graph_anim_structures import create_arrow_to_connect_node, position_layer_list_horizontal, create_anim_tensor, build_anim_backward_graph
from graph.graph_control_structure import NodeLayer, TensorLayer
from graph.graph_raw_representations import Graph, AcyclicGraph
from graph.parse_utils import parse_graph_from_json, create_acyclic_graph
from utils.anim_utils import pulse


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

    def display_caption_for_node(self, destination_node, text="Gradient Accumulation", fill_color="#A7C7E7", direction=DOWN):
        caption = Tex(text, color="#333333", font_size=20).next_to(
            destination_node, direction, buff=0.3)

        # Add a semi-transparent background rectangle with rounded corners
        background = RoundedRectangle(
            width=caption.width + 0.4,
            height=caption.height + 0.2,
            fill_color=fill_color,
            fill_opacity=0.2,
            stroke_width=0.5,
            stroke_color="#BBBBBB",
            corner_radius=0.1
        ).move_to(caption)

        # Group caption and background for cohesive animation
        caption_group = VGroup(background, caption)

        self.add(caption_group)
        self.play(FadeIn(background, run_time=0.5),
                  Write(caption, run_time=1.2))
        self.wait(0.25)
        self.play(FadeOut(caption_group, run_time=0.8))

    def display_gradient_change(self, edge, arrow):
        # display matrix movement
        origin_gradient = self.acyclic_graph.query_anim_tensor(self.acyclic_graph.query_node(
            edge[0]).get_gradient()).copy().scale(0.5)

        destination_gradient = self.acyclic_graph.query_anim_tensor(self.acyclic_graph.query_node(
            edge[1]).get_gradient()).copy().scale(0.5)

        # add temporary glow effect to tensor
        glowing_tensor = origin_gradient.set_color(ORANGE)
        self.add(glowing_tensor)

        # animate tensor moving from origin to destination along arrow with glow
        self.play(
            MoveAlongPath(
                destination_gradient,
                arrow,
                run_time=1.5,  # slightly longer for smoother effect
                rate_func=smooth  # smoother rate function
            ),
            MoveAlongPath(
                glowing_tensor,
                arrow,
                run_time=1.5,
                rate_func=smooth
            ),
            FadeIn(glowing_tensor, run_time=0.5),  # fade in glow at start
        )

        # fade out both tensor and glow
        self.play(
            FadeOut(destination_gradient, run_time=0.5),
            FadeOut(glowing_tensor, run_time=0.5)
        )

    def animate_connection(self):
        for (edge_nr, edge) in enumerate(self.acyclic_graph.get_edges()):
            # get animation node
            origin_node = self.acyclic_graph.query_anim_node(edge[0])
            destination_node = self.acyclic_graph.query_anim_node(edge[1])

            # for root node, starting gradient is required
            if edge_nr == 0:
                # loss text
                loss_text = MathTex(
                    r"\frac{\partial L}{\partial L} =", color="#333333").scale(0.6)
                self.add(loss_text)

                # starting gradient
                backward_start_gradient = self.acyclic_graph.query_anim_tensor(self.acyclic_graph.query_node(
                    edge[0]).get_gradient()).copy().scale(0.5).set_color("#FFC107")
                self.add(backward_start_gradient)

                # group starting gradient
                starting_gradient_group = VGroup(
                    loss_text, backward_start_gradient).arrange(RIGHT, buff=0.15).next_to(origin_node, UP, buff=0.5)

                # create arrow to connect starting gradient to root
                arrow = create_arrow_to_connect_node(
                    start_node=starting_gradient_group, end_node=origin_node, buff=0, mid=True)

                # fade in starting gradient
                self.play(FadeIn(starting_gradient_group, run_time=0.5))

                self.wait(0.25)

                # move gradient to
                self.play(
                    MoveAlongPath(
                        starting_gradient_group,
                        arrow,
                        run_time=1.5,  # slightly longer for smoother effect
                        rate_func=smooth  # smoother rate function
                    ),
                )

                self.wait(0.25)

                # caption for root node to denote starting gradient received
                self.display_caption_for_node(
                    destination_node=origin_node, text="Loss Gradient")

                self.wait(0.25)

                self.play(
                    FadeOut(starting_gradient_group, run_time=0.5),
                )

                self.wait(0.25)

            # arrow and flash to create directional sense
            arrow = create_arrow_to_connect_node(
                start_node=origin_node, end_node=destination_node, buff=0)
            self.add(arrow)
            self.play(Create(arrow))
            self.play(
                ShowPassingFlash(
                    arrow.copy().set_stroke(color="#FFB347", width=6),
                    run_time=1.0,
                    time_width=0.6
                )
            )

            self.wait(0.25)

            # display caption
            if "Accum" in self.acyclic_graph.query_node(edge[1]).get_name():
                self.play(pulse(destination_node,
                                flash_color=ORANGE, base_color=BLUE))

                if self.node_rank[edge[1]] == self.max_rank and self.rank_first_anim_node_map[self.node_rank[edge[1]]][-1][1] == destination_node:
                    self.display_caption_for_node(
                        destination_node=destination_node, text="Accumulate Grad", fill_color=GREEN, direction=UP)
                else:
                    self.display_caption_for_node(
                        destination_node=destination_node, text="Accumulate Grad", fill_color=GREEN)
            else:
                origin_ops_name = self.acyclic_graph.query_node(
                    edge[0]).get_ops_name()
                destination_ops_name = self.acyclic_graph.query_node(
                    edge[1]).get_ops_name()

                compute_caption = f"Derive grad from {origin_ops_name}"
                receive_caption = f"Grad received for {destination_ops_name}"

                self.display_caption_for_node(
                    destination_node=origin_node, text=compute_caption, fill_color=BLUE)

                self.display_caption_for_node(
                    destination_node=destination_node, text=receive_caption, fill_color=BLUE)

            self.wait(0.25)

            # display gradient change
            self.display_gradient_change(edge, arrow)

            self.wait(0.25)

    def backward_graph_construct(self):
        (max_rank, node_rank) = rank_nodes_in_acyclic_graph(self.acyclic_graph)
        self.acyclic_graph.sort_edge(node_rank)
        self.node_rank = node_rank
        self.max_rank = max_rank

        # rank base representation
        rank_first_anim_node_map = create_rank_first_anim_nod_map(
            node_rank, acylic_graph=self.acyclic_graph)

        self.rank_first_anim_node_map = rank_first_anim_node_map

        # scale
        max_height = config.frame_height * 0.95
        total_width = 0.85 * config.frame_width
        spacing = total_width / (max_rank)

        scale_nodes_horizontally(
            total_width, max_rank, self.acyclic_graph.get_anim_node_map())

        for (rank, node_list) in rank_first_anim_node_map.items():
            layer_position = LEFT * (total_width / 2) + \
                (RIGHT * rank * spacing)

            anim_node_list = list(map(lambda x: x[1], node_list))

            self.add(*anim_node_list)

            # create vgroup
            layer = VGroup(*anim_node_list).arrange(DOWN,
                                                    buff=1.5).move_to(ORIGIN).move_to(layer_position)

            scale_vgroup(num_elem=len(anim_node_list),
                         max_height=max_height, layer=layer)

            self.play(FadeIn(layer, lag_ratio=2))

            self.wait(0.5)

    def construct(self):
        self.camera.background_color = WHITE
        introduction = Text("Backward Computation Graph",
                            color=BLACK).move_to(ORIGIN)
        self.play(FadeIn(introduction))
        self.play(FadeOut(introduction))
        self.wait(1.5)

        self.backward_graph_construct()
        self.wait(1)
        self.animate_connection()

        self.wait(1)
        self.clear()
        outtro = Text(
            "All grads propagated to leaf tensors", color=BLACK).move_to(ORIGIN)

        self.play(FadeIn(outtro))
        self.wait(1)
        self.play(FadeOut(outtro))
