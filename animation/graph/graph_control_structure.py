from graph.graph_anim_structures import ForwardNode, BackwardNode
from manim import *


class DirectionInfo:
    def __init__(self, layer, id):
        self.layer = layer
        self.id = id

    def get_layer(self):
        return self.layer

    def get_id(self):
        return self.id

    def get_conneted_mem(self):
        return self.layer.get_mem(self.id)


class EdgeBetweenLayerMembers:
    def __init__(self, layer1, id1, layer2, id2):
        self.origin = DirectionInfo(layer1, id1)
        self.destination = DirectionInfo(layer2, id2)

    def get_origin(self) -> DirectionInfo:
        return self.origin

    def get_destination(self) -> DirectionInfo:
        return self.destination


class ControlLayer:
    def __init__(self, convert=lambda x: x, **convert_args):
        # the animation node
        self.members = []

        # the map between raw and animation
        self.map_raw_to_member_index = {}

        # convert from raw to animation node
        self.convert = convert

        self.edges: list[EdgeBetweenLayerMembers] = []

        self.convert_args = convert_args

        self.is_setup = False

        self.display_group = []

    def __repr__(self):
        return f"{self.get_name()}(members:{len(self.members)}, edges:{len(self.edges)})"

    def get_name(self):
        return "Layer"

    def get_edges(self) -> list[EdgeBetweenLayerMembers]:
        return self.edges

    def get_mem_id_from_raw(self, raw_mem: ForwardNode | BackwardNode) -> int:
        return self.map_raw_to_member_index[raw_mem]

    def get_mem(self, id: int):
        return self.members[id]

    def get_mem_id(self, mem) -> int:
        return self.members.index(mem)

    def get_mem_from_raw(self, raw_mem: ForwardNode | BackwardNode):
        return self.members[self.get_mem_id_from_raw(raw_mem)]

    def get_members(self):
        return self.members

    def get_is_setup(self) -> bool:
        return self.is_setup

    def get_display_group(self) -> VGroup:
        return self.display_group

    def set_edges(self, new_edges: iter):
        self.edges.extend(new_edges)

    def clear_edges(self):
        self.edges = []

    def append_edge(self, new_edge: EdgeBetweenLayerMembers):
        self.edges.append(new_edge)

    def safe_append_mem(self, new_raw_mem: ForwardNode | BackwardNode):
        if new_raw_mem in self.map_raw_to_member_index:
            return
        else:
            new_mem = self.convert(new_raw_mem, **self.convert_args)
            self.members.append(new_mem)
            self.map_raw_to_member_index[new_raw_mem] = len(self.members) - 1

    def safe_append_and_return(self, new_raw_mem: ForwardNode | BackwardNode, with_id=False):
        self.safe_append_mem(new_raw_mem)

        if with_id:
            return (self.map_raw_to_member_index[new_raw_mem], self.get_mem_from_raw(new_raw_mem))
        else:
            return self.get_mem_from_raw(new_raw_mem)

    def format_display(self, scene: Scene):
        scene.add(self.display_group)
        scene.play(FadeIn(self.display_group))

    def setup(self):
        self.display_group = VGroup(
            *self.members).arrange(DOWN, buff=1.5).move_to(ORIGIN)

        max_height = config.frame_height * 0.95

        # Only scale if there's more than one element and it's too tall
        if len(self.members) > 1 and self.display_group.height > max_height:
            self.display_group.scale(max_height / self.display_group.height)

        self.is_setup = True

    def display(self, scene: Scene):
        self.format_display(scene)


class NodeLayer(ControlLayer):
    def __init__(self, convert=lambda x: x, **convert_args):
        super().__init__(convert=convert, **convert_args)

    def get_name(self):
        return "NodeLayer"


class TensorLayer(ControlLayer):
    def __init__(self, convert=lambda x: x, **convert_args):
        super().__init__(convert=convert, **convert_args)

    def get_name(self):
        return "TensorLayer"
