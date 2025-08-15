from graph.graph_anim_structures import ForwardNode, BackwardNode


class DirectionInfo:
    def __init__(layer, id):
        self.layer = layer
        self.id = id


class EdgeBetweenLayerMembers:
    def __init__(self, layer1, id1, layer2, id2):
        self.origin = DirectionInfo(layer1, id1)
        self.destination = DirectionInfo(layer2, id2)

    def get_origin(self) -> DirectionInfo:
        return self.origin

    def get_destination(self) -> DirectionInfo:
        return self.destination


class NodeLayer:
    def __init__(self, convert=lambda x: x, **convert_args):
        # the animation node
        self.members = []

        # the map between raw and animation
        self.map_raw_to_member_index = {}

        # convert from raw to animation node
        self.convert = convert

        self.edges: list[EdgeBetweenLayerMembers] = []

        self.convert_args = convert_args

    def get_edges(self):
        return self.edges

    def get_mem_id(self, raw_mem: ForwardNode | BackwardNode) -> int:
        return self.map_raw_to_member_index[raw_mem]

    def get_mem(self, raw_mem: ForwardNode | BackwardNode):
        return self.members[self.get_mem_id(raw_mem)]

    def get_members(self):
        return self.members

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

    def safe_append_and_return(self, new_raw_mem: ForwardNode | BackwardNode):
        self.safe_append_mem(new_raw_mem)
        return self.get_mem(new_raw_mem)
