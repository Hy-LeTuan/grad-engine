from manim import *
from graph.graph_raw_representations import AcyclicGraph


def rank_nodes_in_acyclic_graph(acyclic_graph: AcyclicGraph) -> (dict[str, int], int):
    """
    Create and return the ranking of all animation nodes in a map. The map is based on the node id.
    """
    node_rank: dict[str, int] = {}
    max_rank = 0

    # initialize ranking map
    for anim_node_id in acyclic_graph.get_anim_node_map().keys():
        node_rank[anim_node_id] = 0

    for edge in acyclic_graph.get_edges():
        origin_node_id = edge[0]
        destination_node_ikd = edge[1]

        node_rank[destination_node_ikd] = max(
            node_rank[destination_node_ikd], node_rank[origin_node_id] + 1)

        max_rank = max(max_rank, node_rank[origin_node_id] + 1)

    node_rank = dict(sorted(node_rank.items(), key=lambda item: item[1]))

    return (max_rank, node_rank)


def scale_nodes_horizontally(total_width: float, max_rank: int, anim_node_map: dict[str, VGroup]):
    max_obj_width = 0

    for anim_node in anim_node_map.values():
        max_obj_width = max(max_obj_width, anim_node.width)

    slot_width = total_width / (max_rank + 1)

    if max_obj_width > slot_width:
        scale_factor = slot_width / max_obj_width
        for anim_node in anim_node_map.values():
            anim_node.scale(scale_factor)


def create_rank_first_anim_nod_map(node_rank: dict[str, int], acylic_graph: AcyclicGraph) -> dict[int, list[(str, VGroup)]]:
    position_node_list: dict[int, list[(str, VGroup)]] = {}

    for (anim_node_id, anim_node_ranking) in node_rank.items():
        if anim_node_ranking in position_node_list:
            node_anim = acylic_graph.query_anim_node(anim_node_id)
            position_node_list[anim_node_ranking].append(
                (anim_node_id, node_anim))
        else:
            node_anim = acylic_graph.query_anim_node(anim_node_id)
            position_node_list[anim_node_ranking] = [(anim_node_id, node_anim)]

    return position_node_list


def scale_vgroup(num_elem: int, max_height: float, layer: VGroup):
    # Only scale if there's more than one element and it's too tall
    if num_elem > 1 and layer.height > max_height:
        layer.scale(
            max_height / layer.height)
