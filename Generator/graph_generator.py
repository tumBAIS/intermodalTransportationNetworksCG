import networkx as nx


def build_graph(instance):
    '''
    Build networkx graph based on temporal/dynamic graph of the instance
    :return: networkx graph
    '''
    G = nx.DiGraph()

    for arc in instance.temp_network.arcs:
        G.add_edge(arc.start, arc.end, weight=arc.cost, cap=arc.capacity, arc_type=arc.arc_type)

    return G
