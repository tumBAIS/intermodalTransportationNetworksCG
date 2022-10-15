import xml.etree.ElementTree as ET


# read an xml file and return the data
def read_xml(path):
    mytree = ET.parse(path)
    return mytree


# return a list of nodes of an xml file
def get_nodes(xml_tree):
    root = xml_tree.getroot()
    nodes = []

    # The first child contains all the nodes
    for node in root[0]:
        nodes.append(node.attrib['id'])

    return nodes


# return a list of all the endnodes of given edges
def get_nodes_by_edges(xml_tree, edges={}):
    nodes = []

    for edge_id in edges.keys():
        new_node = edges[edge_id][0]
        if new_node not in nodes:
            nodes.append(new_node)

        new_node = edges[edge_id][1]
        if new_node not in nodes:
            nodes.append(new_node)
    return nodes


def get_coordinates(xml_tree, nodes):
    root = xml_tree.getroot()
    coordinates = {}
    assert all('pt' in n for n in nodes)
    nodes_copy = [n for n in nodes]
    for element in root[0]:
        if 'pt' not in element.attrib['id']:
            continue
        elif element.attrib['id'] in nodes_copy:
            coordinates[element.attrib['id']] = (float(element.attrib['x']), float(element.attrib['y']))
            nodes_copy.remove(element.attrib['id'])
    return coordinates


# return a list of edges of an xml file
def get_edges(xml_tree):
    root = xml_tree.getroot()
    edges = {}

    # the second child contains all the edges
    for link in root[1]:
        edges[link.attrib['id']] = [link.attrib['from'], link.attrib['to']]
    return edges


# return a dict with edges as keys and capacity as values
def get_edge_capacity(xml_tree, edges={}):
    root = xml_tree.getroot()
    capacity = {}

    # the second child contains all the edges
    for link in root[1]:
        if link.attrib['id'] in edges.keys():
            capacity[link.attrib['id']] = float(link.attrib['capacity'])

    return capacity


# return a dict with edges as keys and calculated costs as values
def get_edge_cost(xml_tree, edges={}):
    root = xml_tree.getroot()
    cost = {}

    # the second child contains all the edges
    for link in root[1]:
        if link.attrib['id'] in edges.keys():
            # The cost is currently the length divided by 80% of the maxspeed
            cost[link.attrib['id']] = round(float(link.attrib['length'])/(0.8*float(link.attrib['freespeed'])), 2)

    return cost


def get_edges_by_mode(xml_tree, modes = []):
    root = xml_tree.getroot()
    edges = {}

    # the second child contains all the edges
    for link in root[1]:
        if any(mode in link.attrib['modes'] for mode in modes) and link.attrib['from'] != link.attrib['to']:
            edges[link.attrib['id']] = [link.attrib['from'], link.attrib['to']]

    return edges


def get_edges_by_mode_city(xml_tree, city, modes = []):
    root = xml_tree.getroot()
    edges = {}

    # the second child contains all the edges
    for link in root[1]:
        if any(mode in link.attrib['modes'] for mode in modes) and link.attrib['from'] != link.attrib['to']:
            edges[link.attrib['id']] = [link.attrib['from'], link.attrib['to']]

    return edges