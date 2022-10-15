import Generator.CSV_parser as csv
import Generator.XML_parser as xml
import Generator.geodata as gd
import pandas as pd
import numpy as np
import math
import networkx as nx


def shortest_paths(G, n):
    return n, nx.single_source_dijkstra(G, n, cutoff=90, weight='weight')


def distance_squared(node1, node2):
    return (node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2


class TimeWindow:
    '''
    Is this even needed anymore?
    '''
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return hash(str(self))

#
# class Route:
#     '''
#     Is this even needed anymore?
#     '''
#     def __init__(self, route_id, start_time):
#         self.id = route_id
#         self.start_time = start_time
#
#     def __eq__(self, other):
#         return self.__dict__ == other.__dict__
#
#     def __hash__(self):
#         return hash(str(self))


class FlatNode:
    '''
    A FlatNode is used for the construction of the static graph of the transportation system
    '''
    def __init__(self, stop, mode, coordinate):
        self.stop = stop
        self.mode = mode
        self.coordinate = coordinate

    def __str__(self):
        return f'{self.stop}, {self.mode}, {self.coordinate}'

    def __eq__(self, other):
        return self.stop == other.stop and self.mode == other.mode

    def __hash__(self):
        return hash((self.stop, self.mode))


class FlatArc:
    '''
    A FlatArc is used for the construction of the static graph of the transportation system
    '''
    def __init__(self, start: FlatNode, end: FlatNode):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start.stop == other.start.stop and self.end.stop == other.end.stop

    def __hash__(self):
        return hash((self.start.stop, self.end.stop, self.start.mode, self.end.mode))


class FlatNetwork:
    '''
    The FlatNetwork is used for the construction of the static graph of the transportation system
    '''
    def __init__(self):
        self.nodes = set()
        self.arcs = set()

    def add_node(self, flat_node: FlatNode):
        self.nodes.add(flat_node)

    def add_arc(self, flat_arc: FlatArc):
        self.arcs.add(flat_arc)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return hash(str(self))


class TempNode:
    '''
    The TempNode is used for the construction of the dynamic graph of the transportation system.
    '''
    __slots__ = ('stop', 'time', 'mode', 'route', 'coordinate', 'hash')

    def __init__(self, stop, t, mode, route, coordinate):
        self.stop = stop
        self.time = t
        self.mode = mode
        self.route = route
        self.coordinate = coordinate

        if isinstance(self.route, set):
            new_hash = hash((self.stop, self.time, frozenset(self.route)))
        else:
            new_hash = hash((self.stop, self.time, self.route))

        self.hash = new_hash

    def __str__(self):
        return f'{self.stop}, {self.time}, {self.mode}, {self.route}, {self.coordinate}'

    # if two TempNodes have the exact same attributes (except coordinates because o rounding errors) they are equal
    def __eq__(self, other):
        return (self.stop == other.stop and self.time == other.time
                and self.mode == other.mode and self.route == other.route)

    # makes the object hashable so it can be used to check if another identical object already lies in the set
    def __hash__(self):
        return self.hash


class TempArc:
    '''
    The TempNode is used for the construction of the dynamic graph of the transportation system.
    '''
    def __init__(self, start: TempNode, end: TempNode, arc_type):
        self.start = start
        self.end = end
        self.cost = end.time - start.time
        self.capacity = None
        self.arc_type = arc_type

    def __str__(self):
        return f'({self.start}) --> {self.end}, cost: {self.cost}, capacity: {self.capacity}'

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __hash__(self):
        if isinstance(self.start.route, set):
            x = frozenset(self.start.route)
        else:
            x = self.start.route

        if isinstance(self.end.route, set):
            y = frozenset(self.end.route)
        else:
            y = self.end.route

        return hash((self.start.stop, self.start.time, x, self.start.stop, self.start.time, y, self.cost))

    def set_cost(self, cost):
        self.cost = cost

    def set_capacity(self, trip_subset, cap_percentage=1):
        '''The capacity of an arc is scaled based on the size of the subset.
        If the subset of passengers of all passengers of our dataset for which the algorithm finds the optimal paths
        has a size of 10% of the subset of all passengers, the capacity of all vehicles is set to 10% of the maximum
        vehicle capacity.'''

        if self.start.mode == self.end.mode == 'bus':
            self.capacity = int(trip_subset * cap_percentage * 60)

        elif self.start.mode == self.end.mode == 'subway':
            self.capacity = int(trip_subset * cap_percentage * 940)

        elif self.start.mode == self.end.mode == 'tram':
            self.capacity = int(trip_subset * cap_percentage * 215)

        elif self.start.mode == self.end.mode == 'rail':
            self.capacity = int(trip_subset * cap_percentage * 400)

        elif self.arc_type in ['source arc', 'sink arc']:
            self.capacity = 1

        # transit and waiting arcs are uncapacitated
        else:
            self.capacity = 9999

        # Make sure that all arcs have at least capacity 1
        if self.capacity == 0:
            self.capacity = 1
            print(f'Raised capacity in arc{self}')


class TempNetwork:
    '''
    The TempNetwork is used for the construction of the dynamic graph of the transportation system.
    '''
    def __init__(self):
        self.nodes = set()
        self.arcs = set()

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return hash(str(self))

    def add_node(self, temp_node: TempNode):
        self.nodes.add(temp_node)
        return

    def add_arc(self, temp_arc: TempArc):
        self.arcs.add(temp_arc)


class Trip:
    '''
    A trip is the demand for one passenger
    '''
    def __init__(self, trip_id, origin, destination, dep_time):
        self.id = trip_id
        self.origin = origin
        self.destination = destination
        self.dep_time = dep_time

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return hash(self.id)


class Instance:
    '''
    An instance contains all relevant information needed to formulate the problem as a minimum cost multi-commodity
    flow problem.
    '''
    def __init__(self, time_window, modes_network, modes_trips, network_file, schedule_file, city, subset):
        self.flat_network = FlatNetwork()
        self.temp_network = TempNetwork()
        self.time_window = time_window
        self.modes_network = modes_network
        self.modes_trips = modes_trips
        self.network_file = network_file
        self.schedule_file = schedule_file
        self.city = city
        self.coordinates = dict()
        self.nearest_joint_nodes = dict()
        self.walking_speed = 1.4 * 60  # in m/minute, this corresponds to 5km/h
        self.waiting_layers = dict()
        self.subset = subset
        self.trips = list()
        self.incidence_matrix = dict()
        self.heuristic_dist = dict()
        self.heuristic = dict()
        self.sources = dict()
        self.sinks = dict()
        self.access = dict()
        self.egress = dict()
        self.stop_modes = dict()


    def __eq__(self, other):
        return self.temp_network == other.temp_network and self.trips == other.trips and \
                self.heuristic == other.heuristic and self.sources == other.sources and self.sinks == other.sinks

    def get_flat_network(self):
        """
        Finds all nodes and edges of the underlying network of the specified transportation modes
        Makes sure that all nodes and edges fully lie in the defined city
        """

        # get all possible edges and nodes
        for mode in self.modes_network:
            edges = xml.get_edges_by_mode(self.network_file, [mode])
            nodes = xml.get_nodes_by_edges(self.network_file, edges)
            nodes_coordinates = xml.get_coordinates(self.network_file, nodes)

            # throw out all nodes/edges which do not fully lie in the city
            nodes_to_remove = list()
            node_in_city = dict()

            for node in nodes:
                node_in_city[node] = gd.point_in_city(self.city, nodes_coordinates[node])

            for edge in list(edges):
                if not (node_in_city[edges[edge][0]]) or not (node_in_city[edges[edge][1]]):
                    del edges[edge]

            for node in nodes:
                if not node_in_city[node]:
                    nodes_to_remove.append(node)

            nodes = [node for node in nodes if node not in nodes_to_remove]

            for node in nodes_to_remove:
                del nodes_coordinates[node]

            # add all remaining nodes to the flat network
            for node in nodes:
                flat_node = FlatNode(node, mode, nodes_coordinates[node])
                self.flat_network.add_node(flat_node)

            # add all remaining edges to the flat network
            for edge in edges:
                start = FlatNode(edges[edge][0], mode, nodes_coordinates[edges[edge][0]])
                end = FlatNode(edges[edge][1], mode, nodes_coordinates[edges[edge][1]])
                self.flat_network.add_arc(FlatArc(start, end))

    def build_temporal_network(self):
        """
        This function builds the temporal expansion of the transportation network based on a schedule file

        Here only the route layers are build
        """
        node_dict = dict()
        for mode in self.modes_network:
            node_dict[mode] = dict()

        # Build an incidence matrix
        for arc in self.flat_network.arcs:
            try:
                node_dict[arc.start.mode][arc.start.stop][arc.end.stop] = None
            except KeyError:
                node_dict[arc.start.mode][arc.start.stop] = dict()
                node_dict[arc.start.mode][arc.start.stop][arc.end.stop] = None

        schedules = xml.read_xml(self.schedule_file)
        root = schedules.getroot()
        factors = (60, 1, 1 / 60)  # needed to convert hh:mm:ss into minute of the day, e.g. 08:20:30 --> 500.5

        used_stops = set()  # if a stop from the flat network is not used, we remove it from the coordinates

        # iterate over all possible routes
        for line in root.iter('transitLine'):
            for transitRoute in line.iter('transitRoute'):
                route_mode = transitRoute.find('transportMode').text
                if route_mode not in self.modes_network:
                    continue

                # get all arcs of the route
                route_id = transitRoute.attrib['id']
                route = transitRoute.find('route')
                route_nodes = list()

                for arc in route.iter('link'):
                    if len(arc.attrib['refId'].split('_')) < 4:
                        route_nodes.append(arc.attrib['refId'])

                # check if any node of the route lies in the flat nodeset of the instance, if not go to next route
                if not any(x in node_dict[route_mode] for x in route_nodes):
                    continue

                # find all departure times of the route starting before the end of the time window
                departure_times = []
                departures = transitRoute.find('departures')
                for departure in departures.iter('departure'):
                    dep_time = sum(i * j for i, j in zip(map(int, departure.attrib['departureTime'].split(':')),
                                                         factors))
                    if dep_time < self.time_window.end + 90:
                        departure_times.append(dep_time)

                # generate all nodes and arcs that are relevant for the temporal expansion
                profile = transitRoute.find('routeProfile')
                arrival_time = []

                # Save the arrival times at the stops of the route, we assume a waiting time of 0
                for stop in profile.iter('stop'):
                    if 'departureOffset' in stop.attrib:
                        travel_time = stop.attrib['departureOffset']
                    elif 'arrivalOffset' in stop.attrib:
                        travel_time = stop.attrib['arrivalOffset']

                    # Based on the arrival times calculate the traveltime of the links
                    arrival_time.append(sum(i * j for i, j in zip(map(int, travel_time.split(':')), factors)))

                # Add the temp nodes for each individual route (path + departure time) to the temp network
                for departure in departure_times:
                    for i in range(len(route_nodes) - 1):

                        if not route_nodes[i] in node_dict[route_mode] or \
                                not route_nodes[i + 1] in node_dict[route_mode][route_nodes[i]]:
                            continue

                        # check if the timed copy of the arc we want to add lies in the time window of the instance
                        if self.time_window.start < departure + arrival_time[i] < self.time_window.end + 90 and \
                                self.time_window.start < departure + arrival_time[i + 1] < self.time_window.end + 90:

                            # add temp start and end node to flat_network_full
                            temp_start = TempNode(self.nearest_joint_nodes[route_nodes[i]], departure + arrival_time[i],
                                                  route_mode, route_id,
                                                  self.coordinates[self.nearest_joint_nodes[route_nodes[i]]])
                            temp_end = TempNode(self.nearest_joint_nodes[route_nodes[i + 1]],
                                                departure + arrival_time[i + 1], route_mode, route_id,
                                                self.coordinates[self.nearest_joint_nodes[route_nodes[i + 1]]])
                            self.temp_network.add_node(temp_start)
                            self.temp_network.add_node(temp_end)

                            # add the temp arc to the arcset of the temp network
                            self.temp_network.add_arc(TempArc(temp_start, temp_end, 'route arc'))

                            used_stops.add(temp_start.stop)
                            used_stops.add(temp_end.stop)

        # remove all stops from the coordinates that are not used in the temporal extension
        for stop in list(self.coordinates.keys()):
            if stop not in used_stops:
                del self.coordinates[stop]

    def join_nodes(self, gtfs_stops):
        """
        Maps all nodes of the flat network to stops of the provided gtfs file
        One stop of the gtfs file can have multiple stops of the flat network since e.g. every rail in a subway stop is
        a unique node in the flat network

        :param gtfs_stops: public transportation stop stops.txt file downloaded from https://gtfs.de/
        """

        from pyproj import Proj, transform

        # convert lat/ lon to epsg 31468 since the network file has a different encoding than the gtfs file
        inProj = Proj('epsg:4326')
        outProj = Proj('epsg:31468')

        # get all stops of the gtfs file and transform the coordinates to the format of the network file
        gtfs = pd.read_csv(gtfs_stops)
        stop_names = gtfs.stop_name.tolist()
        stop_lat = gtfs.stop_lat.tolist()
        stop_lon = gtfs.stop_lon.tolist()
        new_lat, new_lon = transform(inProj, outProj, stop_lat, stop_lon)

        new_coord = list(zip(new_lon, new_lat))
        stops_gtfs = dict(zip(new_coord, stop_names))

        # throw out all stops which do not lie within the city
        for coord in list(stops_gtfs.keys()):
            if not gd.point_in_city(self.city, coord):
                del stops_gtfs[coord]

        # find the nearest gtfs stop to every coordinate of the flat network (can be parallelized if needed)
        nearest_nodes = dict()
        for node in self.flat_network.nodes:
            nearest_node = ('', 9999999)
            for gtfs_coordinate in stops_gtfs.keys():
                dist = distance_squared(node.coordinate, gtfs_coordinate)
                if dist < nearest_node[1]:
                    nearest_node = (stops_gtfs[gtfs_coordinate], dist)

            assert nearest_node != ('', 9999999)
            nearest_nodes[node.stop] = nearest_node[0]

        self.nearest_joint_nodes = nearest_nodes

        # build a dict with key = new stopname, value = list of all nodes that have to be merged to this new stop
        nodes_to_join = dict()
        for node in self.flat_network.nodes:
            if nearest_nodes[node.stop] not in nodes_to_join:
                nodes_to_join[nearest_nodes[node.stop]] = list()

            nodes_to_join[nearest_nodes[node.stop]].append(node)

        new_coordinates = dict()
        for new_stop in nodes_to_join.keys():
            new_coordinates[new_stop] = (np.sum(node.coordinate[0] for node in nodes_to_join[new_stop]) /
                                         len(nodes_to_join[new_stop]),
                                         np.sum(node.coordinate[1] for node in nodes_to_join[new_stop]) /
                                         len(nodes_to_join[new_stop]))

        self.coordinates = new_coordinates

    def add_waiting_layers(self):
        """
        Every distincive stop in the temp network has a copy in the waiting layer.
        A waiting layer consits of temp nodes for every timestep in which a vehicle arrives at this stop
        Between all timely consecutive temp nodes of a waiting layer there exists a temp arc
        """

        waiting_layers = dict()
        for node in self.temp_network.nodes:
            if node.stop not in waiting_layers:
                waiting_layers[node.stop] = list()

            if node.stop not in self.stop_modes:
                self.stop_modes[node.stop] = set()

            self.stop_modes[node.stop].add(node.mode)

            # Get one temp node in the waiting layer for every timestep in which a vehicle arrives at the stop
            if not any(n.time == node.time for n in waiting_layers[node.stop]):
                route_mode = set()
                route_mode.add(node.mode)
                waiting_layers[node.stop].append(
                    TempNode(node.stop, node.time, 'waiting layer', route_mode, node.coordinate))

            # Add all transportation modes at the time interval to the waiting node
            else:
                for n in waiting_layers[node.stop]:
                    if n.time == node.time:
                        n.route.add(node.mode)

        # sort the temp nodes of the waiting layers such that they are ordered by increasing timestep
        for stop in waiting_layers:
            waiting_layers[stop].sort(key=lambda x: x.time)

            # add the waiting arcs to the temp_network
            if len(waiting_layers[stop]) > 1:
                for i in range(len(waiting_layers[stop]) - 1):
                    self.temp_network.add_arc(
                        TempArc(waiting_layers[stop][i], waiting_layers[stop][i + 1], 'waiting arc'))

            # add the waiting layer nodes to the temp network
            for wait_node in waiting_layers[stop]:
                self.temp_network.add_node(wait_node)

        self.waiting_layers = waiting_layers

    def add_transit_arcs(self):
        """
        This function adds the transit arcs from a route to a waiting layer and from a waiting layer to a route

        A passenger arriving in stop s at timestep t can enter the waiting layer at timestep t
        an arc could look like arc = (a,b)
        a = {stop: s, time: t, mode: 'subway', route: '704074_opnv', coordinate: (x,y)}
        b = {stop: s, time: t, mode: 'waiting layer', route: None, coordinate: (x,y)}

        A passenger in the waiting layer at timestep t can enter a route temp node at timestep t,
        the used arc to leave the waiting layer could look like arc = (a, b) with
        a = {stop: s, time: t, mode: 'waiting layer', route: None, coordinate: (x,y)}
        b = {stop: s, time: t, mode: 'subway', route: '704074_opnv', coordinate: (x,y)}

        :param waiting_layer_dict: calculated in function add_waiting_layers_and_transit_arcs
        """

        for node in self.temp_network.nodes:
            if node.mode == 'waiting layer':
                continue

            # Add all transit arcs going from a route layer to a waiting layer
            for wait_node in self.waiting_layers[node.stop]:
                if wait_node.time == node.time:
                    self.temp_network.add_arc(TempArc(node, wait_node, 'transit arc'))
                    break

            # Add all transit arcs going from a waiting layer to a route layer
            for waiting_node in self.waiting_layers[node.stop]:

                # find the latest possible waiting node from which a passenger can switch to the route layer node
                if waiting_node.time == node.time:
                    self.temp_network.add_arc(TempArc(waiting_node, node, 'transit arc'))
                    break

    def add_walking_arcs(self):
        """
        Add walking arcs between all stops with a euclidian distance smaller than 200m
        """

        # calculate the distance between all stops
        stop_dist = dict()
        for s1 in self.coordinates:
            for s2 in self.coordinates:
                if s1 != s2:
                    dist = distance_squared(self.coordinates[s1], self.coordinates[s2])
                    if dist < 40000:
                        if s1 not in stop_dist:
                            stop_dist[s1] = dict()
                        stop_dist[s1][s2] = round(math.sqrt(dist))

        # add the walking arcs to the temp network
        for s1 in stop_dist:
            # make sure the stop is used in the defined time window
            if s1 not in self.waiting_layers:
                continue
            for s2 in stop_dist[s1]:
                if s2 not in self.waiting_layers:
                    continue

                # find the latest possible waiting node from which the passenger can walk to the waiting layer node
                for wait2 in self.waiting_layers[s2]:
                    for i in range(len(self.waiting_layers[s1])):
                        if self.waiting_layers[s1][i].time + round(stop_dist[s1][s2] / self.walking_speed) + 1 \
                                >= wait2.time and i != 0 and self.waiting_layers[s1][i - 1].time + round(
                            stop_dist[s1][s2] / self.walking_speed) + 1 <= wait2.time:

                            self.temp_network.add_arc(TempArc(self.waiting_layers[s1][i - 1], wait2, 'walking arc'))
                            break

    @staticmethod
    def distance_matrix(point, nodes={}):
        '''
        Calculate the squared euclidean distance from a point to a set of nodes
        :return: dict with squared distances
        '''
        np_nodes = np.asarray(list(nodes.values()))
        dist_2 = np.sum((np_nodes - point) ** 2, axis=1)

        return dict(zip(list(nodes.keys()), dist_2))

    def get_trips(self, trip_file, seed):
        """
        Get a subset of trips from the tripfile. All trips must lie in the city, be labeled with the right mode and be
        in the defined timewindow.

        A trip which lies further than 1000m away from the nearest stop is thrown out.
        A passenger must be able to use a vehicle after at least 30 minutes after the arrival at the stop.

        Each trip origin/destination is connected to the network such that the passenger is able to use each
        transportation mode from at least 2 different stations.
        :param trip_file: csv file provided by the Moeckel chair generated by the travel demand tool
        MITO https://www.mos.ed.tum.de/tb/forschung/models/traveldemand/
        """

        file = pd.read_csv(trip_file)
        trips = csv.get_trips_in_time(file, self.time_window.start, self.time_window.end)
        trips = csv.get_trips_by_mode(trips, self.modes_trips)

        # Can be commented out if we use the precomputed datafile of only munich trips
        # trips = csv.get_trips_in_city(trips, self.city)

        trips = csv.get_random_subset(trips, self.subset, seed)

        # add all trips to the instance
        for trip in trips.iterrows():
            new_trip = Trip(trip[1]['id'],
                            (trip[1]['originX'], trip[1]['originY']),
                            (trip[1]['destinationX'], trip[1]['destinationY']),
                            trip[1]['departure_time'])

            # calculate the distance to all used flat nodes
            origin_dist_matrix = self.distance_matrix(new_trip.origin, self.coordinates)

            # remove all entries with a distance > 1000m
            for key in list(origin_dist_matrix.keys()):
                if origin_dist_matrix[key] > 1000000:
                    del origin_dist_matrix[key]

            destination_dist_matrix = self.distance_matrix(new_trip.destination, self.coordinates)
            # remove all entries with a distance > 1000m
            for key in list(destination_dist_matrix.keys()):
                if destination_dist_matrix[key] > 1000000:
                    del destination_dist_matrix[key]

            # If there does not exist a stop within 1000m of the origin or destination we ignore the trip
            if len(origin_dist_matrix) == 0 or len(destination_dist_matrix) == 0:
                continue

            origin_nodes = set()
            modes_possible = dict()
            for mode in self.modes_network:
                modes_possible[mode] = 0

            # connect the origin to the network
            while True:

                # make sure there are still stops left to check
                if len(origin_dist_matrix) == 0:
                    break

                # calculate possible nearest node
                origin_node = min(origin_dist_matrix, key=origin_dist_matrix.get)
                if all(modes_possible[mode] >= 2 for mode in self.stop_modes[origin_node]):
                    del origin_dist_matrix[origin_node]
                    continue

                # find all temp nodes that a passenger can reach
                walking_time = int(math.sqrt(int(origin_dist_matrix[origin_node])) / (1.4 * 60)) + 1
                possible_temp_nodes = [wait_node for wait_node in self.waiting_layers[origin_node] if
                                       new_trip.dep_time + walking_time <= wait_node.time <= new_trip.dep_time +
                                       walking_time + 15]

                modes = set()
                if len(possible_temp_nodes) > 0:

                    # waiting layers are sorted by time and we want to take the first one
                    origin_temp_node = possible_temp_nodes[0]

                    # get all transportation modes that occur in the first 15 minutes after the first possible departure from the origin node
                    for n in possible_temp_nodes:
                        modes = modes | n.route

                        # if all modes occur we can continue
                        if len(modes) == len(self.modes_network):
                            break

                # if all modes of the origin mode are already often enough connected to the source try to find another origin node
                if all(modes_possible[mode] >= 2 for mode in modes):
                    del origin_dist_matrix[origin_node]

                # If the mode is not connected often enough to the source add it to the origin nodes
                else:
                    for mode in modes:
                        modes_possible[mode] += 1

                    origin_nodes.add(origin_temp_node)
                    del origin_dist_matrix[origin_node]

                # if all modes are connected to the source node often enough break the while loop
                if all(modes_possible[mode] >= 2 for mode in self.modes_network):
                    break

            # connect the destination to the network
            destination_nodes = set()
            modes_possible = dict()
            for mode in self.modes_network:
                modes_possible[mode] = 0
            while True:

                # make sure there are still stops left to check
                if len(destination_dist_matrix) == 0:
                    break

                # calculate possible nearest node
                destination_node = min(destination_dist_matrix, key=destination_dist_matrix.get)

                if all(modes_possible[mode] >= 2 for mode in self.stop_modes[destination_node]):
                    del destination_dist_matrix[destination_node]
                    continue

                # find all temp nodes that a passenger can reach
                walking_time = int(math.sqrt(int(destination_dist_matrix[destination_node])) / (1.4 * 60)) + 1
                possible_temp_nodes = [wait_node for wait_node in self.waiting_layers[destination_node] if
                                       new_trip.dep_time <= wait_node.time <= new_trip.dep_time + 90 - walking_time]

                modes = set()
                if len(possible_temp_nodes) > 0:

                    # get all transportation modes that occur in the 15 minutes before the last possible departure to the destination node
                    for waiting_node in possible_temp_nodes:
                        modes = modes | n.route

                        # if all modes occur we can continue
                        if len(modes) == len(self.modes_network):
                            break

                # if all modes of the origin mode are already often enough connected to the source try to find another origin node
                if all(modes_possible[mode] >= 2 for mode in modes):
                    del destination_dist_matrix[destination_node]

                # If the mode is not connected often enough to the source add it to the origin nodes
                else:
                    for mode in modes:
                        modes_possible[mode] += 1

                    # save destination as tuple because we need the walking time to change the arc cost (only for sink)
                    for waiting_node in possible_temp_nodes:
                        destination_nodes.add((waiting_node, walking_time))
                    del destination_dist_matrix[destination_node]
                # if all modes are connected to the source node often enough, break the while loop
                if all(modes_possible[mode] >= 2 for mode in self.modes_network):
                    break

            if len(origin_nodes) >= 1 and len(destination_nodes) >= 1:
                temp_node_origin = TempNode(f'source_{new_trip.id}', new_trip.dep_time, 'source node', None,
                                            new_trip.origin)
                temp_node_dest = TempNode(f'sink_{new_trip.id}', new_trip.dep_time + 90, 'sink node', None,
                                          new_trip.destination)

                self.trips.append(new_trip)

                # Add source and sink nodes
                self.sources[new_trip.id] = temp_node_origin
                self.temp_network.nodes.add(temp_node_origin)
                self.sinks[new_trip.id] = temp_node_dest
                self.temp_network.nodes.add(temp_node_dest)

                # Add all source arcs
                self.access[new_trip.id] = set()
                for origin_node in origin_nodes:
                    self.temp_network.arcs.add(TempArc(temp_node_origin, origin_node, 'source arc'))
                    self.access[new_trip.id].add(origin_node)

                # Add all sink arcs
                self.egress[new_trip.id] = set()
                for dest_node in destination_nodes:
                    # manually change arc cost to walking time
                    sink_arc = TempArc(dest_node[0], temp_node_dest, 'sink arc')
                    sink_arc.set_cost(dest_node[1])
                    self.temp_network.arcs.add(sink_arc)
                    self.egress[new_trip.id].add((dest_node[0].stop, dest_node[1]))

    def set_arc_capacities(self, capacity):
        """
        The arc capacities already depend of the size of the subset of passengers.
        This function can be used to scale the capacity to the users needs.
        :param capacity: between 0 - 1
        """
        for arc in self.temp_network.arcs:
            arc.set_capacity(self.subset, capacity)

    def calculate_incidence_matrix(self):
        """
        Calculate the incidence matrix (IM) of the temporal network.
        For an arc, arc = (a,b)
        IM(a, arc) = 1
        IM(b, arc) = -1
        """
        for node in self.temp_network.nodes:
            self.incidence_matrix[node] = {}

        for arc in self.temp_network.arcs:
            self.incidence_matrix[arc.start][arc] = 1
            self.incidence_matrix[arc.end][arc] = -1

    def build_heuristic_dist(self):
        """
        Find the shortest travel time between (flat) nodes u and v in the network based on the temp network
        :return:
        """

        for arc in self.temp_network.arcs:
            if (arc.start.stop, arc.end.stop) not in self.heuristic_dist:
                self.heuristic_dist[(arc.start.stop, arc.end.stop)] = arc.cost

            elif arc.cost < self.heuristic_dist[(arc.start.stop, arc.end.stop)]:
                self.heuristic_dist[(arc.start.stop, arc.end.stop)] = arc.cost

    def get_heuristic(self):
        """
        Calculate an admissible heuristic for all nodes in the dynamic graph to the destination nodes
        """

        G = nx.DiGraph()

        for (u, v) in self.heuristic_dist:
            if 'source_' not in u:
                G.add_edge(u, v, weight=self.heuristic_dist[(u, v)])

        h = list()
        for n in G:
            if 'sink_' not in n and 'source_' not in n:
                h.append(shortest_paths(G,n))
            else:
                self.heuristic[n] = dict()
                self.heuristic[n][n] = 0

        for (node, dist_dict) in h:
            self.heuristic[node] = dist_dict[0]

    def get_heuristic_fast(self):
        """
        Calculate an admissible heuristic for all nodes in the dynamic graph to the destination nodes in a faster way
        """
        G = nx.DiGraph()

        # build static graph
        for (u, v) in self.heuristic_dist:
            if 'source_' not in u and 'sink_' not in v:
                G.add_edge(u, v, weight=self.heuristic_dist[(u, v)])

        h = dict()
        # Find shortest paths for all pairs
        for n in G.nodes:
            int_res = shortest_paths(G,n)
            h[int_res[0]] = int_res[1]

        # Find minimum distance from all nodes to the sinks (destinations)
        for node, heur in h.items():
            self.heuristic[node] = dict()
            for trip_id, sink in self.sinks.items():
                min_dist_sink = math.inf
                for egress_node in self.egress[trip_id]:
                    try:
                        pos_dist = heur[0][egress_node[0]] + egress_node[1]
                        if pos_dist < min_dist_sink:
                            min_dist_sink = pos_dist

                    except KeyError:
                        continue

                self.heuristic[node][sink.stop] = min_dist_sink

        # Add heuristic value of 0 for all sinks (destinations)
        for id, sink in self.sinks.items():
            self.heuristic[sink.stop] = dict()
            self.heuristic[sink.stop][sink.stop] = 0
