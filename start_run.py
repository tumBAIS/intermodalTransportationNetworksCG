from Solver.column_generation import ContinuousRelaxation
import Generator.graph_generator as gg
from gurobipy import GRB
import Generator.instance_generator as ig
import Generator.XML_parser as xml
import Generator.geodata as gd
import time
import sys

epsilon = 1e-9


def build_instance(seed, capacity, modes='', subset=0.1, time_window=(0, 1440)):
    '''
    Build graph of the multilayered transportation system based on passenger demand of the travel demand modeling
    tool MITO ('data/trips_munich_public_transport.csv') and the schedules of the munich public transportation system
    ('data/schedule.xml')

    :param seed: Used to select random subset of passegers
    :param capacity: Used to scale the capacity
    :param modes: string of transportation modes (s = subway, b = bus, t = tram, r = rail)
    :param subset: size of the passenger subset
    :param time_window: time window of the departure times of the passengers
    :return: instance class which contains all relevant information
    '''

    tw = ig.TimeWindow(time_window[0], time_window[1])
    modes_network = list()
    modes_trips = list()
    if 'b' in modes:
        modes_network.append('bus')
        modes_trips.append('bus')
    if 't' in modes:
        modes_network.append('tram')
        modes_trips.append('tramOrMetro')
    if 's' in modes:
        modes_network.append('subway')
        if not 'tramOrMetro' in modes_trips:
            modes_trips.append('tramOrMetro')
    if 'r' in modes:
        modes_network.append('rail')
        modes_trips.append('train')
    network_file = xml.read_xml('data/network_pt_road.xml')
    city = gd.get_city_gdf('Munich')
    schedule_file = 'data/schedule.xml'
    trip_file = 'data/trips_munich_public_transport.csv'

    instance = ig.Instance(tw, modes_network, modes_trips, network_file, schedule_file, city, subset)
    instance.get_flat_network()
    instance.join_nodes('data/gtfs/stops.txt')
    instance.build_temporal_network()
    instance.add_waiting_layers()
    instance.add_transit_arcs()
    instance.add_walking_arcs()
    instance.get_trips(trip_file, seed)
    instance.set_arc_capacities(capacity)
    instance.calculate_incidence_matrix()
    instance.build_heuristic_dist()
    instance.get_heuristic_fast()
    return instance


def run_price_and_branch(seed, capacity, filter, modes, subset, time_window, dijkstra):
    '''
    Build instance via build_instance() and solve it via Solver/column_generation.py.
    After the column generation terminates, find an integer solution by setting all decision variables of the
    last column generation iteration to be binary with solve_last_iter_int().
    This does not guarentee optimal integer solutions!
    '''
    # Build the instance based on input parameters
    print('Build instance')
    t = time.time()
    instance = build_instance(seed, capacity, modes, subset, time_window)
    t_instance = time.time() - t

    # Build networkx graph based on dynamic graph of the instance
    graph = gg.build_graph(instance)

    # Solve continuous relaxation of the minimum cost multi-commodity flow problem via column generation
    print('Apply column generation')
    t = time.time()
    cr = ContinuousRelaxation(graph, instance.trips, instance.sources, instance.sinks, 91, instance.heuristic)
    cr.build_restricted_masterproblem()
    cr.solve(filter_on=filter, dijkstra=dijkstra)
    t_solve = round(time.time() - t, 2)

    frac_obj = cr.rmp.objVal

    frac_flows = find_frac_flows(cr)

    # Solve the last iteration of the column generation algorithm as a MIP if the solution is not integral
    t = time.time()
    if len(frac_flows) > 0:
        print('\nFind integer solution')
        cr = solve_last_iter_int(cr)

    else:
        print('\nFractional solution is integer')
    t_integer = time.time() - t
    int_obj = cr.rmp.objVal

    # find paths of all passengers in integer solution
    paths = list()
    for flow, var in cr.x.items():
        if var.X == 1:
            paths.append(flow)

    print(f'Instance was build in {round(t_instance, 2)} seconds')
    print(f'The instance was solved in {t_solve} seconds')
    print(f'An integer solution was found in {round(t_integer, 2)} seconds')
    print(f'The optimality gap is {round((int_obj / frac_obj - 1) * 100, 2)} percent')
    print(f'{sum(cr.num_rerun)} pricing problems had to be solved')

    return frac_obj, int_obj, paths


def solve_last_iter_int(lp):
    for flow, v in lp.x.items():
        v.vtype = GRB.BINARY

    mipgap = 0.5 / lp.rmp.objVal
    lp.rmp.setParam('MIPGap', mipgap)
    lp.solve_rmp()
    return lp


def find_frac_flows(lp):
    frac_flows = list()
    for flow, val in lp.x.items():
        if 0 + epsilon < val.X < 1 - epsilon:
            frac_flows.append((flow, val.X))

    return frac_flows


######################## Choose the instance you want to run ##########################
# Choose the dataset 'b' = bus, 's' = subway, 'sbt' = subway-bus-tram
# modes = 's'
modes = sys.argv[1]

# Choose the subset size
# See below which options exist for the different transportation modes
# passengers = 132
passengers = int(sys.argv[2])

# choose a seed betweem 0-9
# seed = 9
seed = int(sys.argv[3])

# Choose whether the filter should be used
# filter_on = True
filter_on = (sys.argv[4] == "True")

# Choose whether A* should be used
# astar = True
astar = (sys.argv[5] == "True")

########################## Do not change from here ###################################
subway_passengers = {132: 0.06,
                     308: 0.14,
                     486: 0.22,
                     662: 0.3}

bus_passengers = {2632: 0.1,
                  7896: 0.3,
                  13160: 0.5,
                  18424: 0.7,
                  23688: 0.9}

bus_subway_tram_passengers = {6255: 0.1,
                              18765: 0.3,
                              31275: 0.5,
                              43785: 0.7,
                              56295: 0.9}

capacity = 1

if modes == 's':
    time_window = (420, 435)
    subset = subway_passengers[passengers]
    print(f'\nInstance: SUBWAY-{passengers}-{seed}, Filter On = {filter_on}, A-star utilized = {astar}')

elif modes == 'b':
    time_window = (420, 540)
    subset = bus_passengers[passengers]

elif modes == 'sbt':
    time_window = (420, 540)
    subset = bus_subway_tram_passengers[passengers]

dijkstra = (astar != True)

run_price_and_branch(seed, capacity, filter_on, modes, subset, time_window, dijkstra)
########################## Do not change until here ###################################