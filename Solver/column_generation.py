import math
import gurobipy as grp
from gurobipy import GRB
import Solver.A_star as A_star
import networkx as nx
import Solver.Dijkstra as Dijkstra

import time


class Flow:

    def __init__(self, path, trip, arcs, reduced_cost):
        self.path = path
        self.trip = trip
        self.arcs = arcs
        self.cost = None
        self.reduced_cost = reduced_cost

    def __eq__(self, other):
        return self.path == other.path and self.trip == other.trip and self.arcs == other.arcs \
               and self.cost == other.cost and self.reduced_cost == other.reduced_cost

    def __hash__(self):
        return hash((self.path[0], tuple(self.path[1]), self.trip, tuple(self.arcs), self.cost, self.reduced_cost))


class ContinuousRelaxation:

    def __init__(self, graph, trips, sources, sinks, dummy_cost, heuristic):

        # General inputs
        self.graph = graph
        self.trips = trips
        self.sources = sources
        self.sinks = sinks
        self.dummy_cost = dummy_cost
        self.heuristic = heuristic

        # Need to define the rmp
        self.dummy_vars = dict()
        self.cap_constr = dict()
        self.conv_constr = dict()
        self.x = dict()
        self.shadow_cap = dict()
        self.shadow_conv = dict()
        for trip in trips:
            self.shadow_conv[trip] = -math.inf
        self.rmp = None
        self.flows = dict()
        for trip in trips:
            self.flows[trip] = list()

        # Used mainly for the filter and the bounds
        self.change = True
        self.lower_bound = [-math.inf]
        self.objVal = list()
        self.rerun_trips = set()
        self.upper_bound = math.inf

        # These attributes are needed for the analysis of the algorithm
        self.num_rerun = list()
        self.time_pp = list()
        self.found_flows = list()
        self.time_filter = list()
        self.explored_nodes_pricing = 0
        self.num_pricing_problems = 0
        self.num_it = -1
        self.solve_time = 0


    @staticmethod
    def heuristic_value(u, v, heuristic):
        '''
        The heuristic dictionary does not store values for u,v with no path. This function assigns these "paths" inf.
        '''
        try:
            return heuristic[u][v]
        except KeyError:
            return math.inf

    def build_restricted_masterproblem(self):
        '''
        Build the restricted masterproblem by using dummy variables
        '''

        self.rmp = grp.Model('RMP')
        self.rmp.setParam('OutputFlag', 0)
        self.rmp.setParam('Presolve', 0)
        self.rmp.setParam('Method', 1)


        self.dummy_vars = {trip: self.rmp.addVar(lb=0, name=f'dummy_{trip.id}') for trip in self.trips}

        # add capacity constraints only for the route arcs (the other arcs are unconstraint)
        self.cap_constr = {(u,v): self.rmp.addConstr(0 * self.dummy_vars[next(iter(self.trips))]
                                                   <= data['cap'], name=f'cap_{u}_{v}')
                           for u,v,data in self.graph.edges(data=True) if data['arc_type'] == 'route arc'}

        # add conv_constraint (here only columns for dummy variable)
        self.conv_constr = {trip: self.rmp.addConstr(self.dummy_vars[trip] >= 1, name=f'conv_constr_{trip}')
                            for trip in self.trips}

        # Add objective function
        obj = sum(self.dummy_cost * self.dummy_vars[trip] for trip in self.trips)
        self.rmp.setObjective(obj, GRB.MINIMIZE)

        # initialize shadow values
        self.shadow_cap = {(u,v): math.inf for u,v,data in self.graph.edges(data=True)
                           if data['arc_type'] == 'route arc'}

        self.shadow_conv = {trip: -math.inf for trip in self.trips}

    def solve_rmp(self):
        '''
        Used to solve the rmp in each iteration of the column generation
        '''
        self.num_it += 1

        # If we do not reset the problem before we solve it,
        # the dual variables of the gurobi model do not update correctly
        self.rmp.reset()

        self.rmp.optimize()

        assert self.rmp.Status == GRB.OPTIMAL

    def solve_pricing_problems(self, list_of_trips, dijkstra=False):
        '''
        Solve the pricing problems of a given list of trips
        '''
        new_flows = list()

        for trip in list_of_trips:
            new_flow = self.pricing_solver(trip, dijkstra=dijkstra)

            if new_flow is not None:
                new_flows.append(new_flow)

        self.add_new_flows(new_flows)
        self.calculate_LB(new_flows)

    def pricing_solver(self, trip, dijkstra=False):
        '''
        Solve the pricing problem for one trip
        Either use Dijkstras' algorithm to solve the shortest path problem (if dijkstra==True) or the A-star
        algorithm (if dijkstra == False)
        :return A new Flow which can be used to add a new column
        '''
        def new_cost(u, v, data):
            try:
                return data['weight'] - self.shadow_cap[(u, v)]
            except KeyError:
                return data['weight']

        if dijkstra:
            try:
                # If we know the shortest path > shadow_conv[trip], we do not find a new flow --> cutoff
                path = Dijkstra.single_source_dijkstra(self.graph, source=self.sources[trip.id],
                                                       target=self.sinks[trip.id],
                                                       cutoff=self.shadow_conv[trip] - 0.0001,
                                                       weight=new_cost)
            except nx.NetworkXNoPath:
                return None

        else:
            try:
                # If we know the shortest path > shadow_conv[trip], we do not find a new flow --> cutoff
                path = A_star.path(self.graph, source=self.sources[trip.id], target=self.sinks[trip.id],
                                   heuristic_dict=self.heuristic,
                                   weight=new_cost,
                                   cutoff=self.shadow_conv[trip] - 0.0001)

            except nx.NetworkXNoPath:
                return None

        path_arcs = list(zip(path[1], path[1][1:]))
        self.num_pricing_problems += 1
        self.explored_nodes_pricing += path[2]
        return Flow(path, trip, path_arcs, path[0] - self.shadow_conv[trip])

    def add_new_flows(self, new_flows):
        '''
        For all new flows found in the pricing problems, add new columns to the rmp
        '''
        self.found_flows.append(len(new_flows))
        new_cols = dict()
        for flow in new_flows:
            assert flow not in self.flows[flow.trip]
            c = grp.Column()

            for (u,v) in flow.arcs:
                if self.graph.edges[(u,v)]['arc_type'] == 'route arc':
                    c.addTerms(1, self.cap_constr[(u,v)])
            c.addTerms(1, self.conv_constr[flow.trip])
            new_cols[flow] = c

            # construct the objective value
            objective = 0
            for a in flow.arcs:
                objective += self.graph[a[0]][a[1]]['weight']

            flow.cost = objective

            # Add the decision variable (column) to the restricted master problem
            self.flows[flow.trip].append(flow)

            self.x[flow] = self.rmp.addVar(vtype=GRB.CONTINUOUS, lb=0, obj=flow.cost,
                                          name=f'flow_{flow.trip.id}_{len(self.flows[flow.trip])}', column=c)

    def calculate_LB(self, new_flows):
        '''
        If the pricing problems of all trips where solved, we can find a new lower bound.
        If only a subset of all pricing problems is solved we can not find a new lower bound
        '''
        if self.rerun_trips != self.trips:
            self.lower_bound.append(-math.inf)
            return
        lb = self.rmp.objVal + sum(f.reduced_cost for f in new_flows)
        self.lower_bound.append(lb)

    def solve(self, filter_on=True, dijkstra=False):
        '''
        Solve the column generation problem
        '''
        t = time.time()
        while True:

            # Used to initiate the algorithm and all dual variables in the gurobi model
            self.solve_rmp()
            self.objVal.append(self.rmp.objVal)

            # Do not use the pricing filter
            if not filter_on:
                self.update_duals()
                if not self.change:
                    break
                self.solve_pricing_problems(self.trips, dijkstra=dijkstra)
                self.num_rerun.append(len(self.trips))

            # Used the pricing filter
            elif filter_on:

                # run the pricing filter
                t_filter = time.time()
                self.pricing_filter()
                self.time_filter.append(time.time()-t_filter)

                # stopping criteria for the column generation
                if len(self.rerun_trips) == 0 or max(self.lower_bound) == self.rmp.objVal:
                    break

                self.num_rerun.append(len(self.rerun_trips))
                t_pp = time.time()

                # Solve the pricing problems for all trips found in the pricing filter
                self.solve_pricing_problems(self.rerun_trips, dijkstra=dijkstra)
                self.time_pp.append(time.time() - t_pp)

        self.solve_time = time.time() - t

    def update_duals(self):
        '''
        Update dual variables used to define arc costs in the pricing problem
        '''

        self.change = False
        for arc in self.shadow_cap:
            if self.shadow_cap[arc] != self.cap_constr[arc].Pi:
                self.change = True
                self.shadow_cap[arc] = self.cap_constr[arc].Pi

        for trip in self.trips:
            if self.shadow_conv[trip] != self.conv_constr[trip].Pi:
                self.change = True
                self.shadow_conv[trip] = self.conv_constr[trip].Pi

    def pricing_filter(self):
        '''
        Find all trips for which a flow/path exists which uses an arc with negativ capacity dual variable
        '''
        self.update_duals()

        # if in the last iteration all trips pricing problems were solved and no new column was added to the basis
        # optimal solution is found
        if self.rerun_trips == self.trips and not self.change:
            self.rerun_trips = set()
            return

        # If the objective value did not change in the ast iteration, run all trips (ensures optimality)
        if len(self.objVal) >= 2 and self.objVal[-1] == self.objVal[-2]:
            self.rerun_trips = self.trips
            return

        bounded_arcs = list()
        rerun = set()
        # Find all arcs with full capacity
        for a, val in self.shadow_cap.items():
            if val < 0:
                bounded_arcs.append(a)

        # find all trips that could use this arc
        for trip, list_flows in self.flows.items():
            for flow in list_flows:
                if len(set(flow.arcs).intersection(bounded_arcs)) > 0:
                    rerun.add(trip)
                    break

        if len(rerun) > 0 and rerun != self.rerun_trips:
            self.rerun_trips = rerun

        else:
            self.rerun_trips = self.trips
