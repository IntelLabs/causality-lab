from causal_discovery_algs.icd import LearnStructICD, unique_element_iterator
from graphical_models import PAG, arrow_head_types as Mark
from itertools import product, combinations, chain


class LearnStructTSICD(LearnStructICD):
    def __init__(self, nodes_sets_list, ci_test, is_homology=True, initial_pag: PAG = None, is_pre_calc_cond_set=False,
                 is_selection_bias=False, is_tail_completeness=True):
        """
        Initialization: create a complete PAG with o--> edges between past-future and o--o between contemporaneous nodes
        :param nodes_sets_list: list of list of nodes.
            first sub-list is the present, second sub-list is one time-step back and so on.
            That is: [ [node names at t=0], [node names at t=-1], ..., [node names at t=-k], where k is max-lag value.
            the order in the sub-lists matters. nodes at the same location in different sub-lists correspond to the
            same node at different time step.
        :param ci_test: a ci test that handles time-series data
        :param is_homology: if True then enforce homology, otherwise do not modify past-past edges (default: True)
        :param initial_pag: if provided, use this graph as the initial graph instead of a complete graph
        :param is_pre_calc_cond_set: if True, conditioning sets are precalculated at the beginning (default: False)
        :param is_selection_bias: is selection bias possibly present? (default: True)
        :param is_tail_completeness: should the algorithm apply rules R8, R9, R10 from Zhang 2008b (default: True)
        """
        win_len = len(nodes_sets_list)  # time steps: t=0, t=-1, ..., t=-(win_len-1)
        assert win_len > 1  # at least two time-steps are expected

        num_nodes_t0 = len(nodes_sets_list[0])  # number of nodes in a single time-step
        assert all((isinstance(nodes_t, list) and len(nodes_t) == num_nodes_t0)
                   for nodes_t in nodes_sets_list)  # check equal length of sub-lists, and that they are list types

        nodes_set = {node for nodes_t_i in nodes_sets_list for node in nodes_t_i}  # unwrap all the nodes into one set
        assert len(nodes_set) == win_len * num_nodes_t0  # duplicate node names are not allowed

        super().__init__(nodes_set=nodes_set, ci_test=ci_test, is_pre_calc_cond_set=is_pre_calc_cond_set,
                         is_selection_bias=is_selection_bias, is_tail_completeness=is_tail_completeness)

        # initialize the graph: o--> edges between nodes in different time-steps, and o--o between contemporaneous nodes
        if initial_pag is None:  # create an initial graph
            initial_pag = PAG(nodes_set)
            initial_pag.create_complete_graph(Mark.Circle, nodes_set)
            past_future_idx = [(t_past, t_future) for t_future in range(win_len - 1)
                               for t_past in range(t_future + 1, win_len)]
            for past_idx, future_idx in past_future_idx:
                for node_past, node_future in product(nodes_sets_list[past_idx], nodes_sets_list[future_idx]):
                    # Note: past instance t-1 has larger indexes than future instance t
                    initial_pag.replace_edge_mark(node_source=node_past, node_target=node_future,
                                                  requested_edge_mark=Mark.Directed)  # past o--> future (may be bidi)

        # copy initial graph and store fixed orientations
        assert isinstance(initial_pag, PAG)
        self.graph.create_empty_graph()
        for node_i, node_j in combinations(initial_pag.nodes_set, 2):  # copy graph. ToDo: consider adding to PAG class
            if initial_pag.is_connected(node_i, node_j):
                self.graph.add_edge(node_i, node_j,
                                    initial_pag.get_edge_mark(node_j, node_i),
                                    initial_pag.get_edge_mark(node_i, node_j))

        self.fixed_orientations = dict()
        for node_i, node_j in combinations(initial_pag.nodes_set, 2):  # store fixed orientations
            if initial_pag.is_connected(node_i, node_j):
                self.fixed_orientations[(node_i, node_j)] = initial_pag.get_edge_mark(node_i, node_j)
                self.fixed_orientations[(node_j, node_i)] = initial_pag.get_edge_mark(node_j, node_i)

        self.is_homology = is_homology
        self.nodes_sets_list = nodes_sets_list
        self.time_window_len = win_len
        self.num_nodes_t = num_nodes_t0

        # create a dictionary for time-stamps of each node
        self.node2time = {node: time_idx for time_idx, nodes_t_i in enumerate(nodes_sets_list) for node in nodes_t_i}

        # create a dictionary for node identity that is encoded in the index in the sub-list
        self.node2idx = {node: node_idx for nodes_t_i in nodes_sets_list for node_idx, node in enumerate(nodes_t_i)}

        # add nodes' dictionaries to the PAG instance (too)
        self.graph.node2time = self.node2time.copy()
        self.graph.node2idx = self.node2idx.copy()

        # create iterators over nodes
        past_nodes_iter = (past_node for time_past in range(1, win_len) for past_node in nodes_sets_list[time_past])
        present_nodes = nodes_sets_list[0]
        self.possible_cross_lag_edges = list(product(iter(present_nodes), past_nodes_iter))  # past-present node pairs
        self.possible_contemporaneous_edges = list(combinations(present_nodes, 2))  # present-time pairs-of-nodes

    def set_fixed_orientations(self):
        for node_i, node_j in combinations(self.graph.nodes_set, 2):
            if self.graph.is_connected(node_i, node_j):
                self.graph.replace_edge_mark(node_source=node_i, node_target=node_j,
                                             requested_edge_mark=self.fixed_orientations[(node_i, node_j)])
                self.graph.replace_edge_mark(node_source=node_j, node_target=node_i,
                                             requested_edge_mark=self.fixed_orientations[(node_j, node_i)])

    def get_edge_homology(self, node_0, node_1) -> list:
        """
        A homology map finds all the node-pairs that are homologous to a given pair
        :param node_0: node at present time
        :param node_1: node in present time (contemporaneous) or in the past (cross-lag)
        :return:
        """
        assert self.node2time[node_0] == 0  # node_0 must be from the current time-stamp
        time_1 = self.node2time[node_1]
        node_1_idx = self.node2idx[node_1]
        node_0_idx = self.node2idx[node_0]  # index of the node in the present (t=0)
        homology = []
        # loop through new time indexes for node_0  and node_i (iteratively move one step back in time)
        for time_0_new, time_1_new in zip(range(1, self.time_window_len - time_1), range(time_1 + 1, self.time_window_len)):
            node_0_new = self._get_node_name(time_0_new, node_0_idx)
            node_1_new = self._get_node_name(time_1_new, node_1_idx)
            homology.append((node_0_new, node_1_new))
        return homology

    def get_ci_homology(self, node_0, node_1, conditioning_tuple) -> list:
        """
        A homology map that finds all the node-pairs & conditioning sets that are homologous.
        Homology is search only for those cases that all node-pair and conditioning sets are inside the time window.
        This result in cases where nodes in the past are not returned as conditionally independent.
        :param node_0: node at present time
        :param node_1: node in present time (contemporaneous) or in the past (cross-lag)
        :param conditioning_tuple: conditioning set in present time or in the past
        :return: a list of tuples, where each tuple corresponds to one homologous triplet (node, node, conditioning set)
        """
        assert self.node2time[node_0] == 0
        time_1 = self.node2time[node_1]
        time_conditioning = [self.node2time[conditioning_node] for conditioning_node in conditioning_tuple]
        node_0_idx = self.node2idx[node_0]  # index of the node in the present (t=0)
        node_1_idx = self.node2idx[node_1]
        conditioning_idx = [self.node2idx[conditioning_node] for conditioning_node in conditioning_tuple]
        past_time = max(max(time_conditioning), time_1)  # furthest back in time. TODO: should be only time_1
        homology = []
        for idx, _ in enumerate(range(past_time+1, self.time_window_len)):
            node_0_new = self._get_node_name(1+idx, node_0_idx)
            node_1_new = self._get_node_name(time_1+1+idx, node_1_idx)
            conditioning_tuple_new = tuple(self._get_node_name(node_time+1+idx, node_idx)
                                           for node_time, node_idx in zip(time_conditioning, conditioning_idx))
            homology.append((node_0_new, node_1_new, conditioning_tuple_new))
        return homology

    def _get_node_name(self, time_idx, node_idx):
        return self.nodes_sets_list[time_idx][node_idx]

    def _pre_calc_conditioning(self, cond_set_size):
        self._pre_calc_conditioning_for_edges(cond_set_size, self.possible_contemporaneous_edges)
        self._pre_calc_conditioning_for_edges(cond_set_size, self.possible_cross_lag_edges)

    def _pre_calc_conditioning_for_edges(self, cond_set_size, possible_edges):  # ToDo: consider moving to ICD class
        for node_i, node_j in possible_edges:
            if self.graph.is_connected(node_i, node_j):
                self.conditioning_set[self.edge_key(node_i, node_j)] = self._get_pdsep_range_sets(
                    node_i, node_j, cond_set_size)

    def _learn_struct_incremental_step(self, cond_set_size=None):
        """
        Learn a single increment, a single ICD step. This treats the generic case for conditioning set sizes >= 2.
        :param cond_set_size: create a list of possible conditioning sets of this size, taking into account the
            removal of previous edges during this step. Ignored if class-member 'pre_calc_pds' is True
        :return: True if the resulting PAG is completed (no more edges can be removed)
        """
        if cond_set_size is None:
            assert self.is_pre_calc_pds is True

        # reduce exogenous
        done1 = self._learn_struct_incremental_for_edges(cond_set_size, self.possible_cross_lag_edges)
        # reduce contemporaneous
        done2 = self._learn_struct_incremental_for_edges(cond_set_size, self.possible_contemporaneous_edges)
        done = done1 and done2

        # Orient edges
        self.orient_graph(complete_orientation=done)
        return done

    def _learn_struct_incremental_for_edges(self, cond_set_size, possible_edges):
        done = True
        for node_i, node_j in possible_edges:
            if not self.graph.is_connected(node_i, node_j):
                continue

            if self.is_pre_calc_pds:
                cond_sets = self.conditioning_set[self.edge_key(node_i, node_j)]
            else:
                cond_sets = self._get_pdsep_range_sets(node_i, node_j, cond_set_size)

            for cond in cond_sets:
                done = False  # reset 'done' signaling to continue to the next ICD-iteration after the current one
                cond_set = cond[0]  # get the set of nodes (in [1] there is the sum-of-minimal-distances)
                cond_tup = tuple(cond_set)
                is_edge_removed = self._test_edge_and_remove_in_homology(node_i, node_j, cond_tup)
                if is_edge_removed:
                    break  # stop searching for independence as we found one and updated the graph accordingly
        return done

    def _test_edge_and_remove_in_homology(self, node_i, node_j, cond_tup) -> bool:
        if self.ci_test.cond_indep(node_i, node_j, cond_tup):
            self.graph.delete_edge(node_i, node_j)  # remove directed/undirected edge
            self.sepset.set_sepset(node_i, node_j, cond_tup)
            if self.is_homology:
                if len(cond_tup) == 0:
                    homology = self.get_edge_homology(node_0=node_i, node_1=node_j)
                    for (node_0, node_1) in homology:
                        self.graph.delete_edge(node_0, node_1)
                        self.sepset.set_sepset(node_0, node_1, cond_tup)
                else:
                    homology = self.get_ci_homology(node_0=node_i, node_1=node_j, conditioning_tuple=cond_tup)
                    for (node_0, node_1, cond_tup_new) in homology:
                        self.graph.delete_edge(node_0, node_1)
                        self.sepset.set_sepset(node_0, node_1, cond_tup_new)
            return True  # edge was removed! (independence was found!)
        else:
            return False  # edge was not removed (independence was not found)

    def _learn_struct_base_step_0(self):
        """
        Execute ICD iteration with r = 0. That is, test unconditional independence between every pair of nodes and
        remove corresponding edges. Then, orient the graph. The result is a 0-representing PAG.

        :return:
        """
        # r = 0: unconditional (marginal) independence tests
        for node_i, node_j in self.possible_cross_lag_edges:
            self._test_edge_and_remove_in_homology(node_i, node_j, ())

        for node_i, node_j in self.possible_contemporaneous_edges:
            self._test_edge_and_remove_in_homology(node_i, node_j, ())

        self.orient_graph(complete_orientation=False)

    def _learn_struct_base_step_1(self):
        """
        Execute ICD iteration with r = 1. That is, test independence between every pair of nodes conditioned on a single
        node, and remove corresponding edges. Then, orient the graph. The result is a 1-representing PAG.

        :return: True if done and no more iterations are required; otherwise False indicating the PAG is not completed.
        """
        source_cpdag = self.graph  # Not a copy!!! Thus, edge deletions affect consequent CI queries

        # r = 1: conditional independence tests order 1
        cond_set_size = 1
        done = True

        # Test Cross-Lag Edges
        for node_i, node_j in self.possible_cross_lag_edges:
            if not source_cpdag.is_connected(node_i, node_j):
                continue

            pot_parents_i = self.graph.find_adjacent_nodes(node_i) - {node_j}
            pot_parents_j = self.graph.find_adjacent_nodes(node_j) - {node_i}

            cond_sets_i = combinations(pot_parents_i, cond_set_size)
            cond_sets_j = combinations(pot_parents_j, cond_set_size)
            cond_sets = unique_element_iterator(  # unique of
                chain(cond_sets_i, cond_sets_j)  # neighbors of node_i OR neighbors of node_j
            )

            for cond_set in cond_sets:
                done = False
                is_edge_removed = self._test_edge_and_remove_in_homology(node_i, node_j, cond_set)
                if is_edge_removed:
                    break  # stop searching for independence as we found one and updated the graph accordingly

        # Test Contemporaneous Edges
        for node_i, node_j in self.possible_contemporaneous_edges:
            if not source_cpdag.is_connected(node_i, node_j):
                continue

            pot_parents_i = self.graph.find_adjacent_nodes(node_i) - {node_j}
            pot_parents_j = self.graph.find_adjacent_nodes(node_j) - {node_i}

            cond_sets_i = combinations(pot_parents_i, cond_set_size)
            cond_sets_j = combinations(pot_parents_j, cond_set_size)
            cond_sets = unique_element_iterator(  # unique of
                chain(cond_sets_i, cond_sets_j)  # neighbors of node_i OR neighbors of node_j
            )

            for cond_set in cond_sets:
                done = False
                is_edge_removed = self._test_edge_and_remove_in_homology(node_i, node_j, cond_set)
                if is_edge_removed:
                    break  # stop searching for independence as we found one and updated the graph accordingly

        self.orient_graph(complete_orientation=done)
        return done

    def orient_graph(self, complete_orientation=True):
        self.graph.reset_orientations(default_mark=Mark.Circle)
        self.set_fixed_orientations()
        self.graph.orient_v_structures(self.sepset)
        self._fill_orientation_homology()
        self.graph.maximally_orient_pattern(rules_set=[1, 2, 3, 4])
        if complete_orientation:
            if self.is_selection_bias:
                self.graph.maximally_orient_pattern(rules_set=[5, 6, 7])  # when selection-bias may be present
            if self.is_tail_completeness:
                self.graph.maximally_orient_pattern(rules_set=[8, 9, 10])  # for tail-completeness
        self._fill_orientation_homology()

    def _fill_orientation_homology(self):
        """
        Two consecutive stages:
            1. Create a list of invariant edge-marks (head or tail)
            2. For each edge-mark set edge-marks throughout the homology
        :return:
        """
        # Step 1: Create a list of invariant edge-marks (head or tail)
        invariant_marks = []  # list of invariant edge marks in the graph
        for node1, node2 in combinations(self.graph.nodes_set, 2):
            if not self.graph.is_connected(node1, node2):
                continue
            edge_mark12 = self.graph.get_edge_mark(node1, node2)
            edge_mark21 = self.graph.get_edge_mark(node2, node1)
            if edge_mark12 != Mark.Circle:
                invariant_marks.append((node1, node2, edge_mark12))
            if edge_mark21 != Mark.Circle:
                invariant_marks.append((node2, node1, edge_mark21))

        # Step 2: For each edge-mark set edge-marks throughout the homology
        for parent, child, edge_mark in invariant_marks:
            parent_time = self.node2time[parent]
            child_time = self.node2time[child]
            min_time = min(parent_time, child_time)
            parent_base = self.nodes_sets_list[parent_time - min_time][self.node2idx[parent]]
            child_base = self.nodes_sets_list[child_time - min_time][self.node2idx[child]]
            if self.node2time[child_base] == 0:
                edge_homology = self.get_edge_homology(child_base, parent_base)
            elif self.node2time[parent_base] == 0:
                edge_homology = self.get_edge_homology(parent_base, child_base)
                edge_homology = [(b, a) for (a, b) in edge_homology]  # reverse a,b
            else:
                raise RuntimeError("unexpected error")
            edge_homology.append((child_base, parent_base))

            for (child_node, parent_node) in edge_homology:
                self.graph.replace_edge_mark(node_source=parent_node, node_target=child_node,
                                             requested_edge_mark=edge_mark)
