from causal_discovery_utils.constraint_based import LearnStructBase
from causal_discovery_algs.pc import LearnStructPC
from graphical_models import PAG, arrow_head_types as Mark
from itertools import combinations


class LearnStructFCI(LearnStructBase):
    def __init__(self, nodes_set, ci_test,
                 is_selection_bias=True, is_tail_completeness=True):
        super().__init__(PAG, nodes_set=nodes_set, ci_test=ci_test)

        assert isinstance(is_selection_bias, bool)
        self.is_selection_bias = is_selection_bias  # if False, orientation rules R5, R6, R7 are not executed.
        assert isinstance(is_tail_completeness, bool)
        self.is_tail_completeness = is_tail_completeness  # if False, orientation rules R8, R9, R10 are not executed

        self.graph.create_complete_graph(Mark.Circle, nodes_set)  # Create a fully connected graph with edges: o--o
        self.pc_alg = LearnStructPC(nodes_set, ci_test)  # initialize a PC object for learning the skeleton
        self.found_D_Sep_link = False  # indicates if the learner removed an edges that the PC stage didn't remove

    def learn_structure(self):
        """
        Learn a partial ancestral graph (PAG) using the fast causal inference (FCI) algorithm
        :return:
        """
        # initial graph is a fully connected one with o--o edges between every pair of nodes
        # learn an initial skeleton using the same procedure as in the PC algorithm
        self._learn_pc_skeleton()

        # the resulting graph consists of only o--o edges
        # find and orient v-structures
        self.graph.orient_v_structures(self.sepset)

        # the resulting graph has only o--o, o-->, or <--> edges
        # find and remove edges between pairs of variables that are d-separated by some subset of Possible-D-SEP sets
        self.found_D_Sep_link = self._refine_pc_skeleton()

        # re-orient
        self.graph.reset_orientations(default_mark=Mark.Circle)
        self.graph.orient_v_structures(self.sepset)
        self.graph.maximally_orient_pattern(rules_set=[1, 2, 3, 4])
        if self.is_selection_bias:
            self.graph.maximally_orient_pattern(rules_set=[5, 6, 7])
        if self.is_tail_completeness:
            self.graph.maximally_orient_pattern(rules_set=[8, 9, 10])

    def _learn_pc_skeleton(self):
        """
        Learn an initial skeleton. This procedure is identical to the one of the PC algorithm
        :return:
        """

        self.pc_alg.learn_skeleton()
        self.sepset.copy_from(self.pc_alg.sepset, self.graph.nodes_set)
        self.graph.create_empty_graph()
        self.graph.copy_skeleton_from_pdag(self.pc_alg.graph)  # create edges with o-marks: X o--o Y
        self.graph.sepset = self.sepset

    def _refine_pc_skeleton(self):
        """
        Refine the skeleton (v-structures are oriented) recovered by the PC algorithm
        using subset of possible-d-sep set.

        :return: True if the graph was modified by this method
        """
        found_indep = False
        pds_list = dict()

        # Prepare the possible-d-sep set for each of the nodes
        for node_x in self.graph.nodes_set:
            pds_list[node_x] = possible_d_sep = self._create_pds_set(node_x)  # self.get_pds(node_x)

        # Test CI for the graph edges
        for node_x in self.graph.nodes_set:
            possible_d_sep = pds_list[node_x]
            adjacent_nodes = self.graph.find_adjacent_nodes(node_x)
            for node_y in adjacent_nodes:
                found_indep |= self._test_ci_increasing(node_x, node_y, possible_d_sep - {node_y})

        return found_indep

    def _test_ci_increasing(self, node_x, node_y, pds_super_set):
        """
        Search for a minimal separating set by gradually increasing conditioning set size.
        :param node_x: a node on one side of the tested edge
        :param node_y: a node on the other side of the tested edge
        :param pds_super_set: a super-set of nodes from which to construct conditioning sets
        :return: True if an edge was deleted, False if no independence was found
        """
        cond_indep = self.ci_test.cond_indep  # for better readability
        for ci_size in range(len(pds_super_set)+1):  # loop over condition set sizes; increasing set sizes
            for cond_set in combinations(pds_super_set, ci_size):  # loop over condition sets of a fixed size
                if cond_indep(node_x, node_y, cond_set):
                    self.graph.delete_edge(node_x, node_y)
                    self.sepset.set_sepset(node_x, node_y, cond_set)
                    return True

        return False

    def _create_pds_set(self, node_edge):
        """
        Construct a possible-d-sep set for node_edge

        :param node_edge: node on the edge being CI tested
        :return: a possible-d-sep
        """

        # Three lists are maintained: "first_nodes", "second_nodes", "neighbors".
        # Corresponding elements from the lists,
        #   "node_1" in "first_nodes",
        #   "node_2" in "second_nodes", and
        #   "node_3" in "neighbors",
        # form a path "node_1" --- "node_2" --- "node_3"
        # If this path is "legal" then "node_2" is in the possible-d-sep set and added to the PDS-tree

        # create an adjacency matrix (ignore edge-marks)
        adj_graph = self.graph.get_adjacency_graph()

        # initialize "first nodes" and "second nodes" lists
        neighbors = adj_graph.get_neighbors(node_edge)
        second_nodes = neighbors.copy()
        first_nodes = [node_edge for _ in range(len(second_nodes))]

        # initialize possible-d-sep list of nodes
        pds_nodes = neighbors.copy()  # initially: the neighbors of the node
        for node_nb in neighbors:
            adj_graph.remove_edge(node_edge, node_nb)  # make sure the search doesn't loop back to the root

        while len(second_nodes) > 0:
            node_1 = first_nodes.pop(0)
            node_2 = second_nodes.pop(0)

            neighbors = adj_graph.get_neighbors(node_2)

            for node_3 in neighbors:
                if self.graph.is_possible_collider(node_x=node_1, node_middle=node_2, node_y=node_3):  # test sub-path
                    adj_graph.remove_edge(node_2, node_3)
                    first_nodes.append(node_2)
                    second_nodes.append(node_3)
                    pds_nodes.append(node_3)

        possible_d_sep_set = set(pds_nodes)
        possible_d_sep_set.discard(node_edge)
        return possible_d_sep_set
