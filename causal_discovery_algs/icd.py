from causal_discovery_utils.constraint_based import LearnStructBase, unique_element_iterator
from graphical_models import PAG, PDSTree, arrow_head_types as Mark
from itertools import combinations, chain


class LearnStructICD(LearnStructBase):
    def __init__(self, nodes_set, ci_test, is_pre_calc_cond_set=False,
                 is_selection_bias=True, is_tail_completeness=True):
        super().__init__(PAG, nodes_set=nodes_set, ci_test=ci_test)

        # initialize ICD
        self.graph.create_complete_graph(Mark.Circle, nodes_set)  # Create a fully connected graph with edges: o--o
        self.test_cond_ancestor = True  # requires nodes in the conditioning set to be possible ancestors

        assert isinstance(is_pre_calc_cond_set, bool)
        self.is_pre_calc_pds = is_pre_calc_cond_set
        assert isinstance(is_selection_bias, bool)
        self.is_selection_bias = is_selection_bias  # if False, orientation rules R5, R6, R7 are not executed.
        assert isinstance(is_tail_completeness, bool)
        self.is_tail_completeness = is_tail_completeness  # if False, orientation rules R8, R9, R10 are not executed
        self.edge_key = lambda x, y: (x, y) if x < y else (y, x)
        self.conditioning_set = {self.edge_key(*edge): set() for edge in combinations(nodes_set, 2)}
        self._state = dict(done=False, cond_set_size=0)

    def learn_structure(self) -> None:
        """
        Learn a partial ancestral graph (PAG) using the iterative causal discovery (ICD) algorithm.

        :return:
        """

        # reset state
        self._state = dict(done=False,  # Latest iteration result
                           cond_set_size=0)  # next iteration r-value: desired search-radius & conditioning-set-size

        done = False
        while not done:
            # Perform ICD single iteration
            done, _ = self.learn_structure_iteration()

    def learn_structure_iteration(self):
        """
        Execute a single ICD-iteration increasing the representation level of the PAG by 1:
            1. Run a single ICD iteration with parameter r (internal)
            2. Prepare for the next iteration: r := r + 1

        :return: a 2-tuple: done, current graph's r-value.
            done is True if ICD concluded and no more iterations are required/allowed.
            At this stage self.graph is an r-representing PAG.
        """

        if self._state['done']:
            raise "ICD already concluded. Cannot run more iterations"

        # -1- Run a single ICD iteration -------------------------------------------------------------------------------
        # for efficiency, handle special cases of ICD iterations (conditioning set is empty or contains a single node)
        if self._state['cond_set_size'] == 0:  # empty conditioning set
            self._learn_struct_base_step_0()
        elif self._state['cond_set_size'] == 1:  # a single node in the conditioning set
            self._state['done'] = self._learn_struct_base_step_1()
        else:  # general ICD iteration
            if self.is_pre_calc_pds:
                self._pre_calc_conditioning(self._state['cond_set_size'])
            self._state['done'] = self._learn_struct_incremental_step(self._state['cond_set_size'])

        r_value = self._state['cond_set_size']  # the graphs representation level

        # -2- Prepare for the next iteration: r := r + 1 ---------------------------------------------------------------
        self._state['cond_set_size'] += 1  # for the next iteration: increase r (radius & conditioning-set-size)

        return self._state['done'], r_value  # return r-value of latest iteration, i.e., self.graph is r-representing

    def _pre_calc_conditioning(self, cond_set_size):
        for node_i, node_j in combinations(self.graph.nodes_set, 2):
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
        cond_indep = self.ci_test.cond_indep
        source_pag = self.graph  # Not a copy!!! thus, edge deletions affect consequent CI queries
        done = True
        for node_i, node_j in combinations(source_pag.nodes_set, 2):
            if not source_pag.is_connected(node_i, node_j):
                continue

            if self.is_pre_calc_pds:
                cond_sets = self.conditioning_set[self.edge_key(node_i, node_j)]
            else:
                cond_sets = self._get_pdsep_range_sets(node_i, node_j, cond_set_size)

            for cond in cond_sets:
                done = False  # reset 'done' signaling to continue to the next ICD-iteration after the current one
                cond_set = cond[0]  # get the set of nodes (in [1] there is the sum-of-minimal-distances)
                cond_tup = tuple(cond_set)
                if cond_indep(node_i, node_j, cond_tup):
                    self.graph.delete_edge(node_i, node_j)  # remove directed/undirected edge
                    self.sepset.set_sepset(node_i, node_j, cond_tup)
                    break  # stop searching for independence as we found one and updated the graph accordingly

        # Orient edges
        # ------------
        if not done:  # re-orient the skeleton only if it was modified
            self.graph.reset_orientations(default_mark=Mark.Circle)
            self.graph.orient_v_structures(self.sepset)  # corresponds to rule R0
            self.graph.maximally_orient_pattern(rules_set=[1, 2, 3, 4])
        else:  # algorithm concluded, orient all edges for obtaining completeness
            if self.is_selection_bias:
                self.graph.maximally_orient_pattern(rules_set=[5, 6, 7])  # when selection-bias may be present
            if self.is_tail_completeness:
                self.graph.maximally_orient_pattern(rules_set=[8, 9, 10])  # for tail-completeness

        return done

    def _learn_struct_base_step_0(self):
        """
        Execute ICD iteration with r = 0. That is, test unconditional independence between every pair of nodes and
        remove corresponding edges. Then, orient the graph. The result is a 0-representing PAG.

        :return:
        """
        cond_indep = self.ci_test.cond_indep
        source_cpdag = self.graph  # Not a copy!!! Thus, edge deletions affect consequent CI queries

        # r = 0: unconditional (marginal) independence tests
        for node_i, node_j in combinations(source_cpdag.nodes_set, 2):
            if cond_indep(node_i, node_j, ()):
                self.graph.delete_edge(node_i, node_j)  # remove directed/undirected edge
                self.sepset.set_sepset(node_i, node_j, ())

        self.graph.orient_v_structures(self.sepset)
        self.graph.maximally_orient_pattern(rules_set=[1, 2, 3, 4])

    def _learn_struct_base_step_1(self):
        """
        Execute ICD iteration with r = 1. That is, test independence between every pair of nodes conditioned on a single
        node, and remove corresponding edges. Then, orient the graph. The result is a 1-representing PAG.

        :return: True if done and no more iterations are required; otherwise False indicating the PAG is not completed.
        """
        cond_indep = self.ci_test.cond_indep
        source_cpdag = self.graph  # Not a copy!!! Thus, edge deletions affect consequent CI queries

        # r = 1: conditional independence tests order 1
        cond_set_size = 1
        done = True
        for node_i, node_j in combinations(source_cpdag.nodes_set, 2):
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
                if cond_indep(node_i, node_j, cond_set):
                    self.graph.delete_edge(node_i, node_j)  # remove directed/undirected edge
                    self.sepset.set_sepset(node_i, node_j, cond_set)
                    break  # stop searching for independence as we found one and updated the graph accordingly

        self.graph.reset_orientations(default_mark=Mark.Circle)
        self.graph.orient_v_structures(self.sepset)
        self.graph.maximally_orient_pattern(rules_set=[1, 2, 3, 4])
        if self.is_selection_bias:
            self.graph.maximally_orient_pattern(rules_set=[5, 6, 7])  # when selection-bias may be present
        if self.is_tail_completeness:
            self.graph.maximally_orient_pattern(rules_set=[8, 9, 10])  # for tail-completeness

        return done

    def _get_pdsep_range_sets(self, node_i, node_j, cond_set_size):
        """
        Create a list of conditioning sets that comply with the ICD-Sep conditions

        :param node_i: node on one side of the tested edge
        :param node_j: node on the other side of the tested edge
        :param cond_set_size: requested conditioning set size (ICD-Sep condition 1)
        :return: a list of conditioning sets to consider when testing CI between node_i and node_j
        """
        # create PDS-trees for the tested nodes
        pds_tree_i, possible_d_sep_i = create_pds_tree(self.graph, node_i, max_depth=cond_set_size)
        pds_tree_j, possible_d_sep_j = create_pds_tree(self.graph, node_j, max_depth=cond_set_size)
        # pds_tree_i, possible_d_sep_i = self._create_pds_tree(node_i, max_depth=cond_set_size)
        # pds_tree_j, possible_d_sep_j = self._create_pds_tree(node_j, max_depth=cond_set_size)

        cond_sets_list_init = pds_tree_i.get_subsets_list(set_nodes=possible_d_sep_i, subset_size=cond_set_size)
        cond_sets_list_init += pds_tree_j.get_subsets_list(set_nodes=possible_d_sep_j, subset_size=cond_set_size)

        cond_sets_list = []
        for cond in cond_sets_list_init:
            cond_set = cond[0]
            if (node_i in cond_set) or (node_j in cond_set):
                continue

            if not self._is_cond_set_possible_ancestor(cond_set, node_i, node_j):
                continue

            cond_sets_list.append(cond)

        # sort the list with respect to the sum-of-minimal-distances
        cond_sets_list.sort(key=lambda x: x[1])
        return cond_sets_list

    def _is_cond_set_possible_ancestor(self, cond_set, node_i, node_j):
        """
        Test ICD-Sep condition 3. That is, test if the all the nodes in the conditioning set are possible ancestors of
        node_i or node_j.

        :param cond_set: the conditioning set under examination
        :param node_i: node on one side of the tested edge
        :param node_j: node on the other side of the tested edge
        :return: True if the condition is satisfied, otherwise False
        """
        for z in cond_set:
            if not ((self.graph.is_possible_ancestor(ancestor_node=z, descendant_node=node_i)) or
                    (self.graph.is_possible_ancestor(ancestor_node=z, descendant_node=node_j))):
                return False
        return True

    # def _create_pds_tree(self, node_root, max_depth=None):
    #     """
    #     Create a PDS-tree rooted at node_root.
    #
    #     :param node_root: root of the PDS tree
    #     :param max_depth: maximal depth of the tree (search radius around the root)
    #     :return: a PDS-tree
    #     """
    #
    #     # Three lists are maintained: "first_nodes", "second_nodes", "neighbors".
    #     # Corresponding elements from the lists,
    #     #   "node_1" in "first_nodes",
    #     #   "node_2" in "second_nodes", and
    #     #   "node_3" in "neighbors",
    #     # form a path "node_1" --- "node_2" --- "node_3"
    #     # If this path is "legal" then "node_2" is in the possible-d-sep set and added to the PDS-tree
    #
    #     pds_tree = PDSTree(node_root)  # initialize
    #
    #     # create an adjacency matrix (ignore edge-marks)
    #     adj_graph = self.graph.get_skeleton_graph()
    #
    #     # initialize "first nodes" and "second nodes" lists
    #     neighbors = adj_graph.get_neighbors(node_root)
    #     second_nodes = neighbors.copy()
    #     first_nodes = [node_root for _ in range(len(second_nodes))]
    #
    #     # initialize possible-d-sep list of nodes
    #     pds_nodes = neighbors.copy()  # initially: the neighbors of the node
    #     for node_nb in neighbors:
    #         adj_graph.remove_edge(node_root, node_nb)  # make sure the search doesn't loop back to the root
    #
    #     # ----- for creating a PDS-tree -----\
    #     if max_depth is None:  # do not limit depth
    #         max_depth = len(self.graph.nodes_set) - 1
    #     # create "first_nodes" and "second_nodes" trees
    #     first_nodes_trees = [pds_tree for _ in range(len(second_nodes))]
    #     for node in pds_nodes:
    #         pds_tree.add_branch(node)  # add nodes to the PDS-tree
    #         second_nodes_trees = pds_tree.children.copy()  # update "node_2 trees" list
    #     # now, both node_1_trees and node_2_trees have corresponding elements
    #     # -End: for creating a PDS-tree -----/
    #
    #     while len(second_nodes) > 0:
    #         node_1 = first_nodes.pop(0)
    #         node_2 = second_nodes.pop(0)
    #
    #         # ----- for creating a PDS-tree -----
    #         node_2_tree = second_nodes_trees.pop(0)
    #         if node_2_tree.depth_level >= max_depth:
    #             continue  # skip the current pair: node_1 *--> node_2 (do not search <--* node_3 )
    #         # -End: for creating a PDS-tree -----
    #
    #         neighbors = adj_graph.get_neighbors(node_2)
    #
    #         for node_3 in neighbors:
    #             if self.graph.is_possible_collider(node_x=node_1, node_middle=node_2, node_y=node_3):  # test sub-path
    #                 adj_graph.remove_edge(node_2, node_3)
    #                 first_nodes.append(node_2)
    #                 second_nodes.append(node_3)
    #                 pds_nodes.append(node_3)
    #
    #                 # ----- for creating a PDS-tree -----
    #                 node_2_tree.add_branch(node_3)
    #                 added_branch = node_2_tree.get_child_branch(node_3)  # get the added child branch
    #                 second_nodes_trees.append(added_branch)
    #                 first_nodes_trees.append(node_2_tree)
    #                 # -End: for creating a PDS-tree -----
    #
    #     possible_d_sep_set = set(pds_nodes)
    #     possible_d_sep_set.discard(node_root)
    #     return pds_tree, possible_d_sep_set


def create_pds_tree(source_pag, node_root, max_depth=None):
    """
    Create a PDS-tree rooted at node_root.

    :param node_root: root of the PDS tree
    :param max_depth: maximal depth of the tree (search radius around the root)
    :return: a PDS-tree
    """

    # Three lists are maintained: "first_nodes", "second_nodes", "neighbors".
    # Corresponding elements from the lists,
    #   "node_1" in "first_nodes",
    #   "node_2" in "second_nodes", and
    #   "node_3" in "neighbors",
    # form a path "node_1" --- "node_2" --- "node_3"
    # If this path is "legal" then "node_2" is in the possible-d-sep set and added to the PDS-tree

    pds_tree = PDSTree(node_root)  # initialize

    # create an adjacency matrix (ignore edge-marks)
    adj_graph = source_pag.get_skeleton_graph()

    # initialize "first nodes" and "second nodes" lists
    neighbors = adj_graph.get_neighbors(node_root)
    second_nodes = neighbors.copy()
    first_nodes = [node_root for _ in range(len(second_nodes))]

    # initialize possible-d-sep list of nodes
    pds_nodes = neighbors.copy()  # initially: the neighbors of the node
    for node_nb in neighbors:
        adj_graph.remove_edge(node_root, node_nb)  # make sure the search doesn't loop back to the root

    # ----- for creating a PDS-tree -----\
    if max_depth is None:  # do not limit depth
        max_depth = len(source_pag.nodes_set) - 1
    # create "first_nodes" and "second_nodes" trees
    first_nodes_trees = [pds_tree for _ in range(len(second_nodes))]
    for node in pds_nodes:
        pds_tree.add_branch(node)  # add nodes to the PDS-tree
        second_nodes_trees = pds_tree.children.copy()  # update "node_2 trees" list
    # now, both node_1_trees and node_2_trees have corresponding elements
    # -End: for creating a PDS-tree -----/

    while len(second_nodes) > 0:
        node_1 = first_nodes.pop(0)
        node_2 = second_nodes.pop(0)

        # ----- for creating a PDS-tree -----
        node_2_tree = second_nodes_trees.pop(0)
        if node_2_tree.depth_level >= max_depth:
            continue  # skip the current pair: node_1 *--> node_2 (do not search <--* node_3 )
        # -End: for creating a PDS-tree -----

        neighbors = adj_graph.get_neighbors(node_2)

        for node_3 in neighbors:
            if source_pag.is_possible_collider(node_x=node_1, node_middle=node_2, node_y=node_3):  # test sub-path
                adj_graph.remove_edge(node_2, node_3)
                first_nodes.append(node_2)
                second_nodes.append(node_3)
                pds_nodes.append(node_3)

                # ----- for creating a PDS-tree -----
                node_2_tree.add_branch(node_3)
                added_branch = node_2_tree.get_child_branch(node_3)  # get the added child branch
                second_nodes_trees.append(added_branch)
                first_nodes_trees.append(node_2_tree)
                # -End: for creating a PDS-tree -----

    possible_d_sep_set = set(pds_nodes)
    possible_d_sep_set.discard(node_root)
    return pds_tree, possible_d_sep_set
