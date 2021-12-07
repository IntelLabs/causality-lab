from itertools import combinations, chain

import graphical_models.arrow_head_types as Mark  # incoming arrow head-types
from graphical_models import PDAG
from causal_discovery_utils.constraint_based import LearnStructBase, unique_element_iterator


class LearnStructRAI(LearnStructBase):
    """
    RAI structure learning algorithm

    Example:
        import pandas as pd
        pd_data = pd.read_csv('Alarm1_s5000_v1.txt', header=None, sep='  ')
        data_alarm = pd_data.values  # a 2D numpy vector. len(data)=num. of cases, len(data[0])=num. of variables
        n_samples, n_vars = data_alarm.shape
        ci_test = CondIndepCMI(dataset=data_alarm, threshold=0.012)
        nodes = set(range(n_vars))
        rai = LearnStructRAI(nodes_set=nodes, ci_test=ci_test)
    """
    def __init__(self, nodes_set, ci_test):
        super().__init__(PDAG, nodes_set=nodes_set, ci_test=ci_test)
        self.graph.create_complete_graph(nodes_set)  # Create a fully connected graph
        self.overwrite_starting_graph = True  # if True, the sequence at which the CIs are tested affects the result

    def learn_structure(self):
        """
        Learn a CPDAG (completed partially directed graph) using the recursive autonomy identification (RAI) algorithm
        :return:
        """
        # initialize for the 1st recursive call
        en_nodes = self.graph.nodes_set
        ex_nodes = set()

        self._learn_recursively(en_nodes=en_nodes, ex_nodes=ex_nodes, order=0)
        self.graph.maximally_orient_pattern((1, 2, 3))

        self._re_orient_skeleton()

    def _re_orient_skeleton(self):
        """
        Remove all edge directions and re-orient using rules R1, R2, R3
        :return:
        """
        cpdag_final = PDAG(self.graph.nodes_set)
        for node in self.graph.nodes_set:
            connected_nodes_set = self.graph.parents(node) | self.graph.undirected_neighbors(node)
            cpdag_final.add_edges(parents_set=connected_nodes_set, target_node=node, arrowhead_type=Mark.Undirected)

        for node in self.graph.nodes_set:
            parents_set = self.graph.parents(node)
            for (parent_i, parent_j) in combinations(parents_set, 2):
                if not self.graph.is_connected(parent_i, parent_j):
                    cpdag_final.orient_edge(source_node=parent_i, target_node=node)  # orient v-structure
                    cpdag_final.orient_edge(source_node=parent_j, target_node=node)

        cpdag_final.maximally_orient_pattern((1, 2, 3))  # use orientation rules R1, R2, and R3
        self.graph = cpdag_final

    def _exit_cond(self, en_nodes, order):
        """
        Check if the max fan-in is lower or equal to the order (exit-cond. is met)
        :param en_nodes: nodes of the sub-graph
        :param order: condition set size of the CI-test
        :return: True if exit condition is met
        """
        for node in en_nodes:
            if self.graph.fan_in(node) > order:  # if a node have a large enough number of parents, exit cond. is false
                return False
        else:
            return True  # didn't find a node with a large enough number of parents for CI test, so exit

    def _learn_recursively(self, en_nodes, ex_nodes, order):
        """
        The folowing steps are preformed:
        1. Refine and orient using CI tests of a specific condition set size
            a. Test CI between exogenous nodes and endogenous nodes, remove edges, and orient edges
            b. Test CI among the endogenous nodes, remove edges, and orient
        3. Identify ancestors and descendant groups
        4. Call recursively for the ancestor and descendant groups with CI order+1
        :param en_nodes: Endogenous nodes
        :param ex_nodes: Exogenous nodes
        :param order: CI test order, i.e., condition set size
        :return:
        """

        # test exit condition
        if self._exit_cond(en_nodes, order):
            return

        # refine and orient with condition set size equal to "order"
        self._refine_and_orient(en_nodes=en_nodes, ex_nodes=ex_nodes, order=order, cond_indep=self.ci_test.cond_indep)

        # split into ancestors/descendant autonomous sub-graphs
        d_nodes, list_of_ancestors_sets, a_nodes = self._split_ancestors_descendant(en_nodes=en_nodes)

        # Recursive calls for each autonomous sub-graph
        for ancestor_set in list_of_ancestors_sets:
            self._learn_recursively(en_nodes=ancestor_set, ex_nodes=ex_nodes,
                                    order=order+1)  # recursive call (ancestor)
        self._learn_recursively(en_nodes=d_nodes, ex_nodes=a_nodes | ex_nodes,
                                order=order+1)  # recursive call (descendant)

    def _refine_and_orient(self, en_nodes, ex_nodes, order, cond_indep=None):
        """
        Refine by removing edges between nodes that are conditionally independent given some set.
        Note: This is a the core element in the RAI and BRAI algorithms which is called recursively.
        :param en_nodes: Endogenous nodes
        :param ex_nodes: Exogenous nodes
        :param order: CI test order (condition set size)
        :param cond_indep: CI test
        :return:
        """
        self._refine_exogenous_effect(en_nodes=en_nodes, ex_nodes=ex_nodes, order=order, cond_indep=cond_indep)
        self.maximally_orient_edges(en_nodes=en_nodes)
        self._refine_endogenous(en_nodes=en_nodes, order=order, cond_indep=cond_indep)
        self.maximally_orient_edges(en_nodes=en_nodes)

    def _split_ancestors_descendant(self, en_nodes):
        """
        Split the nodes into a descendant nodes-set and a list of disconnected ancestor node-sub-sets
        :param en_nodes: set of nodes to split
        :return: descendant set, list of ancestors sub-sets, set of all ancestor nodes
        """
        d_nodes = self._get_lowest_topological_set(en_nodes)
        a_nodes = en_nodes - d_nodes  # sets of nodes that need to be separated
        # list_of_ancestors_sets = self._get_unconnected_subgraphs(all_nodes=a_nodes)  # get unconnected ancestor sets
        list_of_ancestors_sets = self.graph.find_unconnected_subgraphs(a_nodes)  # get unconnected ancestor sets
        return d_nodes, list_of_ancestors_sets, a_nodes

    def _refine_exogenous_effect(self, en_nodes, ex_nodes, order, cond_indep=None):
        """
        Test each edge from an exogenous node to an endogenous node
        :param en_nodes: Endogenous nodes
        :param ex_nodes: Exogenous nodes
        :param order: CI test order,  i.e., condition set size
        :param cond_indep: an oracle that answers conditional independence queries
        :return:
        """
        if cond_indep is None:
            cond_indep = self.ci_test.cond_indep

        if self.overwrite_starting_graph:
            source_cpdag = self.graph  # Not a copy!!! thus, edge deletions affect consequent CI queries
        else:
            source_cpdag = self.graph.copy()  # slower, but removes the dependence on the sequence of CI testing

        for node in en_nodes:
            for ex in ex_nodes:
                if not source_cpdag.is_connected(ex, node):
                    continue

                pot_parents_node = (source_cpdag.parents(node) | source_cpdag.undirected_neighbors(node)) - {ex}
                pot_parents_ex = (source_cpdag.parents(ex) | source_cpdag.undirected_neighbors(ex)) - {node}
                cond_sets_node = combinations(pot_parents_node, order)
                cond_sets_ex = combinations(pot_parents_ex, order)
                cond_sets = unique_element_iterator(
                    chain(cond_sets_node, cond_sets_ex)
                )
                for cset in cond_sets:  # note that cond_sets is a generator of tuples (not sets)
                    if cond_indep(ex, node, cset):  # CI test: test for conditional independence
                        self.graph.delete_edge(ex, node)  # remove the edge ex --> node
                        self.sepset.set_sepset(ex, node, cset)
                        break  # stop searching for independence as we found one and updated the graph accordingly

    def _refine_endogenous(self, en_nodes, order, cond_indep=None):
        """
        Remove edges between pairs of conditionally independent endogenous variables. Condition set consists of nodes
        from endogenous and exogenous nodes and has a specific size. Test edges X --> Y and X --- Y
        :param en_nodes: Endogenous nodes
        :param order: Condition set size
        :param cond_indep: an oracle that answers conditional independence queries
        :return:
        """
        if cond_indep is None:
            cond_indep = self.ci_test.cond_indep

        if self.overwrite_starting_graph:
            source_cpdag = self.graph  # Not a copy!!! thus, edge deletions affect consequent CI queries
        else:
            source_cpdag = self.graph.copy()  # slower, but removes the dependence on the sequence of CI testing

        for node_i, node_j in combinations(en_nodes, 2):
            if not source_cpdag.is_connected(node_i, node_j):
                continue

            pot_parents_i = (source_cpdag.parents(node_i) | source_cpdag.undirected_neighbors(node_i)) - {node_j}
            pot_parents_j = (source_cpdag.parents(node_j) | source_cpdag.undirected_neighbors(node_j)) - {node_i}
            cond_sets_i = combinations(pot_parents_i, order)
            cond_sets_j = combinations(pot_parents_j, order)
            cond_sets = unique_element_iterator(
                chain(cond_sets_i, cond_sets_j)
            )

            for cset in cond_sets:
                if cond_indep(node_i, node_j, cset):
                    self.graph.delete_edge(node_i, node_j)  # remove directed/undirected edge
                    self.sepset.set_sepset(node_i, node_j, cset)
                    break  # stop searching for independence as we found one and updated the graph accordingly

    def _get_lowest_topological_set(self, en_nodes):
        """
        Return the set of nodes having the lowest topological order.
        In a directed edge, the parent has a higher topological order than the child.
        In an undirected edge, both nodes on the end points have equal topological order.
        :param en_nodes:
        :return: A set of nodes having the lowest topological order (equally)
        """

        #return en_nodes # ToDo: Consider removing this section

        # ToDo: Consider removing this section
        if False:
            generic_parents = set()  # parents of someone in the endogenous set
            for node in en_nodes:
                generic_parents.update(self.graph.parents(node) & en_nodes)

            leaves = en_nodes - generic_parents
            # "leaves" set may still contains nodes that are adjacent to "generic_parents" rendering them equal topological
            # order, thus not a descendant. Remove nodes that have a path to a generic-parent through only undirected edges
            lowest_set = set()
            for leaf in leaves:
                if self.graph.is_reachable_any_undirected(leaf, generic_parents, en_nodes):
                    continue
                else:
                    lowest_set.add(leaf)

            return lowest_set

        # ToDo: Consider removing this section
        ordered_sets = self.graph.find_partial_topological_order(en_nodes)
        # return ordered_sets[0]  # ToDo: consider returning this
        if len(ordered_sets) > 1:  # if there two or more sets
            return en_nodes - ordered_sets[-1]  # return all the nodes not in the highest order
        else:
            return ordered_sets[0]  # there is only one set (no separation was found)

    def maximally_orient_edges(self, en_nodes):
        """
        Maximally orient edges starting anywhere but ending (including undirected) at the endogenous nodes.
        First, v-structures are identified and oriented R0. Then, four rules, R1-R4 are repeatedly applied.
        :param en_nodes: Endogenous nodes
        :return:
        """
        # ToDo: Move this to the PAG class
        self.orient_v_structures(en_nodes)  # [R0]: orient v-structures

        self.graph.maximally_orient_pattern([1, 2, 3, 4])

        self.graph.convert_bidirected_to_undirected(en_nodes)  # treat bi-directed (spurious) as undirected

    def orient_v_structures(self, en_nodes):
        """
        Orient edges starting anywhere but ending (including undirected) at the endogenous nodes.
        note that in our case, the situation: X--Z<-Y can happen and node Z is tested if it can be a collider.
        :param en_nodes: The set of nodes from which to search for a collider
        :return:
        """
        # ToDo: Move this to the PAG class
        # create a copy of edges
        pre_neighbors = dict()
        pre_parents = dict()
        for node in self.graph.nodes_set:
            pre_neighbors[node] = self.graph.undirected_neighbors(node).copy()  # undirected neighbors pre graph changes

        # check each node if it can serve as new collider for a disjoint neighbors
        for node_z in en_nodes:
            # check undirected neighbors
            xy_nodes = pre_neighbors[node_z]  # undirected neighbors
            for node_x, node_y in combinations(xy_nodes, 2):
                if self.graph.is_connected(node_x, node_y):
                    continue  # skip this pair as they are connected
                if node_z not in self.sepset.get_sepset(node_x, node_y):
                    self.graph.orient_edge(source_node=node_x, target_node=node_z)  # orient X --> Z
                    self.graph.orient_edge(source_node=node_y, target_node=node_z)  # orient Y --> Z

        for node in self.graph.nodes_set:
            pre_neighbors[node] = self.graph.undirected_neighbors(node).copy()  # undirected neighbors pre graph changes
            pre_parents[node] = self.graph.parents(node).copy()  # undirected parents pre graph changes

        for node_z in en_nodes:
            # check the case when one of the neighbors is already a parent
            parents_z = pre_parents[node_z]  # parents before the graph was modified
            neighbors_z = pre_neighbors[node_z]  # remaining undirected edges X --- Z
            for node_x in neighbors_z:
                for node_y in parents_z:
                    if self.graph.is_connected(node_x, node_y):
                        continue  # skip this pair as they are connected
                    if node_z not in self.sepset.get_sepset(node_x, node_y):
                        self.graph.orient_edge(source_node=node_x, target_node=node_z)  # orient (only) X --> Z
