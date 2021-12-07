from itertools import combinations, chain
from causal_discovery_utils.constraint_based import LearnStructBase, unique_element_iterator
from graphical_models import PDAG


class LearnStructPC(LearnStructBase):
    def __init__(self, nodes_set, ci_test):
        super().__init__(PDAG, nodes_set=nodes_set, ci_test=ci_test)
        self.graph.create_complete_graph(nodes_set)  # Create a fully connected graph
        self.overwrite_starting_graph = True  # if True, the sequence at which the CIs are tested affects the result

    def learn_structure(self):
        """
        Learn a CPDAG (completed partially directed graph) using the PC algorithm
        :return:
        """
        self.learn_skeleton()

        self.orient_v_structures()
        self.graph.convert_bidirected_to_undirected()  # treat bi-directed (spurious) as undirected

        self.graph.maximally_orient_pattern([1, 2, 3])

    def _exit_cond(self, order):
        """
        Check if the max fan-in is lower or equal to the order (exit-cond. is met)
        :param order: condition set size of the CI-test
        :return: True if exit condition is met
        """
        for node in self.graph.nodes_set:
            if self.graph.fan_in(node) > order:  # if a node have a large enough number of parents, exit cond. is false
                return False
        else:
            return True  # didn't find a node with a large enough number of parents for CI test, so exit

    def learn_skeleton(self):
        cond_indep = self.ci_test.cond_indep

        if self.overwrite_starting_graph:
            source_cpdag = self.graph  # Not a copy!!! thus, edge deletions affect consequent CI queries
        else:
            source_cpdag = self.graph.copy()  # slower, but removes the dependence on the sequence of CI testing

        cond_set_size = 0
        while not self._exit_cond(cond_set_size):
            for node_i, node_j in combinations(source_cpdag.nodes_set, 2):
                if not source_cpdag.is_connected(node_i, node_j):
                    continue

                pot_parents_i = source_cpdag.undirected_neighbors(node_i) - {node_j}
                pot_parents_j = source_cpdag.undirected_neighbors(node_j) - {node_i}
                cond_sets_i = combinations(pot_parents_i, cond_set_size)
                cond_sets_j = combinations(pot_parents_j, cond_set_size)
                cond_sets = unique_element_iterator(  # unique of
                    chain(cond_sets_i, cond_sets_j)  # neighbors of node_i OR neighbors of node_j
                )

                for cond_set in cond_sets:
                    if cond_indep(node_i, node_j, cond_set):
                        self.graph.delete_edge(node_i, node_j)  # remove directed/undirected edge
                        self.sepset.set_sepset(node_i, node_j, cond_set)
                        break  # stop searching for independence as we found one and updated the graph accordingly

            cond_set_size += 1  # now go again over all the edges and try to remove using a condition set size +1

    def orient_v_structures(self):
        # ToDo: Move this function to the PDAG class
        # create a copy of edges
        pre_neighbors = dict()
        for node in self.graph.nodes_set:
            pre_neighbors[node] = self.graph.undirected_neighbors(node).copy()  # undirected neighbors pre graph changes

        # check each node if it can serve as new collider for a disjoint neighbors
        for node_z in self.graph.nodes_set:
            # check undirected neighbors
            xy_nodes = pre_neighbors[node_z]  # undirected neighbors
            for node_x, node_y in combinations(xy_nodes, 2):
                if self.graph.is_connected(node_x, node_y):
                    continue  # skip this pair as they are connected
                if node_z not in self.sepset.get_sepset(node_x, node_y):
                    self.graph.orient_edge(source_node=node_x, target_node=node_z)  # orient X --> Z
                    self.graph.orient_edge(source_node=node_y, target_node=node_z)  # orient Y --> Z
