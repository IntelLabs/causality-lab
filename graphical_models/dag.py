import numpy as np
from itertools import combinations
from .basic_graph import Graph
from .undirected_graph import UndirectedGraph
from . import arrow_head_types as Mark

_ErrorCyclicGraph = 'Graph is cyclic'


class DAG(Graph):
    """
    A directed acyclic graph
    Example:
        nodes_set1 = set(range(5))
        dag = DAG(nodes_set1)
        dag.add_edges({0}, 1)
        dag.add_edges({0}, 2)
        dag.add_edges({1, 2}, 3)
        dag.add_edges({3}, 4)

        print('Is acyclic?', dag.is_acyclic())
        print('(0, 4) d-separated by {1, 2}?', dag.dsep(0, 4, {1, 2}))
    """
    def max_parents(self):
        max_parents = 0
        for node in self.nodes_set:
            num_parents = len(self.parents(node))
            if num_parents > max_parents:
                max_parents = num_parents

        return max_parents

    def parents(self, node):
        return self._graph[node]

    def find_children(self, node_parent, nodes_set=None):
        # ToDo: inefficient, should re-impelement
        if nodes_set is None:
            nodes_set = self.nodes_set

        children_set = set()
        for node in nodes_set:
            if node_parent in self._graph[node]:
                children_set.add(node)

        return children_set

    def find_adjacent_nodes(self, node):
        return self.parents(node) | self.find_children(node)

    def is_connected(self, node_i, node_j):
        if (node_i in self.parents(node_j)) or (node_j in self.parents(node_i)):
            return True
        else:
            return False

    def is_ancestor(self, descendant_node, tested_node):
        if descendant_node == tested_node:
            return True  # a node is defined to be its own ancestor

        parents_set = self.parents(descendant_node)
        if len(parents_set) == 0:
            return False  # no parents, descendant_node is a root

        if tested_node in parents_set:
            return True  # found the tested_node

        for parent in parents_set:
            if self.is_ancestor(descendant_node=parent, tested_node=tested_node):
                return True  # found tested_node to be an ancestor of one of the parents
        else:
            return False  # tested_node is not an ancestor of one of the parents

    def is_acyclic(self):
        for node in self._graph:
            parents_set = self.parents(node)
            for parent in parents_set:
                # test if a node is an ancestor of its parents
                if self.is_ancestor(descendant_node=parent, tested_node=node):
                    return False
        else:
            return True

    def is_graph_connected(self, nodes_set=None):
        # ToDo: Check correctness and improve efficiency
        if nodes_set is None:
            nodes_set = self.nodes_set

        assert len(nodes_set) > 1

        nodes_to_reach = nodes_set.copy()  # create a copy (passed by reference)
        starting_nodes = {nodes_to_reach.pop()}  # start from an arbitrary node

        while len(starting_nodes) > 0:
            node_start = starting_nodes.pop()
            parent_nodes = self.parents(node_start) & nodes_to_reach
            nodes_to_reach = nodes_to_reach - parent_nodes

            children_nodes = self.find_children(node_start, nodes_to_reach)
            nodes_to_reach = nodes_to_reach - children_nodes

            if len(nodes_to_reach) == 0:
                return True  # reach all the nodes in the graph

            starting_nodes.update(parent_nodes)
            starting_nodes.update(children_nodes)

        return False

    def get_ancestors(self, node, candidate_nodes=None):
        if candidate_nodes is None:
            candidate_nodes = self.nodes_set
        parents_set = self.parents(node) & candidate_nodes
        if len(parents_set) == 0:
            return {node}

        ancestors = set()
        for parent in parents_set:
            ancestors.update(self.get_ancestors(parent, candidate_nodes - parents_set))

        ancestors.add(node)  # a node is considered its own ancestor
        return ancestors

    def dsep(self, node_i, node_j, condition_set):
        """
        Test d-separation by following these steps:
            1. Find the ancestors of node_i, node_j, and the nodes in the condition_set
            2. moralize the sub-graph consisting of the ancestors, resulting in an undirected sub-graph
            3. test separation by blocking all the undirected paths through the condition_set
        :param node_i:
        :param node_j:
        :param condition_set:
        :return: True if the node_i and node_j are d-separated by condition_set
        """

        # 1. Find the nodes of the ancestors of node_i, node_j, and the nodes in the condition_set
        # a node is defined to be its own ancestor, thus node_i, node_j, and condition_set will be included
        ancestors = set()
        ancestors.update(self.get_ancestors(node_i))
        ancestors.update(self.get_ancestors(node_j))
        for cond_node in condition_set:
            ancestors.update(self.get_ancestors(cond_node))

        # 2. Moralize the sub-graph consisting of the ancestors, resulting in an undirected sub-graph
        moral_graph = UndirectedGraph(ancestors)  # undirected graph
        for node in ancestors:
            parents_set = self.parents(node) & ancestors
            for parent in parents_set:
                moral_graph.add_edge(parent, node)  # create undirected edges between node and its parents
            for (parent_k, parent_l) in combinations(parents_set, 2):
                if not self.is_connected(parent_k, parent_l):
                    moral_graph.add_edge(parent_k, parent_l)  # "marry" unconnected parents by an undirected graph

        # 3. Test separation by blocking all the undirected paths through the condition_set
        return not moral_graph.is_reachable(node_i, node_j, condition_set)

    def convert_to_cpdag(self, cpdag):
        """
        Convert the DAG to a CPDAG by copying the skeleton and v-structures. Then, the remaining undirected edges are
        oriented by rules R1, R2, R3.
        :param cpdag: an externally instantiated PDAG that will be filled with the result
        """
        if not self.is_acyclic():
            raise ValueError(_ErrorCyclicGraph)

        # copy skeleton
        for node in self.nodes_set:
            parents_set = self.parents(node)
            cpdag.add_edges(parents_set=parents_set, target_node=node, arrowhead_type=Mark.Undirected)

        for node in self.nodes_set:
            parents_set = self.parents(node)
            for (parent_i, parent_j) in combinations(parents_set, 2):
                if not self.is_connected(parent_i, parent_j):
                    cpdag.orient_edge(source_node=parent_i, target_node=node)  # orient v-structure
                    cpdag.orient_edge(source_node=parent_j, target_node=node)

        cpdag.maximally_orient_pattern({1, 2, 3})  # use orientation rules R1, R2, and R3

    def get_adj_mat(self, en_nodes_list=None):
        """
        Return the adjacency matrix, a numpy matrix format
        :param en_nodes_list: (optional) an ordered list of edges to which the matrix indexes will correspond.
            A partial list of graph nodes can be provided. The size of the output matrix will be num.nodes X num.nodes.
        :return: 1) Adjacency matrix, and
                 2) if no list was provided as input, also returns an ordered list of node identifiers
        """
        if en_nodes_list is None:
            nodes_sorted_list = sorted(self.nodes_set)
        else:
            assert isinstance(en_nodes_list, list)
            for node in en_nodes_list:
                assert node in self.nodes_set
            nodes_sorted_list = en_nodes_list

        num_nodes = len(nodes_sorted_list)
        adj_mat = np.zeros((num_nodes, num_nodes), dtype=int)
        node_index_map = {node: i for i, node in enumerate(nodes_sorted_list)}

        for node in nodes_sorted_list:
            parents_set = [node_index_map[n] for n in self.parents(node)]
            adj_mat[parents_set, node_index_map[node]] = 1

        # return the proper values
        if en_nodes_list is None:
            return adj_mat, nodes_sorted_list  # return both the adjacency matrix and the ordered list of nodes
        else:
            return adj_mat  # return only the adjacency matrix since the ordered list of nodes was input

    def find_topological_order(self, en_nodes=None) -> list:
        topological_groups = self.find_topological_order_groups(en_nodes)
        return [node for group in topological_groups for node in group]

    def find_topological_order_groups(self, en_nodes=None) -> list:
        if en_nodes is None:
            en_nodes = self.nodes_set

        if len(en_nodes) == 0:
            return []

        parents_set = set()
        for node in en_nodes:
            parents_set.update(self.parents(node) & en_nodes)  # update the set of nodes that are parents of someone

        leaves_set = en_nodes - parents_set  # nodes that are not parents of any endogenous node
        assert len(leaves_set) > 0  # there should be at least one leaf in an acyclic graph

        high_topological_order = self.find_topological_order_groups(parents_set)  # recursive call
        return high_topological_order + [leaves_set]

    # --- functions that modify the graph -----------------------------------------------------------------------------
    def init_from_adj_mat(self, adj_mat: np.ndarray, nodes_order: list = None):
        num_vars = adj_mat.shape[0]
        if nodes_order is not None:
            assert isinstance(nodes_order, list)
            assert num_vars == len(nodes_order)
        else:
            nodes_order = list(range(num_vars))

        self.create_empty_graph()  # delete all pre-existing edges

        parents_list, children_list = adj_mat.nonzero()

        for (parent, child) in zip(parents_list, children_list):  # convert adjacency matrix to DAG
            self.add_edges(
                parents_set={nodes_order[parent]},
                target_node=nodes_order[child]
            )

    def add_edges(self, parents_set, target_node):
        assert isinstance(parents_set, set)

        if len(parents_set - self._graph.keys()) != 0:
            raise ValueError('Parents set includes nodes that are not in the graph')

        if target_node not in self._graph:
            raise ValueError('Target node is not in the graph')

        self._graph[target_node].update(parents_set)
