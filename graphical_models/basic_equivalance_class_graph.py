import numpy as np
from itertools import combinations
from .undirected_graph import UndirectedGraph


class MixedGraph:
    """
    A graph for representing equivalence classes such as CPDAG and PAG
    """
    def __init__(self, nodes_set, edge_mark_types):
        assert isinstance(nodes_set, set)

        self.edge_mark_types = set(edge_mark_types)

        self._graph = dict()
        self.nodes_set = nodes_set
        self.create_empty_graph(self.nodes_set)

    # graph initialization functions ----------------------------------------------------------------------------------
    def create_empty_graph(self, nodes_set=None):
        if nodes_set is None:
            nodes_set = self.nodes_set
        else:
            assert isinstance(nodes_set, set)

        for node in nodes_set:
            self._graph[node] = dict()
            for head_type in self.edge_mark_types:  # loop over arrow head_types
                self._graph[node][head_type] = set()

    def create_complete_graph(self, edge_mark, nodes_set=None):
        if nodes_set is None:
            nodes_set = self.nodes_set
        else:
            assert isinstance(nodes_set, set)

        self.create_empty_graph(nodes_set)  # first, clear all arrow-heads

        for node in nodes_set:
            self._graph[node][edge_mark] = nodes_set - {node}  # connect all nodes into the current node

    # --- graph query functions ---------------------------------------------------------------------------------------
    def is_empty(self, nodes_set=None):
        """
        Test if the graph is empty
        :return: True if the graph is empty; Flase if there exist at least one edge
        """
        if nodes_set is None:
            nodes_set = self.nodes_set

        for node in nodes_set:
            for edge_mark in self.edge_mark_types:
                if self._graph[node][edge_mark]:
                    return False  # an edge was found, graph is not empty
        else:
            return True  # completed looping over all the nodes and didn't find an edge

    def number_of_edges(self, return_missing=False):
        num_edges = 0
        missing_edges = 0
        for node_i, node_j in combinations(self.nodes_set, 2):
            if self.is_connected(node_i, node_j):
                num_edges += 1
            else:
                missing_edges += 1

        if return_missing:
            return num_edges, missing_edges
        else:
            return num_edges

    def is_any_edge_mark(self, node_source, node_target):
        """
        Test if there is any edge-mark at "node_target" on the edge between node_source and node_target
        :param node_source:
        :param node_target:
        :return: True if the is some edge-mark, False otherwise (no edge-mark; not to be confused with undirected-mark)
        """
        for edge_mark in self.edge_mark_types:  # test all edge marks
            if node_source in self._graph[node_target][edge_mark]:
                return True
        else:
            return False

    def get_edge_mark(self, node_parent, node_child):
        for edge_mark in self.edge_mark_types:  # test all edge marks
            if node_parent in self._graph[node_child][edge_mark]:
                return edge_mark
        else:
            return None

    def is_connected(self, node_i, node_j):
        """
        Test if two nodes are adjacent in the graph. That is, if they are connected by any edge type.
        :param node_i:
        :param node_j:
        :return: True if the nodes are adjacent; otherwise, False
        """
        assert node_i != node_j

        for (node_p, node_c) in [(node_i, node_j), (node_j, node_i)]:  # switch roles "parent"-"child"
            for edge_mark in self.edge_mark_types:  # test all edge marks
                if node_p in self._graph[node_c][edge_mark]:
                    return True

        return False

    def is_edge(self, node_i, node_j, edge_mark_at_i, edge_mark_at_j):
        """
        Test the esistance of an edge with the given edge-marks.
        :param node_i:
        :param node_j:
        :param edge_mark_at_i:
        :param edge_mark_at_j:
        :return: True if the specific edge exists; otherwise, False.
        """
        assert (edge_mark_at_i in self.edge_mark_types) and (edge_mark_at_j in self.edge_mark_types)

        if node_j in self._graph[node_i][edge_mark_at_i] and node_i in self._graph[node_j][edge_mark_at_j]:
            return True
        else:
            return False

    def is_graph_connected(self, nodes_set=None):
        # ToDo: Check correctness
        if nodes_set is None:
            nodes_set = self.nodes_set

        assert len(nodes_set) > 1

        nodes_to_reach = nodes_set.copy()  # create a copy (passed by reference)
        starting_nodes = {nodes_to_reach.pop()}  # start from an arbitrary node

        while len(starting_nodes) > 0:
            node_start = starting_nodes.pop()
            adjacent_nodes = self.find_adjacent_nodes(node_start, nodes_to_reach)
            nodes_to_reach = nodes_to_reach - adjacent_nodes
            if len(nodes_to_reach) == 0:
                return True  # reach all the nodes in the graph
            starting_nodes.update(adjacent_nodes)

        return False

    def find_adjacent_nodes(self, node_i, pool_nodes=None, edge_type=None):
        """
        Find all the nodes that are connected in/out of node_i.
        :param node_i:
        :param pool_nodes: a set of nodes from which to find the adjacent ones (default: all graph nodes)
        :param edge_type: a tuples: (alpha, beta) defining the allowed connecting edge,
            where alpha is the edge-mark at node_i and beta is the edge-mark at the neighbors.
            Default is None indicating that any edge-mark is allowed.
        :return:
        """
        if edge_type is None:
            connected_nodes = set()
            for edge_mark in self.edge_mark_types:
                connected_nodes.update(self._graph[node_i][edge_mark])
        else:
            mark_origin = edge_type[0]
            mark_neighbor = edge_type[1]
            connected_nodes = set(filter(
                lambda neighbor: node_i in self._graph[neighbor][mark_neighbor],
                self._graph[node_i][mark_origin]
            ))

        if pool_nodes is not None:
            connected_nodes = connected_nodes & pool_nodes
        return connected_nodes

    def find_reachable_set(self, anchor_node, nodes_pool, edge_type_list):
        """
        Find the set of nodes that are reachable from a node via specific edge-types
        :param anchor_node: A node from which to start reaching
        :param nodes_pool: a set of nodes tested to be reachable
        :param edge_type_list: a list of edge types, e.g., [('<--', '---'), ('<--', '-->')]
        :return: a set of nodes that are reachable from the anchor node
        """
        neighbors_set = set()

        if len(nodes_pool) == 0:
            return neighbors_set

        # find immediate reachable neighbors
        if edge_type_list is None:  # any edge type
            neighbors_set = self.find_adjacent_nodes(anchor_node, nodes_pool, None)
        else:
            for edge_type in edge_type_list:  # specific edge types
                neighbors_set.update(self.find_adjacent_nodes(anchor_node, nodes_pool, edge_type))

        if len(neighbors_set) == 0:
            return neighbors_set

        reachable_set = neighbors_set.copy()
        updated_nodes_pool = nodes_pool - neighbors_set

        for neighbor in neighbors_set:
            neighbor_reach = self.find_reachable_set(neighbor, updated_nodes_pool, edge_type_list)
            reachable_set.update(neighbor_reach)
            updated_nodes_pool.difference_update(neighbor_reach)  # remove neighbor_reach from the pool

        return reachable_set

    def find_unconnected_subgraphs(self, en_nodes=None, sym_edge_mark=None) -> list:
        """
        Find groups of nodes that belong to unconnected sub-graphs (connected component)
        :param en_nodes: Nodes that belong to the (unconnected) graph that need to be clustered
        :param sym_edge_mark: the type of symmetric edges that defines connectivity has the provided edges-mark,
            e.g., Mark.Directed guides the search to consider only bi-directed edges as connectivity.
            Note that if you provide an edge-mark, only symmetric edges are considered, in contrast to the None default
            Default: None, means that any edge qualifies as connection (not just symmetric ones).
        :return: disjoint subsets of en_nodes that belong to distinct sub-graphs (connected components)
        """
        if en_nodes is None:
            en_nodes = self.nodes_set

        connected_sets = []
        nodes = en_nodes.copy()

        edge_type_list = None
        if sym_edge_mark in self.edge_mark_types:
            edge_type_list = [(sym_edge_mark, sym_edge_mark)]

        while len(nodes) > 0:
            node_i = nodes.pop()
            reachable_set = self.find_reachable_set(node_i, nodes, edge_type_list)
            nodes.difference_update(reachable_set)
            reachable_set.add(node_i)
            connected_sets.append(reachable_set)

        return connected_sets

    def get_skeleton_graph(self, en_nodes=None) -> UndirectedGraph:
        if en_nodes is None:
            en_nodes = self.nodes_set

        adj_graph = UndirectedGraph(en_nodes.copy())
        for node_i, node_j in combinations(en_nodes, 2):
            if self.is_connected(node_i, node_j):
                adj_graph.add_edge(node_i, node_j)
        return adj_graph

    # --- graph modification functions --------------------------------------------------------------------------------
    def delete_edge(self, node_i, node_j):
        for edge_mark in self.edge_mark_types:  # loop through all edge marks
            self._graph[node_i][edge_mark].discard(node_j)
            self._graph[node_j][edge_mark].discard(node_i)

    def replace_edge_mark(self, node_source, node_target, requested_edge_mark):
        assert requested_edge_mark in self.edge_mark_types

        # remove any edge-mark
        for edge_mark in self.edge_mark_types:
            self._graph[node_target][edge_mark].discard(node_source)

        # set requested edge-mark
        self._graph[node_target][requested_edge_mark].add(node_source)

    def reset_orientations(self, default_mark, nodes_set=None):
        """
        Reset all orientations, e.g., convert all edges into o--o edges, where "o" is the default edge-mark
        :param default_mark: an edge-mark to place the instead of the existing edge_marks
        :param nodes_set: Only edges between pairs of nodes from this set will be converted (default: all edges)
        :return:
        """
        assert default_mark in self.edge_mark_types
        if nodes_set is None:
            nodes_set = self.nodes_set

        for (node_x, node_y) in combinations(nodes_set, 2):
            if self.is_connected(node_x, node_y):
                self.replace_edge_mark(node_x, node_y, default_mark)
                self.replace_edge_mark(node_y, node_x, default_mark)

    def add_edge(self, node_i, node_j, edge_mark_at_i, edge_mark_at_j):
        """
        Add an edge with the requested edge-marks.
        :param node_i:
        :param node_j:
        :param edge_mark_at_i:
        :param edge_mark_at_j:
        :return:
        """

        assert not self.is_connected(node_i, node_j)  # edge already exists
        assert (edge_mark_at_i in self.edge_mark_types) and (edge_mark_at_j in self.edge_mark_types)

        self._graph[node_i][edge_mark_at_i].add(node_j)
        self._graph[node_j][edge_mark_at_j].add(node_i)

    def get_skeleton_mat(self):
        """
        Return the adjacency matrix of the graph skeleton, a square numpy matrix format.
        :return:
        """
        num_nodes = len(self.nodes_set)
        adj_mat = np.zeros((num_nodes, num_nodes), dtype=np.int)
        node_index_map = {node: i for i, node in enumerate(sorted(list(self.nodes_set)))}

        for node in self._graph:
            for edge_mark in self.edge_mark_types:  # test all edge marks
                for node_p in self._graph[node][edge_mark]:
                    adj_mat[node_index_map[node_p]][node_index_map[node]] = 1

        return adj_mat

    # --- plotting tools ----------------------------------------------------------------------------------------------
    def __str__(self):
        text_print = 'Edge-marks on the graph edges:\n'
        for node in self.nodes_set:
            for edge_mark in self.edge_mark_types:
                if len(self._graph[node][edge_mark]) > 0:
                    text_print += ('Edges: ' + str(node) + ' ' + edge_mark + '*' +
                                   ' ' + str(self._graph[node][edge_mark]) + '\n')
        return text_print
