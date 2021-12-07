from itertools import combinations
from .basic_equivalance_class_graph import MixedGraph
from . import arrow_head_types as Mark


class PDAG(MixedGraph):
    """
    Partially directed graph having two type of arrowheads: directed (--> node) and undirected (--- node)
    """
    def __init__(self, nodes_set):
        super().__init__(nodes_set, [Mark.Undirected, Mark.Directed])
        self.orientation_rules = {
            1: self.orient_by_rule_1,
            2: self.orient_by_rule_2,
            3: self.orient_by_rule_3,
            4: self.orient_by_rule_4
        }

    # --- graph initialization functions ------------------------------------------------------------------------------
    def create_complete_graph(self, nodes_set=None):
        super().create_complete_graph(Mark.Undirected, nodes_set)

    # --- graph query functions ---------------------------------------------------------------------------------------
    def parents(self, target_node):
        """
        Return the (directed) parents of the target node
        :param target_node: the child node
        :return: parents of target_node
        """
        return self._graph[target_node][Mark.Directed]

    def children(self, parent_node, potential_children_set):
        """
        Return the set of children by looping over potential_children_set
        :param parent_node:
        :param potential_children_set: a set from which to search for children
        :return: set of parent_node's children in potential_children_set
        """
        children_set = set()
        for node in potential_children_set:
            if parent_node in self._graph[node][Mark.Directed]:
                children_set.add(node)

        return children_set

    def undirected_neighbors(self, node):
        """
        Return neighbors connected by an un-directed edge
        :param node: a given node
        :return: neighbors of node connected by an un-directed edge
        """
        return self._graph[node][Mark.Undirected]

    def is_sink(self, sink_node, en_nodes=None):
        """
        Test if a node is a sink; that is, no (directed edges) children
        :param sink_node:
        :param en_nodes:
        :return:
        """
        if en_nodes is None:
            en_nodes = self.nodes_set

        for node in en_nodes:
            if sink_node in self.parents(node):
                return False
        else:
            return True  # completed looping over all the nodes and none of them is the sink's child

    def fan_in(self, target_node):
        """
        Return the number of arrow heads (directed and undirected) into a node
        :param target_node: a node
        :return: Fan-in of node target_node
        """
        return len(self.parents(target_node)) + len(self.undirected_neighbors(target_node))

    def get_num_edges(self):
        """
        Count the number of directed and undirected edges in the graph
        :return: Number of edges in the graph
        """
        num_edges = 0.0
        for node in self._graph:
            num_edges += len(self._graph[node][Mark.Directed])
            num_edges += 0.5*len(self._graph[node][Mark.Undirected])

        return int(num_edges)

    def is_reachable_undirected(self, source_node, target_node, en_nodes=None):
        """
        Test if there is a path, consisting of only undirected edges, from source node to target node
        :param source_node: source node
        :param target_node: target node
        :param en_nodes: nodes (a set) that are allowed to be on the path
        :return:
        """
        # Todo: remove obsolete method
        if en_nodes is None:
            en_nodes = self.nodes_set

        neighbors = self.undirected_neighbors(source_node)
        if target_node in neighbors:
            return True

        neighbors_set = neighbors & en_nodes
        past_neighbors = {source_node}
        for node_neighbor in neighbors_set:
            past_neighbors.add(node_neighbor)
            new_en_nodes = en_nodes - past_neighbors
            if self.is_reachable_undirected(node_neighbor, target_node, new_en_nodes):
                return True
        else:
            return False

    def is_reachable_any_undirected(self, source_node, target_set, en_nodes=None):
        """
        Test if there is a path, consisting of only undirected edges, from source node to any node in the target set
        :param source_node: source node
        :param target_set: target node
        :param en_nodes: nodes (a set) that are allowed to be on the path
        :return:
        """
        # Todo: remove obsolete method
        if en_nodes is None:
            en_nodes = self.nodes_set

        neighbors = self.undirected_neighbors(source_node)
        for node_neighbor in neighbors:
            if node_neighbor in target_set:
                return True

        neighbors_set = neighbors & en_nodes
        past_neighbors = {source_node}
        for node_neighbor in neighbors_set:
            past_neighbors.add(node_neighbor)
            new_en_nodes = en_nodes-past_neighbors
            if self.is_reachable_any_undirected(node_neighbor, target_set, new_en_nodes):
                return True
        else:
            return False

    def find_partial_topological_order(self, en_nodes=None):
        """
        Find a topological ordering for groups of nodes; each group consists of nodes by undirected edges
        :param en_nodes: set of endogenous nodes to find topological ordering
        :return: a list of nodes-sets' ordered by topological order
        """
        if en_nodes is None:
            en_nodes = self.nodes_set

        if len(en_nodes) == 0:
            return []

        parents_set = set()
        for node in en_nodes:
            parents_set.update(self.parents(node) & en_nodes)  # update the set of nodes that are parents of someone

        leaves_set = en_nodes - parents_set  # nodes that are not parents of any endogenous node
        if len(leaves_set) == 0:
            return [parents_set]  # couldn't distinguish between different topological orders
        else:
            high_topological_ordering = self.find_partial_topological_order(parents_set)  # recursive call
            return [leaves_set] + high_topological_ordering

    # --- functions that modify the graph -----------------------------------------------------------------------------
    def orient_edge(self, source_node, target_node):
        """
        Modify the graph by orienting an undirected edge source --- target to source --> target
        Note that the existence of an undirected edge is not tested in order to allow
        bi-directed edges (spurious association)
        :param source_node: to be a parent node
        :param target_node: to be a child node
        :return:
        """
        self._graph[target_node][Mark.Directed].add(source_node)  # add a directed arrow head
        self._graph[target_node][Mark.Undirected].discard(source_node)  # remove an undirected arrow head
        self._graph[source_node][Mark.Undirected].discard(target_node)  # remove an undirected arrow head

    def delete_directed_edge(self, source_node, target_node):
        """
        Delete a directed edge
        :param source_node:
        :param target_node:
        :return:
        """
        self._graph[target_node][Mark.Directed].discard(source_node)

    def delete_undirected_edge(self, node_i, node_j):
        """
        Delete an undirected edge
        :param node_i: 1st node
        :param node_j: 2nd node
        :return:
        """
        self._graph[node_i][Mark.Undirected].discard(node_j)
        self._graph[node_j][Mark.Undirected].discard(node_i)

    def delete_edge(self, node_i, node_j):
        self.delete_directed_edge(node_i, node_j)  # delete directed arrow head into node j
        self.delete_directed_edge(node_j, node_i)  # delete directed arrow head into node i
        self.delete_undirected_edge(node_i, node_j)  # delete undirected arrow heads

    def add_edges(self, parents_set, target_node, arrowhead_type=Mark.Undirected):
        if arrowhead_type != Mark.Undirected and arrowhead_type != Mark.Directed:
            raise ValueError

        self._graph[target_node][arrowhead_type] |= parents_set
        if arrowhead_type == Mark.Undirected:
            for parent_node in parents_set:
                self._graph[parent_node][Mark.Undirected].add(target_node)  # reverse edge

    def convert_bidirected_to_undirected(self, nodes=None):
        """
        In some cases, e.g., after orienting v-structures, an edge can be oriented in opposite directions and needs to
        be converted into an undirected edge.
        For example, A --- B --- D --- E and C is a common neighbor B --- C --- D. Then B <-> D is oriented.
        :return:
        """
        if nodes is None:
            nodes = self.nodes_set

        for node_x, node_y in combinations(nodes, 2):
            if node_y in self._graph[node_x][Mark.Directed] and node_x in self._graph[node_y][Mark.Directed]:
                self._graph[node_x][Mark.Directed].discard(node_y)  # remove directed heads
                self._graph[node_y][Mark.Directed].discard(node_x)
                self._graph[node_x][Mark.Undirected].add(node_y)  # add undirected heads
                self._graph[node_y][Mark.Undirected].add(node_x)

    def convert_to_undirected(self, nodes=None):
        if nodes is None:
            nodes = self.nodes_set

        for node_x, node_y in combinations(nodes, 2):
            if node_y in self._graph[node_x][Mark.Directed] or node_x in self._graph[node_y][Mark.Directed]:
                # remove directed heads
                if node_y in self._graph[node_x][Mark.Directed]:
                    self._graph[node_x][Mark.Directed].discard(node_y)
                if node_x in self._graph[node_y][Mark.Directed]:
                    self._graph[node_y][Mark.Directed].discard(node_x)

                self._graph[node_x][Mark.Undirected].add(node_y)  # add undirected heads
                self._graph[node_y][Mark.Undirected].add(node_x)

    def maximally_orient_pattern(self, rules_set, en_nodes=None):
        """
        Maximally orient edges starting anywhere but ending (including undirected) at the endogenous nodes.
        Orientation rules are applied iteratively until no more edges can be oriented.
        This function is generally applied after orienting all the v-structures.
        :param rules_set: a set of indexes of the orientation rules to be used
        :param en_nodes: Endogenous nodes (if none, all the graphs are assumed endogenous)
        :return:
        """
        if en_nodes is None:
            en_nodes = self.nodes_set

        graph_modified = True
        while graph_modified:
            graph_modified = False
            for rule_idx in rules_set:
                rule = self.orientation_rules[rule_idx]
                graph_modified |= rule(en_nodes)

    def orient_by_rule_1(self, en_nodes):
        """
        [R1] Orient Z --> X --- Y into Z --> X --> Y if Z and Y are not connected.
        Orient undirected edges connected to endogenous nodes.
        :param en_nodes: endogenous nodes to be tested
        :return: True if an orientation is found, False if graph is unchanged
        """
        graph_modified = False
        for node_y in en_nodes:
            x_nodes = self.undirected_neighbors(node_y).copy()  # neighbors of the current Y
            for node_x in x_nodes:  # test all undirected edges "into" Y
                for node_z in self.parents(node_x):
                    if not self.is_connected(node_z, node_y):
                        self.orient_edge(source_node=node_x, target_node=node_y)  # orient X --> Y
                        graph_modified = True
                        break  # X --> Y was oriented so stop searching through Z nodes and go to the next X --- Y

        return graph_modified

    def orient_by_rule_2(self, en_nodes):
        """
        [R2] Orient X --- Y into X --> Y if there is a directed path X --> Z --> Y (utilizing acyclic assumption).
        Orient undirected edges connected to endogenous nodes.
        :param en_nodes: endogenous nodes concicting the sub-graph to be oriented
        :return: True if an orientation is found, False if graph is unchanged
        """
        graph_modified = False
        for node_y in en_nodes:
            x_nodes = self.undirected_neighbors(node_y).copy()
            for node_x in x_nodes:
                z_nodes = self.parents(node_y)  # directed parents
                for node_z in z_nodes:
                    if node_x in self.parents(node_z):
                        self.orient_edge(source_node=node_x, target_node=node_y)
                        graph_modified = True
                        break  # X --> Y was oriented so stop searching through Z nodes and go to the next X --- Y

        return graph_modified

    def orient_by_rule_3(self, en_nodes):
        """
        [R3] Orient X --- Y into X --> Y if there exists X --- W --> Y and X --- Z --> Y, where W and Z are disconnected
        Orient undirected edges connected to endogenous nodes.
        :param en_nodes: endogenous nodes
        :return: True if an orientation is found, False if graph is unchanged
        """
        graph_modified = False
        for node_y in en_nodes:
            x_nodes = self.undirected_neighbors(node_y).copy()
            wz_nodes = self.parents(node_y)
            for node_x in x_nodes:
                wz_nodes_of_x = self.undirected_neighbors(node_x).intersection(wz_nodes)  # W,Z neighbors of X
                for node_w, node_z in combinations(wz_nodes_of_x, 2):
                    if self.is_connected(node_w, node_z):
                        continue  # skip as W and Z are connected

                    self.orient_edge(source_node=node_x, target_node=node_y)
                    graph_modified = True
                    break  # X --> Y was oriented so stop searching through Z nodes and go to the next X --- Y

        return graph_modified

    def orient_by_rule_4(self, en_nodes):
        """
        [R4] Orient X --- Y into X --> Y if W --> Z --> Y and X and Z are connected by an undirected edge,
        and W and Y are disconnected.
        Orient undirected edges connected to endogenous nodes.
        :param en_nodes: endogenous nodes
        :return: True if an orientation is found, False if graph is unchanged
        """
        graph_updated = False
        for node_y in en_nodes:
            x_nodes = self.undirected_neighbors(node_y).copy()
            z_nodes = self.parents(node_y)
            for node_x in x_nodes:
                for node_z in z_nodes:
                    if not self.is_connected(node_z, node_x):  # make sure Z and X are connected
                        continue  # skip and search for the next Z for the given X node
                    w_nodes = self.parents(node_z).intersection(self.undirected_neighbors(node_x))
                    if len(w_nodes) > 0:
                        self.orient_edge(source_node=node_x, target_node=node_y)
                        graph_updated = True
                        break

        return graph_updated

    def copy(self):
        """
        Copy graph
        :return: a PDAG copy
        """
        target_pdag = PDAG(self.nodes_set)

        for node in self.nodes_set:
            target_pdag._graph[node][Mark.Undirected] = self._graph[node][Mark.Undirected].copy()
            target_pdag._graph[node][Mark.Directed] = self._graph[node][Mark.Directed].copy()

        return target_pdag

    def delete_edges_not_present_in(self, source_pdag, nodes_set=None):
        if nodes_set is None:
            nodes_set = source_pdag.nodes_set

        for node_i, node_j in combinations(nodes_set, 2):
            if not source_pdag.is_connected(node_i, node_j):
                self.delete_edge(node_i, node_j)

    def add_edges_from(self, source_pdag, en_nodes=None, ex_nodes=None):
        """
        Copy a sub-graph from an externally defined PDAG.
        Note: the function assumes the target sub-graph is empty and does not delete any arrowheads
        :param source_pdag: An externally defined PDAG from which to copy arrowheads
        :param en_nodes: Endogenous nodes of the sub-graph to be copied
        :param ex_nodes: Exogenous nodes to the sub-graph to be copied (edges between these nodes to the endogenous)
        :return:
        """

        assert isinstance(source_pdag, PDAG)

        if en_nodes is None:
            en_nodes = self.nodes_set
        else:
            assert isinstance(en_nodes, set)

        if ex_nodes is None:
            ex_nodes = set()
        else:
            assert isinstance(ex_nodes, set)

        exen_nodes = ex_nodes | en_nodes
        for node in en_nodes:
            parents = source_pdag.parents(node) & exen_nodes
            neighbors = source_pdag.undirected_neighbors(node) & exen_nodes

            self.add_edges(parents_set=parents, target_node=node,
                           arrowhead_type=Mark.Directed)  # add directed arrowheads
            self.add_edges(parents_set=neighbors, target_node=node,
                           arrowhead_type=Mark.Undirected)  # add undirected arrowheads

    def convert_to_dag(self, dag):
        """
        Convert the PDAG to DAG using the algorithm by Dor and Tarsi, 1992
        :param dag: an externally instantiated DAG that will be filled with the result (None if no DAG extension exist)
        """

        def select_node(a_nodes_1, cpdag_1):
            for node_x1 in a_nodes_1:
                if cpdag_1.is_sink(node_x1):
                    x_adjacent = cpdag_1.undirected_neighbors(node_x1) | cpdag_1.parents(node_x1)
                    y_nodes1 = cpdag_1.undirected_neighbors(node_x1)

                    # all the undirected neighbors should be connected to all the adjacencies of x
                    for node_y1 in y_nodes1:
                        for node_adj_x in x_adjacent-{node_y1}:
                            if not cpdag_1.is_connected(node_y1, node_adj_x):
                                break
                        else:
                            # node_y1 is connected to all adjacent nodes; continue to the next node_y1
                            continue

                        break  # second break, initiated by the inner loop break
                    else:
                        # completed looping through all the y_nodes1, and all of them are connected to x adjacent nodes
                        return node_x1, y_nodes1
            else:
                # PDAG does not admit any extension
                return None, None

        cpdag_a = self.copy()
        a_nodes = cpdag_a.nodes_set.copy()

        # copy directed edges to the DAG
        for node in self.nodes_set:
            dag.add_edges(parents_set=self.parents(node), target_node=node)

        # "orient edges" by creating directed edges in the DAG
        while not cpdag_a.is_empty():
            (node_x, y_nodes) = select_node(a_nodes, cpdag_a)
            if node_x is None:
                # PDAG does not admit any DAG extension
                return False

            # add oriented edges to DAG
            dag.add_edges(parents_set=y_nodes, target_node=node_x)

            # disconnect node_x from all of its neighbors in cpdag_a (it doesn't have children in this graph)
            for node_y in y_nodes.copy():
                cpdag_a.delete_undirected_edge(node_y, node_x)
            for parent in cpdag_a.parents(node_x).copy():
                cpdag_a.delete_directed_edge(source_node=parent, target_node=node_x)

            a_nodes.discard(node_x)  # node_x is not to be considered again
        return True