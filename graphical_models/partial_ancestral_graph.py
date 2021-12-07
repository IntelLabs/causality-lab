from causal_discovery_utils import constraint_based
from .basic_equivalance_class_graph import MixedGraph
from .partially_dag import PDAG
from . import arrow_head_types as Mark
from itertools import combinations


class PAG(MixedGraph):
    """
    Partial Ancestral Graph. It has three arrow-head/edge-mark types: 'circle', 'undirected', and 'directed',
    and six edge types: o--o, o---, o-->, --->, <-->, and ----
    """
    def __init__(self, nodes_set):
        super().__init__(nodes_set, [Mark.Circle, Mark.Directed, Mark.Tail])
        self.sepset = constraint_based.SeparationSet(nodes_set)
        self.visible_edges = None  # a set of visible edges, where each element is a tuple: (parent, child)
        self.orientation_rules = {
            1: self.orient_by_rule_1,
            2: self.orient_by_rule_2,
            3: self.orient_by_rule_3,
            4: self.orient_by_rule_4,
            5: self.orient_by_rule_5,  # when selection bias may be present
            6: self.orient_by_rule_6,  # when selection bias may be present
            7: self.orient_by_rule_7,  # when selection bias may be present
            8: self.orient_by_rule_8,  # required for tail-completeness
            9: self.orient_by_rule_9,  # required for tail-completeness
            10: self.orient_by_rule_10,  # required for tail-completeness
        }

    def fan_in(self, target_node):
        """
        Return the number of arrow heads <--* and o--* into a node. Do not count tails (undirected)
        :param target_node: a node
        :return: Fan-in of node target_node
        """
        return len(self._graph[target_node][Mark.Directed]) + len(self._graph[target_node][Mark.Circle])

    def is_collider(self, node_middle, node_x, node_y):
        """
        Test if X *--> middle-node <--* Y, that is, test if middle-node is a collider.
        :param node_middle:
        :param node_x:
        :param node_y:
        :return: True if the middle node is a collider
        """
        pot_parents = self._graph[node_middle][Mark.Directed]
        if (node_x in pot_parents) and (node_y in pot_parents):
            return True
        else:
            return False

    def is_possible_collider(self, node_x, node_middle, node_y):
        """
        Test if node_2 is a possible collider in a completed version of this PAG.
        The given triplet should be either a triangle or the middle node should be a (definite) collider.
        This method is used, for example, to test if node_1 *--* node_2 *--* node_3 is a sub-path of a PDS-path.

        :param node_x: first node
        :param node_middle: middle (second) node
        :param node_y: third node
        :return: 'True' if the middle node is a possible collider
        """
        if node_x == node_y or \
                (not self.is_connected(node_x, node_middle)) or \
                (not self.is_connected(node_middle, node_y)):  # make sure there is a path: X *--* Middle *--* Y
            return False
        return self.is_connected(node_x, node_y) or \
               self.is_collider(node_middle=node_middle, node_x=node_x, node_y=node_y)

    def is_possible_parent(self, potential_parent_node, child_node):
        """
        Test if a node can possibly serve as parent of the given node.
        Make sure that on the connecting edge
            (a) there is no head edge-mark (->) at the tested node and
            (b) there is no tail edge-mark (--) at the given node,
        where variant edge-marks (o) are allowed.
        :param potential_parent_node: the node that is being tested
        :param child_node: the node that serves as the child
        :return:
        """
        if potential_parent_node == child_node:
            return False
        if not self.is_connected(potential_parent_node, child_node):
            return False

        if ((potential_parent_node in self._graph[child_node][Mark.Tail]) or
                (child_node in self._graph[potential_parent_node][Mark.Directed])):
            return False
        else:
            return True

        # if ((potential_parent_node in self._graph[child_node][Mark.Directed] or
        #      potential_parent_node in self._graph[child_node][Mark.Circle]) and
        #     (child_node in self._graph[potential_parent_node][Mark.Tail] or
        #      child_node in self._graph[potential_parent_node][Mark.Circle])):
        #     return True
        # else:
        #     return False

    def find_possible_children(self, parent_node, en_nodes=None):
        if en_nodes is None:
            en_nodes = self.nodes_set - {parent_node}
        potential_child_nodes = set()
        for potential_node in en_nodes:
            if self.is_possible_parent(potential_parent_node=parent_node, child_node=potential_node):
                potential_child_nodes.add(potential_node)

        return potential_child_nodes

    def find_possible_parents(self, child_node, en_nodes=None):
        if en_nodes is None:
            en_nodes = self.nodes_set - {child_node}

        possible_parents = {parent_node for parent_node in en_nodes if self.is_possible_parent(parent_node, child_node)}
        return possible_parents

    def is_parent(self, node_parent, node_child):
        """
        Test if a node is a parent of another parent: that is, there is a directed edge: node_source ---> node_target
        :param node_parent:
        :param node_child:
        :return: True if the relation exists in the graph; otherwise, False
        """
        return ((node_parent in self._graph[node_child][Mark.Directed]) and
                (node_child in self._graph[node_parent][Mark.Tail]))

    def find_parents(self, node_child):
        """
        Find the set of parents, oriented edges parent ---> child
        :param node_child:
        :return:
        """
        parent_nodes = set()
        potential_parents = self._graph[node_child][Mark.Directed]
        for potential_node in potential_parents:
            if node_child in self._graph[potential_node][Mark.Tail]:  # should have a tail at the parent
                parent_nodes.add(potential_node)
        return parent_nodes

    def find_spouses(self, node):
        """
        Find nodes connected with bi-directed edges to the node
        :param node: the node for which spouses are searched
        :return: A set of nodes connected to the node by bi-directed edges
        """
        spouses = set(filter(
            lambda possible_spouse: possible_spouse in self._graph[node][Mark.Directed] and
                                    node in self._graph[possible_spouse][Mark.Directed],
            self.nodes_set
        ))
        return spouses

    def find_visible_edges(self):
        """
        Test every edge in the graph if it is visible. Store the visible edges (parent,child) in visible_edges.
        :return:
        """
        self.visible_edges = set()
        for child in self.nodes_set:
            for parent in self._graph[child][Mark.Directed]:
                if self._test_edge_visible(parent, child):
                    self.visible_edges.add((parent, child))

    def is_edge_visible(self, parent, child):
        """
        Test is a directed edge parent ---> child (specificaly, with edge-marks "tail" and "head") is visible.
        If visibility of edges was not established previously, then find_visible_edges is called first.
        An edge is visible if every DAG in the equivalence class, there is no inducing path between A and B,
        relative to the latent variables, that is into A.
        A graphical criterion from Defenition 8, "Casual Reasoning with Ancestral Graphs", Zhang, JMLR 2008,
        is used to determine visibility.
        :param parent:
        :param child:
        :return:
        """
        assert self.visible_edges is not None  # need to call self.find_visible_edges() first

        if (parent, child) in self.visible_edges:
            return True
        else:
            return False

    def _test_edge_visible(self, node_a, node_b):
        """
        Test if a directed edge A ---> B is "visible". That is, test if every DAG in the equivalence class, there is
        no inducing path between A and B, relative to the latent variables, that is into A.
        A graphical criterion from Defenition 8, "Casual Reasoning with Ancestral Graphs", Zhang, JMLR 2008.
        :param node_a: parent node
        :param node_b: child node
        :return: True if the edge is directed A ---> B and visible
        """

        # make sure the tested edge is: A ---> B
        if not self.is_parent(node_parent=node_a, node_child=node_b):
            return False

        # discard nodes that are connected to node B
        c_nodes = set(filter(
            lambda node: not self.is_connected(node_b, node),  # keep only nodes disconnected from B
            self.nodes_set - {node_b}
        ))
        c_nodes.discard(node_a)
        c_nodes.discard(node_b)

        # test condition 1: C *--> A ---> B and C & B are disconnected
        for node_c in c_nodes:
            if node_c in self._graph[node_a][Mark.Directed]:
                return True

        # test condition 2: there is a collider path between C and A, into A, and every node in between is a parent of B

        # the first node in the collider path must be connected by a bi-directed edge to A
        # This is because the path should be into A and the node needs to be a collider
        spouse_set = self.find_spouses(node_a)

        # every node on the collider path (except the end-points) must be a parent of B
        d_nodes = set(filter(
            lambda node: self.is_parent(node_parent=node, node_child=node_b),
            spouse_set
        ))

        for node_d in d_nodes:
            node_c = self.find_discriminating_path_to_triplet(node_d, node_a, node_b)
            if node_c is not None:
                return True  # found a path, the edge is "visible"
        return False  # no path was found, the edge is invisible

    def find_uncovered_path(self, node_x, node_y, neighbor_x, neighbor_y, en_nodes=None, edge_condition=None):
        """
        Find a path <X, Neighbor_X, ..., Neighbor_Y, Y> such that for every three consecutive nodes <V, U, W>,
        V and W are disconnected.
        In general, the shortest path, excluding the end-points, has length 2 (a total of 4 nodes path);
        however, this function treats a special case when neighbor_x == neighbor_y by immediately (!) returning the path
        [neighbor_X] without (!) testing if node_x and node_y are disconnected, or testing edge_condition.
        :param node_x: one end of the path
        :param node_y: second end of the path
        :param neighbor_x: a node that must be considered as the start of the path: <A, neighbor_a, ..., B>.
        :param neighbor_y: a node that must be considered as the end of the path: <A, ..., neighbor_b, B>.
        :param en_nodes: nodes from which to construct the path
        :param edge_condition: a condition that restricts the relation between two consecutive nodes on the path.
            Note that it does not test the relation between X & Y and their given neighbors, neighbor_x & neighbor_y.
            Example 1: an uncovered circle path:
                edge_condition = lambda in1, in2: pag.is_edge(in1, in2, Mark.Circle, Mark.Circle)
            Example 2: a possible (potentially) directed uncovdered path:
                edge_condition = lambda in1, in2: pag.is_possible_parent(in1, in2)
        :return: path nodes excluding the end-points
        """
        # ToDo: needs to be thoroughly debuged
        if neighbor_x == neighbor_y:
            return [neighbor_x]

        if edge_condition is None:
            edge_condition = self.is_connected

        if en_nodes is None:
            en_nodes = self.nodes_set
        en_nodes = en_nodes - {node_x, node_y, neighbor_x, neighbor_y}

        # Exit condition of the recursion: a trivial uncovered path
        if edge_condition(neighbor_x, neighbor_y) and \
                (not self.is_connected(neighbor_x, node_y)) and \
                (not self.is_connected(node_x, neighbor_y)):
            return [neighbor_x, neighbor_y]  # found a trivial path

        # Find path extensions: node_x --- neighbor_x --- node_c
        # s.t. neighbor_x --- node_c is a qualifying edge and node_x is disconnected from node_c
        c_nodes = {tested_node for tested_node in self.find_adjacent_nodes(neighbor_x, en_nodes)
                   if edge_condition(neighbor_x, tested_node) and not self.is_connected(node_x, tested_node)}

        for node_c in c_nodes:
            path = self.find_uncovered_path(node_x=neighbor_x, node_y=node_y,
                                            neighbor_x=node_c, neighbor_y=neighbor_y,
                                            en_nodes=en_nodes, edge_condition=edge_condition)
            if path is not None:
                return [neighbor_x, *path]
        else:
            return None

    def find_discriminating_path_to_triplet(self, node_a, node_b, node_c, nodes_set=None):
        """
        Find a discriminating path from some node (denoted D) to node C for node B.
        That is, D *--> ? <--> ... <--> A <--* B *--* C
        :param node_a:
        :param node_b:
        :param node_c:
        :param nodes_set:
        :return: Path source node (node D)
        """
        if nodes_set is None:
            nodes_set = self.nodes_set - {node_a, node_b, node_c}  # create a copy

        # assumed: A <--* B *--* C or a path from A to B with all colliders & parents of C; and A ---> C,
        # we need to find D such that D *--> A, D and C are disjoint
        d_nodes = (self._graph[node_a][Mark.Directed] - {node_a, node_b, node_c}) & nodes_set
        new_a_nodes = set()
        for node_d in d_nodes:
            if not self.is_connected(node_d, node_c):  # found a discriminating path from D to C for B, (D, A, B, C)
                return node_d  # the source of the path
            else:
                # if D ---> C and D <--> A, then D becomes the "A" node in the new search triplet
                if self.is_parent(node_d, node_c) and node_a in self._graph[node_d][Mark.Directed]:
                    new_a_nodes.add(node_d)

        # didn't find a minimal discriminating path (containing three edges). Search with the new "A" nodes
        # we have D nodes that are part of the path D *--> A <--* B *--* C and D ---> C and A ---> C
        for new_node_a in new_a_nodes:
            node_d = self.find_discriminating_path_to_triplet(new_node_a, node_b, node_c, nodes_set - {new_node_a})
            if node_d is not None:
                return node_d

        return None  # didn't find a discriminating path

    def is_reachable_any_edgemark(self, source_node, target_node, en_nodes=None):
        """
        Test if there is an undirected path (path ignoring edgemarks) between source and target_node
        :param source_node: a node from which to start
        :param target_node: a node that is searched
        :param en_nodes: set of nodes. Paths are searched only through these nodes.
        :return: True if a path is found, False otherwise
        """
        if en_nodes is None:
            en_nodes = self.nodes_set

        # (exit condition of the recursion)
        # test if the target is in the neighborhood
        neighbors = self.find_adjacent_nodes(source_node, en_nodes)  # find the neighbors from the en_nodes set
        if target_node in neighbors:
            return True

        # recursively test if there is a path from any neighbor to the target node
        past_neighbors = {source_node}
        for node_neighbor in neighbors:
            past_neighbors.add(node_neighbor)
            new_en_nodes = en_nodes - past_neighbors
            if self.is_reachable_any_edgemark(node_neighbor, target_node, new_en_nodes):
                return True
        else:
            return False

    def is_possible_ancestor(self, ancestor_node, descendant_node, en_nodes=None):
        '''
        Test if one node is a possible ancestor of second node.
        This relation is true if there is a path <node_1, X(0), X(1), ..., X(n), node_2> such that for i=1,...,n-1,
        X(i) is a possible parent of X(i+1). Recall that X(i) is a possible parent of X(i+1) if on the edge between them
        there is no head-mark at X(i) and no tail-mark at X(i+1).
        :param ancestor_node:
        :param descendant_node:
        :param en_nodes: considers paths that are consisted of these nodes. That is, a sug-graph is considered.
        :return: True if the relation exists in the graph; otherwise, False
        '''
        if en_nodes is None:
            en_nodes = self.nodes_set

        # (exit condition of the recursion)
        # test if the descendant_node is a child of ancestor_node
        possible_children = self.find_possible_children(ancestor_node, en_nodes)
        if descendant_node in possible_children:
            return True

        # recursively test if there is a potentially directed path from any potential child to the descendant node
        new_en_nodes = en_nodes - {ancestor_node}
        for node_child in possible_children:
            new_en_nodes.discard(node_child)
            if self.is_possible_ancestor(node_child, descendant_node, new_en_nodes):
                return True
        else:
            return False

    def is_set_possible_sink(self, nodes_set, ex_nodes=None):
        """
        Test if a set of nodes have all arrow-heads into it from nodes external to the set
        :param nodes_set: tested set; set of interest
        :param ex_nodes: nodes external (exogenous) to the tested set
        :return: True if there is at least one arrow-head at any external node, for an edge from the tested set.
        """
        if ex_nodes is None:
            ex_nodes = self.nodes_set - nodes_set

        for node in nodes_set:
            if (not self._graph[node][Mark.Tail].isdisjoint(ex_nodes)) or \
                    (not self._graph[node][Mark.Circle].isdisjoint(ex_nodes)):
                return False
        else:
            return True

    def which_reachable_any_edgemark(self, source_set, target_node, en_nodes=None):
        """
        Find a subset of nodes that can reach the target though an undirected path (ignoring edgemarks)
        :param source_set: Set of possible sources. A subset of this set is returned
        :param target_node: Target node to be reached
        :param en_nodes: set of nodes. Paths are searched only through these nodes.
        :return: A subset of the source set containing nodes that have a path to the target
        """
        reach_set = set()  # initialize the subset of nodes that can reach the target
        for node_source in source_set:
            if self.is_reachable_any_edgemark(node_source, target_node, en_nodes):
                reach_set.add(node_source)

        return reach_set

    def find_possible_ancestors(self, descendants_set, en_nodes=None):
        """
        Find the set of possible ancestors of a set of nodes. Note: the input nodes are included in the result
        :param descendants_set: the nodes for which we find the ancestors
        :param en_nodes: the nodes from which to search possible ancestors (sub-graph)
        :return: set of possible ancestors (including the input nodes set)
        """
        # Todo: implement more efficiently
        assert isinstance(descendants_set, (list, tuple, set))  # make sure a set is received and not a single element
        if en_nodes is None:
            en_nodes = self.nodes_set

        ancestors = set(descendants_set)
        for node in descendants_set:  # find all possible ancestors for node
            candidates = en_nodes - ancestors
            for pos_ancestor in candidates:  # search possible ancestors from the nodes that weren't already identified
                if self.is_possible_ancestor(ancestor_node=pos_ancestor, descendant_node=node, en_nodes=en_nodes):
                    ancestors.add(pos_ancestor)
        return ancestors

    def is_m_separated(self, node_x, node_y, condition_set: set):
        """
        Test if X and Y are m-separated given set Z.

        The algorithm is m*-separation (Richardson and Spirtes, 2002).
        1. Find the anterior nodes of X, Y, and Z.
        2. In an empty undirected graph connect every pair of nodes that are collider-connected.
           We apply the following steps:
            2.a. Initialize a moral graph using the graph skeleton
            2.b. Find dc-components
            2.c. Augment each dc-component by adding nodes that have incoming edges ( ---> or o---> edges).
                 Note that the augmented components may overlap. Consider F <--> E <--- A ---> B <--> C. A is shared.
            2.c. In the moral graph connect pair of nodes that are within the same augmented-dc-component
        3. Test separation by blocking all the undirected paths through the condition_set.
        :param node_x:
        :param node_y:
        :param condition_set:
        :return: True if X and Y are m-separated (independent) given Z
        """

        # 1. Find the anterior nodes of X, Y, and Z
        anterior_set = set()
        anterior_set.update(self.find_possible_ancestors({node_x}))
        anterior_set.update(self.find_possible_ancestors({node_y}))
        for node_z in condition_set:
            anterior_set.update(self.find_possible_ancestors({node_z}))

        # 2. In an empty undirected graph connect every pair of nodes that are collider-connected
        moral_graph = self.get_adjacency_graph(anterior_set)  # 2.a. initialize a moral graph using the graph skeleton
        components_list = self.find_definite_c_components(anterior_set)  # 2.b. find dc-components
        for dc_comp in components_list:  # 2.c. augment each dc-component by adding nodes that have incoming edges
            augmented_nodes = set()
            for dc_node in dc_comp:
                augmented_nodes.update(filter(
                    lambda node: node in self._graph[dc_node][Mark.Directed],
                    anterior_set - dc_comp
                ))
            dc_comp.update(augmented_nodes)
        for component in components_list:  # 2.c. connect pair of nodes that are within the same augmented-dc-component
            for node_i, node_j in combinations(component, 2):
                moral_graph.add_edge(node_i, node_j)

        # 3. Test separation by blocking all the undirected paths through the condition_set
        return not moral_graph.is_reachable(node_x, node_y, condition_set)

    def find_definite_c_components(self, en_nodes=None) -> list:
        """
        Find dc-components (definite c-components) in the PAG.
        Essentially, connected components when considering only bi-directed edges.
        See "A Graphical Criterion for Effect Identification in Equivalence Classes of Causal Diagrams",
        (Jaber et al., 2018).
        :param en_nodes:
        :return: a list of dc-components
        """
        if en_nodes is None:
            en_nodes = self.nodes_set

        dc_components = self.find_unconnected_subgraphs(en_nodes, sym_edge_mark=Mark.Directed)
        return dc_components

    def find_union_pc_components_for_node(self, node_x, en_nodes=None, dc_components=None) -> set:
        """
        Given node, X, find all the nodes that are in some possible c-component (pc-component) with X.
        See "A Graphical Criterion for Effect Identification in Equivalence Classes of Causal Diagrams",
        (Jaber et al., 2018).

        Implemented procedure:
        1. Find invisible_neighbors
        2. Find invisible_children: node adjacent to X such that there is an arrow-head at them and
                                    the edge is not visible (invisible)
        3. for each invisible_child, Y, find the dc-component.
            Now we have X *--> Y <--> ... dc-component nodes ... <--->
        4. For each node in the dc-component, Z, find nodes, W, such that Z <--* W
        Essentially, include
            a. W-nodes,
            b. Z-nodes,
            c. Y-nodes, and
            d. invisible_neighbors
        in the pc-component of X.

        :param node_x: the node for which we want to find the pc-component
        :param en_nodes: nodes of the sub-graph in which to operate
        :param dc_components: optional. if not provided, it is calculated each time this method is called
        :return: a set of nodes that are in the same pc-component with node_x
        """
        assert self.visible_edges is not None  # need to call self.find_visible_edges() first

        if en_nodes is None:  # is a sub-graph defined? if not, use the entire graph
            en_nodes = self.nodes_set

        # 1. Find invisible neighbors of x: X *--* neighbor, such that the edge is not marked visible
        invisible_neighbors = set(filter(
            lambda neighbor: (not self.is_edge_visible(parent=node_x, child=neighbor) and
                              not self.is_edge_visible(parent=neighbor, child=node_x)),
            self.find_adjacent_nodes(node_x, pool_nodes=en_nodes)  # all the neighbors of node X
        ))

        # 2. Find invisible children of X: X * --> Y, where the edge is not visible
        invisible_children = set(filter(
            lambda node_y_test: node_x in self._graph[node_y_test][Mark.Directed],  # X *--> Y (Y should be a collider)
            invisible_neighbors
        ))
        pc_component = set()
        pc_component.update(invisible_children)

        # 3. For each invisible_child, Y, find the dc-component
        nodes_valid_dc = set()  # Z nodes. Nodes that are connected by bi-directed edges to invisible children of X
        # Todo: accept global dc-components and modify (possibly split) its components that include exogenous nodes)
        # find dc-components (nodes connected by di-directed edges in the sub-graph)
        if dc_components is None:
            dc_components = self.find_definite_c_components(en_nodes)
        for node_y in invisible_children:
            for dc_comp in dc_components:
                if node_y in dc_comp:  # found a dc-component of node_y, add it to the pc-component of X
                    nodes_valid_dc.update(dc_comp)
                    break  # a node cannot appear in more than one dc-component

        pc_component.update(nodes_valid_dc)

        # 4. For each node in the dc-component, Z, find nodes, W, such that Z <--* W
        nodes_valid_dc.discard(node_x)
        remaining_nodes = en_nodes - pc_component  # nodes that are not in the updated pc-component
        invisible_parents = set()
        for node_z in nodes_valid_dc:
            invisible_parents.update(filter(
                lambda node_w: (node_w in self._graph[node_z][Mark.Directed] and  # Z <--* W
                                not self.is_edge_visible(node_w, node_y)),  # the edge should not be visible
                remaining_nodes
            ))
        pc_component.update(invisible_parents)
        pc_component.update(invisible_neighbors)
        pc_component.add(node_x)
        return pc_component

    def find_union_pc_components_for_set(self, set_x, en_nodes=None) -> set:
        """
        Given set, X, find all the nodes that are in some possible c-component (pc-component) with any node in X.
        See "A Graphical Criterion for Effect Identification in Equivalence Classes of Causal Diagrams",
        (Jaber et al., 2018).

        :param set_x: the set for which we want to find the pc-component
        :param en_nodes: nodes of the sub-graph in which to operate
        :return: a set of nodes that are in the same pc-component with node_x
        """
        if en_nodes is None:
            en_nodes = self.nodes_set

        dc_components = self.find_definite_c_components(en_nodes)

        pc_component = set()
        for node_x in set_x:
            pc_component.update(
                self.find_union_pc_components_for_node(node_x=node_x, en_nodes=en_nodes, dc_components=dc_components)
            )
        return pc_component

    def find_region(self, set_a, en_nodes=None) -> set:
        """
        Find the region of set A in the given sub_graph.
        A region is the union of buckets containing nodes in the pc-component of A.
        :param set_a: the set for which the region is required
        :param en_nodes: endogenous nodes defining the sub-graph
        :return: a region of A: a set of nodes
        """
        pc_component = self.find_union_pc_components_for_set(set_a, en_nodes=en_nodes)
        buckets_list = self.find_buckets_list(en_nodes)
        region = set()
        for node in pc_component:
            for bucket in buckets_list:
                if node in bucket:
                    region.update(bucket)
                    break
        return region

    def find_buckets_list(self, en_nodes=None) -> list:
        """
        Find buckets of nodes, as defined in "Causal Identification under Markov Equivalence" (Jaber et al., 2018).
        A Bucket is a set of nodes such that between ever pair, there exists a circle path.

        :param en_nodes: set of nodes defining the graph
        :return: a list of buckets (not in topological order)
        """
        buckets_list = self.find_unconnected_subgraphs(en_nodes=en_nodes, sym_edge_mark=Mark.Circle)
        return buckets_list

    def get_ordered_buckets(self, en_nodes=None) -> list:
        """
        Find the buckets (circle components) in a the PAG, and return them in a topological sort.
        It is a realization of the PTO algorithm (Jaber et al., IJCAI 2018;
        A Graphical Criterion for Effect Identification in Equivalence Classes of Causal Diagrams).
        :param en_nodes: set of nodes defining the graph
        :return: a topologically sorted list of buckets
        """
        if en_nodes is None:
            en_nodes = self.nodes_set

        buckets_list = self.find_buckets_list(en_nodes=en_nodes)  # unordered list of buckets (circle components)
        ordered_buckets = []
        remaining_nodes = {node for bucket in buckets_list for node in bucket}  # nodes of buckets that need sorting
        while len(buckets_list) > 0:
            idx = 0
            while idx < len(buckets_list):
                bucket = buckets_list[idx]
                if self.is_set_possible_sink(bucket, ex_nodes=remaining_nodes-bucket):
                    bucket = buckets_list.pop(idx)
                    ordered_buckets.append(bucket)
                    remaining_nodes = remaining_nodes - bucket
                    break
                idx += 1
            if len(ordered_buckets) == 0:  # no initial bucket with lowest topological order (sink) was found
                assert len(ordered_buckets) > 0

        ordered_buckets.reverse()
        return ordered_buckets

    # -----------------------------------------------------------------------------------------------------------------
    # --- Methods that modify the graph -------------------------------------------------------------------------------
    def copy_skeleton_from_pdag(self, pdag: PDAG, nodes_set=None):
        """
        Add the skeleton of in an external pdag (ignore edge-marks) to the current PAG (add o--o edges)
        :param pdag: An external PDAG object
        :param nodes_set: a set of nodes that define the graph to copy
        :return:
        """
        assert isinstance(pdag, PDAG)

        if nodes_set is None:
            nodes_set = self.nodes_set

        for node_i, node_j in combinations(nodes_set, 2):
            if pdag.is_connected(node_i, node_j):
                self.add_edge(node_i, node_j, Mark.Circle, Mark.Circle)

    def orient_v_structures(self, sepsets=None):
        """
        Orient X *--* Z *--* Y as X *--> Z <--* if X and Y are disjoint and Z is not in their separation set
        :param sepsets: Separating sets, an instance of the SeparationSet class
        :return:
        """
        assert sepsets is not None
        # check each node if it can serve as a collider for a disjoint neighbors
        for node_z in self.nodes_set:
            # check neighbors
            xy_nodes = self.find_adjacent_nodes(node_z)  # neighbors with some edge-mark at node_z
            for node_x, node_y in combinations(xy_nodes, 2):
                if self.is_connected(node_x, node_y):
                    continue  # skip this pair as they are connected
                if node_z not in sepsets.get_sepset(node_x, node_y):
                    self.replace_edge_mark(
                        node_source=node_x, node_target=node_z, requested_edge_mark=Mark.Directed)  # orient X *--> Z
                    self.replace_edge_mark(
                        node_source=node_y, node_target=node_z, requested_edge_mark=Mark.Directed)  # orient Y *--> Z

    def maximally_orient_pattern(self, rules_set=None):
        """
        Complete orienting graph. It is assumed that all v-structures have been previously oriented
        :param rules_set:
        :return:
        """
        if rules_set is None:
            rules_set = list(self.orientation_rules.keys())

        graph_modified = True
        while graph_modified:
            graph_modified = False
            for rule_idx in rules_set:
                rule = self.orientation_rules[rule_idx]
                graph_modified |= rule()

    # Batch of rules for initial graph orientation [R1, R2, R3, R4] ----------------------------------------------------
    def orient_by_rule_1(self):
        """
        [R1] If A *--> B o--* C, and A & C are not connected, then orient A *--> B ---> C
        :return: True if the graph was modified; otherwise False
        """
        graph_modified = False
        for node_b in self.nodes_set:
            a_nodes = self._graph[node_b][Mark.Directed]
            c_nodes = self._graph[node_b][Mark.Circle].copy()
            for node_a in a_nodes:
                for node_c in c_nodes:
                    if not self.is_connected(node_a, node_c):
                        self.replace_edge_mark(node_source=node_c, node_target=node_b,
                                               requested_edge_mark=Mark.Tail)  # tail edge-mark
                        self.replace_edge_mark(node_source=node_b, node_target=node_c,
                                               requested_edge_mark=Mark.Directed)  # head edge-mark
                        graph_modified = True

        return graph_modified

    def orient_by_rule_2(self):
        """
        [R2] If (1) A *--> B ---> C or (2) A ---> B *--> C, and A *--o C, then orient A *--> C

        :return: True if the graph was modified; otherwise False
        """
        graph_modified = False

        # case (1): If A *--> B ---> C and A *--o C, then orient A *--> C
        for node_b in self.nodes_set:
            a_nodes = self._graph[node_b][Mark.Directed]  # A *--> B
            c_nodes = self._graph[node_b][Mark.Tail]  # B ---* C (we still need to check that B ---> C)

            for node_c in c_nodes:
                if node_b not in self._graph[node_c][Mark.Directed]:  # check if B *--> C (already is B ---* C)
                    continue  # skip this node_c
                # now we are sure that B ---> C
                for node_a in a_nodes:
                    if node_a in self._graph[node_c][Mark.Circle]:  # if A *--o C
                        self.replace_edge_mark(node_source=node_a, node_target=node_c,
                                               requested_edge_mark=Mark.Directed)
                        graph_modified = True

        # case (2): If A ---> B *--> C, and A *--o C, then orient A *--> C
        for node_c in self.nodes_set:
            b_nodes = self._graph[node_c][Mark.Directed].copy()  # B *--> C

            for node_b in b_nodes:
                a_nodes = self._graph[node_b][Mark.Directed]  # A *--> B (we still need to check A ---> B)
                for node_a in a_nodes:
                    if node_b not in self._graph[node_a][Mark.Tail]:  # check if A ---* B (already is A *--> B)
                        continue  # skip this node_x
                    if node_a in self._graph[node_c][Mark.Circle]:  # if A *--o C
                        self.replace_edge_mark(node_source=node_a, node_target=node_c,
                                               requested_edge_mark=Mark.Directed)
                        graph_modified = True

        return graph_modified

    def orient_by_rule_3(self):
        """
        [R3] If A *--> B <--* C and A *--o D o--* C, A & C not connected, D *--o B, then orient D *--> B

        :return: True if the graph was modified; otherwise False
        """
        graph_modified = False
        for node_b in self.nodes_set:
            d_nodes = self._graph[node_b][Mark.Circle].copy()  # D *--o B
            for node_d in d_nodes:
                # find pairs that satisfy (A, C) *--> B and (A, C) *--o D
                ac_nodes = self._graph[node_b][Mark.Directed] & self._graph[node_d][Mark.Circle]
                for (node_a, node_c) in combinations(ac_nodes, 2):
                    if not self.is_connected(node_a, node_c):  # a pair (A,C) exists and is disjoint
                        self.replace_edge_mark(node_source=node_d, node_target=node_b,
                                               requested_edge_mark=Mark.Directed)
                        graph_modified = True

        return graph_modified

    def orient_by_rule_4(self):
        """
        [R4] If a discriminating path between D and C for B, i.e., (D, ..., A, B, C), and B o--* C, then:
            (1) if B in sep-set of (D, C), orient: B ---> C,
            (2) else, orient the triplet A <--> B <--> C

        :return: True if the graph was modified; otherwise False
        """
        graph_modified = False
        for node_b in self.nodes_set:
            c_nodes = self._graph[node_b][Mark.Circle].copy()  # B o--* C
            for node_c in c_nodes:
                potential_a_nodes = self.find_parents(node_c)  # should comply with A ---> C
                for node_a in potential_a_nodes:
                    if node_b in self._graph[node_a][Mark.Directed]:  # should comply with A <--* B
                        # node_x is legal
                        node_d = self.find_discriminating_path_to_triplet(node_a, node_b, node_c)
                        if node_d is not None:
                            # found a discriminating path
                            if node_b in self.sepset.get_sepset(node_d, node_c):
                                # orient B o--* C into B ---> C
                                self.replace_edge_mark(node_source=node_b, node_target=node_c,
                                                       requested_edge_mark=Mark.Directed)
                                self.replace_edge_mark(node_source=node_c, node_target=node_b,
                                                       requested_edge_mark=Mark.Tail)
                            else:
                                # orient A <--> B <--> C
                                self.replace_edge_mark(node_source=node_b, node_target=node_a,
                                                       requested_edge_mark=Mark.Directed)
                                self.replace_edge_mark(node_source=node_a, node_target=node_b,
                                                       requested_edge_mark=Mark.Directed)
                                self.replace_edge_mark(node_source=node_b, node_target=node_c,
                                                       requested_edge_mark=Mark.Directed)
                                self.replace_edge_mark(node_source=node_c, node_target=node_b,
                                                       requested_edge_mark=Mark.Directed)

                            graph_modified = True

        return graph_modified

    # Batch of rules [R5, R6, R7] when considering selection bias ------------------------------------------------------
    def orient_by_rule_5(self):
        """
        [R5] If A o--o B and there is an uncovered circle path <A, X, ..., Y, B>,
        such that (A, Y) are disconnected and (X, B) are disconnected

        :return: True if the graph was modified; otherwise False
        """
        graph_modified = False

        # create a list of all the A o--o B edges in the graph
        var_edges_list = [(node_a, node_b)
                          for node_a in self.nodes_set
                          for node_b in self.nodes_set
                          if self.is_edge(node_a, node_b, Mark.Circle, Mark.Circle)]

        # examine each variant edge
        for (node_a, node_b) in var_edges_list:
            a_neighbors_list = {nb_a for nb_a in self.nodes_set - {node_a, node_b}
                                if self.is_edge(node_a, nb_a, Mark.Circle, Mark.Circle)  # node_x o--o nb_a
                                and not self.is_connected(node_b, nb_a)}  # nb_a not connected to node_y
            b_neighbors_list = {nb_b for nb_b in self.nodes_set - {node_a, node_b}
                                if self.is_edge(node_b, nb_b, Mark.Circle, Mark.Circle)  # node_y o--o nb_b
                                and not self.is_connected(node_a, nb_b)}  # nb_b not connected to node_x

            for neighbor_a in a_neighbors_list:
                for neighbor_b in b_neighbors_list:
                    uncov_circ_path = \
                        self.find_uncovered_path(node_a, node_b,
                                                 neighbor_x=neighbor_a, neighbor_y=neighbor_b, edge_condition=
                                                 lambda in1, in2: self.is_edge(in1, in2, Mark.Circle, Mark.Circle))
                    if uncov_circ_path is not None:
                        # criterion is met
                        graph_modified = True
                        self.reset_orientations(Mark.Tail, {node_a, node_b})
                        full_path = [node_a, *uncov_circ_path, node_b]  # add the end-points, A and B, to the path
                        for idx in range(len(full_path)-1):
                            self.reset_orientations(Mark.Tail, {full_path[idx], full_path[idx+1]})

        return graph_modified

    def orient_by_rule_6(self):
        """
        [R6] If A ---- B o--* C, and A & C may or may not be connected, then orient B ---* C

        :return: True if the graph was modified; otherwise False
        """
        graph_modified = False
        for node_b in self.nodes_set:
            a_nodes = {can_node for can_node in self._graph[node_b][Mark.Tail]
                       if node_b in self._graph[can_node][Mark.Tail]}  # A ---- B
            c_nodes = self._graph[node_b][Mark.Circle].copy()  # B o--* C
            for node_a in a_nodes:
                for node_c in c_nodes:
                    self.replace_edge_mark(node_source=node_c, node_target=node_b,
                                           requested_edge_mark=Mark.Tail)  # tail edge-mark
                    graph_modified = True

        return graph_modified

    def orient_by_rule_7(self):
        """
        [R7] If A ---o B o--* C, and A & C are not connected, then orient tail B ---* C

        :return: True if the graph was modified; otherwise False
        """
        graph_modified = False
        for node_b in self.nodes_set:
            a_nodes = {can_node for can_node in self._graph[node_b][Mark.Circle]
                       if node_b in self._graph[can_node][Mark.Tail]}  # A ---o B
            c_nodes = self._graph[node_b][Mark.Circle].copy()  # B o--* C
            for node_a in a_nodes:
                for node_c in c_nodes:
                    if node_a == node_c:
                        continue
                    if not self.is_connected(node_a, node_c):
                        self.replace_edge_mark(node_source=node_c, node_target=node_b,
                                               requested_edge_mark=Mark.Tail)  # tail edge-mark
                        graph_modified = True

        return graph_modified

    # Batch of rules [R8, R9, R10] -------------------------------------------------------------------------------------
    def orient_by_rule_8(self):
        """
        [R8] If A ---> B ---> C or A ---o B ---> C and A o--> C then orient tail A ---> C

        :return: True if the graph was modified; otherwise False
        """
        graph_modified = False
        for node_b in self.nodes_set:
            a_nodes = {can_node for can_node in self._graph[node_b][Mark.Directed] | self._graph[node_b][Mark.Circle]
                       if node_b in self._graph[can_node][Mark.Tail]}  # A ---> B or A ---o B
            c_nodes = {can_node for can_node in self._graph[node_b][Mark.Tail]
                       if node_b in self._graph[can_node][Mark.Directed]}  # B ---> C
            for node_a in a_nodes:
                for node_c in c_nodes:
                    if self.is_edge(node_a, node_c, Mark.Circle, Mark.Directed):  # A o--> C
                        self.replace_edge_mark(node_source=node_c, node_target=node_a,
                                               requested_edge_mark=Mark.Tail)  # tail edge-mark at A
                        graph_modified = True

        return graph_modified

    def orient_by_rule_9(self):
        """
        [R9] If A o--> C and there is a possibly directed uncovered path <A, B, ..., D, C>, B and C are not connected
                then orient tail A ---> C.

        :return: True if the graph was modified; otherwise False
        """
        graph_modified = False
        for node_a in self.nodes_set:
            c_nodes = {can_node for can_node in self._graph[node_a][Mark.Circle]
                       if node_a in self._graph[can_node][Mark.Directed]}  # A o--> C
            for node_c in c_nodes:
                # look for a possibly directed uncovered path s.t. B and C are not connected (for the given A o--> C
                b_nodes = {can_b for can_b in self.find_possible_children(node_a, self.nodes_set - {node_c, node_a})
                           if not self.is_connected(can_b, node_c)}

                for node_b in b_nodes:
                    d_nodes = self.find_possible_parents(node_c, self.nodes_set - {node_a, node_b, node_c})
                    # search a p.d. uncovered path for <A, B, ..., C>
                    for node_d in d_nodes:
                        pd_path = self.find_uncovered_path(node_x=node_a, node_y=node_c,
                                                           neighbor_x=node_b, neighbor_y=node_d,
                                                           edge_condition=self.is_possible_parent)
                        if pd_path is not None:
                            self.replace_edge_mark(node_source=node_c, node_target=node_a,
                                                   requested_edge_mark=Mark.Tail)  # tail edge-mark at A
                            graph_modified = True
                            return graph_modified

        return graph_modified

    def orient_by_rule_10(self):
        """
        [R10] If A o--> C and B ---> C <---D, and if there are two possibly directed uncovered paths
        <A, E, ..., B>, <A, F, ..., D> s.t. E, F are disconnected, and any of these paths can be a single-edge path,
        A o--> B or A o--> D, then orient tail A ---> C.

        :return: True if the graph was modified; otherwise False
        """
        graph_modified = False

        for node_c in self.nodes_set:
            a_nodes = {can_node for can_node in self.nodes_set - {node_c}
                       if self.is_edge(can_node, node_c, Mark.Circle, Mark.Directed)}  # find A o--> C
            if len(a_nodes) == 0:
                continue  # no A o--> was found for the specific C node, go to the next c_node
            # find B, D such that B ---> C <---D (directed edges)
            bd_nodes = {can_node for can_node in self.nodes_set - {node_c}
                        if self.is_edge(can_node, node_c, Mark.Tail, Mark.Directed)}  # find a pair {D,B} ---> C
            if len(bd_nodes) < 2:
                continue
            for node_a in a_nodes:  # try to orient the tail of this specific A o--> C edge
                a_possible_children = self.find_possible_children(
                    parent_node=node_a, en_nodes=self.nodes_set - {node_a, node_c})  # find A o--{o,>} neighbors
                if len(a_possible_children) < 2:
                    continue  # cannot draw two paths out of A so go to the next A-node

                # now all the nodes are specified test the condition of the rule
                for node_b, node_d in combinations(bd_nodes, 2):
                    # try to construct two p.d. uncovered paths
                    # note that a path <A, E, ...> may end in either B or D. The same for <A, F, ...>
                    for node_e in a_possible_children:
                        for node_f in a_possible_children:
                            if node_e == node_f or self.is_connected(node_e, node_f):
                                continue

                            path_e = self.find_uncovered_path(node_x=node_a, node_y=node_c,
                                                              neighbor_x=node_e, neighbor_y=node_b)
                            if path_e is not None:
                                path_f = self.find_uncovered_path(node_x=node_a, node_y=node_c,
                                                                  neighbor_x=node_f, neighbor_y=node_d)
                                if path_f is not None:
                                    self.replace_edge_mark(node_source=node_c, node_target=node_a,
                                                           requested_edge_mark=Mark.Tail)  # tail edge-mark at A
                                    graph_modified = True
                                    return graph_modified

        return graph_modified
