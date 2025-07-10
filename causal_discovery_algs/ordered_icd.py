from causal_discovery_algs.icd import create_pds_tree
from causal_discovery_utils.constraint_based import LearnStructBase
from graphical_models import PAG, arrow_head_types as Mark


class LearnStructOrderedICD(LearnStructBase):
    def __init__(self, nodes_set, causal_order, ci_test, is_selection_bias=True, is_tail_completeness=True,
                 max_cond_size=None):
        """
        Initialize Ordered-ICD Algorithm
        :param nodes_set: a set of graph nodes
        :param causal_order: a causally ordered list of nodes, from ancestors to descendants, such that
            a node is not an ancestor of nodes preceding it in the list.
        :param ci_test: conditional independence test
        :param is_selection_bias: are selection variables possibly present
        :param is_tail_completeness: use orientation rules to ensure tail completeness.
        :param max_cond_size: largest conditioning set size to use
        """
        super().__init__(PAG, nodes_set=nodes_set, ci_test=ci_test)
        self.graph.create_complete_graph(Mark.Circle, nodes_set)  # Create a fully connected graph with edges: o--o
        if type(causal_order) in [list, tuple]:
            if len(causal_order) == len(nodes_set):
                self.order = list(causal_order)
            else:
                raise "Causal order length should be the same as the number of nodes."
        else:
            raise "Causal order should be a list or a tuple."
        self.is_tail_completeness = is_tail_completeness
        self.is_selection_bias = is_selection_bias

        # dictionary set up for tracking which edges have completed learning.
        self.edge_done = {base_node: dict() for base_node in self.order}
        for ancestor_node, base_node in self.get_causal_pairs():
            self.edge_done[base_node][ancestor_node] = False

        if max_cond_size is not None:
            if max_cond_size < 0 or max_cond_size > (len(causal_order)-2):
                raise "Invalid max_cond_size value."
        self.max_cond_size = max_cond_size

    def learn_structure(self, local_structure_learning=False):
        if local_structure_learning:
            self.learn_structure_local()
        else:
            self.learn_structure_global()

    def get_causal_pairs(self, start_base_idx=1):
        if start_base_idx < 1:
            raise "base node index should be at least 1"

        n = len(self.order)
        for base_idx in range(start_base_idx, n):
            base = self.order[base_idx]
            for anc in self.order[:base_idx]:
                yield anc, base

    def reset_graph_orientations(self):
        """
        Erase all edge marks replacing them with the circle mark.
        """
        for ancestor_node, base_node in self.get_causal_pairs():
            if self.graph.is_connected(ancestor_node, base_node):
                self.graph.replace_edge_mark(node_source=ancestor_node,
                                             node_target=base_node, requested_edge_mark=Mark.Directed)
                self.graph.replace_edge_mark(node_source=base_node,
                                             node_target=ancestor_node, requested_edge_mark=Mark.Circle)

    def _is_cond_set_possible_ancestor(self, cond_set, ancestor_idx, base_idx):
        ancestor_node = self.order[ancestor_idx]
        base_node = self.order[base_idx]
        for z in cond_set:
            anc_en_nodes = set(self.order[:ancestor_idx])
            anc_en_nodes.add(ancestor_node)
            base_en_nodes = set(self.order[:base_idx])
            base_en_nodes.add(base_node)

            if self.graph.is_possible_ancestor(ancestor_node=z, descendant_node=ancestor_node, en_nodes=anc_en_nodes):
                continue
            if self.graph.is_possible_ancestor(ancestor_node=z, descendant_node=base_node, en_nodes=base_en_nodes):
                continue
            return False
        return True

    # --- A Local Structure Learning Approach ----------------------------------------------------------------
    def _learn_node_connection(self, node_idx):
        """
        Test each edge connecting the given node and nodes preceding it (possible ancestors) using CI tests with
        increasing conditioning set sizes up to a value derived from the node position in the causal order.
        :param node_idx: index of the node in the causal order
        :return:
        """
        assert node_idx > 0
        cond_indep = self.ci_test.cond_indep
        base_node = self.order[node_idx]
        max_conditioning_size = node_idx - 1  # tie conditioning size, search radius and position in the causal order
        if self.max_cond_size is not None:
            max_conditioning_size = min(max_conditioning_size, self.max_cond_size)
        anc_nodes = set(self.order[:node_idx])  # possible ancestors of the base node

        # start of with independence tests without conditioning (conditioning size 0)
        for ancestor_idx in range(node_idx):
            ancestor_node = self.order[ancestor_idx]
            if self.edge_done[base_node][ancestor_node]:
                continue
            if cond_indep(ancestor_node, base_node, tuple()):
                self.graph.delete_edge(ancestor_node, base_node)  # remove directed/undirected edge
                self.edge_done[base_node][ancestor_node] = True
        self.orient_graph(anytime=True)
        if max_conditioning_size == 0:
            return

        # next, test with independence tests with one conditioning node (conditioning set size 1)
        for ancestor_node in anc_nodes:
            if not self.graph.is_connected(ancestor_node, base_node):
                continue
            if self.edge_done[base_node][ancestor_node]:
                continue
            pot_parents_base = self.graph.find_adjacent_nodes(base_node, pool_nodes=anc_nodes-{ancestor_node})
            for conditioning_node in pot_parents_base:
                cond_tup = (conditioning_node,)
                if cond_indep(ancestor_node, base_node, cond_tup):
                    self.graph.delete_edge(ancestor_node, base_node)  # remove directed/undirected edge
                    self.sepset.set_sepset(ancestor_node, base_node, cond_tup)
                    self.edge_done[base_node][ancestor_node] = True
                    break
        self.orient_graph(anytime=True)
        if max_conditioning_size == 1:
            return

        # now, test with independence tests with two or more conditioning node (conditioning size > 1)
        en_nodes = set(self.order[:(node_idx+1)])  # base_node + all its possible ancestors
        for cond_size in range(2, max_conditioning_size+1):
            if all(self.edge_done[base_node].values()):  # if all edges into base_node are done
                return
            # print(f'Creating a PDS-Tree for: {base_node}, CI-order: {cond_size}')
            pds_tree, possible_dsep_set = create_pds_tree(source_pag=self.graph, node_root=base_node, en_nodes=en_nodes,
                                                          max_depth=cond_size)
            cond_sets_list_init = pds_tree.get_subsets_list(set_nodes=possible_dsep_set, subset_size=cond_size)
            cond_sets_list_init.sort(key=lambda x: x[1])  # sort by the distance from the root node

            # prune connection between base_node and its possible ancestors using the current conditioning set size
            for ancestor_idx in range(node_idx):
                ancestor_node = self.order[ancestor_idx]
                if self.edge_done[base_node][ancestor_node]:
                    continue
                self.edge_done[base_node][ancestor_node] = True  # temporarily set true, in case no legal cond_set found
                if not self.graph.is_connected(ancestor_node, base_node):
                    continue
                for cond in cond_sets_list_init:
                    cond_set = cond[0]
                    if ancestor_node in cond_set:
                        continue
                    if not self._is_cond_set_possible_ancestor(cond_set, ancestor_idx, node_idx):
                        continue
                    # at this point, a legal cond_set is found
                    self.edge_done[base_node][ancestor_node] = False
                    cond_tup = tuple(cond_set)
                    if cond_indep(ancestor_node, base_node, cond_tup):
                        self.graph.delete_edge(ancestor_node, base_node)  # remove directed/undirected edge
                        self.sepset.set_sepset(ancestor_node, base_node, cond_tup)
                        self.edge_done[base_node][ancestor_node] = True
                        break  # stop searching for independence as we found one and updated the graph accordingly
            self.orient_graph(anytime=True)

    def orient_graph(self, anytime=False):
        # ToDo: confine orientation to a selected set of nodes
        self.reset_graph_orientations()
        # orient edges using rules that preserve the anytime property
        self.graph.orient_v_structures(self.sepset)  # corresponds to rule R0
        self.graph.maximally_orient_pattern(rules_set=[1, 2, 3, 4])
        # if the algorithm concluded, orient all edges for obtaining completeness
        if not anytime:
            if self.is_selection_bias:
                self.graph.maximally_orient_pattern(rules_set=[5, 6, 7])  # when selection-bias may be present
            if self.is_tail_completeness:
                self.graph.maximally_orient_pattern(rules_set=[8, 9, 10])  # for tail-completeness

    def learn_structure_local(self) -> None:
        # tied parameters: conditioning size, search radius, and position in the causal-order
        num_nodes = len(self.graph.nodes_set)
        for idx in range(1, num_nodes):  # skip the first node in the order as it has no observed ancestors
            self._learn_node_connection(node_idx=idx)
        self.orient_graph()  # final orientation to ensure completeness

    # --- A Global Structure Learning Approach ----------------------------------------------------------------
    def _learn_r0_connectivity_node(self, base_node_idx) -> None:
        """
        Independence tests with one conditioning node (conditioning size 1)
        """
        cond_indep = self.ci_test.cond_indep
        base_node = self.order[base_node_idx]
        anc_nodes = self.order[:base_node_idx]
        for ancestor_node in anc_nodes:
            if self.edge_done[base_node][ancestor_node]:
                continue
            if cond_indep(ancestor_node, base_node, tuple()):
                self.graph.delete_edge(ancestor_node, base_node)  # remove directed/undirected edge
                self.edge_done[base_node][ancestor_node] = True

    def _learn_r1_connectivity_node(self, base_node_idx) -> bool:
        """
        Independence tests with one conditioning node (conditioning size 1).
        :return: True if CI testing is complete and additional CI tests are not needed.
        """
        done = True  # temporarily set. If later a valid conditioning set will be found for an edge, set to False
        cond_indep = self.ci_test.cond_indep
        base_node = self.order[base_node_idx]
        anc_nodes = set(self.order[:base_node_idx])  # possible ancestors of the base node
        for ancestor_node in anc_nodes:
            if not self.graph.is_connected(ancestor_node, base_node):
                continue
            if self.edge_done[base_node][ancestor_node]:
                continue
            pot_parents_base = self.graph.find_adjacent_nodes(base_node, pool_nodes=anc_nodes - {ancestor_node})
            for conditioning_node in pot_parents_base:
                cond_tup = (conditioning_node,)
                if cond_indep(ancestor_node, base_node, cond_tup):
                    self.graph.delete_edge(ancestor_node, base_node)  # remove directed/undirected edge
                    self.sepset.set_sepset(ancestor_node, base_node, cond_tup)
                    self.edge_done[base_node][ancestor_node] = True
                    break
                else:
                    done = False  # found at least one conditioning set that did not lead to the removal of the edge
        return done

    def _learn_r_connectivity_node(self, r, base_node_idx):
        """
        Conditional independence tests with a given (r) conditioning nodes (two or more).
        :return: True if CI testing is complete and additional CI tests are not needed.
        """
        max_conditioning_size = base_node_idx - 1  # tie cond. size, search radius and position in the causal order
        if self.max_cond_size is not None:
            max_conditioning_size = min(max_conditioning_size, self.max_cond_size)

        cond_indep = self.ci_test.cond_indep
        base_node = self.order[base_node_idx]
        cond_size = r  # alias
        en_nodes = set(self.order[:(base_node_idx + 1)])  # base_node + all its possible ancestors

        if all(self.edge_done[base_node].values()):  # if all edges into base_node are done
            return True
        # print(f'Creating a PDS-Tree for: {base_node}, CI-order: {cond_size}')
        pds_tree, possible_dsep_set = create_pds_tree(source_pag=self.graph, node_root=base_node, en_nodes=en_nodes,
                                                      max_depth=cond_size)
        cond_sets_list_init = pds_tree.get_subsets_list(set_nodes=possible_dsep_set, subset_size=cond_size)
        cond_sets_list_init.sort(key=lambda x: x[1])  # sort by the distance from the root node

        done = True  # temporarily set. If later a valid conditioning set will be found for an edge, set to False

        # prune connection between base_node and its possible ancestors using the current conditioning set size
        for ancestor_idx in range(base_node_idx):
            ancestor_node = self.order[ancestor_idx]
            if self.edge_done[base_node][ancestor_node]:
                continue
            self.edge_done[base_node][ancestor_node] = True  # temporarily set true, in case no legal cond_set found
            if not self.graph.is_connected(ancestor_node, base_node):
                continue
            for cond in cond_sets_list_init:
                cond_set = cond[0]
                if ancestor_node in cond_set:
                    continue
                if not self._is_cond_set_possible_ancestor(cond_set, ancestor_idx, base_node_idx):
                    continue
                # at this point, a legal cond_set is found
                cond_tup = tuple(cond_set)
                if cond_indep(ancestor_node, base_node, cond_tup):
                    self.graph.delete_edge(ancestor_node, base_node)  # remove directed/undirected edge
                    self.sepset.set_sepset(ancestor_node, base_node, cond_tup)
                    self.edge_done[base_node][ancestor_node] = True
                    break  # stop searching for independence as we found one and updated the graph accordingly
                else:
                    self.edge_done[base_node][ancestor_node] = False
                    done = False
        return done

    def learn_structure_global(self) -> None:
        """
        This function learns the structure in the following manner:

        for order in [0, ..., n-1]:
            for current_node in [causal_order[order+1], ..., causal_order[n-1]]:
                prune link current_node and the nodes preceding it in causal_order

            Now, after going over all nodes using the current value of order:
                orient edges(anytime)

            if no conditioning_set sizes were found:
                break
        On exit, orient the graph with the complete set of orientation rules

        :return: None
        """
        num_nodes = len(self.graph.nodes_set)
        r = 0  # initialize conditioning set size and search radius size

        # Learn with empty conditioning set (unconditional)
        for base_node_idx in range(r + 1, num_nodes):
            self._learn_r0_connectivity_node(base_node_idx)
        self.orient_graph(anytime=True)

        # Learn with conditioned on a single node (conditioning set size == 1)
        r = 1
        for base_node_idx in range(r + 1, num_nodes):
            self._learn_r1_connectivity_node(base_node_idx)  # connection pruning: this base node and its ancestors
        self.orient_graph(anytime=True)

        if self.max_cond_size is not None:
            max_cond_size = self.max_cond_size + 1
        else:
            max_cond_size = (num_nodes - 2) + 1

        done = True
        for r in range(2, max_cond_size):  # [2, ..., num_nodes-2] inclusive
            for base_node_idx in range(r + 1, num_nodes):
                node_node = self._learn_r_connectivity_node(r, base_node_idx)
                done = done and node_node  # a single not-done if enough to turn the graph-done flag to False
            if done:
                break
            self.orient_graph(anytime=True)

        self.orient_graph(anytime=False)  # use the full set of orientation rules
