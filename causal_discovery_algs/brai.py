from causal_discovery_algs.rai import LearnStructRAI
from graphical_models import PDAG, DAG
import numpy as np
from itertools import combinations
import graphical_models.arrow_head_types as Mark  # incoming arrow head-types
from causal_discovery_utils.performance_measures import score_bdeu
from causal_discovery_utils.data_utils import get_var_size
from causal_discovery_utils.constraint_based import SeparationSet


class CookieNode:
    """
    Cookie node: holds one possible graph learned with CI-test up to a specific order.
    Consists of a MultiHypNode (multiple-hypothesis-node) for each slice of the cookie
    (The children of a cookie-node are multi.-hyp. nodes and the children of a multi.-hyp. node are cookie nodes)
    """
    def __init__(self, multi_hyp_ancestors=None, multi_hyp_descendant=None, cpdag=None, slices=None, extra_data=None):
        self.multi_hyp_ancestors = []  # a list of multi.-hyp nodes, each node corresponds to a sub-set of nodes
        self.multi_hyp_descendant = None  # a multi.-hyp node corresponding to the descendant sub-set of nodes
        self.cpdag = None
        self.slices = None
        self.extra_data = extra_data

        if multi_hyp_ancestors is not None:
            for ancestor in multi_hyp_ancestors:
                self.add_multi_hyp_ancestor(ancestor)
        if multi_hyp_descendant is not None:
            self.add_multi_hyp_descendant(multi_hyp_descendant)
        if cpdag is not None:
            self.set_cookie_graph(cpdag)
        if slices is not None:
            self.set_cookie_slices(slices)

    def add_multi_hyp_ancestor(self, node):  # add children corresponding to ancestor-groups
        assert isinstance(node, MultiHypNode)
        self.multi_hyp_ancestors.append(node)

    def add_multi_hyp_descendant(self, node):  # add a child corresponding to the descendant-group
        assert isinstance(node, MultiHypNode)
        self.multi_hyp_descendant = node

    def set_cookie_slices(self, slices):
        assert isinstance(slices, dict)
        self.slices = slices

    def set_cookie_graph(self, cpdag):
        assert isinstance(cpdag, PDAG)
        self.cpdag = cpdag  # a partially refined pdag


class MultiHypNode:
    """
    Multiple-hypothesis node: contains multiple cookie-nodes each describing a set of nodes
    The children of a multi.-hyp. node are cookie-nodes, and the children of a cookie node are multi.-hyp. nodes
    When sampling, only one of the children should be selected
    """
    def __init__(self, cookie_children=None, multi_hyp_data=None, extra_data=None):

        self.children = []  # list of cookie nodes
        self.multi_hyp_data = None
        self.extra_data = extra_data
        self.selected_cookie_idx = None  # will be set each call to sample graph

        if cookie_children is not None:
            self.is_leaf = False
            for cookie in cookie_children:
                self.addCookie(cookie)
        else:  # a leaf node
            self.is_leaf = True

        if multi_hyp_data is not None:
            self.set_data(multi_hyp_data)

    def addCookie(self, node):
        assert isinstance(node, CookieNode)
        self.children.append(node)
        self.is_leaf = False

    def set_data(self, multi_hyp_data):
        assert isinstance(multi_hyp_data, dict)
        assert isinstance(multi_hyp_data['endogenous'], set)
        assert isinstance(multi_hyp_data['exogenous'], set)
        assert isinstance(multi_hyp_data['ci-order'], int)
        if multi_hyp_data['score'] is not None:
            assert isinstance(multi_hyp_data['score'], float)
        if self.is_leaf:
            assert isinstance(multi_hyp_data['graph'], PDAG)
            assert isinstance(multi_hyp_data['sepset'], SeparationSet)
        self.multi_hyp_data = multi_hyp_data

    @property
    def en_nodes(self):
        return self.multi_hyp_data['endogenous']


class LearnStructBRAI(LearnStructRAI):
    def __init__(self, nodes_set, ci_test, num_of_hyp, scoring_data=None, node_size=None, scoring_function=None):
        super().__init__(nodes_set, ci_test)
        self.graph.create_complete_graph(nodes_set)  # Create a fully connected graph
        assert (num_of_hyp > 0)
        self.num_of_hyp = num_of_hyp

        # get data from the CI test
        self.data = ci_test.data.copy()
        self.num_records = ci_test.num_records
        self.num_nodes = ci_test.num_vars

        self.graph_generating_tree = MultiHypNode()

        self.scoring_data = None
        self.node_size = None

        if scoring_function is None:
            # scoring function with arguments: fun(dag, scoring_data, node_sizes, nodes), returns a log-probability
            self.scoring_function = score_bdeu  # default scoring function is BDeu (assumed discrete variables)

        if scoring_data is None:
            self.is_scored = False
        else:
            self.is_scored = True  # turn off if scoring the graph is not required
            self.scoring_data = scoring_data
            if node_size is not None:
                self.node_size = node_size
            elif scoring_data is not None:
                self.node_size = get_var_size(scoring_data)
            else:
                self.node_size = None

    def learn_structure(self):
        """
        Main structure learning function.
        :return: The root of the learned GGT (graph generating tree with a MultiHypNode as root and leaves)
        """

        # initialize for the 1st recursive call
        en_nodes = self.graph.nodes_set
        ex_nodes = set()

        ggt_root = self.learn_recursively(en_nodes=en_nodes, ex_nodes=ex_nodes, order=0)
        self.graph_generating_tree = ggt_root  # root of GGT, a node of type MultiHypNode
        self.sample_cpdag(temperature=0)  # get the MAP CPDAG

    def learn_recursively(self, en_nodes, ex_nodes, order):
        """
        The folowing steps are preformed:
        1. For num_of_hyp:
            a. Create a bootstrap sample of the training data
            b. Refine and orient using CI tests of a specific condition set size
            c. Identify ancestors and descendant groups
            d. Call recursively for the ancestor and descendant groups with CI order+1
        2. Update GGT (tree structure)
        :param en_nodes: Endogenous nodes
        :param ex_nodes: Exogenous nodes
        :param order: CI test order, i.e., condition set size
        :return: A MultiHypNode that is ONE of the following:
            1) a parent of "num_of_hyp" CookieNodes
            2) a leaf node with values: CPDAG sub-graph and Bayesian score
        """

        if self._exit_cond(en_nodes, order):
            # reached a leaf
            if self.is_scored:
                leaf_pdag = PDAG(self.graph.nodes_set)
                leaf_pdag.add_edges_from(self.graph, en_nodes, ex_nodes)
                leaf_dag = DAG(leaf_pdag.nodes_set)
                is_dag = leaf_pdag.convert_to_dag(leaf_dag)
                if is_dag:
                    leaf_score = self.scoring_function(leaf_dag, self.scoring_data, self.node_size, en_nodes)
                else:
                    leaf_score = -float('inf')  # CPDAG does not admit any DAG extension
            else:
                leaf_score = 0

            multi_hyp_data = {
                'endogenous': en_nodes,
                'exogenous': ex_nodes,
                'ci-order': order,
                'graph': self.graph.copy(),
                'sepset': self.sepset.copy(en_nodes | ex_nodes),
                'score': leaf_score
            }
            leaf = MultiHypNode(cookie_children=None, multi_hyp_data=multi_hyp_data)
            return leaf

        cpdag_initial = self.graph  # remember the initial graph because each hypothesis overwrites self.graph
        sepset_initial = self.sepset  # remember the initial sepset because each hypothesis overwrites self.graph

        # generate multiple hypotheses for the further refinement of the graph
        # Recursive calls for each autonomous sub-graph
        cookie_nodes_list = []
        for hyp_id in range(self.num_of_hyp):
            self.graph = cpdag_initial.copy()  # initialize each hypothesis with the initial graph (erase previous hyp.)
            self.sepset = sepset_initial.copy()  # initialize each hypothesis with the initial separation set
            cookie_node = CookieNode()

            # Create a bootstrap sample of the original training data
            idx_sampled = np.random.choice(self.num_records, self.num_records, replace=True)  # sample data records
            self.ci_test.data = self.data[idx_sampled, :]  # set a bootstrap sample to be used for CI testing

            # refine using self.ci_test with condition set size equal to "order", and orient
            self._refine_and_orient(en_nodes=en_nodes, ex_nodes=ex_nodes, order=order)

            # split into ancestors/descendant autonomous sub-graphs (cookie slices: descendant_set, and ancestors_sets)
            descendant_set, ancestors_sets, a_nodes = self._split_ancestors_descendant(en_nodes=en_nodes)

            # record cookie structure
            cookie_slices = {
                'descendant-slice': descendant_set,
                'ancestor-slices': ancestors_sets
            }
            cookie_node.set_cookie_slices(cookie_slices)
            cookie_node.set_cookie_graph(self.graph.copy())

            # learn for ancestors sub-sets
            for ancestor in ancestors_sets:
                multi_hyp_ancestor = self.learn_recursively(en_nodes=ancestor, ex_nodes=ex_nodes,
                                                            order=order + 1)  # recursive call (ancestor)
                cookie_node.add_multi_hyp_ancestor(multi_hyp_ancestor)

            # learn for descendant sub-set
            multi_hyp_descendant = self.learn_recursively(en_nodes=descendant_set, ex_nodes=a_nodes | ex_nodes,
                                                              order=order + 1)  # recursive call (descendant)
            cookie_node.add_multi_hyp_descendant(multi_hyp_descendant)

            cookie_nodes_list.append(cookie_node)

        multi_hyp_data = {'endogenous': en_nodes, 'exogenous': ex_nodes, 'ci-order': order, 'score': None}
        multi_hyp_node = MultiHypNode(cookie_children=cookie_nodes_list, multi_hyp_data=multi_hyp_data)
        return multi_hyp_node

    def sample_cpdag(self, temperature=1):
        self.graph.create_empty_graph()
        self.sepset.erase()
        if self.is_scored:
            score = self._sample_cpdag_recursive(self.graph_generating_tree, self.graph, self.sepset, temperature)
        else:
            self._sample_cpdag_recursive_no_score(self.graph_generating_tree)
            score = None

        self._re_orient_skeleton()
        return score

    def _sample_cpdag_recursive(self, ggt_root, pdag: PDAG, sepset, temperature):
        """
        Update input PDAG and return score
        :param ggt_root: root of GGT (root of the sub-tree)
        :param pdag: PDAG object to be updated
        :param sepset: SeparationSet object to be updated
        :return: (log-)score of the updated portion of the PDAG
        """
        en_nodes = ggt_root.multi_hyp_data['endogenous']
        ex_nodes = ggt_root.multi_hyp_data['exogenous']

        if ggt_root.is_leaf:  # update PDAG and return score
            pdag.add_edges_from(source_pdag=ggt_root.multi_hyp_data['graph'],
                                en_nodes=en_nodes, ex_nodes=ex_nodes)  # get sub-graph stored in the leaf
            sepset.copy_from(source_sepset=ggt_root.multi_hyp_data['sepset'],
                                  nodes=en_nodes | ex_nodes)  # get separation sets stored in the leaf
            return ggt_root.multi_hyp_data['score']

        # we have several cookies. We need to get their scores and then return one w.r.t. the scores

        cookie_pdags = []
        cookie_sepsets = []
        cookie_scores = []
        for cookie_node in ggt_root.children:  # loop through cookies
            pdag_i = PDAG(self.graph.nodes_set)  # create a new PDAG to hold the sampled PDAG of the cookie
            sepset_i = SeparationSet(self.sepset.nodes_set)
            score_i = 0

            # sample a sub-graph for each of the ancestor sub-sets
            for ancestor_multi_hyp_node in cookie_node.multi_hyp_ancestors:
                score_i += self._sample_cpdag_recursive(  # update pdag_i
                    ggt_root=ancestor_multi_hyp_node, pdag=pdag_i, sepset=sepset_i, temperature=temperature)

            # sample a sub-graph for the descendant sub-set
            descendant_multi_hyp_node = cookie_node.multi_hyp_descendant  # for consistent naming
            score_i += self._sample_cpdag_recursive(# updade sepset_i
                ggt_root=descendant_multi_hyp_node, pdag=pdag_i, sepset=sepset_i, temperature=temperature)

            cookie_pdags.append(pdag_i)
            cookie_sepsets.append(sepset_i)
            cookie_scores.append(score_i)

        # sample a cookie
        if temperature > 0:  # temperature in Boltzmann distribution
            max_score = np.max(cookie_scores)

            if max_score != -float('inf'):
                log_scores = cookie_scores - max_score
                scores = np.exp(log_scores/temperature)
                scores /= scores.sum()
                sampled_cookie_idx = np.random.choice(len(scores), p=scores)  # weighted sampling
            else:
                sampled_cookie_idx = np.random.choice(len(cookie_scores))  # uniform sampling
        elif temperature == 0:  # get highest scoring CPDAG
            sampled_cookie_idx = np.argmax(cookie_scores)  # arg-max: select the maximal score
        else:
            sampled_cookie_idx = None  # this case should not happen in normal use
            assert (temperature >= 0)

        ggt_root.selected_cookie_idx = sampled_cookie_idx
        sampled_pdag = cookie_pdags[sampled_cookie_idx]
        sampled_sepset = cookie_sepsets[sampled_cookie_idx]
        sampled_score = cookie_scores[sampled_cookie_idx]
        ggt_root.multi_hyp_data['score'] = sampled_score  # score of the sampled path starting from the sampled cookie

        # update PDAG and sepset, and return score (trash cookie PDAGs and sepsets)
        pdag.add_edges_from(source_pdag=sampled_pdag, en_nodes=en_nodes, ex_nodes=ex_nodes)
        sepset.copy_from(source_sepset=sampled_sepset, nodes=en_nodes | ex_nodes)
        return sampled_score

    def _sample_cpdag_recursive_no_score(self, ggt_root):
        if ggt_root.is_leaf:
            cpdag_leaf = ggt_root.multi_hyp_data['graph']
            sepset_leaf = ggt_root.multi_hyp_data['sepset']
            en_nodes_leaf = ggt_root.multi_hyp_data['endogenous']
            ex_nodes_leaf = ggt_root.multi_hyp_data['exogenous']
            self.graph.add_edges_from(source_pdag=cpdag_leaf,
                                      en_nodes=en_nodes_leaf, ex_nodes=ex_nodes_leaf)  # get sub-graph stored in the leaf
            self.sepset.copy_from(source_sepset=sepset_leaf, nodes=en_nodes_leaf | ex_nodes_leaf)
            return

        cookie_idx = np.random.choice(len(ggt_root.children))
        self.selected_cookie_idx = cookie_idx
        cookie_node = ggt_root.children[cookie_idx]

        # sample a sub-graph for each of the ancestor sub-sets
        for ancestor_multi_hyp_node in cookie_node.multi_hyp_ancestors:
            self._sample_cpdag_recursive_no_score(ancestor_multi_hyp_node)

        # sample a sub-graph for the descendant sub-set
        descendant_multi_hyp_node = cookie_node.multi_hyp_descendant  # for consistent naming
        self._sample_cpdag_recursive_no_score(descendant_multi_hyp_node)

    def calc_graph_uncertainty(self, num_of_samples, threshold, temperature=1):
        graphs_list = []

        # sample cpdags
        for _ in range(num_of_samples):
            self.sample_cpdag(temperature)
            graphs_list.append(self.graph)

        # calculate skeleton
        skeleton = PDAG(nodes_set=self.graph.nodes_set)
        for node_i, node_j in combinations(self.graph.nodes_set, 2):
            count = 0.
            for cpdag in graphs_list:
                if cpdag.is_connected(node_i, node_j):
                    count += 1.

            if count > (threshold*num_of_samples):
                skeleton.add_edges(parents_set={node_i}, target_node=node_j, arrowhead_type=Mark.Undirected)

        return skeleton







