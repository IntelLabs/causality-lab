import numpy as np
from causal_discovery_algs import LearnStructICD
from causal_discovery_algs.icd import create_pds_tree
from causal_discovery_utils.cond_indep_tests import CondIndepParCorr
from causal_discovery_utils.stat_utils import cov_to_corr


class CLEANN:
    def __init__(self, attention_matrix: np.ndarray, num_samples, p_val_th: float, explanation_tester,
                 nodes_set=None, search_minimal=True, structure_learning_class=LearnStructICD):
        """
        Initialize a CLEANN explainer [https://arxiv.org/abs/2310.20307].
        :param attention_matrix: self-attention, square matrix from which to infer inter-token relations.
        :param num_samples: number of samples from which the attention was calculated (e.g., config.hidden_size).
        :param p_val_th: p-value threshold for the partial-correlation-based independence test.
        :param explanation_tester: an externally-defined function that tests if a subset of tokens is an explanation.
            It takes as input: a list of all tokens, a list of indexes of tokens subset to be considered as explanation,
            and the index of the target token for which we are seeking an explanation.
            It outputs True if the explanation is confirmed, otherwise False.
        :param nodes_set: a list of tokens, where each token will be a node in the learned graph.
        :param search_minimal: If True, only the explanations with the minimal size will be returned. If False, all
            the explanations found from the graph will be returned.
        :param structure_learning_class: structure learning class to instantiate. Default: LearnStructICD.
        """

        # calculate correlation matrix from attention matrix
        cov_matrix = np.matmul(attention_matrix, attention_matrix.transpose())  # COV = A @ transpose(A)
        corr_mat = cov_to_corr(cov_matrix)

        # prepare for learning a graph
        num_vars, _ = corr_mat.shape  # number of graph-nodes
        if nodes_set is None:
            nodes_set = set(range(num_vars))
        self.nodes_set = nodes_set
        self.ci_test = CondIndepParCorr(
            threshold=p_val_th, dataset=None, num_records=num_samples, num_vars=num_vars,
            count_tests=True, use_cache=True)
        self.ci_test.correlation_matrix = corr_mat
        self.StructureLearning = structure_learning_class
        self.graph = None

        # initialize for evaluating explanations
        self.results = dict()  # explanations found by the 'explain' method will be stored in this dictionary
        self.is_explanation = explanation_tester
        self._search_minimal = search_minimal

    def learn_graph(self):
        icd_alg = self.StructureLearning(nodes_set=self.nodes_set, ci_test=self.ci_test)  # init structure learner
        icd_alg.learn_structure()
        return icd_alg.graph

    def explain(self, target_node_idx, max_set_size=None, max_range=None):
        """
        Identify an explanation for the given target node. The result is stored in self.results['explanations'].
        :param target_node_idx: index of the node to be explained.
        :param max_set_size: setting a value limits the search to look for explanations
            having at most max_set_size nodes.
        :param max_range: setting a value limits the search to look for explanations
            such that the distance on the graph between the explanation nodes and the target node is at most max_range.
            containing nodes having at most max_range distance on the graph from the target node.
        :return: a list of minimal explanations (all explanations have the same size).
        """
        if target_node_idx in self.results:
            return

        # learn a Graph if one haven't been learned already
        if self.graph is None:
            self.graph = self.learn_graph()

        # create a PDS-tree rooted at the target node
        pds_tree, full_explain_set = create_pds_tree(self.graph, target_node_idx, max_depth=max_range)
        max_pds_tree_depth = pds_tree.get_max_depth()
        results = dict()
        results['pds_tree'] = pds_tree
        results['full_explanation_set'] = full_explain_set
        results['max_pds_tree_depth'] = max_pds_tree_depth

        if max_set_size is None:
            max_size = len(full_explain_set)
        else:
            max_size = max_set_size

        explanations_list = []
        found_explanation = False
        for set_size in range(1, max_size+1):
            sets_list = pds_tree.get_subsets_list(set_nodes=full_explain_set, subset_size=set_size)
            sets_list.sort(key=lambda x: x[1])  # sort with respect to the sum of minimal distances
            for possible_explanation_set in sets_list:
                if self.is_explanation(list(possible_explanation_set[0]), target_node_idx):
                    explanations_list.append(possible_explanation_set)
                    found_explanation = True
            if found_explanation and self._search_minimal:
                break

        results['explanations'] = explanations_list
        self.results[target_node_idx] = results
        return explanations_list
