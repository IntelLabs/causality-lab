from itertools import combinations
import numpy as np
from scipy.special import gammaln
from graphical_models import PDAG, DAG
from causal_discovery_utils.data_utils import calc_stats
from causal_discovery_utils.cond_indep_tests import DSep  # perfect CI oracle, used here to find the true PAG
from causal_discovery_algs.icd import LearnStructICD  # used here to find the true PAG


def find_true_pag(true_dag, true_observed_set):
    perfect_ci_test = DSep(true_dag=true_dag)
    icd_true = LearnStructICD(nodes_set=true_observed_set, ci_test=perfect_ci_test)
    icd_true.learn_structure()  # find the PAG using the ICD algorithm
    return icd_true.graph


def calc_skeleton_accuracy(graph_tested, graph_correct):
    num_true_positive = 0
    num_false_negative = 0  # missing edges
    num_false_positive = 0  # extra edges
    num_true_negative = 0
    num_edges_true = 0  # count the number of edges in the true graph (== false_negative + true_positive)

    for (node_i, node_j) in combinations(graph_correct._graph, 2):

        # calculate edge errors
        if graph_correct.is_connected(node_i, node_j):
            num_edges_true += 1  # count the number of edges in the true PAG
            if graph_tested.is_connected(node_i, node_j):
                num_true_positive += 1
            else:
                num_false_negative += 1

        else:  # there is no edge in the true graph
            if graph_tested.is_connected(node_i, node_j):
                num_false_positive += 1
            else:
                num_true_negative += 1

    edge_precision = num_true_positive / (num_false_positive + num_true_positive)
    edge_recall = num_true_positive / num_edges_true
    edge_f1 = 2 * edge_precision * edge_recall / (edge_precision + edge_recall)  # 2 / (1/recision + 1/recall)
    FPR = num_false_positive / (num_false_positive+num_true_negative)  # false positive rate (FPR)
    FNR = num_false_negative / (num_false_negative+num_true_positive)  # false negative rate (FNR)
    skeleton_accuracy = {
        'edge_precision': edge_precision,
        'edge_recall': edge_recall,
        'edge_F1': edge_f1,
        'FPR': FPR,
        'FNR': FNR
    }
    return skeleton_accuracy


def structural_hamming_distance_cpdag(tested_graph: PDAG, true_graph: PDAG):
    """
    Measure structural hamming distance between two CPDAGs. The following are calculated:
        * Edges (directed or undirected)
            ** Missing: an edge missing from the tested graph but existing in the true graph
            ** Extra:   an edge in the tested graph but missing from the true graph
        * Arrowhead (for edges existing existing on both graphs)
            ** Missing:     undirected in tested graph but directed in true graph
            ** Extra:       directed in tested graph but undirected in true graph
            ** Reversed:    directed in opposite direction

    The total SHD is the sum of all values in the returned dictionary

    :param tested_graph:
    :param true_graph:
    :return: A nested dictionary
    """
    if (not isinstance(true_graph, PDAG)) or (not isinstance(tested_graph, PDAG)):
        raise ValueError

    shd_edge = {'missing': 0, 'extra': 0}
    shd_arrowhead = {'missing': 0, 'extra': 0, 'reversed': 0}
    for (node_i, node_j) in combinations(true_graph._graph, 2):
        if not tested_graph.is_connected(node_i, node_j):  # if edge is missing from the tested graph
            if true_graph.is_connected(node_i, node_j):  # if edge exists in true graph
                shd_edge['missing'] += 1
        elif not true_graph.is_connected(node_i, node_j):  # edge exists in tested graph; check is missing from true
            shd_edge['extra'] += 1
        else:  # edge exists in both true and tested graphs
            # now check direction error
            if node_i in true_graph.undirected_neighbors(node_j):  # if the edge is undirected in the true graph
                if (node_i in tested_graph.parents(node_j)) or (node_j in tested_graph.parents(node_i)):
                    shd_arrowhead['extra'] += 1  # the edge in the tested graph is directed
            elif node_i in tested_graph.undirected_neighbors(node_j):  # directed in true; check if undirected in tested
                shd_arrowhead['missing'] += 1
            else:  # both edges are directed
                (source, target) = (node_i, node_j) if node_i in true_graph.parents(node_j) else (node_j, node_i)
                if target in tested_graph.parents(source):
                    shd_arrowhead['reversed'] += 1  # the edges are not directed in the same direction (i --> j)

    shd_total = sum(shd_edge.values()) + sum(shd_arrowhead.values())

    return {'total': shd_total, 'edge': shd_edge, 'arrowhead': shd_arrowhead}


def score_bdeu(dag: DAG, data, node_size, en_nodes=None):
    """
    Calculate the BDeu score of a DAG
    :param dag: DAG to be scored
    :param data: dataset of discrete random variables from which to calculate the score
    :param node_size: sizes of the nodes: number of possible values for each variable in the dataset
    :param en_nodes: the score will be calculated for the sub-graph induced by these nodes
    :return: BDeu score
    """
    if dag is None:
        return -float('inf')

    assert isinstance(dag, DAG)  # graph must be a DAG

    if en_nodes is None:
        en_nodes = dag.nodes_set

    score = 0
    for node in en_nodes:
        parents = dag.parents(node)
        family = tuple(parents) + (node, )  # a tuple family nodes where the child ("node") is last
        family_sizes = [node_size[node_i] for node_i in family]
        family_data = data[:, family]
        counts = calc_stats(family_data, family_sizes)
        if counts is None:  # memory error
            return -float('inf')
        counts = np.reshape(counts, [-1, family_sizes[-1]], order='F')  # 2nd axis is the states of the child

        prior = np.ones_like(counts)
        prior = prior/prior.sum()

        score += score_family_dirichlet(counts=counts, prior=prior)

    return score


def score_family_dirichlet(counts, prior):
    """
    Score a family: a node and its parents

    :param counts: a matrix of counts, where the 2nd dimension belongs to the child in the family
    :param prior: prior
    :return: score of the family
    """
    lu = (gammaln(prior + counts) - gammaln(prior)).sum(axis=1)
    alpha_ij = prior.sum(axis=1)
    n_ij = counts.sum(axis=1)
    lv = gammaln(alpha_ij) - gammaln(alpha_ij + n_ij)

    family_log_likelihood = (lu + lv).sum()
    return family_log_likelihood


def calc_structural_accuracy_pag(pag_tested, pag_correct):
    """
    Calculate structural accuracy:
      - Edge accuracy: precision and recall
      - Orientation accuracy: number of correctly identified edge-marks (variant:'o--', head:'<--', tail:'---')
    :param pag_tested:
    :param pag_correct:
    :return: a dictionary of the form:
             { 'edge_precision': value, 'edge_recall': value, 'orientation_correctness': value }
    """

    num_orient_correct = 0
    num_orient_total = 0  # number edge-marks in the true PAG

    for (node_i, node_j) in combinations(pag_correct._graph, 2):
        if pag_correct.is_connected(node_i, node_j):
            num_orient_total += 2
            for edge_mark in pag_correct.edge_mark_types:  # check which edge mark is present
                if node_i in pag_correct._graph[node_j][edge_mark] and node_i in pag_tested._graph[node_j][edge_mark]:
                    num_orient_correct += 1
                if node_j in pag_correct._graph[node_i][edge_mark] and node_j in pag_tested._graph[node_i][edge_mark]:
                    num_orient_correct += 1

    edge_accuracy = calc_skeleton_accuracy(pag_tested, pag_correct)
    edge_precision = edge_accuracy['edge_precision']
    edge_recall = edge_accuracy['edge_recall']
    edge_f1 = edge_accuracy['edge_F1']
    causal_accuracy = num_orient_correct/num_orient_total  # percentage of correct orientations

    result = {
        'FPR': edge_accuracy['FPR'],
        'FNR': edge_accuracy['FNR'],
        'edge_precision': edge_precision,
        'edge_recall': edge_recall,
        'edge_F1': edge_f1,
        'orientation_correctness': causal_accuracy
    }
    return result


def calc_skeleton_fnr_fpr(graph_tested, graph_correct):
    # ToDo: implement
    skeleton_acc = calc_skeleton_accuracy(graph_tested=graph_tested, graph_correct=graph_correct)
    res = {'FPR': skeleton_acc['FPR'], 'FNR': skeleton_acc['FNR']}
    return res
