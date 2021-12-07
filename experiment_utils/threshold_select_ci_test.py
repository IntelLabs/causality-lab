from causal_discovery_utils.performance_measures import score_bdeu
from graphical_models import DAG

def search_threshold_bdeu(alg_class, train_data, ci_test_class, th_range_list, use_cache=True):
    """
    A grid-search for the threshold that maximizes the BDeu score when learning a DAG structure
    :param alg_class: Class of the algorithm to be used to learn the graph
    :param train_data: Data that is used for calculating the BDeu score
    :param ci_test_class: Class of CI test for which we are searching a threshold
    :param th_range_list: A list of candidate threshold
    :param use_cache: CI test statistic is cached and re-used when evaluating different thresholds (default=TRUE).
            do not use caching for B-RAI algorithm as it changes the data during its operation.
    :return: The threshold that returned the structure having the highest score, and a list of all candidates scores
    """
    _n_samples, _n_vars = train_data.shape
    _nodes = set(range(_n_vars))
    best_th = float("inf")
    best_score = -float("inf")
    score_list = []

    _ci_test = ci_test_class(dataset=train_data, threshold=None, use_cache=use_cache)  # conditional independence oracle
    for i, _th in enumerate(th_range_list):
        _ci_test.threshold = _th
        _alg = alg_class(nodes_set=_nodes, ci_test=_ci_test)  # algorithm instance
        _alg.learn_structure()  # learn structure
        _dag = DAG(_alg.graph.nodes_set)
        is_dag = _alg.graph.convert_to_dag(_dag)
        if is_dag == True:
            _score = score_bdeu(_dag, train_data, _ci_test.node_size)
        else:
            _score = -float("inf")
        if _score >= best_score:  # if several threshold result in equal scores, get the highest threshold
            best_score = _score
            best_th = _th
        score_list.append(_score)

    return best_th, score_list
