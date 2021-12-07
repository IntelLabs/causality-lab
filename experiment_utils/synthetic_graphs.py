import random
import numpy as np
from graphical_models import DAG


def create_random_dag(num_nodes, expected_neighborhood_size):
    nodes_set = set(range(num_nodes))
    dag = DAG(nodes_set)

    # The probability of the presence of a directed edge is Bernoulli(expected_neighborhood_size/(num_nodes-1))
    for node_parent in range(num_nodes-1):
        for node_child in range(node_parent+1, num_nodes):
            is_edge = random.random() < (expected_neighborhood_size / (num_nodes-1))
            if is_edge:
                dag.add_edges({node_parent}, node_child)

    return dag


def create_random_dag_max_fan(num_nodes, max_fan_in):
    """
    Create a random DAG with bounded fan-in (number of parents per node)
    :param num_nodes: Number of nodes in the graph
    :param max_fan_in: Maximal number of parents per child
    :return: A random DAG with bounded fan-in
    """
    nodes_set = set(range(num_nodes))
    dag = DAG(nodes_set)  # create an empty graph

    for current_node_id in range(1, num_nodes):
        num_parents = random.randint(0, min(current_node_id, max_fan_in))  # sample number of parents
        if num_parents > 0:
            parents = set(random.sample(range(current_node_id), num_parents))  # sample parents
            dag.add_edges(parents_set=parents, target_node=current_node_id)

    return dag


def create_random_connected_dag(num_nodes, expected_neighborhood_size, num_dags_timeout=10000):
    is_conn = False
    acc = 0
    while not is_conn:
        if num_dags_timeout < acc:
            return None  # timed out: did not find a connected DAG

        dag_rand = create_random_dag(num_nodes, expected_neighborhood_size)
        is_conn = dag_rand.is_graph_connected()
        acc += 1

    return dag_rand


def select_latent_variables(graph: DAG):
    """
    Find nodes in a DAG that can serve as latent confounders. They comply with:
        1. They don't have incoming edges (parentless)
        2. Each of the is a parent of at least two nodes
    :param graph: A DAG for which to find the possible latents
    :return: A set of varaibles that can serve as latent confounders
    """
    # find parentless nodes
    parentless_set = set()
    for node in graph.nodes_set:
        if len(graph.parents(node)) == 0:
            parentless_set.add(node)

    # find parentless that have at least 2 children
    parents_set = set()
    for parent in parentless_set:
        if len(graph.find_children(parent)) >= 2:
            parents_set.add(parent)

    return parents_set


def create_random_dag_with_latents(n_nodes, conn_coeff):
    # sample a connected DAG
    dag_samp = create_random_connected_dag(n_nodes, conn_coeff, num_dags_timeout=1000000)

    # find nodes that can serve as latents (parentless, and parents of at least two observed nodes)
    potential_latents = select_latent_variables(dag_samp)

    # sample 50% of the potential latents
    lat_set = set(
        random.sample(potential_latents, len(potential_latents) // 2)
    )
    obs_set = dag_samp.nodes_set - lat_set
    return dag_samp, obs_set, lat_set


def sample_data_from_dag(in_dag, num_samples, min_edge_weight, max_edge_weight):
    """
    Sample data from a linear SEM. A linear SEM is created from a DAG.
    A node is the sum of a normally distributed noise term and the weighted sum of the values of its parents.
    :param in_dag: The DAG structure of the linear SCM
    :param num_samples: number of samples (dataset records)
    :param min_edge_weight: lowest absolute value of edge weight (linear coefficient)
    :param max_edge_weight: highest absolute value of the weight (linear coefficient)
    :return: Sampled dataset in the form of a 2D NumPy array
    """
    # ToDo: create a dedicated module for probabilistic graphical models. These should take graph structure as input
    data = np.random.randn(num_samples, len(in_dag.nodes_set))  # sample noise: N(0,1)
    topological_order = in_dag.find_topological_order()
    for node in topological_order:
        parents_set = in_dag.parents(node)
        for node_parent in parents_set:
            weight_sign = 2 * random.randint(0, 1) - 1  # select positive or negative range for the edge weight
            weight = weight_sign * np.random.uniform(min_edge_weight,
                                                     max_edge_weight)  # considers negative weights as well
            data[:, node] += weight * data[:, node_parent]  # add the linear effect of the parents

    return data
