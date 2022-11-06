import numpy as np


def get_var_size(data):
    num_records, num_vars = data.shape
    node_size = np.zeros((num_vars,), dtype='int')
    for var in range(num_vars):
        node_size[var] = data[:, var].max() + 1
    # node_size = data.max(axis=0)+1  # number of node states (smallest state 0)
    return node_size


def calc_stats(data, var_size, weights=None):
    """
    Calculate the counts of instances in the data
    :param data: a dataset of categorical features
    :param var_size: a vector defining the cardinalities of the features
    :param weights: a vector of non-negative weights for data samples.
    :return:
    """
    sz_cum_prod = [1]
    for node_i in range(len(var_size) - 1):
        sz_cum_prod += [sz_cum_prod[-1] * var_size[node_i]]

    sz_prod = sz_cum_prod[-1] * var_size[-1]

    data_idx = np.dot(data, sz_cum_prod)
    try:
        # hist_count, _ = np.histogram(data_idx, np.arange(sz_prod + 1), weights=weights)
        hist_count = np.bincount(data_idx, minlength=sz_prod, weights=weights)
    except MemoryError as error:
        print('Out of memory')
        return None
    return hist_count


def unroll_temporal_data(data_full, observed_nodes_list, window_len, t_step=1):
    """
    Unroll temporally sorted data samples into the defined time window. For example, if the time window is 2, then
    each two consecutive data samples are concatenated into a single sample. The new samples are sorted temporally.
    :param data_full: Temporally sorted data sample. Each sample consists of jointly measured values.
    :param observed_nodes_list: indexes of columns in the original data that correspond to 'observed' variables.
    :param window_len: Length (number of time-stamps) of the unrolling window (current-time + window_len-1 past-steps).
    :param t_step: Nuber of time-step skips between unrolled samples (default is 1).
    :return: An unrolled temporally sorted data samples.
    """
    n_samples = data_full.shape[0]
    n_contemporaneous_nodes = data_full.shape[1]
    num_nodes_unrolled = window_len * n_contemporaneous_nodes  # number of variables in a single unrolled sample

    # calculate the starting time index for each unrolled sample
    starts = [xid * n_contemporaneous_nodes for xid in np.arange(0, n_samples + 1 - window_len, t_step)]

    # create unrolled data
    data_full_unrolled = np.zeros((len(starts), num_nodes_unrolled))  # initialize unrolled data
    for i, st in enumerate(starts):
        data_full_unrolled[i] = data_full.flat[st:st + num_nodes_unrolled]

    # indexes of variables in the unrolled data
    _nodes_sets_list_full = np.reshape(range(num_nodes_unrolled), (window_len, n_contemporaneous_nodes))
    _nodes_sets_list = [_nodes_sets_list_full[i, observed_nodes_list].tolist() for i in range(window_len)]  # observed

    return data_full_unrolled, _nodes_sets_list_full, _nodes_sets_list
