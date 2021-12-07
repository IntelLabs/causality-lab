import numpy as np


def get_var_size(data):
    num_records, num_vars = data.shape
    node_size = np.zeros((num_vars,), dtype='int')
    for var in range(num_vars):
        node_size[var] = data[:, var].max() + 1
    # node_size = data.max(axis=0)+1  # number of node states (smallest state 0)
    return node_size


def calc_stats(data, var_size):
    """
    Calculate the counts of instances in the data
    :param data:
    :param var_size:
    :return:
    """
    sz_cum_prod = [1]
    for node_i in range(len(var_size) - 1):
        sz_cum_prod += [sz_cum_prod[-1] * var_size[node_i]]

    sz_prod = sz_cum_prod[-1] * var_size[-1]

    data_idx = np.dot(data, sz_cum_prod)
    try:
        hist_count, _ = np.histogram(data_idx, np.arange(sz_prod + 1))
    except MemoryError as error:
        print('Out of memory')
        return None
    return hist_count
