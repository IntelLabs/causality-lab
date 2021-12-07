from itertools import combinations


class LearnStructBase:
    """
    Base class for constraint-based structure learning algorithms
    """
    def __init__(self, graph_class, nodes_set, ci_test):
        if not isinstance(nodes_set, set):
            raise ValueError('nodes_set should be a set type')
        self.ci_test = ci_test
        self.sepset = SeparationSet(nodes_set)
        self.graph = graph_class(nodes_set)  # e.g., graph_class=PDAG (for CPDAG under causal sufficiency) or PAG
        self.graph.sepset = self.sepset  # link the algorithm's updated sepset to the graph (by reference)

    def get_graph(self):
        return self.graph.get_graph()


class SeparationSet:
    def __init__(self, nodes_set):
        self._sepset = dict()
        self.nodes_set = nodes_set
        for (i, j) in combinations(nodes_set, 2):
            hkey = self.get_hash_key(i, j)
            self._sepset[hkey] = set()

    @staticmethod
    def get_hash_key(node_1, node_2):
        """
        Get the hash key used to store separation sets
        :param node_1:
        :param node_2:
        :return: Hash key for sepset dictionary
        """
        hkey = (node_1, node_2) if node_1 < node_2 else (node_2, node_1)
        return hkey

    def erase(self):
        for key in self._sepset:
            self._sepset[key] = set()

    def set_sepset(self, node_1, node_2, sepset):
        hkey = self.get_hash_key(node_1, node_2)
        self._sepset[hkey] = set(sepset)

    def get_sepset(self, node_1, node_2):
        hkey = self.get_hash_key(node_1, node_2)
        return self._sepset[hkey]

    def copy(self, nodes=None, target_sepset=None):
        if nodes is None:
            nodes = self.nodes_set

        if target_sepset is None:
            target_sepset = SeparationSet(nodes)

        for (i, j) in combinations(nodes, 2):
            hkey_source = self.get_hash_key(i, j)
            hkey_target = target_sepset.get_hash_key(i, j)
            target_sepset._sepset[hkey_target] = self._sepset[hkey_source].copy()  # create a copy of the separation-set

        return target_sepset

    def copy_from(self, source_sepset, nodes):
        """
        Selectively copy values from another SeparationSet instance
        :param source_sepset: source SeparationSet instance
        :param nodes: Nodes of interest (separation sets for pairs of these nodes will be copied)
        :return:
        """
        for (i, j) in combinations(nodes, 2):
            hkey_target = self.get_hash_key(i, j)
            self._sepset[hkey_target] = source_sepset.get_sepset(i, j).copy()  # copy separation-sets from external


def unique_element_iterator(chained_iterators):
    """
    return the unique instances of the chained iterators
    :param an iterator with possibly repeating elements, e.g., chained_iterators: chain(iter_a, iter_b)
    :return: an iterator with unique (unordered) elements
    """
    seen = set()
    for e in chained_iterators:
        if e in seen:
            continue

        seen.add(e)
        yield e
