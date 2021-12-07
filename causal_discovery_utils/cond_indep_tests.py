# This file contains conditional independence tests

import math
import numpy as np
from causal_discovery_utils.data_utils import calc_stats
from causal_discovery_utils.data_utils import get_var_size
from graphical_models import DAG, UndirectedGraph
from scipy.stats import norm


class CacheCI:
    """
    A cache for CI tests.
    """
    def __init__(self, num_vars=None):
        """
        Initialize cache
        :param num_vars: Number of variables; if None, cache is not initialized
        """
        if num_vars is None:
            self._cache = None
        else:
            self._cache = dict()
            # for each pair create a dictionary that holds the cached ci test. The sorted condition set is the hash key
            for i in range(num_vars - 1):
                for j in range(i + 1, num_vars):
                    hkey, _ = self.get_hkeys(i, j, ())  # get a key for the (i, j) pair (simply order them)
                    self._cache[hkey] = dict()

    def get_hkeys(self, x, y, zz):
        """
        Return a keys for hashing variable-pair and for the condition set
        :param x: 1st variable
        :param y: 2nd variable
        :param zz: Set of variables that consist of the condition set
        :return:
        """
        hkey = (x, y) if x < y else (y, x)
        hkey_cond_set = tuple(sorted(zz))
        return hkey, hkey_cond_set

    def set_cache_result(self, x, y, zz, res):
        """
        Set (override previous value) a result to be cached
        :param x: 1st variable
        :param y: 2nd variable
        :param zz: Variables that consists of the condition set
        :param res: Result to be cached
        :return:
        """
        assert self._cache is not None

        hkey, hkey_cond_set = self.get_hkeys(x, y, zz)  # get keys for hashing
        self._cache[hkey][hkey_cond_set] = res  # cache, override previous result

    def get_cache_result(self, x, y, zz):
        """
        Get previously cached result
        :param x: 1st variable
        :param y: 2nd variable
        :param zz: Variables that consists of the condition set
        :return: Cached result. None if nothing was cached previously
        """
        if self._cache is None:  # is cache data structure was initialized?
            return None

        hkey, hkey_cond_set = self.get_hkeys(x, y, zz)

        if hkey not in self._cache.keys():  # check if variable-pair cache data structure was created
            return None

        if hkey_cond_set not in self._cache[hkey].keys():  # check is result was ever cached
            return None

        return self._cache[hkey][hkey_cond_set]

    def del_cache(self, x, y, zz):
        """
        Removed cached entry.
        :param x: 1st variable
        :param y: 2nd variable
        :param zz: Variables that consists of the condition set
        :return: Cached result that was deleted
        """
        if self._cache is None:  # is cache data structure was initialized?
            return None

        hkey, hkey_cond_set = self.get_hkeys(x, y, zz)

        if hkey not in self._cache.keys():  # check if variable-pair cache data structure was created
            return None

        if hkey_cond_set not in self._cache[hkey].keys():  # check is result was ever cached
            return None

        return self._cache[hkey].pop(hkey_cond_set)


class DSep:
    """
    An optimal CI oracle that uses the true DAG and returns d-separation result
    """
    def __init__(self, true_dag: DAG, count_tests=False, use_cache=False, verbose=False):
        assert isinstance(true_dag, DAG)
        self.true_dag = true_dag

        self.verbose = verbose

        num_nodes = len(true_dag.nodes_set)

        self.count_tests = count_tests
        if count_tests:
            self.test_counter = [0 for _ in range(num_nodes-1)]
        else:
            self.test_counter = None

        self.is_cache = use_cache
        if use_cache:
            self.cache_ci = CacheCI(num_nodes)
        else:
            self.cache_ci = CacheCI(None)

    def cond_indep(self, x, y, zz):
        res = self.cache_ci.get_cache_result(x, y, zz)

        if res is None:
            res = self.true_dag.dsep(x, y, zz)
            if self.verbose:
                print('d-sep(', x, ',', y, '|', zz, ')', '=', res)
            if self.is_cache:
                self.cache_ci.set_cache_result(x, y, zz, res)
            if self.count_tests:
                self.test_counter[len(zz)] += 1  # update counter only if the test was not previously cached
        return res


class StatCondIndep:
    def __init__(self,
                 dataset, threshold, database_type,
                 retained_edges=None, count_tests=False, use_cache=False, verbose=False):
        """
        Base class for statistical conditional independence tests
        :param dataset:
        :param threshold:
        :param retained_edges: an undirected graph containing edges between nodes that are dependent (not to be tested)
        :param count_tests: if True, count the number of CI test queries (default: False). Mainly for debug
        """
        self.verbose = verbose

        data = np.array(dataset, dtype=database_type)
        num_records, num_vars = data.shape

        if retained_edges is None:
            self.retained_graph = UndirectedGraph(set(range(num_vars)))
            self.retained_graph.create_empty_graph()
        else:
            self.retained_graph = retained_edges

        node_size = get_var_size(data)

        self.data = data
        self.num_records = num_records
        self.num_vars = num_vars
        self.node_size = node_size
        self.threshold = threshold

        # Initialize counter of CI tests per conditioning set size
        self.count_tests = count_tests
        if count_tests:
            self.test_counter = [0 for _ in range(num_vars-1)]
        else:
            self.test_counter = None

        # Initialize cache
        self.is_cache = use_cache
        if use_cache:
            self.cache_ci = CacheCI(num_vars)
        else:
            self.cache_ci = CacheCI(None)

    def cond_indep(self, x, y, zz):
        if self.is_edge_retained(x, y):
            return False  # do not test and return: "not independent"

        statistic = self.cache_ci.get_cache_result(x, y, zz)

        if statistic is None:
            statistic = self.calc_statistic(x, y, zz)  # calculate correlation level
            self._debug_process(x, y, zz, statistic)
            self._cache_it(x, y, zz, statistic)

        res = statistic > self.threshold  # test if p-value is greater than the threshold
        return res

    def calc_statistic(self, y, x, zz):
        return None  # you must override this function in inherited classes

    def _debug_process(self, x, y, zz, res):
        """
        Handles all tasks required for debug
        """
        if self.verbose:
            print('Test: ', 'CI(', x, ',', y, '|', zz, ')', '=', res)
        if self.count_tests:
            self.test_counter[len(zz)] += 1

    def _cache_it(self, x, y, zz, res):
        """
        Handles all task required after calculating the CI statistic
        """
        if self.is_cache and (res is not None):
            self.cache_ci.set_cache_result(x, y, zz, res)

    def is_edge_retained(self, x, y):
        return self.retained_graph.is_connected(x, y)


class CondIndepParCorr(StatCondIndep):
    def __init__(self, threshold, dataset, retained_edges=None, count_tests=False, use_cache=False):
        super().__init__(dataset, threshold, np.float, retained_edges, count_tests, use_cache)
        self.correlation_matrix = np.corrcoef(self.data.T)
        self.data = None  # no need to store the data, as we have the correlation matrix

    def calc_statistic(self, x, y, zz):
        corr_coef = self.correlation_matrix  # for readability
        if len(zz) == 0:
            par_corr = corr_coef[x, y]
        elif len(zz) == 1:
            z = zz[0]
            par_corr = (
                (corr_coef[x, y] - corr_coef[x, z]*corr_coef[y, z]) /
                np.sqrt((1-np.power(corr_coef[x, z], 2)) * (1-np.power(corr_coef[y, z], 2)))
            )
        else:  # zz contains 2 or more variables
            all_var_idx = (x, y) + zz
            corr_coef_subset = corr_coef[np.ix_(all_var_idx, all_var_idx)]
            inv_corr_coef = -np.linalg.pinv(corr_coef_subset)  # consider using pinv instead of inv
            par_corr = inv_corr_coef[0, 1] / np.sqrt(abs(inv_corr_coef[0, 0]*inv_corr_coef[1, 1]))

        z = np.log1p(2*par_corr / (1-par_corr))  # log( (1+par_corr)/(1-par_corr) )

        val_for_cdf = abs(
            np.sqrt(self.num_records - len(zz) - 3) *
            0.5 * z
        )

        statistic = 2*(1-norm.cdf(val_for_cdf))
        return statistic


class CondIndepCMI(StatCondIndep):
    def __init__(self, dataset, threshold, retained_edges=None, count_tests=False, use_cache=False):
        super().__init__(dataset, threshold, np.int, retained_edges, count_tests, use_cache)

    def cond_indep(self, x, y, zz):
        res = super().cond_indep(x, y, zz)
        return not res  # invert the decision because the statistic is correlation level and not p-value

    def calc_statistic(self, x, y, zz):
        """
        Calculate conditional mutual information for discrete variables
        :param x: 1st variable (index)
        :param y: 2nd variable (index)
        :param zz: condition set, a tuple. e.g., if zz contains a single value zz = (val,)
        :return: Empirical conditional mutual information
        """
        all_var_idx = (x, y) + zz
        dd = self.data[:, all_var_idx]
        sz = [self.node_size[node_i] for node_i in all_var_idx]

        hist_count = calc_stats(data=dd, var_size=sz)
        if hist_count is None:  # memory error
            return 0
        hist_count = np.reshape(hist_count, [sz[0], sz[1], -1], order='F')  # 3rd axis is the states of condition set

        xsize, ysize, csize = hist_count.shape

        # Calculate conditional mutual information
        cmi = 0
        for zi in range(csize):
            cnt = hist_count[:, :, zi]
            cnum = cnt.sum()
            for node_i in range(self.node_size[x]):
                for node_j in range(self.node_size[y]):
                    if cnt[node_i, node_j] > 0:
                        cnt_val = cnt[node_i, node_j]
                        cx = cnt[:, node_j].sum()  # sum over y for specific x-state
                        cy = cnt[node_i, :].sum()  # sum over x for specific y-state

                        lg = math.log(cnt_val*cnum / (cx * cy))
                        cmi_ = lg*cnt_val/self.num_records
                        cmi += cmi_
        return cmi
