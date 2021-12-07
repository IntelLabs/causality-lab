from unittest import TestCase
from graphical_models import DAG
import numpy as np

class TestDAG(TestCase):
    def test_find_topological_order_groups(self):
        latent = {'A', 'B', 'C', 'D', 'E'}
        observed = {'Sun', 'W', 'T', 'Z', 'V', 'X', 'U', 'Y'}
        dag = DAG(latent | observed)
        dag.add_edges(parents_set={'A', 'B', 'T', 'V'}, target_node='X')
        dag.add_edges(parents_set={'E', 'U'}, target_node='Y')
        dag.add_edges(parents_set={'B', 'Z'}, target_node='U')
        dag.add_edges(parents_set={'E', 'D'}, target_node='V')
        dag.add_edges(parents_set={'Sun', 'C', 'D'}, target_node='Z')
        dag.add_edges(parents_set={'W', 'C'}, target_node='T')
        dag.add_edges(parents_set={'W', 'A'}, target_node='Sun')

        # correct topological order (found by recursively eliminating leaves)
        correct_groups_order = [{'W', 'A'}, {'D', 'C', 'Sun'}, {'B', 'E', 'Z'}, {'V', 'U', 'T'}, {'X', 'Y'}]

        group_list = dag.find_topological_order_groups(latent | observed)
        for group_test, group_correct in zip(group_list, correct_groups_order):
            self.assertSetEqual(group_test, group_correct)

    def test_get_adj_mat(self):
        nodes_list = [2, 4, 6, 1]
        dag = DAG(nodes_set=set(nodes_list))
        dag.add_edges(parents_set={2, 4}, target_node=6)  # a v-structure
        dag.add_edges(parents_set={6}, target_node=1)

        adj_mat1, nodes_list1 = dag.get_adj_mat()
        self.assertListEqual(nodes_list1, sorted(nodes_list))
        adj_true = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0]])
        self.assertFalse(np.any(adj_true != adj_mat1))

        adj_mat2 = dag.get_adj_mat(en_nodes_list=nodes_list)
        adj_true = np.array([[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        self.assertFalse(np.any(adj_true != adj_mat2))

        adj_mat3 = dag.get_adj_mat(en_nodes_list=[6, 2, 4, 1])
        adj_true = np.array([[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
        self.assertFalse(np.any(adj_true != adj_mat3))

        latent = {'A', 'B', 'C', 'D', 'E'}
        observed = {'Sun', 'W', 'T', 'Z', 'V', 'X', 'U', 'Y'}
        dag = DAG(latent | observed)
        dag.add_edges(parents_set={'A', 'B', 'T', 'V'}, target_node='X')
        dag.add_edges(parents_set={'E', 'U'}, target_node='Y')
        dag.add_edges(parents_set={'B', 'Z'}, target_node='U')
        dag.add_edges(parents_set={'E', 'D'}, target_node='V')
        dag.add_edges(parents_set={'Sun', 'C', 'D'}, target_node='Z')
        dag.add_edges(parents_set={'W', 'C'}, target_node='T')
        dag.add_edges(parents_set={'W', 'A'}, target_node='Sun')
        adj = dag.get_adj_mat(en_nodes_list=['A', 'B', 'C', 'D', 'E', 'Sun', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
        adj_true = np.array([
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])
        self.assertFalse(np.any(adj_true != adj))




