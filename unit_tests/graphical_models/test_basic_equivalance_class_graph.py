from unittest import TestCase
from graphical_models import PAG, DAG, arrow_head_types as Mark
from itertools import combinations


class TestECGraph(TestCase):
    def test_find_adjacent_nodes(self):
        # define a PAG: 0 is a common member of all other disconnected nodes
        nodes = set(range(10))
        pag = PAG(nodes)
        pag.add_edge(0, 1, Mark.Tail, Mark.Directed)
        pag.add_edge(0, 6, Mark.Tail, Mark.Directed)
        pag.add_edge(0, 4, Mark.Tail, Mark.Directed)
        pag.add_edge(0, 3, Mark.Directed, Mark.Directed)
        pag.add_edge(0, 5, Mark.Directed, Mark.Directed)
        pag.add_edge(0, 2, Mark.Circle, Mark.Directed)
        pag.add_edge(0, 7, Mark.Directed, Mark.Tail)
        pag.add_edge(0, 8, Mark.Circle, Mark.Circle)
        pag.add_edge(0, 9, Mark.Tail, Mark.Tail)

        # test common use (consider any edge-mark/edge-type)
        neighbors = pag.find_adjacent_nodes(0, nodes)
        self.assertSetEqual(nodes - {0}, neighbors)
        neighbors = pag.find_adjacent_nodes(0, nodes - {1, 2})
        self.assertSetEqual(nodes - {0, 1, 2}, neighbors)

        # find neighbors connected via specific edge-type
        neighbors = pag.find_adjacent_nodes(0, None, (Mark.Tail, Mark.Tail))
        self.assertSetEqual({9}, neighbors)
        neighbors = pag.find_adjacent_nodes(0, None, (Mark.Tail, Mark.Directed))
        self.assertSetEqual({1, 6, 4}, neighbors)
        neighbors = pag.find_adjacent_nodes(0, None, (Mark.Directed, Mark.Directed))
        self.assertSetEqual({3, 5}, neighbors)
        neighbors = pag.find_adjacent_nodes(0, None, (Mark.Directed, Mark.Tail))
        self.assertSetEqual({7}, neighbors)
        neighbors = pag.find_adjacent_nodes(0, None, (Mark.Directed, Mark.Circle))
        self.assertSetEqual(set(), neighbors)
        neighbors = pag.find_adjacent_nodes(0, None, (Mark.Circle, Mark.Circle))
        self.assertSetEqual({8}, neighbors)
        neighbors = pag.find_adjacent_nodes(0, None, (Mark.Circle, Mark.Directed))
        self.assertSetEqual({2}, neighbors)
        neighbors = pag.find_adjacent_nodes(0, None, (Mark.Circle, Mark.Tail))
        self.assertSetEqual(set(), neighbors)

    def test_find_reachable_set(self):
        nodes = set(range(10))
        pag = PAG(nodes)
        pag.add_edge(5, 1, Mark.Tail, Mark.Directed)
        pag.add_edge(5, 2, Mark.Directed, Mark.Circle)
        pag.add_edge(1, 4, Mark.Tail, Mark.Directed)
        pag.add_edge(1, 9, Mark.Tail, Mark.Directed)
        pag.add_edge(1, 3, Mark.Circle, Mark.Circle)
        pag.add_edge(4, 2, Mark.Circle, Mark.Tail)
        pag.add_edge(1, 2, Mark.Circle, Mark.Circle)
        pag.add_edge(2, 6, Mark.Circle, Mark.Directed)
        pag.add_edge(6, 7, Mark.Circle, Mark.Circle)
        pag.add_edge(7, 8, Mark.Tail, Mark.Directed)
        pag.add_edge(9, 0, Mark.Directed, Mark.Directed)

        reachable = pag.find_reachable_set(anchor_node=5, nodes_pool=nodes,
                                           edge_type_list=[(Mark.Tail, Mark.Directed), (Mark.Circle, Mark.Directed)])
        self.assertSetEqual({1, 4, 9}, reachable)

    def test_find_unconnected_subgraphs(self):
        print('Test finding unconnected sub-graph')
        nodes = set(range(10))
        pag = PAG(nodes)
        pag.add_edge(0, 4, Mark.Directed, Mark.Circle)
        pag.add_edge(4, 2, Mark.Directed, Mark.Directed)
        pag.add_edge(2, 8, Mark.Directed, Mark.Directed)
        pag.add_edge(3, 5, Mark.Directed, Mark.Directed)
        pag.add_edge(6, 5, Mark.Circle, Mark.Tail)
        pag.add_edge(7, 9, Mark.Circle, Mark.Circle)

        sub_graphs = pag.find_unconnected_subgraphs()
        self.assertEqual(4, len(sub_graphs))
        self.assertIn({0, 2, 4, 8}, sub_graphs)
        self.assertIn({3, 5, 6}, sub_graphs)
        self.assertIn({7, 9}, sub_graphs)
        self.assertIn({1}, sub_graphs)

        # test on a subset of nodes. Nodes outside this subset are considered "blocking" nodes
        sub_graphs = pag.find_unconnected_subgraphs(en_nodes=nodes-{2,1})
        self.assertEqual(4, len(sub_graphs))
        self.assertIn({8}, sub_graphs)
        self.assertIn({0, 4}, sub_graphs)
        self.assertIn({3, 5, 6}, sub_graphs)
        self.assertIn({7, 9}, sub_graphs)

        dc_components = pag.find_unconnected_subgraphs(en_nodes=nodes, sym_edge_mark=Mark.Directed)
        self.assertEqual(7, len(dc_components))
        self.assertIn({0}, dc_components)
        self.assertIn({1}, dc_components)
        self.assertIn({2, 4, 8}, dc_components)
        self.assertIn({3, 5}, dc_components)
        self.assertIn({6}, dc_components)
        self.assertIn({7}, dc_components)
        self.assertIn({9}, dc_components)

        dc_components = pag.find_unconnected_subgraphs(en_nodes=nodes-{2}, sym_edge_mark=Mark.Directed)
        self.assertEqual(8, len(dc_components))
        self.assertIn({0}, dc_components)
        self.assertIn({1}, dc_components)
        self.assertIn({4}, dc_components)
        self.assertIn({8}, dc_components)
        self.assertIn({3, 5}, dc_components)
        self.assertIn({6}, dc_components)
        self.assertIn({7}, dc_components)
        self.assertIn({9}, dc_components)
