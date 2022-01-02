import unittest
from graphical_models import PAG, DAG
import graphical_models.arrow_head_types as Mark
from itertools import combinations
import numpy as np

class TestPAG(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pass

    def test_find_uncovered_path(self):
        print('Testing uncovered path search')
        node_set = set(range(10))
        pag = PAG(node_set)

        node_a = 3
        node_b = 9
        neighbor_a = 4
        neighbor_b = 2

        pag.create_complete_graph(Mark.Circle)
        p = pag.find_uncovered_path(node_a, node_b, neighbor_x=neighbor_a, neighbor_y=neighbor_b)
        self.assertIsNone(p)

        pag.create_empty_graph()
        pag.add_edge(3, 4, Mark.Circle, Mark.Circle)
        pag.add_edge(4, 5, Mark.Circle, Mark.Directed)
        pag.add_edge(4, 6, Mark.Circle, Mark.Circle)
        pag.add_edge(5, 0, Mark.Circle, Mark.Circle)
        pag.add_edge(0, 8, Mark.Circle, Mark.Circle)
        pag.add_edge(8, 1, Mark.Circle, Mark.Directed)
        pag.add_edge(1, 2, Mark.Circle, Mark.Circle)
        pag.add_edge(2, 9, Mark.Circle, Mark.Circle)

        pag.add_edge(7, 5, Mark.Circle, Mark.Circle)
        pag.add_edge(7, 8, Mark.Circle, Mark.Circle)
        pag.add_edge(5, 8, Mark.Circle, Mark.Circle)

        p = pag.find_uncovered_path(node_a, node_b, neighbor_x=neighbor_a, neighbor_y=neighbor_b)
        self.assertListEqual(p, [4, 5, 8, 1, 2])

        p = pag.find_uncovered_path(node_a, node_b, neighbor_x=neighbor_a, neighbor_y=neighbor_b,
                                    edge_condition=lambda in1, in2: pag.is_edge(in1, in2, Mark.Circle, Mark.Circle))
        self.assertIsNone(p)

        p = pag.find_uncovered_path(node_a, node_b, neighbor_x=neighbor_a, neighbor_y=neighbor_b,
                                    edge_condition=lambda in1, in2: pag.is_possible_parent(in1, in2))
        self.assertListEqual(p, [4, 5, 8, 1, 2])

        pag.replace_edge_mark(5, 8, Mark.Tail)
        p = pag.find_uncovered_path(node_a, node_b, neighbor_x=neighbor_a, neighbor_y=neighbor_b,
                                    edge_condition=lambda in1, in2: pag.is_possible_parent(in1, in2))
        self.assertIsNone(p)

    def test_orient_by_rule_5(self):
        print('Testing orientation rule R5')

        # test trivial case
        node_set = set(range(4))
        pag = PAG(node_set)
        pag.add_edge(0, 1, Mark.Circle, Mark.Circle)
        pag.add_edge(1, 2, Mark.Circle, Mark.Circle)
        pag.add_edge(2, 3, Mark.Circle, Mark.Circle)
        pag.add_edge(3, 0, Mark.Circle, Mark.Circle)

        self.assertTrue(pag.is_edge(0, 1, Mark.Circle, Mark.Circle))
        self.assertTrue(pag.is_edge(1, 2, Mark.Circle, Mark.Circle))
        self.assertTrue(pag.is_edge(2, 3, Mark.Circle, Mark.Circle))
        self.assertTrue(pag.is_edge(3, 0, Mark.Circle, Mark.Circle))
        pag.orient_by_rule_5()
        self.assertTrue(pag.is_edge(0, 1, Mark.Tail, Mark.Tail))
        self.assertTrue(pag.is_edge(1, 2, Mark.Tail, Mark.Tail))
        self.assertTrue(pag.is_edge(2, 3, Mark.Tail, Mark.Tail))
        self.assertTrue(pag.is_edge(3, 0, Mark.Tail, Mark.Tail))

        # test trivial path length + 1
        node_set = set(range(5))
        pag = PAG(node_set)
        pag.add_edge(0, 1, Mark.Circle, Mark.Circle)
        pag.add_edge(1, 2, Mark.Circle, Mark.Circle)
        pag.add_edge(2, 3, Mark.Circle, Mark.Circle)
        pag.add_edge(3, 4, Mark.Circle, Mark.Circle)
        pag.add_edge(4, 0, Mark.Circle, Mark.Circle)

        self.assertTrue(pag.is_edge(0, 1, Mark.Circle, Mark.Circle))
        self.assertTrue(pag.is_edge(1, 2, Mark.Circle, Mark.Circle))
        self.assertTrue(pag.is_edge(2, 3, Mark.Circle, Mark.Circle))
        self.assertTrue(pag.is_edge(3, 4, Mark.Circle, Mark.Circle))
        self.assertTrue(pag.is_edge(4, 0, Mark.Circle, Mark.Circle))
        pag.orient_by_rule_5()
        self.assertTrue(pag.is_edge(0, 1, Mark.Tail, Mark.Tail))
        self.assertTrue(pag.is_edge(1, 2, Mark.Tail, Mark.Tail))
        self.assertTrue(pag.is_edge(2, 3, Mark.Tail, Mark.Tail))
        self.assertTrue(pag.is_edge(3, 4, Mark.Tail, Mark.Tail))
        self.assertTrue(pag.is_edge(4, 0, Mark.Tail, Mark.Tail))

        # test partially triangulated
        node_set = set(range(5))
        pag = PAG(node_set)
        pag.add_edge(0, 1, Mark.Circle, Mark.Circle)
        pag.add_edge(1, 2, Mark.Circle, Mark.Circle)
        pag.add_edge(2, 3, Mark.Circle, Mark.Circle)
        pag.add_edge(3, 4, Mark.Circle, Mark.Circle)
        pag.add_edge(4, 0, Mark.Circle, Mark.Circle)
        pag.add_edge(2, 4, Mark.Circle, Mark.Circle)  # triangle: 2,4,3

        self.assertTrue(pag.is_edge(0, 1, Mark.Circle, Mark.Circle))
        self.assertTrue(pag.is_edge(1, 2, Mark.Circle, Mark.Circle))
        self.assertTrue(pag.is_edge(2, 3, Mark.Circle, Mark.Circle))
        self.assertTrue(pag.is_edge(3, 4, Mark.Circle, Mark.Circle))
        self.assertTrue(pag.is_edge(4, 0, Mark.Circle, Mark.Circle))
        pag.orient_by_rule_5()
        self.assertTrue(pag.is_edge(0, 1, Mark.Tail, Mark.Tail))
        self.assertTrue(pag.is_edge(1, 2, Mark.Tail, Mark.Tail))
        self.assertTrue(pag.is_edge(2, 3, Mark.Circle, Mark.Circle))
        self.assertTrue(pag.is_edge(3, 4, Mark.Circle, Mark.Circle))
        self.assertTrue(pag.is_edge(4, 0, Mark.Tail, Mark.Tail))
        self.assertTrue(pag.is_edge(2, 4, Mark.Tail, Mark.Tail))

        # test triangulated graph -- no change is expected
        node_set = set(range(5))
        pag = PAG(node_set)
        pag.add_edge(0, 1, Mark.Circle, Mark.Circle)
        pag.add_edge(1, 2, Mark.Circle, Mark.Circle)
        pag.add_edge(2, 3, Mark.Circle, Mark.Circle)
        pag.add_edge(3, 4, Mark.Circle, Mark.Circle)
        pag.add_edge(4, 0, Mark.Circle, Mark.Circle)
        pag.add_edge(2, 4, Mark.Circle, Mark.Circle)  # triangle: 2,4,3
        pag.add_edge(2, 0, Mark.Circle, Mark.Circle)  # triangle: 2,0,1

        self.assertTrue(pag.is_edge(0, 1, Mark.Circle, Mark.Circle))
        self.assertTrue(pag.is_edge(1, 2, Mark.Circle, Mark.Circle))
        self.assertTrue(pag.is_edge(2, 3, Mark.Circle, Mark.Circle))
        self.assertTrue(pag.is_edge(3, 4, Mark.Circle, Mark.Circle))
        self.assertTrue(pag.is_edge(4, 0, Mark.Circle, Mark.Circle))
        pag.orient_by_rule_5()
        self.assertTrue(pag.is_edge(0, 1, Mark.Circle, Mark.Circle))
        self.assertTrue(pag.is_edge(1, 2, Mark.Circle, Mark.Circle))
        self.assertTrue(pag.is_edge(2, 3, Mark.Circle, Mark.Circle))
        self.assertTrue(pag.is_edge(3, 4, Mark.Circle, Mark.Circle))
        self.assertTrue(pag.is_edge(4, 0, Mark.Circle, Mark.Circle))

    def test_orient_by_rule_6(self):
        print('Testing orientation rule R6')
        # test trivial case
        node_set = set(range(5))
        pag = PAG(node_set)
        pag.add_edge(0, 1, Mark.Tail, Mark.Tail)
        pag.add_edge(1, 2, Mark.Circle, Mark.Tail)
        pag.add_edge(1, 3, Mark.Circle, Mark.Directed)
        pag.add_edge(1, 4, Mark.Circle, Mark.Circle)
        pag.orient_by_rule_6()
        self.assertTrue(pag.is_edge(1, 2, Mark.Tail, Mark.Tail))
        self.assertTrue(pag.is_edge(1, 3, Mark.Tail, Mark.Directed))
        self.assertTrue(pag.is_edge(1, 4, Mark.Tail, Mark.Circle))

        # test partially triangulated
        node_set = set(range(5))
        pag = PAG(node_set)
        pag.add_edge(0, 1, Mark.Circle, Mark.Circle)
        pag.add_edge(1, 2, Mark.Circle, Mark.Circle)
        pag.add_edge(2, 3, Mark.Circle, Mark.Circle)
        pag.add_edge(3, 4, Mark.Circle, Mark.Circle)
        pag.add_edge(4, 0, Mark.Circle, Mark.Circle)
        pag.add_edge(2, 4, Mark.Circle, Mark.Circle)  # triangle: 2,4,3

        self.assertTrue(pag.is_edge(0, 1, Mark.Circle, Mark.Circle))
        self.assertTrue(pag.is_edge(1, 2, Mark.Circle, Mark.Circle))
        self.assertTrue(pag.is_edge(2, 3, Mark.Circle, Mark.Circle))
        self.assertTrue(pag.is_edge(3, 4, Mark.Circle, Mark.Circle))
        self.assertTrue(pag.is_edge(4, 0, Mark.Circle, Mark.Circle))
        pag.orient_by_rule_5()
        pag.orient_by_rule_6()
        self.assertTrue(pag.is_edge(2, 3, Mark.Tail, Mark.Circle))
        self.assertTrue(pag.is_edge(4, 3, Mark.Tail, Mark.Circle))

    def test_orient_by_rule_7(self):
        print('Testing orientation rule R7')
        # test trivial case
        node_set = set(range(5))
        pag = PAG(node_set)
        pag.add_edge(0, 1, Mark.Tail, Mark.Circle)
        pag.add_edge(1, 2, Mark.Circle, Mark.Tail)
        pag.add_edge(1, 3, Mark.Circle, Mark.Directed)
        pag.add_edge(1, 4, Mark.Circle, Mark.Circle)
        pag.orient_by_rule_7()
        self.assertTrue(pag.is_edge(1, 2, Mark.Tail, Mark.Tail))
        self.assertTrue(pag.is_edge(1, 3, Mark.Tail, Mark.Directed))
        self.assertTrue(pag.is_edge(1, 4, Mark.Tail, Mark.Circle))

    def test_orient_by_rule_8(self):
        print('Testing orientation rule R8')
        node_set = set(range(3))
        pag = PAG(node_set)
        pag.add_edge(0, 1, Mark.Tail, Mark.Directed)
        pag.add_edge(1, 2, Mark.Tail, Mark.Directed)
        pag.add_edge(0, 2, Mark.Circle, Mark.Directed)

        pag.orient_by_rule_8()
        self.assertTrue(pag.is_edge(0, 2, Mark.Tail, Mark.Directed))

        node_set = set(range(3))
        pag = PAG(node_set)
        pag.add_edge(0, 1, Mark.Tail, Mark.Circle)
        pag.add_edge(1, 2, Mark.Tail, Mark.Directed)
        pag.add_edge(0, 2, Mark.Circle, Mark.Directed)

        pag.orient_by_rule_8()
        self.assertTrue(pag.is_edge(0, 2, Mark.Tail, Mark.Directed))

    def test_orient_by_rule_9(self):
        print('Testing orientation rule R9')
        node_set = set(range(5))
        pag = PAG(node_set)
        pag.add_edge(0, 1, Mark.Circle, Mark.Directed)
        pag.add_edge(0, 2, Mark.Circle, Mark.Circle)
        pag.add_edge(2, 3, Mark.Circle, Mark.Directed)
        pag.add_edge(3, 4, Mark.Circle, Mark.Circle)
        pag.add_edge(4, 1, Mark.Circle, Mark.Circle)

        pag.orient_by_rule_9()
        pag.orient_by_rule_9()
        self.assertTrue(pag.is_edge(0, 1, Mark.Tail, Mark.Directed))
        self.assertTrue(pag.is_edge(0, 2, Mark.Circle, Mark.Circle))
        self.assertTrue(pag.is_edge(2, 3, Mark.Tail, Mark.Directed))
        self.assertTrue(pag.is_edge(3, 4, Mark.Circle, Mark.Circle))
        self.assertTrue(pag.is_edge(4, 1, Mark.Circle, Mark.Circle))

        # no edges should be oriented
        pag = PAG(node_set)
        pag.add_edge(0, 1, Mark.Circle, Mark.Directed)
        pag.add_edge(0, 2, Mark.Circle, Mark.Circle)
        pag.add_edge(2, 3, Mark.Circle, Mark.Directed)
        pag.add_edge(3, 4, Mark.Directed, Mark.Circle)  # an arrow-head into 3 breaking the p.d path
        pag.add_edge(4, 1, Mark.Circle, Mark.Circle)

        pag.orient_by_rule_9()
        pag.orient_by_rule_9()
        self.assertTrue(pag.is_edge(0, 1, Mark.Circle, Mark.Directed))
        self.assertTrue(pag.is_edge(0, 2, Mark.Circle, Mark.Circle))
        self.assertTrue(pag.is_edge(2, 3, Mark.Tail, Mark.Directed))
        self.assertTrue(pag.is_edge(3, 4, Mark.Directed, Mark.Circle))
        self.assertTrue(pag.is_edge(4, 1, Mark.Circle, Mark.Circle))

    def test_orient_by_rule_10(self):
        print('Testing orientation rule R10')

        # test trivial case
        node_set = set(range(4))
        pag = PAG(node_set)
        pag.add_edge(0, 2, Mark.Circle, Mark.Directed)
        pag.add_edge(0, 1, Mark.Circle, Mark.Directed)
        pag.add_edge(0, 3, Mark.Circle, Mark.Directed)
        pag.add_edge(1, 2, Mark.Tail, Mark.Directed)
        pag.add_edge(3, 2, Mark.Tail, Mark.Directed)

        pag.orient_by_rule_10()
        self.assertTrue(pag.is_edge(0, 2, Mark.Tail, Mark.Directed))  # modified edge: oriented a tail
        self.assertTrue(pag.is_edge(0, 1, Mark.Circle, Mark.Directed))
        self.assertTrue(pag.is_edge(0, 3, Mark.Circle, Mark.Directed))
        self.assertTrue(pag.is_edge(1, 2, Mark.Tail, Mark.Directed))
        self.assertTrue(pag.is_edge(3, 2, Mark.Tail, Mark.Directed))

        # test trivial case - no orientation is expected
        pag = PAG(node_set)
        pag.add_edge(0, 2, Mark.Circle, Mark.Directed)
        pag.add_edge(0, 1, Mark.Circle, Mark.Directed)
        pag.add_edge(0, 3, Mark.Circle, Mark.Directed)
        pag.add_edge(1, 2, Mark.Tail, Mark.Directed)
        pag.add_edge(3, 2, Mark.Tail, Mark.Directed)
        pag.add_edge(1, 3, Mark.Circle, Mark.Circle)  # bridge the two paths, violating the R10 antecedent

        pag.orient_by_rule_10()
        self.assertTrue(pag.is_edge(0, 2, Mark.Circle, Mark.Directed))  # non-modified edge
        self.assertTrue(pag.is_edge(0, 1, Mark.Circle, Mark.Directed))
        self.assertTrue(pag.is_edge(0, 3, Mark.Circle, Mark.Directed))
        self.assertTrue(pag.is_edge(1, 2, Mark.Tail, Mark.Directed))
        self.assertTrue(pag.is_edge(3, 2, Mark.Tail, Mark.Directed))
        self.assertTrue(pag.is_edge(1, 3, Mark.Circle, Mark.Circle))

        # test with one trivial p.d. path and one uncovered p.d. path
        node_set = set(range(6))
        pag = PAG(node_set)
        pag.add_edge(0, 2, Mark.Circle, Mark.Directed)
        pag.add_edge(0, 3, Mark.Circle, Mark.Directed)
        pag.add_edge(3, 2, Mark.Tail, Mark.Directed)
        pag.add_edge(1, 2, Mark.Tail, Mark.Directed)
        pag.add_edge(0, 4, Mark.Circle, Mark.Circle)
        pag.add_edge(4, 5, Mark.Circle, Mark.Directed)
        pag.add_edge(5, 1, Mark.Circle, Mark.Circle)

        pag.orient_by_rule_10()
        self.assertTrue(pag.is_edge(0, 2, Mark.Tail, Mark.Directed))  # modified edge: oriented a tail
        self.assertTrue(pag.is_edge(0, 3, Mark.Circle, Mark.Directed))
        self.assertTrue(pag.is_edge(3, 2, Mark.Tail, Mark.Directed))
        self.assertTrue(pag.is_edge(1, 2, Mark.Tail, Mark.Directed))
        self.assertTrue(pag.is_edge(0, 4, Mark.Circle, Mark.Circle))
        self.assertTrue(pag.is_edge(4, 5, Mark.Circle, Mark.Directed))
        self.assertTrue(pag.is_edge(5, 1, Mark.Circle, Mark.Circle))

        # test with one trivial p.d. path and one uncovered p.d. path, and an edge that falsifies the antecedent
        node_set = set(range(6))
        pag = PAG(node_set)
        pag.add_edge(0, 2, Mark.Circle, Mark.Directed)
        pag.add_edge(0, 3, Mark.Circle, Mark.Directed)
        pag.add_edge(3, 2, Mark.Tail, Mark.Directed)
        pag.add_edge(1, 2, Mark.Tail, Mark.Directed)
        pag.add_edge(0, 4, Mark.Circle, Mark.Circle)
        pag.add_edge(4, 5, Mark.Circle, Mark.Directed)
        pag.add_edge(5, 1, Mark.Circle, Mark.Circle)
        pag.add_edge(4, 3, Mark.Circle, Mark.Circle)  # falsify by connecting E and F nodes, adjacent to A

        pag.orient_by_rule_10()
        self.assertTrue(pag.is_edge(0, 2, Mark.Circle, Mark.Directed))  # modified edge: oriented a tail
        self.assertTrue(pag.is_edge(0, 3, Mark.Circle, Mark.Directed))
        self.assertTrue(pag.is_edge(3, 2, Mark.Tail, Mark.Directed))
        self.assertTrue(pag.is_edge(1, 2, Mark.Tail, Mark.Directed))
        self.assertTrue(pag.is_edge(0, 4, Mark.Circle, Mark.Circle))
        self.assertTrue(pag.is_edge(4, 5, Mark.Circle, Mark.Directed))
        self.assertTrue(pag.is_edge(5, 1, Mark.Circle, Mark.Circle))

        # test with two uncovered p.d. paths
        node_set = set(range(7))
        pag = PAG(node_set)
        pag.add_edge(0, 2, Mark.Circle, Mark.Directed)
        pag.add_edge(0, 3, Mark.Circle, Mark.Directed)
        pag.add_edge(1, 2, Mark.Tail, Mark.Directed)
        pag.add_edge(0, 4, Mark.Circle, Mark.Circle)
        pag.add_edge(4, 5, Mark.Circle, Mark.Directed)
        pag.add_edge(5, 1, Mark.Circle, Mark.Circle)
        pag.add_edge(3, 6, Mark.Circle, Mark.Circle)
        pag.add_edge(6, 2, Mark.Tail, Mark.Directed)

        pag.orient_by_rule_10()
        self.assertTrue(pag.is_edge(0, 2, Mark.Tail, Mark.Directed))  # modified edge: oriented a tail
        self.assertTrue(pag.is_edge(0, 3, Mark.Circle, Mark.Directed))
        self.assertTrue(pag.is_edge(1, 2, Mark.Tail, Mark.Directed))
        self.assertTrue(pag.is_edge(0, 4, Mark.Circle, Mark.Circle))
        self.assertTrue(pag.is_edge(4, 5, Mark.Circle, Mark.Directed))
        self.assertTrue(pag.is_edge(5, 1, Mark.Circle, Mark.Circle))
        self.assertTrue(pag.is_edge(3, 6, Mark.Circle, Mark.Circle))
        self.assertTrue(pag.is_edge(6, 2, Mark.Tail, Mark.Directed))

    def test_find_possible_ancestors(self):
        print('Testing find possible ancestors')
        nodes = set(['X1', 'X2', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5'])
        pag = PAG(nodes)  # PAG from Jaber et al., 2019, figure 2
        pag.add_edge('X1', 'X2', Mark.Directed, Mark.Directed)
        pag.add_edge('X1', 'Y1', Mark.Tail, Mark.Directed)
        pag.add_edge('X1', 'Y3', Mark.Directed, Mark.Directed)

        pag.add_edge('X2', 'Y1', Mark.Directed, Mark.Directed)
        pag.add_edge('X2', 'Y2', Mark.Tail, Mark.Directed)

        pag.add_edge('Y1', 'Y5', Mark.Directed, Mark.Circle)
        pag.add_edge('Y2', 'Y3', Mark.Directed, Mark.Directed)
        pag.add_edge('Y3', 'Y4', Mark.Directed, Mark.Circle)
        pag.add_edge('Y4', 'Y5', Mark.Circle, Mark.Circle)

        expected_possible_ancestors = {
            'X1': {'X1'},
            'X2': {'X2'},
            'Y1': {'Y1', 'X1', 'Y4', 'Y5'},
            'Y2': {'Y2', 'X2'},
            'Y3': {'Y3', 'Y4', 'Y5'},
            'Y4': {'Y4', 'Y5'},
            'Y5': {'Y5', 'Y4'}}

        for n in pag.nodes_set:
            an = pag.find_possible_ancestors(descendants_set={n})
            self.assertEqual(an, expected_possible_ancestors[n])

    def test_find_pc_component_for_node(self):
        print('Testing find pc-component')
        nodes = {0, 1}
        pag = PAG(nodes)
        pag.add_edge(0, 1, Mark.Circle, Mark.Circle)
        pag.find_visible_edges()
        pc_comp_x = pag.find_union_pc_components_for_node(0)
        self.assertSetEqual({0, 1}, pc_comp_x)

        # test find_pc_component_for_set
        nodes = set(range(15))
        pag = PAG(nodes)
        pag.add_edge(1, 0, Mark.Tail, Mark.Directed)
        pag.add_edge(1, 3, Mark.Circle, Mark.Directed)
        pag.add_edge(4, 3, Mark.Directed, Mark.Directed)
        pag.add_edge(0, 8, Mark.Circle, Mark.Directed)
        pag.add_edge(9, 8, Mark.Directed, Mark.Directed)
        pag.add_edge(9, 10, Mark.Directed, Mark.Directed)
        pag.add_edge(14, 10, Mark.Circle, Mark.Directed)
        pag.add_edge(11, 9, Mark.Circle, Mark.Directed)
        pag.add_edge(11, 13, Mark.Directed, Mark.Directed)
        pag.add_edge(11, 12, Mark.Directed, Mark.Directed)
        pag.add_edge(5, 0, Mark.Directed, Mark.Directed)
        pag.add_edge(5, 6, Mark.Directed, Mark.Directed)
        pag.add_edge(7, 6, Mark.Directed, Mark.Directed)
        pag.add_edge(0, 2, Mark.Circle, Mark.Directed)
        pag.find_visible_edges()

        pc_comp_x = pag.find_union_pc_components_for_set(set_x={0})
        self.assertSetEqual({0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 14}, pc_comp_x)

    def test_find_region(self):
        print('Testing find region')
        nodes = set(range(9))
        pag = PAG(nodes)
        pag.add_edge(0, 5, Mark.Circle, Mark.Directed)
        pag.add_edge(0, 6, Mark.Circle, Mark.Circle)
        pag.add_edge(0, 1, Mark.Circle, Mark.Circle)
        pag.add_edge(6, 7, Mark.Circle, Mark.Directed)
        pag.add_edge(1, 2, Mark.Circle, Mark.Directed)
        pag.add_edge(4, 2, Mark.Circle, Mark.Directed)
        pag.add_edge(3, 2, Mark.Circle, Mark.Directed)
        pag.add_edge(3, 8, Mark.Circle, Mark.Circle)
        pag.find_visible_edges()

        region = pag.find_region({4, 7})
        self.assertSetEqual({0, 1, 2, 3, 4, 6, 7, 8}, region)

    def test_find_discriminating_path_to_triplet(self):
        print('Testing find discriminating path')
        nodes = set(range(5))
        pag = PAG(nodes)
        pag.add_edge(0, 1, Mark.Tail, Mark.Directed)
        pag.add_edge(1, 2, Mark.Tail, Mark.Directed)
        pag.add_edge(1, 4, Mark.Tail, Mark.Directed)
        pag.add_edge(2, 3, Mark.Directed, Mark.Directed)
        pag.add_edge(4, 3, Mark.Directed, Mark.Directed)

        res = pag.find_discriminating_path_to_triplet(2, 3, 4)
        self.assertIsNone(res)  # node 1 is not a collider on the path

        # modify: node 1 becomes a collider on the path
        pag.replace_edge_mark(node_source=2, node_target=1, requested_edge_mark=Mark.Directed)
        res = pag.find_discriminating_path_to_triplet(2, 3, 4)
        self.assertIsNotNone(res)  # node 1 is not a collider on the path

    def test_is_m_separated(self):
        print('Testing m-separation')
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

        pag = PAG(observed)
        pag.add_edge('X', 'Sun', Mark.Directed, Mark.Directed)
        pag.add_edge('W', 'Sun', Mark.Tail, Mark.Directed)
        pag.add_edge('W', 'T', Mark.Tail, Mark.Directed)
        pag.add_edge('Z', 'T', Mark.Directed, Mark.Directed)
        pag.add_edge('V', 'Z', Mark.Directed, Mark.Directed)
        pag.add_edge('Y', 'V', Mark.Directed, Mark.Directed)
        pag.add_edge('X', 'U', Mark.Directed, Mark.Directed)
        pag.add_edge('Sun', 'Z', Mark.Tail, Mark.Directed)
        pag.add_edge('Z', 'U', Mark.Tail, Mark.Directed)
        pag.add_edge('V', 'X', Mark.Tail, Mark.Directed)
        pag.add_edge('T', 'X', Mark.Tail, Mark.Directed)
        pag.add_edge('U', 'Y', Mark.Tail, Mark.Directed)

        for i, j in combinations(observed, 2):
            for r in range(len(observed)-2):
                for cond_set in combinations(observed - {i, j}, r):
                    # print(i, j, cond_set, dag.dsep(i, j, set(cond_set)), pag.is_m_separated(i, j, set(cond_set)))
                    self.assertEqual(
                        dag.dsep(i, j, set(cond_set)),
                        pag.is_m_separated(i, j, set(cond_set))
                    )

    def test_get_adj_mat(self):
        # create PAG
        node_set = set(range(10))
        pag = PAG(node_set)
        pag.add_edge(3, 4, Mark.Circle, Mark.Circle)
        pag.add_edge(4, 5, Mark.Circle, Mark.Directed)
        pag.add_edge(4, 6, Mark.Circle, Mark.Circle)
        pag.add_edge(5, 0, Mark.Circle, Mark.Circle)
        pag.add_edge(0, 8, Mark.Circle, Mark.Circle)
        pag.add_edge(8, 1, Mark.Circle, Mark.Directed)
        pag.add_edge(1, 2, Mark.Circle, Mark.Circle)
        pag.add_edge(2, 9, Mark.Circle, Mark.Circle)
        pag.add_edge(7, 5, Mark.Circle, Mark.Circle)
        pag.add_edge(7, 8, Mark.Circle, Mark.Circle)
        pag.add_edge(5, 8, Mark.Circle, Mark.Circle)

        # convert PAG to adjacency matrix
        adj_mat = pag.get_adj_mat()

        # total of 11 edges
        self.assertEqual(np.sum(adj_mat > 0), 11*2)

        # check result
        arrow_type_map = dict()
        arrow_type_map[None] = 0
        arrow_type_map[Mark.Circle] = 1
        arrow_type_map[Mark.Directed] = 2
        arrow_type_map[Mark.Tail] = 3

        for node1 in pag.nodes_set:
            for node2 in pag.nodes_set:
                edge_mark = pag.get_edge_mark(node1, node2)
                self.assertEqual(adj_mat[node1, node2] , arrow_type_map[edge_mark])


    def test_init_from_adj_mat(self):
        # create PAG
        node_set = set(range(10))
        pag = PAG(node_set)
        pag.add_edge(3, 4, Mark.Circle, Mark.Circle)
        pag.add_edge(4, 5, Mark.Circle, Mark.Directed)
        pag.add_edge(4, 6, Mark.Circle, Mark.Circle)
        pag.add_edge(5, 0, Mark.Circle, Mark.Circle)
        pag.add_edge(0, 8, Mark.Circle, Mark.Circle)
        pag.add_edge(8, 1, Mark.Circle, Mark.Directed)
        pag.add_edge(1, 2, Mark.Circle, Mark.Circle)
        pag.add_edge(2, 9, Mark.Circle, Mark.Circle)
        pag.add_edge(7, 5, Mark.Circle, Mark.Circle)
        pag.add_edge(7, 8, Mark.Circle, Mark.Circle)
        pag.add_edge(5, 8, Mark.Circle, Mark.Circle)

        # convert PAG to adjacency matrix
        adj_mat = pag.get_adj_mat()

        # create PAG from adjacency matrix
        new_pag = PAG(node_set)
        new_pag.init_from_adj_mat(adj_mat)
        new_adj_mat = new_pag.get_adj_mat()

        # check result
        arrow_type_map = dict()
        arrow_type_map[None] = 0
        arrow_type_map[Mark.Circle] = 1
        arrow_type_map[Mark.Directed] = 2
        arrow_type_map[Mark.Tail] = 3

        for node1 in pag.nodes_set:
            for node2 in pag.nodes_set:
                edge_mark = pag.get_edge_mark(node1, node2)
                self.assertEqual(new_adj_mat[node1, node2] , arrow_type_map[edge_mark])


if __name__ == '__main__':
    unittest.main()
