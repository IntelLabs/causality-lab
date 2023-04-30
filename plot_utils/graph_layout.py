import numpy as np
from math import sqrt, cos, sin


class BaseLayout:
    """
    Base class for layouts.
    """
    def __init__(self, graph, win_left_right, win_top_bottom, group_sort=None):
        self.num_nodes = len(graph.nodes_set)
        self.left = win_left_right[0]
        self.right = win_left_right[1]
        self.top = win_top_bottom[0]
        self.bottom = win_top_bottom[1]
        self.group_sort = group_sort
        self.graph = graph

        # set initial positions
        if group_sort is not None:
            nodes_order = [node for group in group_sort for node in group]
            xpos = np.linspace(self.left, self.right, self.num_nodes)
            ypos = np.linspace(self.top, self.bottom, self.num_nodes)
        else:
            nodes_order = list(graph.nodes_set)
            xpos = np.random.uniform(self.left, self.right, self.num_nodes)*0.2
            ypos = np.random.uniform(self.top, self.bottom, self.num_nodes)*0.2

        # dictionary with nodes as keys and [x,y] position numpy-vector as value
        self.pos = dict()
        for idx in range(self.num_nodes):
            self.pos[nodes_order[idx]] = np.array([xpos[idx], ypos[idx]])  # np.array([xpos[idx], ypos[idx]])


class ForceDirectedLayout(BaseLayout):
    def __init__(self, graph, win_left_right, win_top_bottom, group_sort=None, k_const=None, num_iterations=100):
        """
        Node positions are calculated using the force-directed algorithm by Fruchterman and Reingold (1991)
        :param graph: a graph to draw
        :param win_left_right: a tuple (left, right) with border coordinates
        :param win_top_bottom: a tuple (top, bottom) with border coordinates
        :param group_sort: a topologically sorted list of groups  of nodes.
                           Nodes at the beginning of the list will tend to be higher in the layout.
        :param k_const: constant of repulsion and attraction forces.
                        (default is None indicating to automatically determine the value)
        :param num_iterations: number of iterations (default: 100; can be reduced to improve runtime)
        """
        super().__init__(graph, win_left_right, win_top_bottom, group_sort)

        np.random.seed(123)
        self.num_iter = num_iterations

        if k_const is None:
            k_const = sqrt((self.right-self.left)*(self.bottom-self.top) / self.num_nodes)  # area / number of nodes
        self.k_const = k_const

        self.attraction = lambda dist: dist*dist / self.k_const
        self.repulsion = lambda dist: self.k_const*self.k_const / dist

        init_layout = CircleLayout(graph, tuple(m for m in win_left_right), tuple(m for m in win_top_bottom), group_sort)
        init_pos = init_layout.calc_layout()

    def _calc_repulsive_forces(self):
        repulse = {node: np.zeros(2, dtype=float) for node in self.graph.nodes_set}  # dictionary
        for node_i in self.graph.nodes_set:
            repulse[node_i] = np.zeros(2, dtype=float)
            for node_j in self.graph.nodes_set-{node_i}:
                dv = self.pos[node_i] - self.pos[node_j]
                dist = np.sqrt(dv[0]*dv[0] + dv[1]*dv[1])  # Euclidean distance
                dist = max(dist, 1)
                repulse[node_i] += (dv / dist) * self.repulsion(dist)
        return repulse

    def _calc_attracting_forces(self):
        attract = {node: np.zeros(2, dtype=float) for node in self.graph.nodes_set}  # dictionary
        for node_i in self.graph.nodes_set:
            for node_j in self.graph.find_adjacent_nodes(node_i):  # loop through neighbors
                dv = self.pos[node_i] - self.pos[node_j]
                dist = np.sqrt(dv[0] * dv[0] + dv[1] * dv[1])  # Euclidean distance
                dist = max(dist, 1)
                attract[node_i] -= (dv / dist) * self.attraction(dist)
        return attract

    def calc_layout(self, num_iterations=None):
        """
        Main function for calculating the layout.
        :param num_iterations: default is None, indicating to use the class attribute value.
        :return: return the final position values (also set the class attribute).
        """
        if num_iterations is None:
            num_iterations = self.num_iter
        first_max_disp = self.k_const
        for i in range(num_iterations):
            max_disp = first_max_disp / (i+1)
            repulsing = self._calc_repulsive_forces()
            attracting = self._calc_attracting_forces()
            for node in self.graph.nodes_set:
                disp = (repulsing[node] + attracting[node])
                disp_norm = np.sqrt(disp[0]*disp[0] + disp[1]*disp[1])
                self.pos[node] += (disp/disp_norm) * min(disp_norm, max_disp)
                self.pos[node][0] = min(self.right, max(self.left, self.pos[node][0]))
                self.pos[node][1] = min(self.bottom, max(self.top, self.pos[node][1]))
        return self.pos


class CircleLayout(BaseLayout):
    def __init__(self, graph, win_left_right, win_top_bottom, group_sort=None):
        super().__init__(graph, win_left_right, win_top_bottom, group_sort)

    def calc_layout(self):
        if self.group_sort is not None:
            nodes_order = [node for group in self.group_sort for node in group]
        else:
            nodes_order = list(self.graph.nodes_set)

        xrad = (self.right - self.left)/2
        yrad = (self.bottom - self.top)/2
        center = ((self.right + self.left)/2, (self.bottom + self.top)/2)  # 2-tuple

        angle_list = np.linspace(0, 2*np.pi, self.num_nodes+1)
        for idx in range(self.num_nodes):
            node = nodes_order[idx]
            angle = angle_list[idx]
            self.pos[node][0] = center[0] + xrad * cos(angle)
            self.pos[node][1] = center[1] + yrad * sin(angle)

        return self.pos


class ColumnLayout(BaseLayout):
    def __init__(self, graph, win_left_right, win_top_bottom, group_sort=None):
        super().__init__(graph, win_left_right, win_top_bottom, group_sort)

    def calc_layout(self):
        if self.group_sort is None:
            nodes_order = list(self.graph.nodes_set)
            group_sort = [nodes_order]
        else:  # self.group_sort is not None
            group_sort = self.group_sort

        # find the size of the largest group of nodes
        max_group_len = 0
        for group in group_sort:
            if len(group) > max_group_len:
                max_group_len = len(group)

        # calculate positions. First group is left-most
        n_groups = len(group_sort)
        y_offset = self.bottom
        if n_groups > 1:
            x_step = (self.right - self.left) / (n_groups-1)
            x_offset = self.left
        else:
            x_step = 0
            x_offset = (self.right + self.left) / 2.
        y_step = (self.top - self.bottom) / (max_group_len - 1)
        for i_group, group in enumerate(group_sort):
            for i_node, node in enumerate(group):
                self.pos[node][0] = x_step * i_group + x_offset
                self.pos[node][1] = y_step * i_node + y_offset

        return self.pos
