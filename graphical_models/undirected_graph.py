from .basic_graph import Graph

_ErrorUnknownNode = 'Node is not in the graph'


class UndirectedGraph(Graph):
    """
    An undirected graphical model.
    """

    # --- graph modification functions --------------------------------------------------------------------------------
    def add_edge(self, node_i, node_j):
        if (node_i not in self.nodes_set) or (node_j not in self.nodes_set):
            raise ValueError(_ErrorUnknownNode)

        self._graph[node_i].add(node_j)
        self._graph[node_j].add(node_i)

    def remove_edge(self, node_i, node_j):
        if (node_i not in self.nodes_set) or (node_j not in self.nodes_set):
            raise ValueError(_ErrorUnknownNode)

        self._graph[node_i].discard(node_j)
        self._graph[node_j].discard(node_i)

    def disconnect_node(self, node):
        neighbors_set = self._graph[node]
        for neighbor in neighbors_set:
            self.remove_edge(neighbor, node)

    # --- graph query functions ---------------------------------------------------------------------------------------
    def is_reachable(self, node_start, node_end, visited_nodes_in=None):
        """
        Test if there is a path between two nodes (node_start, node_end) not passing through the given set of nodes.
        :param node_start: one end-point of the tested path
        :param node_end: second end-point of the tested path
        :param visited_nodes_in: (forbidden nodes) set of nodes that block the tested paths
            (e.g., nodes that were already visited)
        :return: True if a path is found
        """
        if node_start == node_end:
            return True  # reach the destination

        if visited_nodes_in is None:
            visited_nodes = {node_start}
        else:
            visited_nodes = set(visited_nodes_in)  # create a copy and ensure it's of set type
            visited_nodes.add(node_start)

        unvisited_neighbors = self._graph[node_start] - visited_nodes
        for neighbor in unvisited_neighbors:
            if self.is_reachable(neighbor, node_end, visited_nodes):
                return True  # found a path from an (unvisited) neighbor to the target node
            visited_nodes.add(neighbor)
        else:
            return False  # went through all the neighbors and didn't find a path to the target
