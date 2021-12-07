from itertools import combinations
import math


_ErrorChildNotExist = 'The child branch does not exist'
_ErrorAddExistBranch = "The child already exists in the PDS-tree"


class PDSTree:
    """
    A tree structure for storage and retrieval of Possible-D-Sep nodes.
    """
    def __init__(self, node_root):
        self.origin = node_root
        self.children = []
        self.dict_child = {}  # dictionary that maps a node to the index of the child in the children list
        self.depth_level = 0

    def get_child_branch(self, child_origin):
        if child_origin not in self.dict_child:
            raise ValueError(_ErrorChildNotExist)

        child_idx = self.dict_child[child_origin]
        return self.children[child_idx]

    def add_branch(self, branch_root):
        """
        Add a child node (it will serve as the root of a tree originating from it)
        :param branch_root: a node identifier
        :return:
        """
        if branch_root in self.dict_child:
            raise ValueError(_ErrorAddExistBranch)

        self.dict_child[branch_root] = len(self.children)  # create an index value for this child (not a key)
        pds_tree_child = PDSTree(branch_root)
        pds_tree_child.depth_level = self.depth_level + 1
        self.children.append(pds_tree_child)  # add the child to the list of children

    def get_max_depth(self):  # TODO: correct this function to retern a 0-based depth (root is 0)
        """
        Get the maximal depth (number of nodes)
        :return: maximal depth: number of nodes from the root to the deepest leaf (inclusive)
        """
        if len(self.children) == 0:  # a leaf node
            return 1

        max_child_depth = 1
        for child in self.children:
            current_child_depth =  child.get_max_depth()  # max depth of the tree originating from the current child
            if current_child_depth > max_child_depth:
                max_child_depth = current_child_depth

        return max_child_depth + 1

    def get_minimal_distance(self, node):
        minimal_dist = math.inf
        for child_branch in self.children:
            if child_branch.origin == node:
                return child_branch.depth_level  # a child is found, subsequent iteration will return greater or equal
            else:
                dist = child_branch.get_minimal_distance(node)
                minimal_dist = min(minimal_dist, dist)

        return minimal_dist  # in case no children or node not in the immediate children

    def is_pds_path(self, subset_nodes):
        if len(subset_nodes) == 0:
            return True
        for branch_x in self.children:
            if branch_x.origin in subset_nodes:
                path_found = branch_x.is_pds_path(subset_nodes - {branch_x.origin})
                if path_found:
                    return True
        else:
            return False

    def is_legal_cond_set(self, subset_nodes):
        """
        Test ICD-Sep condition 2-b: for every node in the conditioning set there exists a pds path such that
        all the nodes on the path are also members of the same conditioning set.
        :param subset_nodes: conditioning set to be inspected
        :return: True is the conditioning set complies with ICD-Sep condition 2-b.
        """
        # check if evey node in the subset_nodes is reachable from the root using paths composed of only subset_nodes
        for node in subset_nodes:
            if not self.is_reachable(node, possible_path_nodes=subset_nodes):  # ICD-Sep Condition 2-b
                return False
        else:
            return True

    def is_reachable(self, target_node, possible_path_nodes):
        if len(possible_path_nodes) == 0:
            return False
        for branch_x in self.children:
            if branch_x.origin == target_node:
                return True
            if branch_x.origin in possible_path_nodes:
                is_found = branch_x.is_reachable(target_node, possible_path_nodes)
                if is_found:
                    return True

        return False

    def get_subsets_list(self, set_nodes, subset_size):
        min_dist = {node: self.get_minimal_distance(node) for node in set_nodes}  # minimal distances given set_nodes

        subsets_list = []  # each element in this list is a 2-element list [ {subsets}, distance ]

        # create a list of all legal subsets
        for subset_nodes in combinations(set_nodes, subset_size):
            if self.is_legal_cond_set(subset_nodes):
                # sum minimal distances
                dist_sum = 0
                for node in subset_nodes:
                    dist_sum += min_dist[node]

                subsets_list.append([set(subset_nodes), dist_sum])

        return subsets_list
