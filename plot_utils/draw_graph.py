from matplotlib import pyplot as plt
from matplotlib import patches
from math import sqrt
from .graph_layout import ForceDirectedLayout, CircleLayout, ColumnLayout
from graphical_models import DAG, PAG, arrow_head_types as Mark
from itertools import combinations
import numpy as np


def draw_edge(axes, pos0, pos1, node_radius,
              edge_mark_0=None, edge_mark_1=None, line_color='black', fill_color='white',
              text=None, text_color='black', font=None, is_curved=False):
    """
    Draw an edge of a PAG.
    :param axes: a matplotlib axes object
    :param pos0: position of the first node (x,y tuple)
    :param pos1: position of the second node (x,y tuple)
    :param node_radius: radius of the node
    :param edge_mark_0: edge mark at the first node
    :param edge_mark_1: edge mark at the second node
    :param line_color: color of the arrow. If set to 'auto', edges are color according to their type
    :param fill_color: color the will be filled inside a 'o' edge mark
    :param text: a text to place on the edge (e.g., 'v' to mark a 'visible' edge)
    :param text_color: color of the text (if given)
    :param font: a dictionary describing the font.
        Example: font = {'fontfamily': 'Times', 'fontsize': 10, 'fontweight': 'bold', 'fontstyle': 'italic'}
    """
    circle_edgemark_rad = node_radius / 4
    dx = pos1[0] - pos0[0]
    dy = pos1[1] - pos0[1]
    node_distance = sqrt(dx**2 + dy**2)
    cos_angle = dx / node_distance
    sin_angle = dy / node_distance
    offset0 = circle_edgemark_rad+node_radius if edge_mark_0 == Mark.Circle else node_radius
    offset1 = circle_edgemark_rad+node_radius if edge_mark_1 == Mark.Circle else node_radius
    x0 = pos0[0] + offset0 * cos_angle
    y0 = pos0[1] + offset0 * sin_angle
    x1 = pos1[0] - offset1 * cos_angle
    y1 = pos1[1] - offset1 * sin_angle

    if is_curved:
        # connectionstyle ="arc3,rad=" + str(-node_distance / 4. )
        connectionstyle = "arc3,rad=" + str(4.0*node_radius / node_distance)
    else:
        connectionstyle = "arc3,rad=0"

    if edge_mark_0 is None and edge_mark_1 is None:
        axes.add_patch(patches.FancyArrowPatch((x0, y0), (x1, y1),
                                               edgecolor=line_color,
                                               facecolor=line_color,
                                               arrowstyle='-|>',
                                               mutation_scale=20, shrinkA=0, shrinkB=0,
                                               connectionstyle=connectionstyle
                                               )
                       )
    else:
        if edge_mark_0 == Mark.Directed and edge_mark_1 == Mark.Directed:
            arrow_style = '<|-|>'
        elif edge_mark_0 == Mark.Directed and edge_mark_1 != Mark.Directed:
            arrow_style = '<|-'
        elif edge_mark_0 != Mark.Directed and edge_mark_1 == Mark.Directed:
            arrow_style = '-|>'
        else:
            arrow_style = '-'

        if line_color.lower() == 'auto':
            if edge_mark_0 == Mark.Circle and edge_mark_1 == Mark.Circle:  # o--o
                line_color = 'red'
            elif (edge_mark_0 == Mark.Directed and edge_mark_1 == Mark.Tail) or \
                    (edge_mark_0 == Mark.Tail and edge_mark_1 == Mark.Directed):  # --->
                line_color = 'black'
            elif edge_mark_0 == Mark.Directed and edge_mark_1 == Mark.Directed:  # <-->
                line_color = 'blue'
            elif edge_mark_0 == Mark.Tail and edge_mark_1 == Mark.Tail:  # ----
                line_color = 'blueviolet'
            else:  # o-->
                line_color = 'limegreen'

        axes.add_patch(patches.FancyArrowPatch((x0, y0), (x1, y1),
                                               edgecolor=line_color,
                                               facecolor=line_color,
                                               arrowstyle=arrow_style,
                                               mutation_scale=20, shrinkA=0, shrinkB=0,
                                               connectionstyle=connectionstyle
                                               # connectionstyle="angle3,angleA=30,angleB=-30"
                                               # connectionstyle="arc3,rad=-0.5"
                                               )
                       )

        if edge_mark_0 == Mark.Circle:
            axes.add_patch(patches.Circle((x0, y0), circle_edgemark_rad, facecolor=fill_color, edgecolor=line_color))
        if edge_mark_1 == Mark.Circle:
            axes.add_patch(patches.Circle((x1, y1), circle_edgemark_rad, facecolor=fill_color, edgecolor=line_color))

    if text is not None:
        if font is None:
            font_dict = {
                # 'fontfamily' : 'Times',
                'fontsize': 10,
                'fontweight': 'bold'
            }
        else:
            assert isinstance(font, dict)
            font_dict = font

        xc = (x1+x0)/2
        yc = (y1+y0)/2
        axes.text(xc, yc, str(text),
                  horizontalalignment='center', verticalalignment='center',
                  backgroundcolor=fill_color, color=text_color, fontdict=font_dict)


def draw_node(axes, pos, node_radius,
              line_color='black', fill_color='white', node_name='', text_color='black', font=None,
              contour=None):
    """
    Draw a node at a specified location
    :param axes: a matplotlib axes object
    :param pos: position to place the node
    :param node_radius: node radius
    :param line_color: border color of the node
    :param fill_color: fill color of the node
    :param node_name: text to place inside the node (node is not resized
    :param text_color: color of the text
    :param font: a dictionary describing the font.
        Example: font = {'fontfamily': 'Times', 'fontsize': 14, 'fontweight': 'normal', 'fontstyle': 'italic'}
    :param contour: shape of the node: 'circle' or 'rectangle'
    """
    if font is None:
        font_dict = {
            # 'fontfamily': 'Times',
            'fontsize': 14,
            'fontweight': 'normal',
            'fontstyle': 'italic'
        }
    else:
        assert isinstance(font, dict)
        font_dict = font

    if contour is not None:
        assert isinstance(contour, str)  # contour is a string 'rectangle' or 'circle'
        assert contour.lower() in {'rectangle', 'circle'}
        if contour.lower() == 'rectangle':
            axes.add_patch(patches.Rectangle((pos[0]-node_radius, pos[1]-node_radius), node_radius * 2, node_radius * 2,
                                             facecolor=fill_color, edgecolor=line_color))
        elif contour.lower() == 'circle':
            axes.add_patch(patches.Circle(pos, node_radius, facecolor=fill_color, edgecolor=line_color))
        else:
            raise ValueError('Unsupported node contour.')
    else:  # is no contour is defined
        axes.add_patch(patches.Circle(pos, node_radius, facecolor=fill_color, edgecolor=line_color))

    # node text
    axes.text(pos[0], pos[1], str(node_name), horizontalalignment='center', verticalalignment='center',
              color=text_color, fontdict=font_dict)


def draw_graph(graph, latent_nodes=None, selection_nodes=None, bkcolor='white', fgcolor='black', line_color='auto',
               layout_type=None, node_labels=None, top=1, right=1):
    """
    Draw a graph. Currently supported graph types are DAG and PAG. Matplotlib is used as a backend.
    :param graph: the graph to be plotted
    :param latent_nodes: a set of nodes that are considered latents.
        In the case of DAGs they are drawn differently from other nodes
    :param selection_nodes: a set of nodes that are considered selection variables.
        In the case of DAGs they are drawn differently from other nodes
    :param bkcolor: background color of the node
    :param fgcolor: foreground color of the node
    :param line_color: color of the node contour and text
    :param layout_type: type of node position layout: 'circular' or 'force' (default; force-directed algorithm)
    :param node_labels: a mapping from node ID's to desired labels in the rendered graph.
    :return:
    """
    assert isinstance(graph, (DAG, PAG))
    if selection_nodes is None:
        selection_nodes = set()
    if latent_nodes is None:
        latent_nodes = set()
    if node_labels is None:
        node_labels = {}
    for node in graph.nodes_set:
        if node not in node_labels:
            node_labels[node] = node

    bottom = 0
    # top = 1
    left = 0
    # right = 1
    node_radius = 0.04
    width = right - left
    height = top - bottom
    fig = plt.figure()
    ax = fig.add_axes([left, bottom, width, height], frameon=False, aspect=1.)
    ax.set_axis_off()

    factor = 1000
    default_layout = ForceDirectedLayout(graph, (-factor, factor), (-factor, factor), num_iterations=100)
    if layout_type is None:
        factor = 1000
        layout = default_layout
    else:
        assert isinstance(layout_type, str)
        if layout_type == 'circular':
            layout = CircleLayout(graph, (-factor, factor), (-factor, factor))
        elif layout_type == 'force':
            layout = default_layout
        else:
            raise ValueError("Unsupported layout type")

    nodes_pos = layout.calc_layout()
    # normalize positions
    for node in graph.nodes_set:
        nodes_pos[node] = nodes_pos[node] / factor
        nodes_pos[node] = nodes_pos[node] * (1 - 4 * node_radius)  # squeeze to add margins (node radius)
        nodes_pos[node] = (nodes_pos[node] + 1) / 2

    for node in graph.nodes_set:
        if node in latent_nodes:
            contour = 'rectangle'
            fg = bkcolor
            bk = fgcolor
        elif node in selection_nodes:
            contour = 'rectangle'
            fg = fgcolor
            bk = bkcolor
        else:
            contour = 'circle'
            fg = fgcolor
            bk = bkcolor
        draw_node(ax, nodes_pos[node], node_radius=node_radius, node_name=node_labels[node], contour=contour,
                  line_color=fgcolor, fill_color=bk, text_color=fg)

    if isinstance(graph, PAG):
        for node_i, node_j in combinations(graph.nodes_set, 2):
            if graph.is_connected(node_i, node_j):
                text = None
                if graph.visible_edges is not None:
                    if (node_i, node_j) in graph.visible_edges or (node_j, node_i) in graph.visible_edges:
                        text = 'v'
                draw_edge(ax,
                          nodes_pos[node_i],
                          nodes_pos[node_j],
                          node_radius,
                          graph.get_edge_mark(node_parent=node_j, node_child=node_i),
                          graph.get_edge_mark(node_parent=node_i, node_child=node_j),
                          line_color=line_color, fill_color=bkcolor, text=text)
    elif isinstance(graph, DAG):
        if line_color == 'auto':
            line_color = 'black'
        for child_node in graph.nodes_set:
            for parent_node in graph.parents(child_node):
                draw_edge(ax,
                          nodes_pos[parent_node], nodes_pos[child_node],
                          node_radius, line_color=line_color)
    plt.show()
    return fig


def draw_temporal_graph(graph, nodes_set_list, ignore_homology=True, latent_nodes=None, selection_nodes=None,
                        column_labels=None, row_labels=None,
                        bkcolor='white', fgcolor='black', line_color='auto'):
    assert isinstance(graph, (PAG, DAG))
    if selection_nodes is None:
        selection_nodes = set()
    if latent_nodes is None:
        latent_nodes = set()

    text_color = 'black'
    group_sort = nodes_set_list
    if group_sort is None:
        nodes_order = list(graph.nodes_set)
        group_sort = [nodes_order]
    group_sort = list(reversed(group_sort))  # reverse the order of groups, thus past is first and future last

    if column_labels is not None:
        assert len(column_labels) == len(group_sort)
        for s in column_labels:
            assert isinstance(s, str)
        column_labels = list(reversed(column_labels))  # reverse the order of labels, thus past is first and future last

    if row_labels is not None:
        assert len(row_labels) == len(group_sort[-1])  # same length as the first group
        for s in row_labels:
            assert isinstance(s, str)

    font_dict = {
        # 'fontfamily': 'Times',
        'fontsize': 14,
        'fontweight': 'normal',
        'fontstyle': 'italic'
    }

    num_groups = len(group_sort)
    node_time = dict()
    node_y_id = dict()
    for group_id, group in enumerate(group_sort):
        for y_id, node in enumerate(group):
            node_time[node] = group_id - num_groups + 1
            node_y_id[node] = y_id

    bottom = 0
    top = 1
    left = 0
    right = 1
    node_radius = 0.04
    width = right - left
    height = top - bottom
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_axes([left, bottom, width, height], frameon=False, aspect=1.)
    ax.set_axis_off()

    layout = ColumnLayout(graph, (-1, 1), (-1, 1), group_sort)
    nodes_pos = layout.calc_layout()
    # normalize positions
    for node in graph.nodes_set:
        nodes_pos[node] = nodes_pos[node] * (1 - 6 * node_radius)  # squeeze to add margins (node radius)
        nodes_pos[node] = (nodes_pos[node] + 1) / 2  # scale to range [0, 1]

    # make room for row and column labels
    if row_labels is not None:
        for node in graph.nodes_set:
            nodes_pos[node][0] = nodes_pos[node][0] * (1-4*node_radius) + 4*node_radius
    if column_labels is not None:
        for node in graph.nodes_set:
            nodes_pos[node][1] = nodes_pos[node][1] * (1-2*node_radius) + 2*node_radius

    # plot a box around the present-time
    bot_present_pos = nodes_pos[nodes_set_list[0][-1]]  # bottom node position
    top_present_pos = nodes_pos[nodes_set_list[0][0]]  # top node position
    present_box_bl = (bot_present_pos[0] - 2 * node_radius,
                      bot_present_pos[1] - 1.5 * node_radius)

    ax.add_patch(patches.Rectangle(present_box_bl, 4 * node_radius,
                                   top_present_pos[1] - bot_present_pos[1] + 3 * node_radius,
                                   fill=False, color='gray', linestyle='-'))

    # display row labels
    if row_labels is not None:
        nodes = group_sort[-1]
        for i, label in enumerate(row_labels):
            ax.text(0, nodes_pos[nodes[i]][1], label, horizontalalignment='left', verticalalignment='center',
                    color=text_color, fontdict=font_dict)

    # display column labels
    if column_labels is not None:
        x_vals = sorted(list({nodes_pos[node][0] for node in graph.nodes_set}))
        for i, label in enumerate(column_labels):
            ax.text(x_vals[i], node_radius*2, label, horizontalalignment='center', verticalalignment='center',
                    color=text_color, fontdict=font_dict)

    for node in graph.nodes_set:
        if node in latent_nodes:
            contour = 'rectangle'
            fg = bkcolor
            bk = fgcolor
        elif node in selection_nodes:
            contour = 'rectangle'
            fg = fgcolor
            bk = bkcolor
        else:
            contour = 'circle'
            fg = fgcolor
            bk = bkcolor
        draw_node(ax, nodes_pos[node], node_radius=node_radius, node_name=str(node), contour=contour,
                  line_color=fgcolor, fill_color=bk, text_color=fg)

    present_nodes = set(nodes_set_list[0])

    if isinstance(graph, PAG):
        for node_i, node_j in combinations(graph.nodes_set, 2):
            if ignore_homology and node_i not in present_nodes and node_j not in present_nodes:
                continue
            if graph.is_connected(node_i, node_j):
                text = None
                if graph.visible_edges is not None:
                    if (node_i, node_j) in graph.visible_edges or (node_j, node_i) in graph.visible_edges:
                        text = 'v'
                if abs(node_time[node_i] - node_time[node_j]) > 1:
                    is_curved = True
                elif abs(node_y_id[node_i] - node_y_id[node_j]) > 1:
                    is_curved = True
                else:
                    is_curved = False
                draw_edge(ax,
                          nodes_pos[node_i], nodes_pos[node_j], node_radius,
                          graph.get_edge_mark(node_parent=node_j, node_child=node_i),
                          graph.get_edge_mark(node_parent=node_i, node_child=node_j),
                          is_curved=is_curved,
                          line_color=line_color, fill_color=bkcolor, text=text)
    elif isinstance(graph, DAG):
        if line_color == 'auto':
            line_color = 'black'
        for child_node in graph.nodes_set:
            for parent_node in graph.parents(child_node):
                if ignore_homology and parent_node not in present_nodes and child_node not in present_nodes:
                    continue
                draw_edge(ax,
                          nodes_pos[parent_node], nodes_pos[child_node],
                          node_radius, line_color=line_color, is_curved=True)
    plt.show()
    return fig
