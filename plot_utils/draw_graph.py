from matplotlib import pyplot as plt
from matplotlib import patches
from math import sqrt
from .graph_layout import ForceDirectedLayout, CircleLayout
from graphical_models import DAG, PAG, arrow_head_types as Mark
from itertools import combinations


def draw_edge(axes, pos0, pos1, node_radius,
              edge_mark_0=None, edge_mark_1=None, line_color='black', fill_color='white',
              text=None, text_color='black', font=None):
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

    if edge_mark_0 is None and edge_mark_1 is None:
        axes.add_patch(patches.FancyArrowPatch((x0, y0), (x1, y1),
                                               edgecolor=line_color,
                                               facecolor=line_color,
                                               arrowstyle='-|>',
                                               mutation_scale=20, shrinkA=0, shrinkB=0
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
               layout_type=None):
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
    :return:
    """
    assert isinstance(graph, (DAG, PAG))
    if selection_nodes is None:
        selection_nodes = set()
    if latent_nodes is None:
        latent_nodes = set()

    bottom = 0
    top = 1
    left = 0
    right = 1
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
        draw_node(ax, nodes_pos[node], node_radius=node_radius, node_name=str(node), contour=contour,
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
