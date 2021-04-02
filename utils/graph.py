import matplotlib.pyplot as plt
import networkx as nx
import graphviz as gv
import numpy as np
import random


LETTERS = tuple(map(chr, range(97, 123)))  # lowercase alphabet
NODES, LABELS = {}, set()  # {id: (node, label)}, {label}

def label(node, lb=None):
    nid = id(node)
    if nid not in NODES:
        if np.shape(node):
            if lb and nid not in LABELS:
                a = lb
            else:
                a = random.choice(LETTERS)
                while a in LABELS:
                    a += random.choice(LETTERS)
            lb = '%s%s' % (a, list(np.shape(node)))
            LABELS.add(lb)
        else:
            try:
                lb = '%.2e' % float(node)
            except:
                lb = str(type(node)) if hasattr(node, 'apply') else str(node)
        NODES[nid] = (node, lb)
    else: lb = NODES[nid][1]
    return lb

# def show_ete(lst):
#     from ete3 import Tree
#     def newick(lst):
#         node, *children = lst
#         name = node_label(node)
#         if children:
#             return '({},({}))'.format(name, ','.join(map(newick, children)))
#         else:
#             return name
#     Tree(newick(lst) + ';').show()

# def show_ascii(lst):
#     from treelib import Node, Tree
#     t = Tree()
#     def add_nodes(parent, lst):
#         node, *children = lst
#         label = node_label(node)
#         nid = str(random.random())
#         t.create_node(label, nid, parent=parent)
#         for child in children:
#             add_nodes(nid, child)
#         return nid
#     add_nodes(None, lst)
#     t.show()


def to_netx(lst):
    def add_edges(lst):
        node, *children = lst
        for child in children:
            label(child[0]), label(node)
            g.add_edge(id(child[0]), id(node))
            add_edges(child)
    g = nx.DiGraph()
    add_edges(lst)
    n = len(g.nodes)
    pos = nx.fruchterman_reingold_layout(g, k=2/np.sqrt(n))
    return g, pos

def show_plt(lst):
    g, pos = to_netx(lst)
    fig = plt.figure(figsize=[10, 7.5])
    labels = {i: NODES[i][1] for i in g.nodes}
    nx.draw(g, labels=labels, **graph_cfg)
    plt.gca().margins(0.20)
    plt.show()
    return fig 

graph_cfg = {
    "font_size": 8,
    "node_size": 2000,
    "node_color": "white",
    "edgecolors": "black",
    "linewidths": 0.8,
    "width": 0.8,
    "with_labels": True,
}

def show_plotly(lst):
    import plotly.graph_objects as pg
    from itertools import starmap
    
    def edge_node_pos(g):
        for edge in g.edges:
            yield pos[edge[0]]
            yield pos[edge[1]]
            yield None, None

    g, pos = to_netx(lst)
    edge_x, edge_y = np.transpose(list(edge_node_pos(g)))
    node_x, node_y = np.transpose([pos[n] for n in g.nodes])
    
    node_types = [bool(np.ndim(NODES[n][0])) for n in g.nodes]
    marker_syms = [plotly_cfg.markersymbols[t] for t in node_types]
    colors = [plotly_cfg.colors[t] for t in node_types]
    labels = [NODES[n][1] for n in g.nodes]
    info = [str(NODES[n][0]) for n in g.nodes]
            
    # edge_trace = pg.Scatter(
    #     x=edge_x, y=edge_y,
    #     line=plotly_cfg.linestyle,
    #     mode='lines')
    node_trace = pg.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=labels,
        textfont_size=plotly_cfg.fontsize,
        hoverinfo='text',
        hovertext=info,
        marker_symbol=marker_syms,
        marker=dict(size=plotly_cfg.markersize,
                    color=colors,
                    line_width=2))
    
    # arrows
    def arrow(x0, y0, x1, y1, sq0, sq1):
        s = plotly_cfg.arrow_margin
        k = plotly_cfg.markersize / s
        dx, dy = x1-x0, y1-y0
        def shift(x, y, sq):
            if sq: dr = max(abs(dx), abs(dy))
            else: dr = np.sqrt(dx**2 + dy**2)
            return dx/dr, dy/dr
        [(dx0, dy0), (dx1, dy1)] = starmap(shift, [(x0, y0, sq0), (x1, y1, sq1)])
        x0 += k * dx0; x1 -= k * dx1
        y0 += k * dy0; y1 -= k * dy1
        return dict(x=x0, ax=x1, y=y0, ay=y1, xref='x', yref='y',
                    axref='x', ayref='y', arrowwidth=1, arrowsize=1.2,
                    showarrow=True, arrowhead=2)
    arrows = [arrow(*pos[v], *pos[u], sq0=np.ndim(NODES[v][0]), sq1=np.ndim(NODES[u][0]))
              for u, v in g.edges]
    
    fig = pg.Figure(data=[node_trace],
                    layout=pg.Layout(
        title='<br>Computation graph of a toych network',
        titlefont=dict(size=14),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        annotations=arrows)
    )
    fig.show(); return fig

class plotly_cfg:
    colors = ['#ca4', '#28d']
    markersymbols = ['circle', 'square']
    fontsize = 10
    markersize = 40
    linestyle = dict(width=0.8, color='#888')
    arrow_margin = 400
    
    
def show_dot(lst):    
    def add_edges(lst):
        def add_node(n):
            nid = hex(id(n))
            shape = 'box' if np.shape(n) else 'oval'
            g.node(nid, label(n), shape=shape)
            return nid
        node, *children = lst
        for child in children:
            g.edge(add_node(child[0]), add_node(node))
            add_edges(child)
    g = gv.Digraph('A toych computation graph')
    add_edges(lst)
    # g.render(format='png')
    return g


def compgraph(param):
    def dfs(y, visited={None}):
        try: ctx = y._ctx
        except: return [y]
        if ctx in visited: return [y]
        visited.add(ctx)
        return [y, [ctx, *[dfs(x, visited) for x in ctx.inputs]]]
    return dfs(param)

def show_compgraph(param, type='dot'):
    """available types: ['plt', 'plotly', 'dot']."""
    show = eval('show_' + type)
    return show(compgraph(param))
