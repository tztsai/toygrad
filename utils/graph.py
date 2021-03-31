import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math
import random


def show_ete(lst):
    from ete3 import Tree
    def newick(lst):
        node, *children = lst
        name = node_label(node)
        if children:
            return '({},({}))'.format(name, ','.join(map(newick, children)))
        else:
            return name
    Tree(newick(lst) + ';').show()


def show_ascii(lst):
    from treelib import Node, Tree
    t = Tree()
    def add_nodes(parent, lst):
        node, *children = lst
        label = node_label(node)
        nid = str(random.random())
        t.create_node(label, nid, parent=parent)
        for child in children:
            add_nodes(nid, child)
        return nid
    add_nodes(None, lst)
    t.show()


def to_netx(lst):
    def add_edges(lst):
        node, *children = lst
        for child in children:
            g.add_edge(node_label(child[0]), node_label(node))
            add_edges(child)
    g = nx.DiGraph()
    add_edges(lst)
    # pos = nx.spectral_layout(g)
    pos = nx.planar_layout(g)
    pos = nx.kamada_kawai_layout(g, pos=pos)
    # pos = nx.circular_layout(g)
    # pos = nx.spring_layout(g, k=1, pos=pos)
    return g, pos

    
def show_mpl(lst):
    options = {
        "font_size": 15,
        "node_size": 2000,
        "node_color": "white",
        "edgecolors": "black",
        "linewidths": 2,
        "width": 2,
        "with_labels": True,
    }
    g, pos = to_netx(lst)
    fig = plt.figure(figsize=[8, 6])
    nx.draw(g, pos=pos, **options)
    plt.gca().margins(0.20)
    plt.show()
    

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
    
    node_types = [n in PARAMS for n in g.nodes]
    marker_syms = [plotly_cfg.markersymbols[t] for t in node_types]
    colors = [plotly_cfg.colors[t] for t in node_types]
    info = [str(PARAMS.get(lb, lb)) for lb in g.nodes]
            
    # edge_trace = pg.Scatter(
    #     x=edge_x, y=edge_y,
    #     line=plotly_cfg.linestyle,
    #     mode='lines')

    node_trace = pg.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=list(g.nodes),
        textfont_size=plotly_cfg.fontsize,
        hoverinfo='text',
        hovertext=info,
        marker_symbol=marker_syms,
        marker=dict(size=plotly_cfg.markersize,
                    color=colors,
                    line_width=2))
    
    # arrows
    def arrow(x0, y0, x1, y1, sq0, sq1):
        s = 500
        k = plotly_cfg.markersize / s
        dx, dy = x1-x0, y1-y0
        def shift(x, y, sq):
            if sq: dr = max(abs(dx), abs(dy))
            else: dr = math.sqrt(dx**2 + dy**2)
            return dx/dr, dy/dr
        [(dx0, dy0), (dx1, dy1)] = starmap(shift, [(x0, y0, sq0), (x1, y1, sq1)])
        x0 += k * dx0; x1 -= k * dx1
        y0 += k * dy0; y1 -= k * dy1
        return dict(x=x0, ax=x1, y=y0, ay=y1, xref='x', yref='y',
                    axref='x', ayref='y', arrowwidth=1, arrowsize=1.2,
                    showarrow=True, arrowhead=2)
    arrows = [arrow(*pos[v], *pos[u], sq0=v in PARAMS, sq1=u in PARAMS) for u, v in g.edges]
    
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

    fig.show()
    # globals().update(locals())
    

class plotly_cfg:
    colors = ['#ca4', '#28d']
    markersymbols = ['circle', 'square']
    fontsize = 14
    markersize = 50
    linestyle = dict(width=0.8, color='#888')
    

LETTERS = tuple(map(chr, range(97, 123)))
PARAMS, LABELS = {}, {}
def node_label(node):
    if np.shape(node):
        if node in LABELS:
            a = LABELS[node]
        else:
            a = random.choice(LETTERS)
            while a in PARAMS:
                a += random.choice(LETTERS)
        lb = '%s%s' % (a, list(np.shape(node)))
        LABELS[node] = a
        PARAMS[lb] = node
        return lb
    else:
        return str(node)
    
    
def compgraph(param):
    def dfs(y, visited={None}):
        try: ctx = y._ctx
        except: return [y]
        if ctx in visited: return [y]
        visited.add(ctx)
        op = ctx.__class__
        return [y, [op, *[dfs(x, visited) for x in ctx.parents]]]
    return dfs(param)


def show_compgraph(param, type='mpl'):
    show = eval('show_' + type)
    return show(compgraph(param))
