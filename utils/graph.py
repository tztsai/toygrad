import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
import random


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
            add_node(child[0])
            add_node(node)
            g.add_edge(id(child[0]), id(node))
            add_edges(child)
    g = nx.DiGraph()
    add_edges(lst)
    # pos = nx.spectral_layout(g)
    # pos = nx.circular_layout(g)
    # pos = nx.planar_layout(g)
    pos = nx.kamada_kawai_layout(g)#, pos=pos)
    pos = nx.spring_layout(g, k=10/np.sqrt(g.size()), pos=pos)
    return g, pos

    
LETTERS = tuple(map(chr, range(97, 123)))
PARAM_LABELS, LABELS = {}, {}

def getlabel(node, label=None):
    nid = id(node)
    if nid not in LABELS:
        if label and nid not in PARAM_LABELS:
            a = label
        else:
            a = random.choice(LETTERS)
            while a in PARAM_LABELS.values():
                a += random.choice(LETTERS)
        lb = '%s%s' % (a, list(np.shape(node)))
        PARAM_LABELS[nid] = lb
        return lb
        
def add_node(node, label=None):
    nid = id(node)
    if nid in LABELS:
        return
    elif np.shape(node):
        lb = label(node)
    else:
        try:
            x = float(node)
            lb = '%.2e' % x
        except:
            lb = str(node)
    LABELS[nid] = lb

    
def show_plt(lst):
    g, pos = to_netx(lst)
    fig = plt.figure(figsize=[8, 6])
    nx.draw(g, pos=pos, labels=LABELS, **graph_cfg)
    plt.gca().margins(0.20)
    plt.show()
    
graph_cfg = {
    "font_size": 10,
    "node_size": 2500,
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
    
    node_types = [n in LABELS[n] in PARAM_LABELS for n in g.nodes]
    marker_syms = [plotly_cfg.markersymbols[t] for t in node_types]
    colors = [plotly_cfg.colors[t] for t in node_types]
    info = [str(PARAM_LABELS.get(lb := LABELS[n], lb)) for n in g.nodes]
            
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
    arrows = [arrow(*pos[v], *pos[u], sq0=v in PARAM_LABELS, sq1=u in PARAM_LABELS) for u, v in g.edges]
    
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
    
    
def compgraph(param):
    def dfs(y, visited={None}):
        try: ctx = y._ctx
        except: return [y]
        if ctx in visited: return [y]
        visited.add(ctx)
        op = ctx.__class__
        return [y, [op, *[dfs(x, visited) for x in ctx.inputs]]]
    return dfs(param)


def show_compgraph(param, type='plt'):
    show = eval('show_' + type)
    return show(compgraph(param))
