import numpy as np
import matplotlib.pyplot as plt
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


def show_netx(lst):
    import networkx as nx
    options = {
        "font_size": 15,
        "node_size": 2000,
        "node_color": "white",
        "edgecolors": "black",
        "linewidths": 2,
        "width": 2,
        "with_labels": True,
    }
    
    def add_edges(lst):
        node, *children = lst
        for child in children:
            g.add_edge(node_label(child[0]), node_label(node))
            add_edges(child)
            
    g = nx.DiGraph()
    add_edges(lst)
    fig = plt.figure(figsize=[8, 6])
    nx.draw(g, **options)
    plt.gca().margins(0.20)
    plt.show()

LETTERS = tuple(map(chr, range(65, 91)))
NODE_LABELS = {}
def node_label(node):
    if np.shape(node):
        if node in NODE_LABELS:
            a = NODE_LABELS[node]
        else:
            a = random.choice(LETTERS)
            while a in NODE_LABELS:
                a += random.choice(LETTERS)
            NODE_LABELS[node] = a
        return '%s%s' % (a, list(np.shape(node)))
    else:
        return str(node)

def show_compgraph(param, type='ascii'):
    "If you just want a list, pass type=list."
    def dfs(y, visited={None}):
        try: ctx = y._ctx
        except: return [y]
        if ctx in visited: return [y]
        visited.add(ctx)
        op = ctx.__class__
        return [y, [op, *[dfs(x, visited) for x in ctx.parents]]]
    show = eval('show_' + type)
    return show(dfs(param))
