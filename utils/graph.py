import graphviz as gv
import numpy as np
import random
from core import Function, Param


LETTERS = tuple(map(chr, range(97, 123)))  # lowercase alphabet
NODES, LABELS = {}, set()  # {id: (node, label)}, {label}

def nodelabel(node):
    nid = id(node)
    if nid not in NODES:
        if isinstance(node, Param) and node.size > 1:
            try: a = node.name
            except: a = random.choice(LETTERS)
            while a in LABELS:
                a += random.choice('1234567890')
            LABELS.add(a)
            lb = '%s%s' % (a, list(np.shape(node)))
        else:
            try:
                lb = '%.2e' % float(node)
            except:
                lb = str(type(node)) if isinstance(node, Function) else str(node)
        NODES[nid] = (node, lb)
    else: lb = NODES[nid][1]
    return lb
    
def dot_graph(graph):    
    def add_edges(graph):
        def add_node(n):
            nid = hex(id(n))
            shape = 'box' if np.shape(n) else 'oval'
            g.node(nid, nodelabel(n), shape=shape)
            return nid
        node, *children = graph
        for child in children:
            g.edge(add_node(child[0]), add_node(node))
            add_edges(child)
    g = gv.Digraph('A toych computation graph')
    add_edges(graph)
    return g

def compgraph(param):
    def dfs(y, visited={None}):
        try:
            ctx = y._outer_ctx
        except:
            try: ctx = y._ctx
            except: return [y]
        if ctx in visited: return [y]
        visited.add(ctx)
        return [y, [ctx, *[dfs(x, visited) for x in ctx.inputs]]]
    return dfs(param)

def show_compgraph(param, filename=None):
    dot = dot_graph(compgraph(param))
    dot.render(filename=filename, format='png', view=True, cleanup=bool(filename))
    return dot
