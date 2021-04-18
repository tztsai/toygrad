import graphviz as gv
import numpy as np
import random
import os
import numbers
from core import Function, Param


NODES, LABELS = {}, set()  # {id: (node, label)}, {label}

def nodelabel(node):
    if not isinstance(node, np.ndarray):
        if isinstance(node, Function):
            return str(node) if node.wait_inputs else str(type(node))
        return str(node)
    nid = id(node)
    if nid not in NODES:
        if isinstance(node, np.ndarray) and node.size > 2:
            a = node.name if hasattr(node, 'name') else 'array'
            # while a in LABELS:
            #     a += random.choice('1234567890')
            # LABELS.add(a)
            lb = '%s%s' % (a, list(np.shape(node)))
        else:
            try:
                lb = '%.2e' % float(node)
            except:
                lb = str(node)
        NODES[nid] = (node, lb)
    else:
        lb = NODES[nid][1]
    return lb

def dot_graph(graph, graphname='_temp_graph'):
    def add_edges(graph):
        def add_node(n):
            nid = str(random.random()) if isinstance(n, numbers.Number) else hex(id(n))
            shape = 'box' if np.shape(n) else 'oval'
            g.node(nid, nodelabel(n), shape=shape)
            return nid
        node, *children = graph
        for child in children:
            g.edge(add_node(child[0]), add_node(node))
            add_edges(child)
    g = gv.Digraph(graphname)
    add_edges(graph)
    return g

def compgraph(param):
    def search(y, visited={None}):
        try:
            ctx = y._outer_ctx
        except:
            try: ctx = y._ctx
            except: return [y]
        if ctx in visited: return [y]
        visited.add(ctx)
        ret = [y, [ctx, *[search(x, visited) for x in ctx.inputs]]]
        if ctx.parent:
            ret[1].insert(0, ApplyNode())
            ret[1][1] = [ctx.parent]
        return ret
    return search(param)

class ApplyNode:
    def __str__(self): return 'apply'


def show_compgraph(param, filename=None):
    dot = dot_graph(compgraph(param))
    dot.render(filename=filename, format='png', view=True, cleanup=bool(filename))
    for f in os.listdir():
        if '_temp_graph' in f: os.remove(f)
    return dot
