import graphviz as gv
import numpy as np
import random
import os
import numbers
from toych.core import Function, Param


NODES = {}  # {id: (node, label)}

def nodelabel(node):
    if not isinstance(node, np.ndarray):
        if isinstance(node, Function):
            return str(type(node))
        return str(node)
    nid = id(node)
    if nid not in NODES:
        if isinstance(node, np.ndarray) and node.size > 2:
            if not isinstance(node, Param):
                node = node.view(Param)
            lb = node.simple_repr()
        else:
            try:
                lb = '%.2e' % float(node)
            except:
                lb = str(node)
        NODES[nid] = (node, lb)
    else:
        lb = NODES[nid][1]
    return lb

def dot_graph(graph):
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
    g = gv.Digraph()
    add_edges(graph)
    return g

def deepwalk(param):
    def walk(y, visited={None}):
        try:
            ctx = y._outer_ctx
        except:
            try: ctx = y._ctx
            except: return [y]
        if ctx in visited: return [y]
        visited.add(ctx)
        ret = [y, [ctx, *[walk(x, visited) for x in ctx.inputs]]]
        if ctx.need_init:
            ret[1].insert(0, ApplyNode())
            ret[1][1] = [ctx]
        return ret
    return walk(param)

class ApplyNode:
    def __str__(self): return 'apply'


def show_graph(param, filename=None):
    dot = dot_graph(deepwalk(param))
    dot.render(filename=filename, format='png', view=True, cleanup=bool(filename))
    return dot
