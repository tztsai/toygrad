import graphviz as gv
import numpy as np
import random
import numbers
from toych.core import Function, Context, array_repr


def nodelabel(node):
    if isinstance(node, np.ndarray):
        if not getattr(node, 'name', 1): repr(node)
        return array_repr(node)
    else:
        if isinstance(node, Function) and not node.need_init:
            return repr(type(node))
        return repr(node)

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
        if isinstance(ctx, Context):
            ret = [y, [ctx.getfunc()], *[walk(x, visited) for x in ctx.inputs]]
        else:
            ret = [y, [ctx, *[walk(x, visited) for x in ctx.inputs]]]
        return ret
    return walk(param)

def show_graph(param, filename=None):
    dot = dot_graph(deepwalk(param))
    dot.render(filename=filename, format='png', view=True, cleanup=bool(filename))
    return dot
