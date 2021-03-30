import numpy as np
import random


def ete_tree(lst):
    from ete3 import Tree
    def newick(lst):
        if type(lst) is not list: lst = [lst]
        node, *children = lst
        if np.shape(node):
            name = 'Par%s' % list(np.shape(node))
        else:
            name = str(node)
        if children:
            return '({},({}))'.format(name, ','.join(map(newick, children)))
        else:
            return name
    return Tree(newick(lst) + ';')


def ascii_tree(lst):
    from treelib import Node, Tree
    t = Tree()
    letters = list(map(chr, range(65, 91)))
    names = {}
    def add_nodes(parent, lst):
        if type(lst) is not list: lst = [lst]
        node, *children = lst
        if np.shape(node):
            i = id(node)
            if i in names:
                a = names[i]
            else:
                a = random.choice(letters)
                letters.remove(a)
                names[i] = a
            name = '%s%s' % (a, list(np.shape(node)))
        else:
            name = str(node)
        i = random.randrange(1000)
        root = t.create_node(name, i, parent=parent)
        for child in children:
            add_nodes(i, child)
        return i
    add_nodes(None, lst)
    return t
