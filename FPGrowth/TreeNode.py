# Gökhan Has - 161044067
# CSE 454 - DATA MINING
# ASSIGNMENT 03
# TreeNode.py


class TreeNode(object):
    def __init__(self, name, count, parent):
        self.value = name
        self.count = count
        self.parent = parent
        self.nodelink = None
        self.children = {}

    def increase_count(self, inc_count=1):
        self.count += inc_count

    def add_child(self, item):
        tree_node = TreeNode(item)
        if item in self.children:
            self.children[item].increase_count()
        else:
            self.children[item] = tree_node
        return tree_node

    def __str__(self):
        return "Name: {} Value: {} Children : {}".format(self.name, self.value, ', '.join(self.children.keys()))

