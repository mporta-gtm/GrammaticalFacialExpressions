""" Basic graph using Action Units as input."""
import sys
sys.path.insert(0, '')
sys.path.extend(['../'])

import numpy as np

from graph import tools

# Number of graph nodes
num_node = 18
# Self connections between nodes
self_link = [(i, i) for i in range(num_node)]
# Connections in inward direction
inward = [(i, k) for i in range(num_node) for k in range(num_node)]
# Connections in outward direction
outward = [(j, i) for (i, j) in inward]
# All connections
neighbor = inward + outward



class AdjMatrixGraph:
    """ Class to encapsulate adjacency matrix."""
    def __init__(self, *args, **kwargs):
        ''' Create adjacency matrix.'''
        # Get variables
        self.num_nodes = num_node
        self.edges = neighbor
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        # Compute binary adjacency matrix
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        # Compute binary adjacency matrix with self-loops
        self.A_binary_with_I = tools.get_adjacency_matrix(
            self.edges + self.self_loops, self.num_nodes)


if __name__ == '__main__':
    # Show adjacency matrix
    graph = AdjMatrixGraph()
    A_binary = graph.A_binary
    import matplotlib.pyplot as plt
    print(A_binary)
    plt.matshow(A_binary)
    plt.show()
                                              