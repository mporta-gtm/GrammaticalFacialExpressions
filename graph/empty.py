""" Graph using all the 68 landmarks but none connection."""
import sys
sys.path.insert(0, '')
sys.path.extend(['../'])

import numpy as np

from graph import tools

# Joint index:
# {0,  right ear}, {17, right eyebrow},    {34, nose contour},      {51, outer lips},       
# {1,  contour},   {18, right eyebrow},    {35, nose contour},      {52, outer lips},
# {2,  contour},   {19, right eyebrow},    {36, right eye},         {53, outer lips},
# {3,  contour},   {20, right eyebrow},    {37, right eye},         {54, outer lips},
# {4,  contour},   {21, right eyebrow},    {38, right eye},         {55, outer lips},
# {5,  contour},   {22, left eyebrow},     {39, right eye},         {56, outer lips},   
# {6,  contour},   {23, left eyebrow},     {40, right eye},         {57, outer lips},   
# {7,  contour},   {24, left eyebrow},     {41, right eye},         {58, outer lips},   
# {8,  chin},      {25, left eyebrow},     {42, left eye},          {59, outer lips},   
# {9,  contour},   {26, left eyebrow},     {43, left eye},          {60, inner lips},   
# {10, contour},   {27, nose trunk},       {44, left eye},          {61, inner lips},   
# {11, contour},   {28, nose trunk},       {45, left eye},          {62, inner lips},   
# {12, contour},   {29, nose trunk},       {46, left eye},          {63, inner lips},   
# {13, contour},   {30, nose trunk},       {47, left eye},          {64, inner lips},  
# {14, contour},   {31, nose contour},     {48, outer lips},        {65, inner lips},  
# {15, contour},   {32, nose contour},     {49, outer lips},        {66, inner lips},  
# {16, left ear},  {33, nose contour},     {50, outer lips},        {67, inner lips}

# Number of landmarks used
num_node = 68
# Self connections
self_link = [(i, i) for i in range(num_node)]
# Inward connections
inward = []
# Outward connections
outward = []
# All connections
neighbor = inward + outward


class AdjMatrixGraph:
    """ Class to encapsulate the adjacency matrix """
    def __init__(self, *args, **kwargs):
        ''' Create adjacency matrix'''
        # Initialize variables
        self.num_nodes = num_node
        self.edges = neighbor
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        # Create binary adjacency matrix
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        # Create binary adjacency matrix with self loops
        self.A_binary_with_I = tools.get_adjacency_matrix(
            self.edges + self.self_loops, self.num_nodes)


if __name__ == '__main__':
    # Display adjacency matrix
    graph = AdjMatrixGraph()
    A_binary = graph.A_binary
    import matplotlib.pyplot as plt
    print(A_binary)
    plt.matshow(A_binary)
    plt.show()
