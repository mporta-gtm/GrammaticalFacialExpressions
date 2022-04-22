""" Basic layer for graph convolutions in MSG3D """
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.activation import activation_factory


class MLP(nn.Module):
    """ Multi Layer Perceptron with 2D convolutions for GCNs"""
    def __init__(self, in_channels, out_channels, activation='relu', dropout=0):
        ''' Initialize the layers
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param activation: activation function
        :param dropout: dropout rate'''
        super().__init__()
        channels = [in_channels] + out_channels
        self.layers = nn.ModuleList()
        # Define blocks of layers
        for i in range(1, len(channels)):
            # Add dropout layer
            if dropout > 0.001:
                self.layers.append(nn.Dropout(p=dropout))
            # Add convolutional layer
            self.layers.append(nn.Conv2d(channels[i-1], channels[i], kernel_size=1))
            # Add batch normalization layer
            self.layers.append(nn.BatchNorm2d(channels[i]))
            # Add activation layer
            self.layers.append(activation_factory(activation))

    def forward(self, x):
        ''' Forward pass through the layers'''
        # Input shape: (N,C,T,V)
        for layer in self.layers:
            x = layer(x)
        return x

