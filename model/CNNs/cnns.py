""" Custom defined Convolutional Neural Networks"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model1(nn.Module):
    """ Model with 1 convolutional layer and 2 linear layers."""
    def __init__(self, in_channels, num_class, num_point, num_person,
                 hidden_neurons_1=64, hidden_neurons_2=256, window_size=100,
                 dropout=0.3, **kwargs):
        ''' Initialize the model layers
        :param in_channels: number of input channels
        :param num_class: number of output classes
        :param num_point: number of landmarks used
        :param num_person: number of persons analyzed     
        :param hidden_neurons_1: number of filters for the convolution layer
        :param hidden_neurons_2: number of hidden units in the linear layers
        :param window_size: length of temporal windows
        :param dropout: dropout rate
        '''
        super(Model1, self).__init__()
        input_size = num_person * hidden_neurons_1 * num_point * window_size
        self.c1 = nn.Conv2d(in_channels, hidden_neurons_1,
                            kernel_size=5, padding=(2, 2))
        self.fc2 = nn.Linear(input_size, hidden_neurons_2)
        self.fc3 = nn.Linear(hidden_neurons_2, num_class)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        ''' Forward pass of the model'''
        N, C, T, V, M = x.size()
        x = x.view(N, C, T, V * M)
        x = F.relu(self.c1(x))
        x = x.view(N, -1)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return self.activation(x)


class Model2(nn.Module):
    """ Model with 2 convolutional layers and 2 linear layers."""
    def __init__(self, in_channels, num_class, num_point, num_person,
                 hidden_neurons_1=64, hidden_neurons_2=256, window_size=100,
                 dropout=0.3, **kwargs):
        ''' Initialize the model layers
        :param in_channels: number of input channels
        :param num_class: number of output classes
        :param num_point: number of landmarks used
        :param num_person: number of persons analyzed     
        :param hidden_neurons_1: number of filters for the convolution layer
        :param hidden_neurons_2: number of hidden units in the linear layers
        :param window_size: length of temporal windows
        :param dropout: dropout rate
        '''
        super(Model2, self).__init__()
        hidden_size = 256
        
        input_size = num_person * hidden_neurons_2 * num_point * window_size
        self.c1 = nn.Conv2d(in_channels, hidden_neurons_1,
                            kernel_size=5, padding=(2, 2))
        self.c2 = nn.Conv2d(hidden_neurons_1, hidden_neurons_2,
                            kernel_size=3, padding=(1, 1))
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_class)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        ''' Forward pass of the model'''
        N, C, T, V, M = x.size()
        x = x.view(N, C, T, V * M)
        x = F.relu(self.c1(x))
        x = self.dropout(x)
        x = F.relu(self.c2(x))
        x = x.view(N, -1)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return self.activation(x)


class Model3(nn.Module):
    """ Model with an adapted convolutional layer and 2 linear layers.
    Adapted convolution uses filters with shape matching the input data."""
    def __init__(self, in_channels, num_class, num_point, num_person,
                 hidden_neurons_1=64, hidden_neurons_2=256, window_size=100,
                 dropout=0.3, **kwargs):
        ''' Initialize the model layers
        :param in_channels: number of input channels
        :param num_class: number of output classes
        :param num_point: number of landmarks used
        :param num_person: number of persons analyzed     
        :param hidden_neurons_1: number of filters for the convolution layer
        :param hidden_neurons_2: number of hidden units in the linear layers
        :param window_size: length of temporal windows
        :param dropout: dropout rate
        '''
        super(Model3, self).__init__()
        input_size = num_person * hidden_neurons_1 * window_size
        self.c1 = nn.Conv2d(in_channels, hidden_neurons_1,
                            kernel_size=(5, num_point), padding=(2, 0))
        self.bn = nn.BatchNorm2d(hidden_neurons_1)
        self.fc2 = nn.Linear(input_size, hidden_neurons_2)
        self.fc3 = nn.Linear(hidden_neurons_2, num_class)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        ''' Forward pass of the model'''
        N, C, T, V, M = x.size()
        x = x.view(N, C, T, V * M)
        x = F.relu(self.bn(self.c1(x)))
        x = x.view(N, -1)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return self.activation(x)


class Model4(nn.Module):
    """ Model with three convolutional layers and 2 linear layers."""
    def __init__(self, in_channels, num_class, num_point, num_person,
                 hidden_neurons_1=64, hidden_neurons_2=256, window_size=100,
                 dropout=0.3, **kwargs):
        ''' Initialize the model layers
        :param in_channels: number of input channels
        :param num_class: number of output classes
        :param num_point: number of landmarks used
        :param num_person: number of persons analyzed     
        :param hidden_neurons_1: number of filters for the convolution layer
        :param hidden_neurons_2: number of hidden units in the linear layers
        :param window_size: length of temporal windows
        :param dropout: dropout rate
        '''
        super(Model4, self).__init__()
        num_filters_2 = hidden_neurons_1 * 2
        
        input_size = num_person * num_filters_2 * num_point * window_size
        self.c1 = nn.Conv2d(in_channels, hidden_neurons_1,
                            kernel_size=5, padding=(2, 2))
        self.c2 = nn.Conv2d(hidden_neurons_1, num_filters_2,
                            kernel_size=3, padding=(1, 1))
        self.c3 = nn.Conv2d(num_filters_2, num_filters_2,
                            kernel_size=3, padding=(1, 1))
        self.fc2 = nn.Linear(input_size, hidden_neurons_2)
        self.fc3 = nn.Linear(hidden_neurons_2, num_class)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        ''' Forward pass of the model'''
        N, C, T, V, M = x.size()
        x = x.view(N, C, T, V * M)
        x = F.relu(self.c1(x))
        x = self.dropout(x)
        x = F.relu(self.c2(x))
        x = self.dropout(x)
        x = F.relu(self.c3(x))
        x = x.view(N, -1)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return self.activation(x)


class Model5(nn.Module):
    """ Model with an adapted convolutional layer, a classic convolutional
    layer and 2 linear layers.
    Adapted convolution uses filters with shape matching the input data."""
    def __init__(self, in_channels, num_class, num_point, num_person,
                 hidden_neurons_1=64, hidden_neurons_2=256, window_size=100,
                 dropout=0.3, **kwargs):
        ''' Initialize the model layers
        :param in_channels: number of input channels
        :param num_class: number of output classes
        :param num_point: number of landmarks used
        :param num_person: number of persons analyzed     
        :param hidden_neurons_1: number of filters for the convolution layer
        :param hidden_neurons_2: number of hidden units in the linear layers
        :param window_size: length of temporal windows
        :param dropout: dropout rate
        '''
        super(Model5, self).__init__()
        num_filters_2 = hidden_neurons_1 * 2
        input_size = num_person * num_filters_2 * window_size
        self.c1 = nn.Conv2d(in_channels, hidden_neurons_1,
                            kernel_size=(3, num_point), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(hidden_neurons_1)
        self.c2 = nn.Conv2d(hidden_neurons_1, num_filters_2,
                            kernel_size=(9, 1), padding=(4, 0))
        self.bn2 = nn.BatchNorm2d(num_filters_2)
        self.fc2 = nn.Linear(input_size, hidden_neurons_2)
        self.fc3 = nn.Linear(hidden_neurons_2, num_class)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        ''' Forward pass of the model'''
        N, C, T, V, M = x.size()
        x = x.view(N, C, T, V * M)
        x = F.relu(self.bn1(self.c1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.c2(x)))
        x = x.view(N, -1)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return self.activation(x)


class Model33(nn.Module):
    """ Model with an adapted convolutional layer, an adaptive pooling
    and two linear layers.
        Adapted convolution uses filters with shape matching the input data."""
    def __init__(self, in_channels, num_class, num_point, num_person,
                 hidden_neurons_1=64, hidden_neurons_2=256, window_size=100,
                 dropout=0.3, **kwargs):
        ''' Initialize the model layers
        :param in_channels: number of input channels
        :param num_class: number of output classes
        :param num_point: number of landmarks used
        :param num_person: number of persons analyzed     
        :param hidden_neurons_1: number of filters for the convolution layer
        :param hidden_neurons_2: number of hidden units in the linear layers
        :param window_size: length of temporal windows
        :param dropout: dropout rate
        '''
        super(Model33, self).__init__()

        input_size = num_person * hidden_neurons_1 * window_size
        self.c1 = nn.Conv2d(in_channels, hidden_neurons_1,
                            kernel_size=(5, num_point), padding=(2, 0))
        self.bn = nn.BatchNorm2d(hidden_neurons_1)
        self.pool = nn.AdaptiveMaxPool1d(window_size)
        self.fc2 = nn.Linear(input_size, hidden_neurons_2)
        self.fc3 = nn.Linear(hidden_neurons_2, num_class)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        ''' Forward pass of the model'''
        N, C, T, V, M = x.size()
        x = x.view(N, C, T, V * M)
        x = F.relu(self.bn(self.c1(x)))
        x = self.pool(torch.squeeze(x, -1))
        x = x.view(N, -1)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return self.activation(x)