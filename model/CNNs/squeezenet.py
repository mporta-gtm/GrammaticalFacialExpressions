""" SqueezeNet implementation from torchvision."""
from torchvision import models
import torch


class Model(torch.nn.Module):
    """ Model encapsulation."""
    def __init__(self, in_channels, num_class, *args, **kwargs):
        ''' Initialize the model layers
        :param num_class: number of classes
        :param in_channels: number of input channels
        The remaining parameters are keep for compatibility but not used
        in this model'''
        super(Model, self).__init__()
        self.model = models.squeezenet1_1(pretrained=False)
        self.model.features[0] = torch.nn.Conv2d(
            in_channels, 64, kernel_size=(3, 3), stride=(2, 2))
        self.model.classifier[1] = torch.nn.Conv2d(512, num_class,
            kernel_size=(1, 1), stride=(1, 1))
        self.model.train()

    def forward(self, x):
        ''' Forward pass of the model'''
        N, C, T, V, M = x.size()
        x = x.view(N, C, T, V * M)
        return self.model(x)
