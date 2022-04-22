""" MobileNet v2 implementation from torchvision"""
from torchvision import models
import torch



class Model(torch.nn.Module):
    """Model encapsulation"""
    def __init__(self, in_channels, num_class, *args, **kwargs):
        ''' Initialize the model layers
        :param num_class: number of classes
        :param in_channels: number of input channels
        The remaining parameters are keep for compatibility but not used
        in this model'''
        super(Model, self).__init__()
        self.model = models.mobilenet_v2(pretrained=False)
        self.model.features[0] = torch.nn.Conv2d(
            in_channels, 32, kernel_size=(3, 3), stride=(2, 2),
            padding=(1, 1), bias=False)
        self.model.classifier[-1] = torch.nn.Linear(
            in_features=1280, out_features=num_class, bias=True)
        self.model.classifier.add_module('softmax', torch.nn.Softmax(dim=1))
        self.model.train()

    def forward(self, x):
        ''' Forward pass of the model'''
        N, C, T, V, M = x.size()
        x = x.view(N, C, T, V * M)
        return self.model(x)
