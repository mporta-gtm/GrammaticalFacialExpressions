""" Utilities and helpers for training and evaluation of models."""
import torch
import random
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def import_class(name):
    '''Import a class dynamically by name'''
    # Split the class name into a module and class name
    components = name.split('.')
    mod = __import__(components[0])
    # Dive into components to get the class
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def count_params(model):
    '''Count the number of parameters in a model'''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def my_collate(batch):
    '''Custom collate function for dealing with batches of images'''
    # Read samples
    data = [torch.from_numpy(item[0]).float() for item in batch]
    # Read labels
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    # Read indexes
    index = [item[2] for item in batch]
    index = torch.LongTensor(index)
    # Return as tuple
    return [data, target, index]

def create_confusion_matrix(y_true, y_pred, dataset, normalize=False,
                          title=None, simple=False, cmap=plt.cm.Blues):
        """ Plot the confusion matrixes for the obtained predictions"""
        # Define title
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'
        # Read classes names and labels
        classes = dataset.classes
        labels = dataset.encoder.transform(classes)
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        # Normalize confusion matrix
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # Plot confusion matrix
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        # Set colorbar
        ax.figure.colorbar(im, ax=ax)
        # Set axis ticks
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        # Format tick labels
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return fig

def perclass_accuracy(y_true, y_pred, dataset):
    """ Compute the accuracy of the predictions for each class"""
    # Read classes names and labels
    classes = dataset.classes
    labels = dataset.encoder.transform(classes)
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # Obtain individual accuracies from matrix diagonal
    with np.errstate(divide='ignore', invalid='ignore'):
        acc = np.diag(cm) / cm.sum(1)
    return {classes[i]: acc[i] for i in range(len(acc))}
