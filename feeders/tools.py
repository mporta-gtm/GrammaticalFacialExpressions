""" Tools and utilities for data feeding and pre-processing"""
import numpy as np


def flip(data):
    ''' Flip the data in spatial dimension
    :param data: data with shape (C, T, V, M)'''
    new_data = data.copy()
    if np.random.rand() > 0.5:
        new_data[0] *= -1
    return new_data

