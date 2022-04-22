""" Data feeder using facial landmarks for MSG3D.
It should work with any evaluation method."""
import sys

from torch.utils import data
sys.path.extend(['../'])

import torch
import pickle
import numpy as np
import re
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from collections import Counter

from feeders import tools

# Look Up Table to simplify LSE_GFE labels
SIMPLE_ENCODER = {'q.partial': 'int', 'q.polar': 'int', 'q.other': 'int',
                  'n.L-R': 'neg', 'n.others': 'neg', 'None': 'none'}
# LSE_GFE simplified labels
SIMPLE_CLASSES = ['int', 'neg', 'none']

class Feeder(Dataset):
    """ Class to handle data retrieving."""
    def __init__(self, data_path, flip=False, normalization=False):
        ''' Initialize variables
        :param data_path: Path to the pickle file containing a dict of samples and labels
        :param flip: Horizontally flip the landmarks randomly
        :param normalization: If true, normalize input landmarks
        '''

        self.data_path = data_path
        self.normalization = normalization
        self.flip = flip

        self.load_data()
        
    def load_data(self):
        ''' Load data from pickle file'''
        # Read data file
        try:
            with open(self.data_path, 'rb') as f:
                data_dict = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.data_path, 'rb') as f:
                data_dict = pickle.load(f, encoding='latin1')
        # Read ids of collaborators
        if 'LSE_GFE' in self.data_path:
            self.person = [re.search(r'p\d{4}', xx).group(0).strip()
                        for xx in data_dict['name']]
        else:
            self.person = [xx.split('_')[0] for xx in data_dict['name']]
        # Read labels names
        self.target = data_dict['label']
        self.classes = list(set(self.target))
        # Encode classes for singleclass classification
        self.encoder = LabelEncoder()
        self.encoder.fit(self.classes)
        self.label = self.encoder.transform(self.target)
        # Save samples
        self.data = data_dict['sample']
        # Normalize data
        if self.normalization:
            self.normalize()


    def normalize(self):
        ''' Normalize landmarks to [-1, 1] with respect to the nose'''
        data = np.array(self.data)
        if len(data.shape) > 1:
            N, C, T, V = data.shape
            center = data[..., 30]
            data = data - center[..., None]
            m = np.max(np.abs(data), axis=(2, 3))
            data = data / m[:, :, None, None]
        self.data = data
            
    def __str__(self):
        ''' Print information about the dataset '''
        return f"{len(self)} samples distributed as {Counter(self.target).most_common()}"
    
    def get_dist(self, indices):
        ''' Print information about the train or test dataset '''
        samples = np.array(self.target)[indices]
        return f"{len(indices)} samples distributed as {Counter(samples).most_common()}"

    def __len__(self):
        ''' Return the number of samples '''
        return len(self.label)

    def __iter__(self):
        ''' Iterate through the dataset '''
        return self

    def __getitem__(self, index):
        ''' Retrieve sample and label given an index.'''
        # Obtain sample and label
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        # Pre-process and augment data
        if self.flip:
            data_numpy = tools.flip(data_numpy)

        return data_numpy[..., None], label, index

    def top_k(self, score, top_k, indexes):
        ''' Obtain top k accuracy given a score and indexes'''
        # Sort scores
        rank = score.argsort()
        # Obtain groundtruth labels
        labels = self.label[indexes]
        # Compute top k hits
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(labels)]
        # Compute and return accuracy
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    ''' Import a class given its name '''
    # Split name in components
    components = name.split('.')
    # Dive into components to retrieve desired class
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def test(data_path, vid=None, graph=None, is_3d=False):
    '''
    vis the samples using matplotlib
    :param data_path:
    :param vid: the id of sample
    :param graph:
    :param is_3d: when vis NTU, set it True
    :return:
    '''
    import matplotlib.pyplot as plt
    loader = torch.utils.data.DataLoader(
        dataset=Feeder(data_path),
        batch_size=64,
        shuffle=False,
        num_workers=2)

    if vid is not None:
        sample_name = loader.dataset.sample_name
        sample_id = [name.split('.')[0] for name in sample_name]
        index = sample_id.index(vid)
        data, label, index = loader.dataset[index]
        data = data.reshape((1,) + data.shape)

        # for batch_idx, (data, label) in enumerate(loader):
        N, C, T, V, M = data.shape

        plt.ion()
        fig = plt.figure()
        if is_3d:
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        if graph is None:
            p_type = ['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'k.', 'k.', 'k.']
            pose = [
                ax.plot(np.zeros(V), np.zeros(V), p_type[m])[0] for m in range(M)
            ]
            ax.axis([-1, 1, -1, 1])
            for t in range(T):
                for m in range(M):
                    pose[m].set_xdata(data[0, 0, t, :, m])
                    pose[m].set_ydata(data[0, 1, t, :, m])
                fig.canvas.draw()
                plt.pause(0.001)
        else:
            p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
            import sys
            from os import path
            sys.path.append(
                path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
            G = import_class(graph)()
            edge = G.inward
            pose = []
            for m in range(M):
                a = []
                for i in range(len(edge)):
                    if is_3d:
                        a.append(ax.plot(np.zeros(3), np.zeros(3), p_type[m])[0])
                    else:
                        a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m])[0])
                pose.append(a)
            ax.axis([-1, 1, -1, 1])
            if is_3d:
                ax.set_zlim3d(-1, 1)
            for t in range(T):
                for m in range(M):
                    for i, (v1, v2) in enumerate(edge):
                        x1 = data[0, :2, t, v1, m]
                        x2 = data[0, :2, t, v2, m]
                        if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                            pose[m][i].set_xdata(data[0, 0, t, [v1, v2], m])
                            pose[m][i].set_ydata(data[0, 1, t, [v1, v2], m])
                            if is_3d:
                                pose[m][i].set_3d_properties(data[0, 2, t, [v1, v2], m])
                fig.canvas.draw()
                # plt.savefig('/home/lshi/Desktop/skeleton_sequence/' + str(t) + '.jpg')
                plt.pause(0.01)

