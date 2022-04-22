""" Train and evaluate a model using Leave One Subject Out Cross Validation
, repeat the process several times and compute the mean and standard deviation
of the resulting metrics.
Updated from main_loso_repeat_new to reset model parameters in a proper way."""
#!/usr/bin/env python
from __future__ import print_function
import os
import time
import yaml
import random
import pickle
import shutil
import inspect
import argparse
from collections import OrderedDict, defaultdict

import torch
import numpy as np
from torch import nn
from torch import optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
# import apex
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (cohen_kappa_score, matthews_corrcoef,
                             log_loss, f1_score)
# To create confusion matrix without showing the plot
import matplotlib
matplotlib.use('Agg')
# Custom functions in utils.py
from utils import *


def init_seed(seed):
    '''Initialize random seed'''
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_parser():
    '''Parse arguments'''
    parser = argparse.ArgumentParser(description='MS-G3D')

    parser.add_argument(
        '--work-dir',
        type=str,
        required=False,
        help='the work folder for storing results')
    parser.add_argument('--model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/train_msg3d.yaml',
        help='path to the configuration file')
    parser.add_argument(
        '--assume-yes',
        action='store_true',
        help='Say yes to every prompt')

    parser.add_argument(
        '--phase',
        default='train',
        help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if true, the classification score will be stored')

    parser.add_argument(
        '--seed',
        type=int,
        default=random.randrange(200),
        help='random seed')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=1,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--eval-start',
        type=int,
        default=1,
        help='The epoch number to start evaluating models')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    parser.add_argument(
        '--feeder',
        default='feeder.feeder',
        help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=os.cpu_count(),
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')

    parser.add_argument(
        '--model',
        default=None,
        help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    parser.add_argument(
        '--amp-opt-level',
        type=int,
        default=1,
        help='NVIDIA Apex AMP optimization level')

    parser.add_argument(
        '--base-lr',
        type=float,
        default=0.01,
        help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument(
        '--optimizer',
        default='SGD',
        help='type of optimizer')
    parser.add_argument(
        '--nesterov',
        type=str2bool,
        default=False,
        help='use nesterov or not')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='training batch size')
    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=256,
        help='test batch size')
    parser.add_argument(
        '--forward-batch-size',
        type=int,
        default=16,
        help='Batch size during forward pass, must be factor of --batch-size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--optimizer-states',
        type=str,
        help='path of previously saved optimizer states')
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='path of previously saved training checkpoint')
    parser.add_argument(
        '--num-repeats',
        type=int,
        default=30,
        help='Num of repetitions to perform')

    return parser
  

class Processor():
    """Processor for keypoints-based Action Recgnition"""

    def __init__(self, arg):
        '''Initialize the Processor'''
        self.arg = arg
        if arg.phase == 'train':
            # Check log dir
            if os.path.isdir(arg.work_dir):
                print(f'log_dir {arg.work_dir} already exists')
                if arg.assume_yes:
                    answer = 'y'
                else:
                    answer = input('delete it? [y]/n:')
                if answer.lower() in ('y', ''):
                    shutil.rmtree(arg.work_dir)
                    os.mkdir(arg.work_dir)
                    print('Dir removed:', arg.work_dir)
                else:
                    print('Dir not removed:', arg.work_dir)
            logdir = os.path.join(arg.work_dir, 'trainlogs')
            # Create metric writers
            self.writer = {}
            self.writer['train'] = SummaryWriter(os.path.join(logdir, 'train'), 'train')
            self.writer['test'] = SummaryWriter(os.path.join(logdir, 'val'), 'val')

        print(f"Log dir: {logdir}")
        self.save_arg()
        # Create backup of Model, Feeder and main.py
        Model = import_class(self.arg.model)
        Feeder = import_class(self.arg.feeder)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        shutil.copy2(inspect.getfile(Feeder), self.arg.work_dir)
        shutil.copy2(os.path.join('.', __file__), self.arg.work_dir)

        # Initialize results containers
        self.train_acc = {}
        self.val_targets = {}
        self.val_logits = {}

    def load_model(self):
        '''Load defined model'''
        # Read graphic device
        output_device = self.arg.device[0] if type(
            self.arg.device) is list else self.arg.device
        self.output_device = output_device
        # Read model definition
        Model = import_class(self.arg.model)
        # Instantiate model and initialize loss
        self.model = Model(**self.arg.model_args).cuda(output_device)
        loss = nn.CrossEntropyLoss()
        self.loss = loss.cuda(output_device)
        self.print_log(f'Model total number of params: {count_params(self.model)}')
        # Load model weights
        if self.arg.weights:
            # Read global step
            try:
                self.global_step = int(self.arg.weights[:-3].split('-')[-1])
            except:
                print('Cannot parse global_step from model weights filename')
                self.global_step = 0
            # Read weights
            self.print_log(f'Loading weights from {self.arg.weights}')
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)
            # Format weights
            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])
            # Delete not desired weights
            for w in self.arg.ignore_weights:
                if weights.pop(w, None) is not None:
                    self.print_log(f'Sucessfully Remove Weights: {w}')
                else:
                    self.print_log(f'Can Not Remove Weights: {w}')
            # Load weights to model
            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                self.print_log('Can not find these weights:')
                for d in diff:
                    self.print_log('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

        # Select graphic device
        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.print_log(f'{len(self.arg.device)} GPUs available, using DataParallel')
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device
                )

    def load_param_groups(self):
        """
        Template function for setting different learning behaviour
        (e.g. LR, weight decay) of different groups of parameters
        """
        self.param_groups = defaultdict(list)

        for name, params in self.model.named_parameters():
            self.param_groups['other'].append(params)

        self.optim_param_groups = {
            'other': {'params': self.param_groups['other']}
        }

    def load_optimizer(self):
        '''Load optimizer. Can be Adam or SGD.'''
        # Load parameters
        params = list(self.optim_param_groups.values())
        # Instantiate optimizer
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                params,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                params,
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError('Unsupported optimizer: {}'.format(self.arg.optimizer))

        # Load optimizer states if any
        if self.arg.checkpoint is not None:
            self.print_log(f'Loading optimizer states from: {self.arg.checkpoint}')
            self.optimizer.load_state_dict(torch.load(self.arg.checkpoint)['optimizer_states'])
            current_lr = self.optimizer.param_groups[0]['lr']
            self.print_log(f'Starting LR: {current_lr}')
            self.print_log(f'Starting WD1: {self.optimizer.param_groups[0]["weight_decay"]}')
            if len(self.optimizer.param_groups) >= 2:
                self.print_log(f'Starting WD2: {self.optimizer.param_groups[1]["weight_decay"]}')

    def load_lr_scheduler(self):
        '''Load multi step learning rate scheduler.'''
        # Instantiate scheduler
        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=self.arg.step, gamma=0.1)
        # Load checkpoint
        if self.arg.checkpoint is not None:
            scheduler_states = torch.load(self.arg.checkpoint)['lr_scheduler_states']
            self.print_log(f'Loading LR scheduler states from: {self.arg.checkpoint}')
            self.lr_scheduler.load_state_dict(scheduler_states)
            self.print_log(f'Starting last epoch: {scheduler_states["last_epoch"]}')
            self.print_log(f'Loaded milestones: {scheduler_states["last_epoch"]}')

    def load_data(self, dataset, train_ids, test_ids):
        '''Instantiate training and validation dataloaders.'''
        # Create a dictionary with one entry per data split
        self.data_loader = dict()
        # Create function to give workers different seeds
        def worker_seed_fn(worker_id):
            return init_seed(self.arg.seed + worker_id + 1)
        # Create training dataloader
        if self.arg.phase == 'train':
            sampler = torch.utils.data.SubsetRandomSampler(train_ids)
            # Instantiate dataloader
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=dataset,
                sampler=sampler,
                batch_size=self.arg.batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=worker_seed_fn,
                collate_fn=my_collate
            )
            self.print_log(f"Train dataset:\n {dataset.get_dist(train_ids)} ")
        # Select sampler with or without batch balancing
        sampler = torch.utils.data.SubsetRandomSampler(test_ids)
        # Instantiate dataloader
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=worker_seed_fn,
            collate_fn=my_collate
            )
        self.print_log(f"Test dataset:\n {dataset.get_dist(test_ids)} ")

    def save_arg(self):
        '''
        Save the config file in the log folder.'''
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open(os.path.join(self.arg.work_dir, 'config.yaml'), 'w') as f:
            yaml.dump(arg_dict, f)

    def print_time(self):
        '''Print elapsed time formatted.'''
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log(f'Local current time: {localtime}')

    def print_log(self, s, print_time=True):
        '''Format and print log messages.'''
        # Add timestamp
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            s = f'[ {localtime} ] {s}'
        # Print to log file
        if self.arg.print_log:
            with open(os.path.join(self.arg.work_dir, 'log.txt'), 'a') as f:
                print(s, file=f)

    def record_time(self):
        '''Record current time.'''
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        '''Obtain elapsed time since last call to this function.'''
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def save_states(self, epoch, states, out_folder, out_name):
        '''Save training states to log dir.'''
        out_folder_path = os.path.join(self.arg.work_dir, out_folder)
        out_path = os.path.join(out_folder_path, out_name)
        os.makedirs(out_folder_path, exist_ok=True)
        torch.save(states, out_path)

    def save_checkpoint(self, epoch, out_folder='checkpoints'):
        '''
        Save checkpoint to log dir.'''
        state_dict = {
            'epoch': epoch,
            'optimizer_states': self.optimizer.state_dict(),
            'lr_scheduler_states': self.lr_scheduler.state_dict(),
        }

        checkpoint_name = f'checkpoint-{epoch}-fwbz{self.arg.forward_batch_size}-{int(self.global_step)}.pt'
        self.save_states(epoch, state_dict, out_folder, checkpoint_name)

    def save_weights(self, epoch, out_folder='weights'):
        '''
        Save model weights to log dir.'''
        state_dict = self.model.state_dict()
        weights = OrderedDict([
            [k.split('module.')[-1], v.cpu()]
            for k, v in state_dict.items()
        ])

        weights_name = f'weights-{epoch}-{self.fold}.pt'
        self.save_states(epoch, weights, out_folder, weights_name)

    def train(self, epoch, save_model=False):
        '''Train the model for one epoch.'''
        # Change to train mode
        self.model.train()
        # Select train dataloader
        loader = self.data_loader['train']
        # Initialize timer
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        # Create container for loss values
        loss_values = []
        # Load current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        # Iterate over batches
        self.print_log(f'Training epoch: {epoch + 1}, LR: {current_lr:.4f}')
        process = tqdm(loader, dynamic_ncols=True, leave=False, desc='Batches')
        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            # Load batch data to GPU
            with torch.no_grad():
                try:
                    data = torch.stack(data).float().cuda(self.output_device)
                except RuntimeError:
                    data = [D.cuda(self.output_device) for D in data]
                label = label.long().cuda(self.output_device)
            # Save elapsed time
            timer['dataloader'] += self.split_time()
            # Reset optimizer gradients
            self.optimizer.zero_grad()
            ############## Gradient Accumulation for Smaller Batches ##############
            # Split batch into smaller batches yo optmize memory usage
            real_batch_size = self.arg.forward_batch_size
            splits = len(data) // real_batch_size
            assert len(data) % real_batch_size == 0, \
                'Real batch size should be a factor of arg.batch_size!'
            if isinstance(data, list):
                assert real_batch_size == 1, \
                    'Real batch size should be one for segments with variable length'
            for i in range(splits):
                left = i * real_batch_size
                right = left + real_batch_size
                batch_data, batch_label = data[left:right], label[left:right]
                if isinstance(batch_data, list):
                    assert len(batch_data) == 1, 'Error in forward batch size'
                    batch_data = batch_data[0].unsqueeze(0)
                # Forward pass
                output = self.model(batch_data)
                if isinstance(output, tuple):
                    output, l1 = output
                    l1 = l1.mean()
                else:
                    l1 = 0
                # Compute loss
                loss = self.loss(output, batch_label) / splits
                # Backward pass
                loss.backward()
                # Save loss values
                loss_values.append(loss.item())
                # Save elapsed time
                timer['model'] += self.split_time()

            #####################################
            # Advance optimizer step
            self.optimizer.step()

            # Save elapsed time
            timer['statistics'] += self.split_time()

            # Delete output/loss after each batch since it may introduce extra mem during scoping
            # https://discuss.pytorch.org/t/gpu-memory-consumption-increases-while-training/2770/3
            del output
            del loss


        # Computes statistics of time consumption
        proportion = {
            k: f'{int(round(v * 100 / sum(timer.values()))):02d}%'
            for k, v in timer.items()
        }
        # Print time consumption and mean loss
        mean_loss = np.mean(loss_values)
        num_splits = self.arg.batch_size // self.arg.forward_batch_size
        self.print_log(f'\tMean training loss: {mean_loss:.4f} (BS {self.arg.batch_size}: {mean_loss * num_splits:.4f}).')
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        # PyTorch > 1.2.0: update LR scheduler here with `.step()`
        # and make sure to save the `lr_scheduler.state_dict()` as part of checkpoint
        self.lr_scheduler.step()

        if save_model:
            # save training checkpoint & weights
            self.save_weights(epoch + 1)
            self.save_checkpoint(epoch + 1)

    def eval(self, epoch, save_score=False, loader_name=['test'],
             wrong_file=None, result_file=None):
        '''Evaluate the model during training phase.'''
        # Skip evaluation if too early
        if epoch + 1 < self.arg.eval_start:
            return
        # Open results files if needed
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        # Create containers for classification values
        y_true, y_pred = [], []
        # Avoid computing gradients
        with torch.no_grad():
            # Load model to device
            self.model = self.model.cuda(self.output_device)
            # Change to evalation mode
            self.model.eval()
            self.print_log(f'Eval epoch: {epoch + 1}')
            # Iterate over evaluation splits
            for ln in loader_name:
                # Initialize containers
                loss_values = []
                score_batches = []
                indexes = []
                step = 0
                # Iterate over batches
                process = tqdm(self.data_loader[ln], dynamic_ncols=True, leave=False, desc='Eval')
                for batch_idx, (data, label, index) in enumerate(process):
                    # Save samples indexes
                    indexes.extend(index)
                    # Load data to device
                    try:
                        data = torch.stack(data).float().cuda(self.output_device)
                    except RuntimeError:
                        assert len(data) == 1, \
                            'Test batch size should be one for segments with variable length'
                        data = data[0].unsqueeze(0).cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    # Forward pass
                    output = self.model(data)
                    if isinstance(output, tuple):
                        output, l1 = output
                        l1 = l1.mean()
                    else:
                        l1 = 0
                    # Compute loss
                    loss = self.loss(output, label)
                    # Save scores and loss values
                    score_batches.append(output.data.cpu().numpy())
                    loss_values.append(loss.item())
                    # Compute predictions
                    _, predict_label = torch.max(output.data, 1)
                    # Advance step
                    step += 1
                    # Update results files if needed
                    if wrong_file is not None or result_file is not None:
                        predict = list(predict_label.cpu().numpy())
                        true = list(label.data.cpu().numpy())
                        for i, x in enumerate(predict):
                            if result_file is not None:
                                f_r.write(str(x) + ',' + str(true[i]) + '\n')
                            if x != true[i] and wrong_file is not None:
                                f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')
                    # Save groundtruth and predictions
                    y_true.extend(list(label.data.cpu().numpy()))
                    y_pred.extend(list(predict_label.cpu().numpy()))
                # Format scores
                score = np.concatenate(score_batches)
                # Compute mean loss
                loss = np.mean(loss_values)
                # Compute accuracy
                accuracy = self.data_loader[ln].dataset.top_k(score, 1, indexes)
                # Save scores and labels of last epoch of each fold
                if epoch + 1 == self.arg.num_epoch:
                    if ln == 'test':
                        self.val_logits[self.fold] = score
                        labels = self.data_loader[ln].dataset.label[indexes]
                        self.val_targets[self.fold] = labels
                    elif ln == 'train':
                        self.train_acc[self.fold] = accuracy
                # Save resutls in tensorboard writer
                if self.arg.phase == 'train':
                    self.writer[ln].add_scalar(f'loss_{self.fold}', loss, epoch)
                    self.writer[ln].add_scalar(f'acc_{self.fold}', accuracy, epoch)
                    kappa = cohen_kappa_score(y_true, y_pred)
                    self.writer[ln].add_scalar('cohen_kappa', kappa, epoch)
                    self.writer[ln].add_figure(
                        f"Confusion matrix {self.fold}", 
                        create_confusion_matrix(y_true, y_pred,
                                                self.data_loader[ln].dataset),
                        epoch)
                    self.writer[ln].add_scalar('epoch', epoch + 1, self.global_step)
                    self.writer[ln].add_scalar('lr', self.lr, self.global_step)

                # Print some logs
                self.print_log(f'\tMean {ln} loss of {len(self.data_loader[ln])} batches: {np.mean(loss_values)}.')
                for k in self.arg.show_topk:
                    self.print_log(f'\tTop {k}: {100 * self.data_loader[ln].dataset.top_k(score, k, indexes):.2f}%')
            # Save scores 
            if save_score:
                score_dict = defaultdict(list)
                for key, value in zip(self.data_loader[ln].dataset.sample_name, score):
                    score_dict[key].append(value)
                with open('{}/epoch{}_{}_score.pkl'.format(self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

        # Empty cache after evaluation
        torch.cuda.empty_cache()
    
    def start(self):
        '''Start the training and evaluation process.'''
        # Train phase
        if self.arg.phase == 'train':
            # self.print_log(f'Parameters:\n{pprint.pformat(vars(self.arg))}\n')
            self.print_log(f'Model total number of params: {count_params(self.model)}')
            # Update global step
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            # Repeat training and evaluation for N epochs
            for epoch in tqdm(range(self.arg.start_epoch, self.arg.num_epoch), dynamic_ncols=True, leave=False, desc='Epochs'):
                save_model = ((epoch + 1) % self.arg.save_interval == 0) or (epoch + 1 == self.arg.num_epoch)
                self.train(epoch, save_model=save_model)
                self.eval(epoch, save_score=self.arg.save_score, loader_name=['test', 'train'])
            # Print some info
            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Forward Batch Size: {self.arg.forward_batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            # Reset variables
            self.global_step = 0
            self.lr = self.arg.base_lr
        # Test phase
        elif self.arg.phase == 'test':
            wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')

            self.print_log(f'Model:   {self.arg.model}')
            self.print_log(f'Weights: {self.arg.weights}')

            self.eval(
                epoch=0,
                save_score=self.arg.save_score,
                loader_name=['test'],
                wrong_file=wf,
                result_file=rf
            )

            self.print_log('Done.\n')

    def reset(self):
        '''Reset model hyper-parameters.'''
        # Load model
        self.load_model()
        self.load_param_groups()
        self.load_optimizer()
        self.load_lr_scheduler()
        # Initialize training
        self.global_step = 0
        self.lr = self.arg.base_lr
        self.load_optimizer()
        self.load_lr_scheduler()

    def reset_metrics(self):
        '''Reset metrics containers.'''
        self.train_acc = {}
        self.val_targets = {}
        self.val_logits = {}


def str2bool(v):
    '''Parse common expressions to boolean values'''
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def print_results(filename, processor, dataset, repeat, metrics):
    '''
    Compute and print some metrics and statistics for each repetition.
        filename - path of file to save results
        processor - processor instance to read stored results
        dataset - dataset instance to read groundtruth labels
        repeat - current repetition number
        metrics - dict to store repetition results
        '''
    with open(filename, 'a') as f:
        # Print number of repetition
        f.write(f'Repeat {repeat + 1}\n\n')
        logits, targets, accs = [], [], []
        # Build scores and grountruths matrixes
        for fold, values in processor.val_logits.items():
            logits.append(values)
            targets.append(processor.val_targets[fold])
            accs.append(processor.train_acc[fold])
        logits = np.concatenate(logits)
        targets = np.concatenate(targets)
        # Mean training accuracy and deviation
        f.write("Training accuracy: ")
        train_acc_mean = np.mean(accs)
        train_acc_dev = np.std(accs)
        f.write(f"{train_acc_mean:.4f} +/- {train_acc_dev:.4f}\n")
        metrics['train_acc'].append(train_acc_mean)
        # Mean validation accuracy and deviation
        pred = logits.argmax(axis=-1)
        val_acc = (pred == targets).mean()
        f.write(f'Accuracy: {val_acc:.4f}\n')
        metrics['test_acc'].append(val_acc)
        # Mean per class validation accuracies
        pcacc = perclass_accuracy(targets, pred, dataset)
        for k, v in pcacc.items():
            f.write(f'Accuracy ({k}): {v:.4f}\n')
        # F-1 score
        f1 = f1_score(targets, pred, average='weighted')
        f.write(f'Weighted f1: {f1*100:.4f}\n')
        metrics['f1'].append(f1)
        # Cohens Kappa
        kappa = cohen_kappa_score(targets, pred)
        f.write(f'Cohens kappa: {kappa:.4f}\n')
        metrics['kappa'].append(val_acc)
        # Matthews Corelation Coefficient
        mcc = matthews_corrcoef(targets, pred)
        metrics['mcc'].append(mcc)
        f.write(f'Matthews correlation coefficient: {mcc:.4f}\n')
        # Loss
        loss = log_loss(targets, logits)
        f.write(f'Log loss: {loss:.4f}\n\n\n')
        metrics['loss'].append(loss)

    # Print confusion matrix
    fig = create_confusion_matrix(targets, pred, dataset)
    if not os.path.isdir(filename.replace('results.txt', "confusion_matrix/")):
        os.mkdir(filename.replace('results.txt', "confusion_matrix/"))
    fig.savefig(filename.replace('results.txt',
                                 f"confusion_matrix/repeat_{repeat + 1}.png"))


def print_summary(metrics, filename, model_name):
    '''
    Compute and print estimation of metrics over repetitions.
        metrics - dict of per-repetition results
        filename - path of file to save results
        model_name - name of the model
        '''
    write_header = not os.path.exists(filename)
    with open(filename, 'a') as f:
        if write_header:
            f.write('Model name, ')
            for key in metrics.keys():
                f.write(key)
                f.write(', ')
            f.write('\n')
        f.write(model_name)
        f.write(', ')
        for key, values in metrics.items():
            f.write(f'{np.mean(values):.4f} +/- {np.std(values):.4f}, ')
        f.write('\n')


def main():
    '''Main function'''
    # Create parser
    parser = get_parser()

    # Load parameters from configuration file
    p = parser.parse_args(['--config', './config/train_msg3d.yaml'])
    if p.config is not None:
        with open(p.config, 'r') as f:
            user_args = yaml.safe_load(f)
        defined_args = vars(p).keys()
        for k in user_args.keys():
            if k not in defined_args:
                print('WRONG ARG:', k)
                assert (k in defined_args)
        parser.set_defaults(**user_args)
    arg = parser.parse_args()
    # Initialize seeds
    init_seed(arg.seed)
    # Initialize datafeeder
    Feeder = import_class(arg.feeder)
    dataset=Feeder(**arg.train_feeder_args)
    # Initialize processor
    processor = Processor(arg)
    # Create metrics container
    metrics = defaultdict(list)
    # Repeat N times LOSO-CV
    for repeat in tqdm(range(arg.num_repeats), dynamic_ncols=True, leave=False, desc='Repeats'):
        processor.print_log(f"\n\nRepetition: {repeat}")
        # Initialize LOSO-CV
        kfold = LeaveOneGroupOut()
        # Iterate over fols
        for _, (train_ids, test_ids) in tqdm(
                enumerate(kfold.split(dataset.data, dataset.label, dataset.person)),
                dynamic_ncols=True, leave=False, desc='LOSO',
                total=kfold.get_n_splits(groups=dataset.person)):
            # Check persons in the test fold
            test_person = set([dataset.person[id] for id in test_ids])
            if len(test_person) > 1:
                print(f"[ERROR]: More than one person in test set\n{test_person}")
                break
            else:
                test_person = test_person.pop()
            if len(test_ids) < 10:
                processor.print_log(f"Person {test_person} with {len(test_ids)} samples skipped (too few samples)")
                continue
            if test_person.startswith('p0') and test_person not in LSEGFE_PERSONS:
                processor.print_log(f"Person {test_person} with {len(test_ids)} samples skipped (not in subset)")
                continue
            processor.print_log(f"\nLOSO fold: {test_person}")
            # Reset hyper-parameters
            processor.reset()
            processor.fold = test_person
            # Load train and evaluation data
            processor.load_data(dataset, train_ids, test_ids)
            # Start training
            processor.start()
        # Save repetition results
        print_results(os.path.join(arg.work_dir, 'results.txt'),
                                processor, dataset, repeat, metrics)
        # Reset processor containers
        processor.reset_metrics()
    # Print summary of results
    sum_dir = os.path.abspath(os.path.join(arg.work_dir, os.pardir))
    summary_path = os.path.join(sum_dir, 'summary.csv')
    model_name = os.path.basename(os.path.normpath(arg.work_dir)).replace('_', '.')
    print_summary(metrics, summary_path, model_name)
    print_summary(metrics, os.path.join(arg.work_dir, 'results.txt'), model_name)



if __name__ == '__main__':
    # Set of LSEGFE persons for LOSO-CV
    LSEGFE_PERSONS = ['p0003', 'p0006', 'p0036', 'p0039', 'p0037', 'p0041',
                     'p0028', 'p0013', 'p0025', 'p0026', 'p0004']
    main()