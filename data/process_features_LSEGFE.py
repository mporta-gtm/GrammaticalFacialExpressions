""" Read LSE_GFE OpenFace extracted features files, sample them in windows
of selected length and store them in a format valid for our models."""
import os
import re
import pickle
import argparse
import pandas
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm
from scipy.signal import resample

# Loop Up Table of column names for different data types
COLS = {
    '2Dlands': ('x', 'y'),
    '3Dlands': ('X', 'Y', 'Z'),
    'AUs': ('AU',)
}


def get_parser():
    ''' Argument parser. '''
    parser = argparse.ArgumentParser()
    # Path to OpenFace extracted feature files
    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Path to OpenFace csv files'
    )
    # Path to files saved info
    parser.add_argument(
        '--info-path',
        required=True,
        type=str,
        help='Path to samples labels'
    )
    # Output folder for processed samples
    parser.add_argument(
        '--out-folder',
        type=str,
        required=True,
        help='Folder to store processed samples'
    )
    # Type of features to obtain
    parser.add_argument(
        '--data-type',
        type=str,
        default='2Dlands',
        choices=['2Dlands', '3Dlands', 'AUs'],
        help='Type of data to load (default: landmarks 2D)'
    )
    # Length of sampling window
    parser.add_argument(
        '--sample-length',
        type=int,
        default=2000,
        help='Window time (in milliseconds)'
    )
    # Size of output sample
    parser.add_argument(
        '--output-size',
        type=int,
        default=60,
        help='Output size (in number of frames)'
    )
    # Whether to split dataset in train and test
    parser.add_argument(
        '--split',
        action='store_true',
        help='To split in test and train or not'
    )

    return parser


def load_info(path):
    ''' Load samples info. '''
    # Load info file
    with open(path, 'rb') as f:
        info = pickle.load(f, encoding='latin1')
    # Check file fields
    assert len(info['name']) == len(info['label'])
    assert len(info['framerate']) == len(info['name'])
    assert len(info['start']) == len(info['name'])
    assert len(info['start']) == len(info['end'])

    return info


def read_openface(filepath, cols_names):
    ''' Read OpenFace csv file. 
    :param filepath: Path to csv file
    :param cols_names: Column names to read
    :return x: Samples as numpy array'''
    # Read csv file as pandas dataframe
    df = pandas.read_csv(filepath, index_col=False)
    # Get desired columns
    cols = [k for k in df.keys() if k.startswith(cols_names, 1)]
    # Get kength of sample
    T = len(df[cols].values)  # pylint: disable=E1136
    # Reshape sample to (channels, length, n_features)
    x = df[cols].values.reshape(T, len(cols_names), -1).transpose(1, 0, 2)  # pylint: disable=E1136  # noqa: E501
    return x


def train_test_split(p, data, out_path):
    ''' Split data in train and test.
    :param p: List of collaborator IDs for test split
    :param data: Data to split
    :param out_path: Base path to save split data '''
    # Create test and train containers
    train = {key: [] for key in data.keys()}
    test = {key: [] for key in data.keys()}
    # Iterate over samples
    for idx, name in enumerate(data['name']):
        # Select corresponding set in function of collaborator ID
        dic = test if any(pp in name for pp in p) else train
        # Add sample fields to corresponding set
        for key, value in data.items():
            dic[key].append(value[idx])
    # Save test data to disk
    path = out_path.replace('_data', '_test_data')
    with open(path, 'wb') as f:
        pickle.dump(test, f)
    # Save train data to disk
    path = out_path.replace('_data', '_train_data')
    with open(path, 'wb') as f:
        pickle.dump(train, f)


def delete_index(info, idx):
    ''' Delete and entry from info dict.
    :param info: Info dict
    :param idx: Index to delete
    :return info: Info dict without deleted entry '''
    for k in info.keys():
        info[k].pop(idx)
    return info


def sample_video(x, video, info, data, length, size):
    ''' Sample video.
    :param x: Features extracted from video
    :param video: Video index
    :param info: Info dict
    :param length: Length of sampling window
    :param data: Dict of processed samples
    :param size: Size of resampled segments
    :return data: Dict of processed samples '''
    # Check labelled event doesnt surpass video length
    if any([end * info['framerate'][video] * 1e-3 > x.shape[1] for end in info['end'][video]]):
        print(info['name'][video], file=DEBUG)
        return data
    # Discard not annotated videos
    if not info['label'][video]:
        return data
    # Use full video as sample
    if length == -1:
        # Add sample info to saved data
        data['sample'].append(x)
        data['label'].append(info['label'][video])
        data['name'].append(info['name'][video])
        return data
    # Use full annotated event as sample
    elif length == 0:
        window = False
        length = size / info['framerate'][video] * 1e3
    # Compute window size
    else:
        window = True
        w_size = round(length * info['framerate'][video] * 1e-3) 
    # Create None sample when video starts without any event
    if min(info['start'][video]) > length:
        # Resample segment
        sample = (resample(x[:, 0:w_size], size, axis=1) if window else
                  x[:, 0:size])
        # Add sample info to saved data
        data['sample'].append(sample)
        data['label'].append('None')
        data['name'].append(info['name'][video])
    # Iterate over annotations
    for index, label in enumerate(info['label'][video]):
        # Check label is valid
        if label in ['i.xeral', 'n.resto']:
            continue
        # Find first and last frame indices
        start = round(info['start'][video][index]
                      * info['framerate'][video] * 1e-3)
        end = round(info['end'][video][index]
                    * info['framerate'][video] * 1e-3)
        # Crop and resample event with defined window size
        if window:
            w_start = max(0, (start + end - w_size) // 2)
            w_end = min(w_start + w_size, x.shape[1])
            sample = resample(x[:, w_start:w_end], size, axis=1)
        # Use full event as sample
        else:
            sample = x[:, start:end]
        # Add sample info to saved data
        data['sample'].append(sample)
        data['label'].append(label)
        data['name'].append(info['name'][video])

    return data


def main(args):
    ''' Iterate over videos and sample labelled events '''
    # Read dataset base name
    dataset_name = os.path.basename(os.path.normpath(args.data_path))
    # Obtain names of CSV fields of interest
    cols_names = COLS[args.data_type]
    # Load videos info
    info = load_info(args.info_path)
    # Create samples container
    out = defaultdict(list)
    # Iterate over videos
    for ii, name in tqdm(enumerate(info['name']), total=len(info['name'])):
        # Exclude non-deaf/interpeters videos
        if 'p0030' in name or 'p0001' in name or 'p0029' in name:
            continue
        # Get video name
        file = os.path.join(args.data_path, name + '.csv')
        if os.path.isfile(file):
            # Read openface file of selected video
            x = read_openface(file, cols_names)
            # Sample labelled events in video
            out = sample_video(x, ii, info, out, args.sample_length, args.output_size)
        else:
            # Print error and delete video from info
            print(f"File not found {file}")
            idx = info['name'].index(name)
            info = delete_index(info, idx)
    # Save samples to disk
    out_path = os.path.join(args.out_folder, f'{dataset_name}_truncated_data.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)
    # Split samples in train and test
    if args.split:
        # Define collaborator IDs for test split
        p = ['p0004', 'p0015', 'p0029', 'p0039']
        train_test_split(p, out, out_path)


if __name__ == "__main__":
    # Parse arguments
    PARSER = get_parser()

    ARGS = PARSER.parse_args()
    # Create folders
    if not os.path.exists(ARGS.out_folder):
        os.makedirs(ARGS.out_folder)
    # Create debug file
    DEBUG = open(os.path.join(ARGS.out_folder, 'feature_processing.log'), 'w')
    print('ARGS:', *ARGS.__dict__.items(), sep='\n', file=DEBUG)
    DEBUG.flush()
    # Run main
    main(ARGS)
    DEBUG.close()
