""" Proccess videos of the BUHMAP dataset with OpenFace.
Save openface logs and video info (label, name, framerate and length) to disk.
"""
import os
import argparse
import pickle
import cv2
import subprocess
import re
from collections import defaultdict
from pympi import Elan
from tqdm import tqdm

# Look Up Table for BUHMAP labels
LABELS = {1: 'neutral', 2: 'head-lr', 3: 'head-up', 4: 'head-fwd',
            5: 'sadness', 6: 'head-ud', 7: 'happiness', 8: 'happy-ud'}

def get_parser():
    ''' Argument parser.'''
    parser = argparse.ArgumentParser()
    # Path to dataset folder
    # Expects to find a single folder with all the BUHMAP videos named as
    # in the original distribution.
    # Example: abuzer_1_1.avi
    parser.add_argument(
        '--video-folder',
        type=str,
        required=True,
        help='Path to videos root folder'
    )
    # Openface Feature Extractor
    # Example: ../OpenFace/build/bin/FeatureExtraction
    parser.add_argument(
        '--openface-path',
        type=str,
        required=True,
        help='Path to OpenFace feature extraction executable'
    )
    # Openface logs folder
    parser.add_argument(
        '--out-folder',
        type=str,
        required=True,
        help='Folder to store extracted features'
    )
    # Path to output folder for video info
    parser.add_argument(
        '--label-folder',
        type=str,
        required=True,
        help='Folder to store extracted labels'
    )

    return parser


def call_openface(video_path, of_path, out_path, log_file):
    ''' Call OpenFace feature extractor on a video file.
    :param video_path: Path to video file
    :param out_path: Path to output folder
    :param log_file: Path to log file'''
    # Define command line
    cm = [of_path, "-f", video_path, "-2Dfp", "-3Dfp",
           "-pose", "-aus", "-out_dir", out_path]
    # Execute command
    try:
        with open(log_file, 'a') as f:
            subprocess.run(cm, stdout=f, stderr=f, check=True)
    except subprocess.CalledProcessError as _:
        print(f"[Error] OpenFace error processing:\n{video_path}")
        return False
    else:
        return True


def save_info(info, label, filename, video_path):
    ''' Save video info to dictionary
    :param info: Dictionary to store info
    :param label: Video label
    :param filename: Name of video
    :param video_path: Path to video file'''
    # Save video label
    info['label'].append(label.strip())
    # Save video name
    info['name'].append(filename)
    # Extract and save video FPS
    vid = cv2.VideoCapture(video_path)
    info['framerate'].append(vid.get(cv2.CAP_PROP_FPS))
    # Save video length
    info['length'].append(vid.get(cv2.CAP_PROP_FRAME_COUNT ))
    return info


def main(args):
    ''' Iterate over videos and extract features and labels'''
    # Read dataset name
    dataset = os.path.basename(os.path.normpath(args.out_folder))
    # Create output files paths
    label_out_path = os.path.join(args.label_folder,
                                  f'{dataset}_info.pkl')
    log_file = os.path.join(
                args.out_folder, "OpenFaceExtractionLog.txt")
    with open(log_file, 'w') as f:
        pass
    # Create data container
    info = defaultdict(list)
    # Iterate over dataset videos
    for filename in tqdm(os.listdir(args.video_folder)):
        # Get file extension
        name, ext = os.path.splitext(filename)
        if ext != '.avi':
            continue
        # Read annotation data
        id, label = name.split('_')[:2]
        label = LABELS[int(label)]
        # Extract landmarks
        video_path = os.path.join(args.video_folder, filename)
        if not os.path.exists(video_path):
            video_path = None              
            print(f"[WARNING] File not found:\n{video_path}")
        if video_path:
            success = call_openface(video_path, args.openface_path, args.out_folder, log_file)
            # Save info
            if success:
                info = save_info(info, label, name, video_path)
    # Write info to disk
    with open(label_out_path, 'wb') as f:
        pickle.dump(info, f)


if __name__ == "__main__":
    # Parse arguments
    PARSER = get_parser()

    ARGS = PARSER.parse_args()
    # Create folders
    if not os.path.exists(ARGS.out_folder):
        os.makedirs(ARGS.out_folder)
    if not os.path.exists(ARGS.label_folder):
        os.makedirs(ARGS.label_folder)

    main(ARGS)
