""" Proccess videos of the LSE_GFE dataset with OpenFace.
Save openface logs and video info (events labels, event beginning and end,
name and framerate) to disk."""
import os
import argparse
import pickle
import cv2
import subprocess
import re
from collections import defaultdict
from pympi import Elan
from tqdm import tqdm


def get_parser():
    ''' Argument parser.'''
    parser = argparse.ArgumentParser()
    # Path to dataset folder
    # Expects to find an set of folders with the following structure:
    #   - video_folder
    #       - pxxxx
    #           - wxxxx
    #               - filename.xxx
    parser.add_argument(
        '--video-folder',
        type=str,
        required=True,
        help='Path to videos root folder'
    )
    # Path to annotations folder
    # Expects to find a single folder with all the annotations in format .eaf
    # The files names should match the names of the corresponding videos
    parser.add_argument(
        '--elan-folder',
        type=str,
        required=True,
        help='Path to elan files'
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


def read_labels(path):
    '''Read ELAN labels for a given eaf file.'''
    eaf = Elan.Eaf(path)
    labels = eaf.get_annotation_data_for_tier('CNM_tipo')

    return labels


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


def save_info(info, labels, filename, video_path):
    ''' Save video info to dictionary
    :param info: Dictionary to store info
    :param labels: Video labels
    :param filename: Name of video
    :param video_path: Path to video file'''
    info['label'].append([lab[-1].strip() for lab in labels])
    info['start'].append([lab[0] for lab in labels])
    info['end'].append([lab[1] for lab in labels])
    info['name'].append(filename)
    # Extract FPS
    vid = cv2.VideoCapture(video_path)
    info['framerate'].append(vid.get(cv2.CAP_PROP_FPS))
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
    # Iterate over dataset files
    for eafname in tqdm(os.listdir(args.elan_folder)):
        # Get video name
        filename, _ = os.path.splitext(eafname)
        # Get collaborator and sentence IDs
        person = re.search(r'p\d{4}', filename).group(0)
        word = re.search(r'w\d{4}', filename).group(0)
        # Extract labels
        labs = read_labels(os.path.join(args.elan_folder, eafname))
        if not labs:
            continue
        # Find video path
        video_path = os.path.join(args.video_folder, person, word,
                                  filename + '.mov')
        if not os.path.exists(video_path):
            video_path = os.path.join(args.video_folder, person, word,
                                      filename + '.mp4')
            if not os.path.exists(video_path):
                video_path = None              
                print(f"[WARNING] File not found:\n{video_path}")
        # Extract landmarks
        if video_path:
            success = call_openface(video_path, args.openface_path, args.out_folder, log_file)
            # Save info
            if success:
                info = save_info(info, labs, filename, video_path)
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
