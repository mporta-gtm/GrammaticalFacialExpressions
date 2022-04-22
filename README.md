# GrammaticalFacialExpressions
Classification of Grammatical Facial Expressions through Neural Networks based in facial landmarks and action units.

PyTorch implementation of "Facial motion analysis beyond emotional expressions". [[arXiv]()]
[//]: # <img src="imgs/cropped-fixed-standup.gif" width="24%"><img src="imgs/cropped-fixed-clap.gif" width="24%">


## Dependencies

- Python >= 3.6
- PyTorch >= 1.2.0
- CUDA Toolkit
- PyYAML, tqdm, tensorboardX, opencv
- matplotlib, numpy, pandas, pickle, scipy 
- pympi, re, argparse, scikit learn, torchvision (0.4)

## Data Preparation

### Download Datasets

There are 2 datasets to download:

- Boğaziçi University Head Motion Analysis Project Database (BUHMAP)
- LSE_GFE_UVIGO

#### BUHMAP

Download dataset here: https://www.cmpe.boun.edu.tr/pilab/pilabfiles/databases/buhmap/files/videos.zip


#### LSE_GFE_UVIGO

1. Download dataset from: TODO

### Data Preprocessing

#### Directory Structure

BUHMAP videos should be stored in the same folder keeping their original names.
On the other hand, LSE_GFE videos should be structured
```
- base_folder/
  - pxxxx/
    - wxxxx/
      - filename
        ....
      -filename
    - wxxxx/
      ...
    - wxxxx/
  - pxxxx/
    ...
  - pxxxx
  
```
Furthermore, annotations files for LSE_GFE should be placed on a single folder in ELAN format and the files names should match the names of the corresponding videos.

#### Generating Data

1. Execute: 'extract_features_DATASET.py --[desired_parameters (check code)]'

2. Execute 'process_features_DATASET.py --[desired_parameters (check code)]'


## Pretrained Models

TODO


## Training & Testing

- The general training template command:
`python3 train_loso_repeat.py --config <config file>`

- Use the default config file from `./config` to train/test different datasets or create/modify new configurations.

- Example: `python3 train_loso_repeat.py --config ./config/train_msg3d.yaml`


## Acknowledgements

This repo is based on [MS-G3D](https://github.com/kenziyuliu/MS-G3D)

Thanks to the original authors for their work!


## Citation

Please cite this work if you find it useful:

TODO

## Contact
Please email `mporta@gts.uvigo.es` for further questions
