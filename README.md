# GrammaticalFacialExpressions
Classification of Grammatical Facial Expressions through Neural Networks based in facial landmarks and action units.

PyTorch implementation of "Facial motion analysis beyond emotional expressions". [[arXiv](TODO)]
<!--- <img src="imgs/cropped-fixed-standup.gif" width="24%"><img src="imgs/cropped-fixed-clap.gif" width="24%"> --->


## Dependencies

- Python >= 3.6
- PyTorch >= 1.2.0
- CUDA Toolkit
- PyYAML, tqdm, tensorboardX, opencv
- matplotlib, numpy, pandas, pickle, scipy 
- pympi, re, argparse, scikit learn, torchvision (0.4)

## Data Preparation

### Download Datasets

This model was tested over 2 datasets_

- Boğaziçi University Head Motion Analysis Project Database (BUHMAP)
- LSE_GFE_UVIGO

#### BUHMAP

Buhmap dataset can be directly downloaded here: https://www.cmpe.boun.edu.tr/pilab/pilabfiles/databases/buhmap/files/videos.zip.
Moreover, the features extracted with OpenFace (2D and 3D facial landmarks as well as Action UnitS) are shared in folder data/BUHMAP_Features. The corresponding annotation data is available in data/BUHMAP_Features/BUHMAP_info.pkl.


#### LSE_GFE_UVIGO

Full UVIGO dataset can't be shared due to data protection matters. Regardless, the extracted features (2D and 3D facial landamarks as well as Action Units) are shared in folder data/LSE_GFE_Features. The annotation data is available in data/LSE_GFE_Features/LSEGFE_info.pkl.

### Data Preprocessing

#### Directory Structure

BUHMAP videos should be stored in the same folder keeping their original names in order to obtain their features using our scripts.
On the other hand, LSE_GFE videos should be structured as follows:
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

1. To obtain the features (landmarks or action units) from the videos (only for BUHMAP case) execute: 
  'extract_features_DATASET.py --[desired_parameters (check code)]'

2. To process the extracted features to a suitable format for the model data feeder execute 
  'process_features_DATASET.py --[desired_parameters (check code)]'
  indicating the base folder of the extracted features files.


## Pretrained Models

To enable fast replication of the model, a set of trained weigths for each dataset are shared in the folder trained_model. Such weights correspond to msg3d models with 1 single scale for both spatial and 3D convolutions and were trained using an empty graph and data augmentation by horizontal flipping.


## Training

- The general training template command:
`python3 train_loso_repeat.py --config <config file>`

- Use the default config file from `./config` to train/test different datasets or create/modify new configurations.

- Example: `python3 train_loso_repeat.py --config ./config/train_msg3d.yaml`


## Acknowledgements

This repo is based on [MS-G3D](https://github.com/kenziyuliu/MS-G3D).

BUHMAP dataset was created by [Boğaziçi University](https://www.cmpe.boun.edu.tr/pilab/pilabfiles/databases/buhmap/)

Thanks to the original authors for their work!


## Citation

Please cite this work if you find it useful:
https://www.mdpi.com/1424-8220/22/10/3839

@Article{s22103839,
AUTHOR = {Porta-Lorenzo, Manuel and Vázquez-Enríquez, Manuel and Pérez-Pérez, Ania and Alba-Castro, José Luis and Docío-Fernández, Laura},
TITLE = {Facial Motion Analysis beyond Emotional Expressions},
JOURNAL = {Sensors},
VOLUME = {22},
YEAR = {2022},
NUMBER = {10},
ARTICLE-NUMBER = {3839},
URL = {https://www.mdpi.com/1424-8220/22/10/3839},
ISSN = {1424-8220},
DOI = {10.3390/s22103839}
}

## Contact
Please email `mporta@gts.uvigo.es` for further questions
