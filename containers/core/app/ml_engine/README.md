# SSD: Single Shot MultiBox Detector

## Overview

This repository is PyTorch implementation of [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325). Dataset shoud be prepared in Pascal VOC format.

## Requirement

- Python 3.6 or higher

## Install

Clone this repository.

```bash
git clone https://github.com/Nextremer/CVTeam_SSD.git
```

Install necessary libraries.

```bash
# This requirements.txt has unnecessary libraries
# since developed enviromnent was constructed based on Anaconda
cd SSD
pip install -r requirements.txt
```

Store backbone weights ([vgg16_reducedfc.pth](https://drive.google.com/open?id=1RF0aeewgQX86UrLqSlbcDYn-gIS6E9so)) in any directory and specify the path to `vgg16_reducedfc.pth` on `_basenet` in [config.py](https://github.com/Nextremer/CVTeam_SSD/blob/master/config.py).

## Usage

Only the following patterns to load trained weights are supported.

|Support|Train on|Detect on|
|:---:|:---:|:---:|
|:heavy_check_mark:|Single-GPU|Single-GPU|
|:heavy_check_mark:|Multi-GPU|Single-GPU|
|Not supported|Single-GPU|Multi-GPU|
|:heavy_check_mark:|Multi-GPU|Multi-GPU|

### Train

1. Modify `Requirements` in [config.py](https://github.com/Nextremer/CVTeam_SSD/blob/master/config.py) at first.

    ```bash
    # Requirements : model
    _basenet = 'PATH/TO/vgg16_reducedfc.pth'
    # Set tuple of classes
    _basenet = 'PATH/TO/vgg16_reducedfc.pth'
    _classes = ('class1',
                'class2',
                'class3')
    # List of colors corresponding to above classes
    _bbox_colors = [(255, 0, 0),
                    (0, 255, 0),
                    (0, 0, 255)]
    # Requirements : preprocess (Not inplemented)
    _preprocess_input_dir = 'PATH/TO/VOC_FORMAT/DIRECTORY'
    _preprocess_save_dir = None
    # Requirements : train
    _train_input_dir = 'PATH/TO/VOC_FORMAT/DIRECTORY'
    _train_save_dir = None # If None, use default directory
    ```

2. Set hyperparameters for train in [config.py](https://github.com/Nextremer/CVTeam_SSD/blob/master/config.py).

    ```bash
    self.train = {
        'input_dir': _train_input_dir, # Not need to modify
        'save_dir': _train_save_dir, # Not need to modify
        'loss_function': {
            'loss_type': 'SSD', # Not need to modify
            'jaccard_threshold': 0.5,
            'neg_pos': 3, # 'neg_pos':1 (Ratio of Hard Negative Mining)
        },
        'resume_weight_path': '',
        'num_workers': 0,
        'batch_size': 64,
        'epoch': 50,
        'shuffle': True,
        'split_random_seed': 0,
        'weight_save_period': 5,
        'optimizer': {
            'type': 'sgd',
            'lr': 1e-4,
            'wait_decay_epoch': 10,
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'gamma': 0.1,
            'T_max': 50
        }
    }
    ```

3. Run script in train mode.

    ```bash
    python execute.py train [-g GPU_ID]
    ```

    If train on multi-GPU, separate GPU IDs with commas.

    ```bash
    # Example: Use two GPUs (0 and 1)
    python execute.py train -g 0,1
    ```

### Detect

1. Set path to trained weights at the `trained_weight_path` in the `config.json` created in train phase.

    ```bash
    "detect": {
        "trained_weight_path": "", # <- Specify trained model path here
        "nms": {
            "n_top": 200,
            "threshold": 0.45
        },
        "conf_threshold": 0.5,
        "visualize": true,
        "save_results": true,
        "box_bgr_colors": [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255]
        ]
    }
    ```

2. Change other configurations above `detect`. Especialy `conf_threshold` affects the final results of detection.

3. Run script in detection mode.

    __For images__

    ```bash
    python execute.py detect -c /PATH/TO/config.json -x /INPUT/DIR [-y /OUTPUT/DIR]
    ```

    __For WebCam__

    ```bash
    # Not inplemented
    python execute.py webcam -c /PATH/TO/config.json
    ```

## Data

### Directory structure

Prepare data in __[PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)__ format as following:

```text
{DATA_HOME}
    ├─ Annotations
    │    ├─ {filename1}.xml
    │    ├─ {filename2}.xml
    │    ├─ {filename3}.xml
    │    └─ ...
    ├─ ImageSets
    │    └─ Main
    │        ├─ train.txt
    │        ├─ val.txt
    │        └─ test.txt
    └─ JPEGImages
         ├─ {filename1}.jpg
         ├─ {filename2}.jpg
         ├─ {filename3}.jpg
         └─ ...
```

### JPEGImages

All image files have the same name with XML files in `Annotations` directory.

### ImageSets

Each file contains filename __without__ extensions.

#### train.txt (Ex. n_train=80)

```text
{filename1}
{filename2}
{filename3}
...
{filename80}
```

#### val.txt (Ex. n_val=10)

```text
{filename81}
{filename82}
{filename83}
...
{filename90}
```

#### test.txt (Ex. n_test=10)

```text
{filename91}
{filename92}
{filename93}
...
{filename100}
```

### Annotations

XML file example of `{filename1}`.xml

```xml
<annotation>
    <folder>JPEGImages</folder>
    <filename>{filename1}.jpg</filename>
    <path>/PATH/TO/{DATA_HOME}/JPEGImages/{filename1}.jpg</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>1224</width>
        <height>370</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>person</name>
        <pose>Unspecified</pose>
        <difficult>0</difficult>
        <bndbox>
            <xmin>712</xmin>
            <ymin>143</ymin>
            <xmax>811</xmax>
            <ymax>308</ymax>
        </bndbox>
    </object>
    <object>
        <name>car</name>
        <pose>Unspecified</pose>
        <difficult>0</difficult>
        <bndbox>
            <xmin>101</xmin>
            <ymin>150</ymin>
            <xmax>494</xmax>
            <ymax>234</ymax>
        </bndbox>
    </object>
</annotation>
```

## Author

- __GitHub:__ [Toshiyuki KITA](https://github.com/t-kita)
- __Email:__ toshiyuki.kita@nextremer.com
