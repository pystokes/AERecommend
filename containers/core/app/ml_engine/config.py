#!/usr/bin/python3
# -*- coding: utf-8 -*-

from logging import getLogger
from pathlib import Path
from attrdict import AttrDict
import torch

logger = getLogger('ML_ENGINE')

class Config(object):

    def __init__(self):

        # Requirements : model
        _input_size = 224 # Must be larger than 2**5=32 because encoder is ResNet50
        # Requirements : preprocess
        _preprocess_input_dir = 'PATH/TO/VOC_FORMAT/DIRECTORY'
        _preprocess_save_dir = None
        # Requirements : train
        _train_input_dir = 'PATH/TO/VOC_FORMAT/DIRECTORY'
        _train_input_dir = '../../../../../_data_storage/barcode/barcodes_00010/train/normal/'
        _train_save_dir = None
        _train_split_ratio = 0.7 # Ratio of train data
        # Requirements : webapp
        _debug_mode = True

        self.model = {
            # General
            'input_size': _input_size,
            'leakyReLU_factor': 0.2,
            'rgb_means': (104.0, 117.0, 123.0),
        }

        self.preproces = {
            'input_dir': _preprocess_input_dir,
            'save_dir': _preprocess_save_dir,
        }

        self.train = {
            'input_dir': _train_input_dir,
            'save_dir': _train_save_dir,
            'resume_weight_path': '',
            'num_workers': 0,
            'train_split_ratio': _train_split_ratio,
            'batch_size': 2,
            'epoch': 50,
            'shuffle': True,
            'split_random_seed': 0,
            'weight_save_period': 5,
            'optimizer': {
                'type': 'adam',
                'lr': 0.001,
                'betas': (0.9, 0.999),
                'eps': 1e-08,
                'weight_decay': 1000,
                'amsgrad': False,
                'wait_decay_epoch': 0,
                'T_max': 10
            }
        }

        self.detect = {
            'trained_weight_path': '',
            'debug_mode': _debug_mode,
            'visualize': True,
            'save_results': True,
        }

        assert not (self.train['optimizer']['wait_decay_epoch'] % self.train['weight_save_period']), 'wait_decay_epoch must be multiples of weight_save_period.'


    def build_config(self):

        config = {
            'model': self.model,
            'preprocess': self.preproces,
            'train': self.train,
            'detect': self.detect,
        }

        logger.info(config)

        return AttrDict(config)


if __name__ == '__main__':

    from pprint import pprint

    config = Config().build_config()
    pprint(config)
