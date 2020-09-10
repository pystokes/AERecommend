#!/usr/local/bin/python

from pathlib import Path

import cv2

class FileIO(object):

    def __init__(self):
        pass

    @classmethod
    def make_dir(self, dir_path):
        Path(dir_path).mkdir(exist_ok=True)

    @classmethod
    def save_image(self, img, save_dir):
        save_name = Path(save_dir).joinpath('source.jpg')
        cv2.imwrite(str(save_name), img)
        return save_name
