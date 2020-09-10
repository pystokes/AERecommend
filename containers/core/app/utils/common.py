#!/usr/local/bin/python

import json
from pathlib import Path

from attrdict import AttrDict
import cv2
import flask
import numpy as np

from config import Config

class Utils(object):

    def __init__(self):
        pass

    @classmethod
    def check_content_type_json(self, request):

        if request.headers['Content-Type'] != 'application/json':
            print(request.headers['Content-Type'])
            return flask.jsonify({'status': 'error',
                                  'status_code': 400,
                                  'message': 'Content-Type must be application/json.'})
        
        print(request.json)
        return flask.jsonify({'status': 'ok',
                              'status_code': 200,
                              'message': 'Content-Type has passed.'})

    @classmethod
    def load_config(self, config_path):

        if Path(config_path).exists:
            with open(config_path, 'r') as f:
                config = json.load(f)
            config = AttrDict(config)
        else:
            config = Config().build_config()
        
        return config

    @classmethod
    def get_image(self, request):

        stream = request.files['image'].stream
        img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, 1)

        return img
