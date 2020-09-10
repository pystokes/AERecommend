#!/usr/local/bin/python
import flask
from flask import Flask, request

from ml_engine.execute import Executor
from utils.file_io import FileIO as io
from utils.common import Utils as utils

app = Flask(__name__)

@app.route('/', methods=['GET'])
def check_from_brawser():
    """
    This method is for just checking from brawser
    """
    return 'Application server return response correctly.'

@app.route('/api/agritech/store_image', methods=['POST'])
def store_image():

    try:
        img = utils.get_image(request)

        save_path = request.form.get('save_path')
        io.make_dir(save_path)
        io.save_image(img, save_path)

        return flask.jsonify({'status': 'ok',
                              'status_code': 200,
                              'message': 'Success to save image.'})
    except:
        return flask.jsonify({'status': 'error',
                              'status_code': 500,
                              'message': 'Failed to save image.'})

@app.route('/api/agritech/preprocess', methods=['POST'])
def preprocess():

    res = utils.check_content_type_json(request)
    return res

@app.route('/api/agritech/train', methods=['POST'])
def train():

    res = utils.check_content_type_json(request)
    return res

@app.route('/api/agritech/evaluate', methods=['POST'])
def evaluate():

    res = utils.check_content_type_json(request)
    return res

@app.route('/api/agritech/predict', methods=['POST'])
def predict():

    res = utils.check_content_type_json(request)

    json_data = request.get_json()

    config = utils.load_config(json_data['config_path'])
    executor = Executor(json_data['exec_type'], config, json_data['y_dir'])

    model, device = executor.load_model(json_data['gpu_id'])
    print(model)
    print(device)
    
    """
    ADD PROCESS TO DETECT HERE
    """

    return res


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
