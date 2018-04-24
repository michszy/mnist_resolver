from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob

import json

import re
import numpy as np
from flask import Flask, json
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.wsgi import WSGIServer

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.models import model_from_json
from os import getcwd

app = Flask(__name__)


SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
json_url = os.path.join(SITE_ROOT, 'static', 'model.json')
json_data = json.load(open(json_url))

h5_url = os.path.join(SITE_ROOT, "static", "model.h5")

print(json_url)
print (h5_url)

with open(json_url, 'r') as f:
    model = model_from_json(f.read())

model.load_weights(h5_url)




print('Model loaded. Start serving')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(28,28))
    x = image.img_to_array(img_path)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode="caffe")
    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=["GET", "POST"])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
        pred_class = preds.argmax(axis=-1)
        pred_class = decode_predictions(preds, top=1)
        result = str(pred_class[0][0][1])
        return result
    return None


if __name__ == '__main__':
    app.run()
