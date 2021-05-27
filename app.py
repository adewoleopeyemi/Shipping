# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 04:07:24 2020

@author: adewole opeyemi
"""

from flask import Flask, request
import os
import numpy as np
import flasgger
from flasgger import Swagger
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from preprocessing.preprocess import preprocess_one_image, preprocess_one_video


app=Flask(__name__)
Swagger(app)
port = int(os.environ.get('PORT', 5000))


@app.route('/predict_video_file', methods=['POST'])
def predict_video_file():
    '''Let's predict if nude content
    This is using docstrings for specifications.
    ---
    parameters:
        - name: file
          in: formData
          type: file
          required: true
        
    responses:
        200:
            description: The output values
    '''
    model = load_model("trainedmodelsh5/deploytestnudity.h5")
    vid = request.files['file']
    vid.save('videofile.mp4')
    frames=preprocess_one_video('videofile.mp4')
    preds = []
    for frame in frames:
        prediction = model.predict(frame.reshape((1, 124, 124, 3)))
        if prediction > 0.5:
            a=1
            preds.append(a)
        else:
            preds.append(0)

    i = 1
    for pred in preds:
        if pred ==1:
            i+=1

    return "There are %i nude frames in this video, please take them down"%i


@app.route('/predict_image_file', methods=['POST'])
def predict_image_file():
    '''Let's predict if nude content
    This is using docstrings for specifications.
    ---
    parameters:
        - name: file
          in: formData
          type: file
          required: true

    responses:
        200:
            description: The output values
    '''
    model = load_model("trainedmodelsh5/deploytestnudity.h5")
    img = preprocess_one_image(request.files['file'])
    prediction = model.predict(img.reshape((1, 124, 124, 3)))
    if prediction >0.98:
        return 'The uploaded image contains nude contents and is not allowed with confidence level ', prediction
    else:
        return "The uploaded image doesn't contain any form of nudity you look good to go"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
