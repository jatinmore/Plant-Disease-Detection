from unittest import result
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import keras

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from .sk import Flask
from . import app
from flask import render_template ,redirect, url_for, request

from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

model =tf.keras.models.load_model('/best_model.h5',compile=False)
print('Model loaded')

# ref = dict(zip(list(train.class_indices.values()),list(val.class_indices.keys())))

def prediction(path,model):
    img = load_img(path, target_size = (256,256))
    i = img_to_array(img)
    im = preprocess_input(i)
    print(im.shape)
    img = np.expand_dims(im, axis=0)
    pred = np.argmax(model.predict(img))
    # print(f"the image belongs to {ref[pred]}")
    s="the image belongs to{ref[pred]}"
    return s


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        f_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(f_path)

        # Make prediction
        preds = prediction(f_path, model)
        print(preds[0])
    # return result

@app.route('/success',methods=['POST'])   
def success():
    if request.method == 'POST':
        f=request.files['file']

        pred = model.predict(img)




    # if __name__ == '__main__':

    # # app.run(port=5002, debug=True)

    # # Serve the app with gevent
    #     http_server = WSGIServer(('', 5000), app)
    #     http_server.serve_forever()
    #     app.run()