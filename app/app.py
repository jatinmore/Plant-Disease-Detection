from unittest import result
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import keras
import json
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from flask import Flask
from flask import render_template ,redirect, url_for, request

from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


app = Flask(__name__)

model=keras.models.load_model('../best_model.h5')
with open('../plant.json')as s:
    s_data=json.load(s)

def prediction(path,model):
    img = load_img(path, target_size = (256,256))
    i = img_to_array(img)
    im = preprocess_input(i)
    print(im.shape)
    img = np.expand_dims(im, axis=0)
    pred = np.argmax(model.predict(img))
    n_p=str(pred)
    # print(f"the image belongs to {ref[pred]}")
    return pred

@app.route('/')  
def upload():  
    return render_template("index.html")  

@app.route('/home') 
def home():
    return render_template('index.html');

@app.route('/success', methods = ['POST'])  
def success():  
       if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        f_path = os.path.join(
            basepath, 'static', secure_filename("untitled.jpg"))
        f.save(f_path)

        # Make prediction
        preds = prediction(f_path, model)
        # print(preds[0])
        n_p=str(preds)
      
        # return render_template('success.html',prediction=preds)
        return render_template('success.html',prediction=str(preds)+' '+s_data[n_p]['name'] +' '+s_data[n_p]['dec'] )


if __name__ == "__main__":
    app.run(debug=True)