# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:02:30 2019

@author: Yan
"""
# best ressources are https://scotch.io/bar-talk/processing-incoming-request-data-in-flask
# https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html?source=post_page
# https://towardsdatascience.com/deploying-a-keras-deep-learning-model-as-a-web-application-in-p-fc0f2354a7ff


from generate_data import word_to_array
from keras.models import load_model
from flask import Flask, request, render_template
import tensorflow as tf


app = Flask(__name__)

def load():
    global models
    models = {}
    models['CNN'] = load_model('modelCNN.hdf5')
    models['RNN'] = load_model('modelRNN.hdf5')
    models['FF'] = load_model('modelFF.hdf5')
    global graph
    graph = tf.get_default_graph()
    
    
@app.route('/', methods=['GET', 'POST'])

def predict():
    if request.method == 'POST':
        word = request.form.get('word')
        model_name = request.form.get('model')
        model = models[model_name]
        if model_name == 'FF':
            arr = word_to_array(word).reshape(1,26*12)
        else:
            arr = word_to_array(word).reshape(1,26,12)
        with graph.as_default():
            prediction = model.predict(arr)
    
        return render_template('results.html', model=model_name, fr=round(100*prediction[0, 0],2), 
                               en=round(100*prediction[0, 1],2), es=round(100*prediction[0, 2],2))
                
    return render_template('app.html')
             
             


if __name__ == '__main__':
    print('Loading model...')
    load()
    app.run(debug=False)
