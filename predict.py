# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 12:57:45 2019

@author: Yan
"""
from keras.models import load_model
import argparse
from generate_data import word_to_array


parser = argparse.ArgumentParser(description='Predict the language that a word is written in')

parser.add_argument('word', help='word to classify')
parser.add_argument('-m', '--model')

args = parser.parse_args()

word = args.word
flatten = False


if args.model == 'FF':
    model = load_model('modelFF.hdf5')
    flatten = True
    print('Using FF')
elif args.model == 'RNN':
    model = load_model('modelRNN.hdf5')
    print('Using RNN')
elif args.model == 'CNNsep':
    model = load_model('modelCNNsep.hdf5')
    print('Using CNNsep')
else:
    model = load_model('modelCNN.hdf5')
    print('Using CNN')


def predict_word(word, model, flatten=False):
    if flatten:
        arr = word_to_array(word).reshape(1,26*12)
    else:
        arr = word_to_array(word).reshape(1,26,12)
    prediction = model.predict(arr)
    print('Francais: ', round(100*prediction[0, 0],2), '%')
    print('English: ', round(100*prediction[0, 1],2), '%')
    print('Espanol: ', round(100*prediction[0, 2],2), '%')
    
    
predict_word(word, model, flatten)