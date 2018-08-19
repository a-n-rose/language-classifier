import keras
from keras.models import model_from_json
import pandas as pd
import numpy as np
import sqlite3
from sqlite3 import Error
import time
from ann_visualizer.visualize import ann_viz



#load json model:
classifier_weights = 'weigths_name.h5'
model = 'model_name.json'
json_file = open(model,'r')
classifier_json = json_file.read()
json_file.close()
classifier = model_from_json(classifier_json)
#load weights:
classifier.load_weights(classifier_weights)
print("Loaded model from disk")

#try visualizing it
ann_viz(classifier,view=True, filename='VISUSALIZE_MFCC_ANN_ENG_GERM',title='English German Classifier')
