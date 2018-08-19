import keras
from keras.models import model_from_json
import pandas as pd
import numpy as np
import sqlite3
from sqlite3 import Error
import time
from ann_visualizer.visualize import ann_viz



#load json model:
classifier_name = 'ANN_DB_sp_mfcc_TABLE_mfcc_40_numMFCC40_batchsize100_epochs50_numrows2000000_English_German_numlayers3_normalizedWstd_mean'
classifier_weights = classifier_name+'.h5'
model = classifier_name+'.json'
json_file = open(model,'r')
classifier_json = json_file.read()
json_file.close()
classifier = model_from_json(classifier_json)
#load weights:
classifier.load_weights(classifier_weights)
print("Loaded model from disk")

#try visualizing it
ann_viz(classifier,view=True, filename='Visualize_'+classifier_name,title='English German Classifier')
