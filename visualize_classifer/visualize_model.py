import keras
from keras.models import model_from_json
import pandas as pd
import numpy as np
from ann_visualizer.visualize import ann_viz
import glob
import os


#batch visualize:
for model in glob.glob('*.json'):
    
    #load json model:
    classifier_name = os.path.splitext(model)[0]
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
    ann_viz(classifier,view=False, filename='Visualize_'+classifier_name,title='English German Classifier')
    
print("Models have been visualized")
