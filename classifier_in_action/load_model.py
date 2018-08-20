'''
uploads previously saved ANN model, specific for the English-German MFCC ANN model
runs the model with "new" data and prints accuracy
'''
import glob
import os
import keras
from keras.models import model_from_json
import pandas as pd
import numpy as np
import sqlite3
from sqlite3 import Error
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix

database = 'sp_mfcc_test.db'
table = 'mfcc_40'


def create_connection(db_file):
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
        
    return None

def create_cursor(conn):
    try:
        return conn.cursor()
    except Error as e:
        print(e)

def table2dataframe(c,table,lim = None):
    try:
        if lim:
            limit = str(lim)
            c.execute("SELECT * FROM {} LIMIT {}".format(table,lim))
            data = c.fetchall()
            df = pd.DataFrame(data)
            return df
        else:
            c.execute("SELECT * FROM {}".format(table))
            data = c.fetchall()
            df = pd.DataFrame(data)
            return df
    except Error as e:
        print(e)
    
    return None

start = time.time()

print("connecting to database")
conn = create_connection(database)
c = create_cursor(conn)

try:
    print("\n\nLoading data from \nDATABASE: '{}'\nTABLE: '{}'\n".format(database,table))

    check_variables = input("\nIMPORTANT!!!!\nAre the items listed above correct? (Y or N): ")
    if 'y' in check_variables.lower():
        
        #import new data
        print("collecting new data --> df")
        df_new = table2dataframe(c,table)

        X = df_new.iloc[:,:40].values
        y = df_new.iloc[:,-1].values

        #normalize 
        mean = np.mean(X,axis=0)
        std = np.std(X,axis=0)
        X = (X-mean)/std

        #make English German --> 0 and 1
        labelencoder_y = LabelEncoder()
        y = labelencoder_y.fit_transform(y)

        #feature scaling
        sc = StandardScaler()
        X = sc.fit_transform(X)


        #collect all models in models directory and test with new data:
        for model in glob.glob('./models/*.json'):
            model_name = os.path.splitext(model)[0]
            json_file = open(model_name+'.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights(model_name+".h5")
            print("Loaded model from disk")

            # evaluate loaded model on new data
            try:
                loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                score = loaded_model.evaluate(X, y, verbose=0)
                print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
            except Exception as e:
                print(e)
                
            y_pred = loaded_model.predict(X)
            y_pred = (y_pred > 0.5)
            y_test=y.astype(bool)
            cm = confusion_matrix(y, y_pred)
            print("Confusion Matrix: {}".format(model_name))
            print(cm)
    
    else:        
        print("Check the variables first, then run the script.")

except Exception as e:
    print(e)
    
finally:
    if conn:
        conn.close()

