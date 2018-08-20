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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

database = 'sp_mfcc_test.db'
table = 'mfcc_40'
dependent_variables = ['English','German']


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

def table2dataframe(c,table,var,lim = None):
    try:
        if lim:
            limit = str(lim)
            c.execute("SELECT * FROM {} WHERE label='{}' LIMIT {}".format(table,var,lim))
            data = c.fetchall()
            df = pd.DataFrame(data)
            return df
        else:
            c.execute("SELECT * FROM {} WHERE label='{}'".format(table,var))
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
        
        df_new = pd.DataFrame()
        for i in range(len(dependent_variables)):
            var = dependent_variables[i]
            label_encoded = i
            print(var)
            print(label_encoded)
            df_var = table2dataframe(c,table,var,500)
            df_varcols = df_var.columns
            df_var[df_varcols[-1]] = label_encoded
            df_new = df_new.append(df_var,ignore_index=True)

        #based on the number of MFCCs used in training Ive seen so far:
        if num_mfcc == 40 or num_mfcc == 20 or num_mfcc == 13:
            a = 0
            b = num_mfcc
        #these leave out the first coefficient (dealing w amplitude/volume)
        elif num_mfcc == 39 or num_mfcc==19 or num_mfcc == 12:
            a = 1
            b = num_mfcc+1
        else:
            print("No options for number of MFCCs = {}".format(num_mfcc))
            print("Please choose from 40,39,20,19,13, or 12")
            
        X = df_new.iloc[:,a:b].values
        y = df_new.iloc[:,-1].values

        #normalize 
        mean = np.mean(X,axis=0)
        std = np.std(X,axis=0)
        X = (X-mean)/std

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
