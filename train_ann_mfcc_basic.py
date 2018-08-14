'''
train ANN with mfccs
basic version: using only train and test datasets (not yet validation)
'''

import numpy as np
import pandas as pd
import sqlite3

import logging
import logging.handlers
logger = logging.getLogger(__name__)
from pympler import tracker

#import dataset
database = 'sp_mfcc.db'
table = 'mfcc_40'
batchsize = 10
epochs = 10
modelname = 'mfcc_matched_eng_germ'#just an example


if __name__ == '__main__':
    try:
        tr_tot = tracker.SummaryTracker()
        
        #default format: severity:logger name:message
        #documentation: https://docs.python.org/3.6/library/logging.html#logrecord-attributes 
        log_formatterstr='%(levelname)s , %(asctime)s, "%(message)s", %(name)s , %(threadName)s'
        log_formatter = logging.Formatter(log_formatterstr)
        logging.root.setLevel(logging.DEBUG)
        #logging.basicConfig(format=log_formatterstr,
        #                    filename='/tmp/tradinglog.csv',
        #                    level=logging.INFO)
        #for logging infos:
        file_handler_info = logging.handlers.RotatingFileHandler('trainANN_loginfo.csv',
                                                                  mode='a',
                                                                  maxBytes=1.0 * 1e6,
                                                                  backupCount=200)
        #file_handler_debug = logging.FileHandler('/tmp/tradinglogdbugger.csv', mode='w')
        file_handler_info.setFormatter(log_formatter)
        file_handler_info.setLevel(logging.INFO)
        logging.root.addHandler(file_handler_info)
        
        
        #https://docs.python.org/3/library/logging.handlers.html
        #for logging errors:
        file_handler_error = logging.handlers.RotatingFileHandler('trainANN_logerror.csv', mode='a',
                                                                  maxBytes=1.0 * 1e6,
                                                                  backupCount=200)
        file_handler_error.setFormatter(log_formatter)
        file_handler_error.setLevel(logging.ERROR)
        logging.root.addHandler(file_handler_error)
        
        #for logging infos:
        file_handler_debug = logging.handlers.RotatingFileHandler('trainANN_logdbugger.csv',
                                                                  mode='a',
                                                                  maxBytes=2.0 * 1e6,
                                                                  backupCount=200)
        #file_handler_debug = logging.FileHandler('/tmp/tradinglogdbugger.csv', mode='w')
        file_handler_debug.setFormatter(log_formatter)
        file_handler_debug.setLevel(logging.DEBUG)
        logging.root.addHandler(file_handler_debug)

        
        
        
        #initialize database
        conn = sqlite3.connect(database)
        c = conn.cursor()
        
        check_variables = input("\nIMPORTANT!!!!\nHave you checked the global variables at the top of this script? Especially how the model should be saved? It would be a pity to save over a previous model... (Y or N): ")
        if 'y' in check_variables.lower():

            #import train (1), validate(2) and test (3) data sets 
            c.execute("SELECT * FROM {} WHERE dataset='{}'".format(table,1))
            dataset_train = c.fetchall()
            df_train = pd.DataFrame(dataset_train)
            
            #c.execute("SELECT * FROM {} WHERE dataset='{}'".format(table,2))
            #dataset_validate = c.fetchall()
            #df_validate = pd.DataFrame(dataset_validate)
            
            c.execute("SELECT * FROM {} WHERE dataset='{}'".format(table,3))
            dataset_test = c.fetchall()
            df_test = pd.DataFrame(dataset_test)
            
            #should apply limits/ensure a healthy balance
            X_train = df_train.iloc[:,:40].values
            y_train = df_train.iloc[:,-1].values
            
            #X_val = df_validate.iloc[:,:40].values
            #y_val = df_validate.iloc[:,-1].values
            
            X_test = df_test.iloc[:,:40].values
            y_test = df_test.iloc[:,-1].values


            from sklearn.preprocessing import LabelEncoder, OneHotEncoder
            labelencoder_y = LabelEncoder()
            y_train = labelencoder_y.fit_transform(y_train)
            #y_val =labelencoder_y.fit_transform(y_val)
            y_test = labelencoder_y.fit_transform(y_test)

            
            #feature scaling
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            #X_val = sc.transform(X_val)

            #ANN
            import keras
            from keras.models import Sequential
            from keras.layers import Dense
            
            classifier = Sequential()
            classifier.add(Dense(activation = 'relu',units=20,input_dim=X_train.shape[1],kernel_initializer='uniform'))
            
            classifier.add(Dense(activation='relu',units=20,kernel_initializer='uniform'))
            
            classifier.add(Dense(activation='sigmoid',units=1,kernel_initializer='uniform'))
            
            classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
            
            #numbers: batchsize and epochs
            classifier.fit(X_train,y_train,batchsize,epochs)
            
            y_pred = classifier.predict(X_test)
            y_pred = (y_pred > 0.5)
            
            from sklearn.metrics import confusion_matrix
            y_test = y_test.astype(bool)
            cm = confusion_matrix(y_test,y_pred)
            score = classifier.evaluate(X_train,y_train,verbose=0)
            acc = "%s: %.2f%%" % (classifier.metrics_names[1], score[1]*100)
            print(acc)
            
            model_json = classifier.to_json()
            with open(modelname+'.json','w') as json_file:
                json_file.write(model_json)
            classifier.save_weights(modelname+'.h5')
            print('Done!')
        else:
            print_message = "\nRun the script after you check the global variables."
            print(print_message)
            logging.info(print_message)
    except sqlite3.Error as e:
        logging.exception("Database error: %s" % e)
    except Exception as e:
        logging.exception("Error occurred: %s" % e)
    finally:
        if conn:
            conn.close()

