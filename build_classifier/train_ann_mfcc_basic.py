'''
train ANN with mfccs
basic version: using only train and test datasets (not yet validation)
'''

import numpy as np
import pandas as pd
import sqlite3
import time
import os
from ann_visualizer.visualize import ann_viz

import logging
import logging.handlers
logger = logging.getLogger(__name__)
from pympler import tracker

#import dataset
database = 'sp_mfcc.db'
database_name = os.path.splitext(database)[0]
table = 'mfcc_40'
num_mfcc = 40
batchsize = 100
epochs = 50
#number of layers in NN, including input and output layers:
tot_layers = 3
tot_numrows = 100000
percentage_train = 0.8 #maintaining 80% train and 20% test
percentage_test = 0.2
dependent_variables = ['English','German']
var_names = ', '.join(dependent_variables)
var_names_underscore = '_'.join(dependent_variables)
noise_level = 0  #options: 0   0.25    0.5    0.75    1   1.25   None
noise_type='matched'
type_nn = 'ANN'
if noise_level == None:
    noise_level_id = 'ALL'
else:
    noise_level_id = noise_level
modelname = '{}_DB_{}_TABLE_{}_numMFCC{}_batchsize{}_epochs{}_numrows{}_{}_numlayers{}_normalizedWstdmean_noiselevel{}{}'.format(type_nn,database_name,table,num_mfcc,batchsize,epochs,tot_numrows,var_names_underscore,tot_layers,noise_level_id,noise_type)#might be overkill...



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
        
        
        print("\n\nLoading data from \nDATABASE: '{}'\nTABLE: '{}'\n".format(database,table))
        print("Dependent variables are: {}".format(var_names))
        print("Number of MFCCs used: {}".format(num_mfcc))
        print("Total number of rows pulled from the table: {}".format(tot_numrows))
        print("Model will be saved as: {}".format(modelname))
        print("Total number of layers in the model (including input and output): {}".format(tot_layers))
        print("Batchsize: {}".format(batchsize))
        print("Epochs: {}".format(epochs))
        print("Type of model: {}".format(type_nn))
        
        check_variables = input("\nIMPORTANT!!!!\nAre the items listed above correct? (Y or N): ")
        if 'y' in check_variables.lower():

            prog_start = time.time()
            logging.info(prog_start)
            
            #calculate number of rows from training and test data:
            if percentage_train + percentage_test == 1.0:
                num_train_rows = int(tot_numrows*percentage_train)
                num_test_rows = int(tot_numrows*percentage_test)
            else:
                num_train_rows = int(tot_numrows*0.8)
                num_test_rows = int(tot_numrows*0.2)
            
            
            variable_train_rows = int(num_train_rows/len(dependent_variables))
            variable_test_rows = int(num_test_rows/len(dependent_variables))
            
            print("Collecting data..")
            
            df_train = pd.DataFrame()
            df_test = pd.DataFrame()
        
            
            for i in range(len(dependent_variables)):
                var = dependent_variables[i]
                print("Current variable's data being collected = {}".format(var))
                label_encoded = i
                print("Label this variable is encoded as: {}".format(label_encoded))
                #import data sets 
                if noise_level:
                        
                    c.execute("SELECT * FROM {} WHERE dataset='{}' AND noiselevel='{}' AND label='{}' LIMIT {}".format(table,1,noise_level,var,variable_train_rows))
                    data = c.fetchall()
                    train = pd.DataFrame(data)
                    col_names = train.columns
                    print("Label was '{}'".format(train[col_names[-1]][0]))
                    train[col_names[-1]] = label_encoded
                    print("Now label is '{}'".format(train[col_names[-1]][0]))
                    df_train = df_train.append(train,ignore_index=True)
                    
                    
                    c.execute("SELECT * FROM {} WHERE dataset='{}' AND noiselevel='{}' AND label='{}' LIMIT {}".format(table,3,noise_level,var,variable_test_rows))
                    data = c.fetchall()
                    test = pd.DataFrame(data)
                    col_names = test.columns
                    test[col_names[-1]] = label_encoded
                    df_test = df_test.append(test,ignore_index=True)
                else:
                    c.execute("SELECT * FROM {} WHERE dataset='{}' AND label='{}' LIMIT {}".format(table,1,var,variable_train_rows))
                    data = c.fetchall()
                    train = pd.DataFrame(data)
                    col_names = train.columns
                    train[col_names[-1]] = label_encoded
                    df_train = df_train.append(train,ignore_index=True)
                    
                    
                    c.execute("SELECT * FROM {} WHERE dataset='{}' AND label='{}' LIMIT {}".format(table,3,var,variable_test_rows))
                    data = c.fetchall()
                    test = pd.DataFrame(data)
                    col_names = test.columns
                    test[col_names[-1]] = label_encoded
                    df_test = df_test.append(test,ignore_index=True)
            print("Converting dataframe as matrix")
            
            
            if num_mfcc == 40:
                a = 0
                b = 40
            elif num_mfcc == 39:
                a = 1
                b = 40
            elif num_mfcc == 13:
                a = 0
                b = 13
            elif num_mfcc == 12:
                a = 1
                b = 13
            else:
                print("No options for number of MFCCs = {}".format(num_mfcc))
                print("Please choose from 40,39,13,12")

                #should apply limits/ensure a healthy balance
            X_train = df_train.iloc[:,a:b].values
            y_train = df_train.iloc[:,-1].values
            

            X_test = df_test.iloc[:,a:b].values
            y_test = df_test.iloc[:,-1].values
                
            
            #Normalize
            print("Normalizing data")
            mean = np.mean(X_train, axis=0)
            std = np.std(X_train,axis=0)
            X_train = (X_train-mean)/std
            X_test = (X_test-mean)/std
            
            #feature scaling
            print("Scaling data")
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            

            #ANN
            import keras
            from keras.models import Sequential
            from keras.layers import Dense
            
            print("Building the model..")
            #set up model variables:
            classifier = Sequential()
            classifier_name = 'sequential'
            input_dim = X_train.shape[1]
            kernel_initializer = 'uniform'
            activation_layers = 'relu'
            activation_output = 'sigmoid'
            num_labels = len(np.unique(y_train))
            if num_labels == 2:
                num_labels = 1
            units_layers = int((input_dim+num_labels)/2) 
            units_output = num_labels
            
            #optimization
            optimizer = 'adam'
            loss = 'binary_crossentropy'
            metrics = ['accuracy']

            classifier.add(Dense(activation = activation_layers,units=units_layers,input_dim=input_dim,kernel_initializer=kernel_initializer))
            
            classifier.add(Dense(activation=activation_layers,units=units_layers,kernel_initializer=kernel_initializer))
            
            classifier.add(Dense(activation=activation_output,units=units_output,kernel_initializer=kernel_initializer))
            
            print("Training the model..")
            classifier.compile(optimizer=optimizer,loss=loss,metrics=metrics)
            
            #numbers: batchsize and epochs
            classifier.fit(X_train,y_train,batchsize,epochs)
            
            y_pred = classifier.predict(X_test)
            y_pred = (y_pred > 0.5)
            
            from sklearn.metrics import confusion_matrix
            y_test = y_test.astype(bool)
            cm = confusion_matrix(y_test,y_pred)
            cm_info = "Confusion Matrix:\n{}".format(cm)
            logging.info(cm_info)
            print(cm_info)
            
            t_eng, f_eng, f_germ, t_germ = confusion_matrix(y_test,y_pred).ravel()
            print("True English: {}".format(t_eng))
            print("False English: {}".format(f_eng))
            print("True German: {}".format(t_germ))
            print("False German: {}".format(f_germ))

            score = classifier.evaluate(X_train,y_train,verbose=0)
            acc = "%s: %.2f%%" % (classifier.metrics_names[1], score[1]*100)
            print("Model Accuracy:")
            print(acc)
            logging.info("Model Accuracy: {}".format(acc))
            
            print('Saving Model')
            model_json = classifier.to_json()
            with open(modelname+'.json','w') as json_file:
                json_file.write(model_json)
            classifier.save_weights(modelname+'.h5')
            print('Done!')
            elapsed_time_hours = (time.time()-prog_start)/3600
            timepassed_message = 'Elapsed time in hours: {}'.format(elapsed_time_hours)
            
            labels_encoded = [i for i in range(len(dependent_variables))]
            info_message = "\n\nFinished Training Model: {}\n{}\nPurpose: to classify data as {} (encoded respectively as: {})\n\nData Used for Training: \nDatabase = '{}'\nTable = '{}'\nNumber of rows for training data (per dependent variable) = {}\nNumber of rows for test data (per dependent variable) = {}\nTotal number of rows used = {}   \n\nModel Specifications:\nClassifer = {}\nInput dimensions = {}\nKernel initializer = {}\nActivation (layers) = {}\nActivation (output) = {}\nNumber of units (layers) = {}\nNumber of output units = {}\nOptimizer = {}\nLoss = {}\nMetrics = {}\nNumber of layers (including input and output layers) = {}\n\n{}\n ".format(modelname,acc,var_names,labels_encoded,database,table,variable_train_rows,variable_test_rows,tot_numrows, classifier_name,input_dim,kernel_initializer,activation_layers,activation_output,units_layers,units_output,optimizer,loss,metrics,tot_layers,timepassed_message)
            print(info_message)
            logging.info(info_message)
            
            
        else:
            print_message = "\nRun the script after you check the global variables."
            print(print_message)
            logging.info(print_message)
    except sqlite3.Error as e:
        print("Database error {}".format(e))
        logging.exception("Database error: %s" % e)
    except Exception as e:
        print("Error occurred: {}".format(e))
        logging.exception("Error occurred: %s" % e)
    finally:
        if conn:
            conn.close()

