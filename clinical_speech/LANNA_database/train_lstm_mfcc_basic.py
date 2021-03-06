'''
train ANN, LSTM with mfccs
basic version: few layers and using only train and test datasets (not yet validation)
'''

import numpy as np
import pandas as pd
import sqlite3
import time
import os

from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten
from sklearn.metrics import confusion_matrix

import logging.handlers
from my_logger import start_logging, get_date
from pympler import tracker

#for logging:
script_purpose = 'trainLSTM_LANNA' #will name logfile 
current_filename = os.path.basename(__file__)
session_name = get_date() #make sure this session has a unique identifier - link to model name and logging information



def reshape_data_3d(dependent_variables_list,df,start_col, end_col,num_sequences,num_mfcc):
    '''
    Expects the last column of dataframe to contain label data
    and that the labels correspond to their indices in the 'dependent variables' list
    '''
    X_reshape = False
    for label in range(len(dependent_variables_list)):
        label_col = df.columns[-1]
        #separate values based on label
        df_sep = df[df[label_col]==label]
        #put dataframe values into matrix
        X_temp = df_sep.iloc[:,start_col:end_col].values
        y_temp = df_sep.iloc[:,-1].values
        num_samp = X_temp.shape[0]
        #make sure number of samples can be evenly divided by number of sequences:
        #necessary to reshape data
        if num_samp % num_sequences != 0:
            num_samp -= (num_samp % num_sequences)
            X_temp = X_temp[:num_samp,:]
            y_temp = y_temp[:num_samp]
        #new number of samples after consideration of sequences
        new_samp = int(num_samp/num_sequences)
        #reshape data
        #(samples, number_seqeunces, number_features)
        X_3d = X_temp.reshape((new_samp,num_sequences,num_mfcc))
        y_3d = y_temp[:new_samp]
        
        if X_reshape is False:
            X = X_3d
            y = y_3d
            X_reshape = True
        else:
            X = np.concatenate((X,X_3d),axis=0)
            y = np.concatenate((y,y_3d),axis=0)
    return X, y
            


#import dataset
database = 'MFCC_extraction_LANNA_VAD.db'
database_name = os.path.splitext(database)[0]
table = 'mfcc_40'
num_mfcc = 40
batchsize = 100
epochs = 50
#number of layers in NN, including input and output layers:
tot_layers = 3
tot_numrows = 2000000
percentage_train = 0.8 #maintaining 80% train and 20% test
percentage_test = 0.2
dependent_variables = ['Healthy','Patients']  #options: ['English','German','Russian']    ['English','German']
if len(dependent_variables) == 2:
    classification = 'binary'
elif len(dependent_variables) >2:
    classification = 'multiple'
var_names = ', '.join(dependent_variables)
var_names_underscore = '_'.join(dependent_variables)
noise_level = 0 #options: 0   0.25    0.5    0.75    1   1.25   None
noise_type='none'#should be value in column 'noisegroup' options: 'matched', 'none', 'random' 
noisegroup = noise_type
type_nn = 'LSTM' #options: 'ANN' 'LSTM'
num_sequences = 20 #if ANN: None; if LSTM 20 (for example)
if noise_level == None:
    noise_level_id = '_ALL_'
else:
    noise_level_id = '_{}_'.format(noise_level)
modelname = '{}_DB_{}_TABLE_{}_numMFCC{}_batchsize{}_epochs{}_numrows{}_{}_numlayers{}_numsequences{}_normalizedWstdmean_noiselevel{}{}'.format(type_nn,database_name,table,num_mfcc,batchsize,epochs,tot_numrows,var_names_underscore,tot_layers,num_sequences,noise_level_id,noise_type)



if __name__ == '__main__':
    try:        
        #initialize database
        conn = sqlite3.connect(database)
        c = conn.cursor()
        
        
        print("\n\nLoading data from \nDATABASE: '{}'\nTABLE: '{}'\n".format(database,table))
        print("Dependent variables are: {}".format(var_names))
        print("Number of MFCCs used: {}".format(num_mfcc))
        print("Total number of rows pulled from the table: {}".format(tot_numrows))
        print("Sequence count fed to the network (relevant for LSTM): {}".format(num_sequences))
        print("Model will be saved as: {}".format(modelname))
        print("Total number of layers in the model (including input and output): {}".format(tot_layers))
        print("Batchsize: {}".format(batchsize))
        print("Epochs: {}".format(epochs))
        print("Type of model: {}".format(type_nn))
        print("Type of classification: {}".format(classification))
        
        check_variables = input("\nIMPORTANT!!!!\nAre the items listed above correct? (Y or N): ")
        if 'y' in check_variables.lower():

            start_logging(script_purpose)
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
                print("This variable is encoded as: {}".format(label_encoded))
                #import data sets 
                if noise_level != None:
                        
                    c.execute("SELECT * FROM {} WHERE dataset='{}' AND noisegroup='{}' AND noiselevel='{}' AND label='{}' LIMIT {}".format(table,1,noisegroup,noise_level,var,variable_train_rows))
                    data = c.fetchall()
                    train = pd.DataFrame(data)
                    col_names = train.columns
                    print("Label was '{}'".format(train[col_names[-1]][0]))
                    train[col_names[-1]] = label_encoded
                    print("Now label is '{}'".format(train[col_names[-1]][0]))
                    df_train = df_train.append(train,ignore_index=True)
                    
                    
                    c.execute("SELECT * FROM {} WHERE dataset='{}' AND noisegroup='{}' AND noiselevel='{}' AND label='{}' LIMIT {}".format(table,3,noisegroup,noise_level,var,variable_test_rows))
                    data = c.fetchall()
                    test = pd.DataFrame(data)
                    col_names = test.columns
                    test[col_names[-1]] = label_encoded
                    df_test = df_test.append(test,ignore_index=True)
                else:
                    c.execute("SELECT * FROM {} WHERE dataset='{}' AND noisegroup='{}' AND label='{}' LIMIT {}".format(table,1,noisegroup,var,variable_train_rows))
                    data = c.fetchall()
                    train = pd.DataFrame(data)
                    col_names = train.columns
                    print("Label was '{}'".format(train[col_names[-1]][0]))
                    train[col_names[-1]] = label_encoded
                    print("Now label is '{}'".format(train[col_names[-1]][0]))
                    df_train = df_train.append(train,ignore_index=True)
                    
                    
                    c.execute("SELECT * FROM {} WHERE dataset='{}' AND noisegroup='{}' AND label='{}' LIMIT {}".format(table,3,noisegroup,var,variable_test_rows))
                    data = c.fetchall()
                    test = pd.DataFrame(data)
                    col_names = test.columns
                    test[col_names[-1]] = label_encoded
                    df_test = df_test.append(test,ignore_index=True)
            
                
            conn.close()
            print("Data has been extracted and database has been closed.")
            print("Converting dataframe to matrix")
            
            
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


            #Reshape data for LSTM
            if type_nn == 'LSTM':
                X_train, y_train = reshape_data_3d(dependent_variables,df_train,a,b,num_sequences,num_mfcc)
                X_test, y_test = reshape_data_3d(dependent_variables,df_test,a,b,num_sequences,num_mfcc)
                        
            #if not LSTM, no reshaping of data is necessary
            else:
                X_train = df_train.iloc[:,a:b].values
                y_train = df_train.iloc[:,-1].values
                
                X_test = df_test.iloc[:,a:b].values
                y_test = df_test.iloc[:,-1].values
                
                
            print("Shape of {} X_train data = {}\nShape of y_train data = {}".format(type_nn,X_train.shape,y_train.shape))
            print("Shape of {} X_test data = {}\nShape of y_test data = {}".format(type_nn,X_test.shape,y_test.shape))
            
            #Normalize
            print("Normalizing data")
            mean = np.mean(X_train, axis=0)
            std = np.std(X_train,axis=0)
            X_train = (X_train-mean)/std
            X_test = (X_test-mean)/std
            
            #feature scaling
            print("Scaling data")
            #different scaling measures for 2d vs 3d data
            if len(X_train.shape) <= 2:
                sc = StandardScaler()
                X_train = sc.fit_transform(X_train)
                X_test = sc.transform(X_test)
            elif len(X_train.shape) == 3:
                scalers = {}
                for i in range(X_train.shape[2]):
                    scalers[i] = StandardScaler()
                    X_train[:,:,i] = scalers[i].fit_transform(X_train[:,:,i])
                for i in range(X_test.shape[2]):
                    X_test[:,:,i] = scalers[i].transform(X_test[:,:,i])
    
            #categorical data for multiple classes:
            if len(dependent_variables) > 2:
                y_train = keras.utils.to_categorical(y_train, num_classes=len(dependent_variables))
                y_test = keras.utils.to_categorical(y_test, num_classes=len(dependent_variables))  
                
            print("Shape of {} X_train data (post preprocessing) = {}\nShape of y_train data = {}".format(type_nn,X_train.shape,y_train.shape))
            print("Shape of {} X_test data (post preprocessing) = {}\nShape of y_test data = {}".format(type_nn,X_test.shape,y_test.shape))

        
            print("Building the model..")
            #set up model variables:

            #general variables for all models (binary vs multiple classifiers):
            num_labels = len(dependent_variables)
            if num_labels == 2:
                num_labels = 1
            if num_labels == 1:
                activation_output = 'sigmoid'
                loss = 'binary_crossentropy'
            else:
                activation_output = 'softmax'
                loss = 'categorical_crossentropy'
            
            metrics = ['accuracy']

            classifier = Sequential()
            classifier_name = 'sequential'

            
            units_output = num_labels
            
        
            #specific to simple ANN
            if type_nn == 'ANN':

                input_dim = X_train.shape[1]
                units_layers = int((input_dim+num_labels)/2) 
                kernel_initializer = 'uniform'
                activation_layers = 'relu'
                
                #optimization
                optimizer = 'adam'

                classifier.add(Dense(activation = activation_layers,units=units_layers,input_dim=input_dim,kernel_initializer=kernel_initializer))
                
                classifier.add(Dense(activation=activation_layers,units=units_layers,kernel_initializer=kernel_initializer))
                
                classifier.add(Dense(activation=activation_output,units=units_output,kernel_initializer=kernel_initializer))
                            
            
            #specific to LSTM
            elif type_nn == 'LSTM':
                input_dim = X_train.shape[2]
                optimizer = 'rmsprop'
                kernel_initializer = None
                units_layers = None
                activation_layers = 100  #num memory neurons
                classifier.add(LSTM(activation_layers,return_sequences = True, input_shape=(num_sequences,input_dim)))
                classifier.add(Flatten())
                classifier.add(Dense(num_labels,activation=activation_output))
                classifier.compile(loss=loss,optimizer=optimizer,metrics = metrics)
                
            print("Training the model..")
            classifier.compile(optimizer=optimizer,loss=loss,metrics=metrics)  
            #numbers: batchsize and epochs
            classifier.fit(X_train,y_train,batchsize,epochs)
            
            
            score = classifier.evaluate(X_test,y_test,verbose=0)
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
            info_message = "\n\nFinished Training Model: {}\n{}\nPurpose: to classify data as {} (encoded respectively as: {})\n\nData Used for Training: \nDatabase = '{}'\nTable = '{}'\nNumber of rows for training data (per dependent variable) = {}\nNumber of rows for test data (per dependent variable) = {}\nTotal number of rows used = {}   \n\nModel Specifications:\nClassifer = {}\nInput dimensions = {}\nKernel initializer = {}\nActivation (layers) = {}\nNumber of Sequences = {}\nActivation (output) = {}\nNumber of units (layers) = {}\nNumber of output units = {}\nOptimizer = {}\nLoss = {}\nMetrics = {}\nNumber of layers (including input and output layers) = {}\n\n{}\n ".format(modelname,acc,var_names,labels_encoded,database,table,variable_train_rows,variable_test_rows,tot_numrows, classifier_name,input_dim,kernel_initializer,activation_layers,num_sequences,activation_output,units_layers,units_output,optimizer,loss,metrics,tot_layers,timepassed_message)
            print(info_message)
            logging.info(info_message)
                
        
            #look at the confusion matrix of binary data
            if len(dependent_variables) == 2:
                y_pred = classifier.predict(X_test)
                y_pred = (y_pred > 0.5)
                y_test = y_test.astype(bool)
                cm = confusion_matrix(y_test,y_pred)
                cm_info = "Confusion Matrix:\n{}".format(cm)
                logging.info(cm_info)
                print(cm_info)
                t_lang1, f_lang1, f_lang2, t_lang2 = confusion_matrix(y_test,y_pred).ravel()
                print("True {}: {}".format(dependent_variables[0],t_lang1))
                print("False {}: {}".format(dependent_variables[0], f_lang1))
                print("True {}: {}".format(dependent_variables[1], t_lang2))
                print("False {}: {}".format(dependent_variables[1], f_lang2))
        
        
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
            print("Database has been closed")
