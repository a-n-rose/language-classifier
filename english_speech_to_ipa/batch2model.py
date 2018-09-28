'''
Useful links:
https://medium.com/@fromtheast/implement-fit-generator-in-keras-61aa2786ce98
'''

import os

import numpy as np
import pandas as pd
from sqlite3 import Error

from sql_data import Connect_db
from batch_prep import Batch_Data

from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Embedding, TimeDistributed, Activation, Dropout

from Errors import Error, DatabaseLimitError, ValidateDataRequiresTestDataError, ShiftLargerThanWindowError, TrainDataMustBeSetError, EmptyDataSetError, MFCCdataNotFoundError
from generator import KerasBatchGenerator

#needed for logging 
import time
import logging.handlers
from my_logger import start_logging, get_date
#for logging:
script_purpose = 'trainmodel_ipamfcc_english' #will name logfile 
current_filename = os.path.basename(__file__)
session_name = get_date() #make sure this session has a unique identifier - link to model name and logging information


if __name__=="__main__":
    start_logging(script_purpose)
    prog_start = time.time()
    logging.info(prog_start)
    
    database = 'speech_wnoise_ipa_mfcc.db'
    table_ipa = 'speech_as_ipa'
    table_mfcc = 'speech_as_mfcc'
    
    #IMPORTANT VARIABLES:
    ipa_window = 3
    window_shift = 1
    batch_size = 20
    num_mfcc = 40
    withnoise = False
    
    print("Window size: {}\nShift size: {}\nBatch size (num sequences): {}\nNumber of features: {}\nNoise added to features: {}".format(ipa_window,window_shift,batch_size,num_mfcc,withnoise))
    logging.info("\nWindow size: {}\nShift size: {}\nBatch size (num sequences): {}\nNumber of features: {}\nNoise added to features: {}".format(ipa_window,window_shift,batch_size,num_mfcc,withnoise))
    
    #table where combined datasets will be saved
    
    #VARIOUS TABLES TO CHOOSE FROM:
    #table w win=3,shift=3,ipacharsonly=false
    #table_final = 'english_40mfcc_ipawindow3_ipashift3_1label_datasets20batches_idclasses'
    
    ##table w win=3,shift=1,ipacharsonly=False
    #table_final = 'english_40mfcc_ipawindow3_ipashift1_1label_datasets20batches_idclasses_ipacharsonlyFalse'
    
    ##table w win=3 shift=1, ipacharsonly=True
    #table_final = 'english_40mfcc_ipawindow3_ipashift1_1label_datasets20batches_idclasses_ipacharsonly'
    
    #table w win=3 shift=3, ipacharsonly=True
    table_final = 'english_40mfcc_ipawindow3_ipashift1_1label_datasets20batches_idclasses_ipacharsonly'
    
    db = Connect_db(database,table_ipa,table_mfcc,table_final)

    db_msg = "Database where data is pulled from: {}".format(database)
    tb_msg = "Table where data is pulled from: {}".format(table_final)
    
    logging.info(db_msg)
    logging.info(tb_msg)
    print(db_msg)
    print(tb_msg)
    

    try: 
        #The following commands allow me to know how many classes of IPA combinations I have and such.
        data_ipa = db.sqldata2df(db.table_ipa,limit=1000000)
        logging.info("Loaded data from table {}".format(table_ipa))
        x_ipa = data_ipa.values
        bp = Batch_Data(data_ipa = x_ipa)
        #setup dataset dict (which number is associated w train/validate/test)
        bp.train_val_test()
        train_label = bp.get_dataset_value(bp.str_train)
        val_label = bp.get_dataset_value(bp.str_val)
        test_label = bp.get_dataset_value(bp.str_test)
        
        #get IPA values given ipa window and shift (how many classes I have)        

        ipa_list,num_classes_local,num_classes_total = bp.doc_ipa_present(ipa_window,window_shift,ipa_chars_only=True)
        #define batches... just to be sure

        bp.def_batch(batch_size)
        logging.info("Number of classes in dataset: {}".format(num_classes_local))
        logging.info("Number of total possible classes (various combinations of IPA characters in sets of 3): {}".format(num_classes_total))

        
        ############ get data: ############
        
        df_train = db.sqldata2df(table_final,column_value_list=[['dataset',train_label]])
        df_val = db.sqldata2df(table_final,column_value_list=[['dataset',val_label]])
        df_test = db.sqldata2df(table_final,column_value_list=[['dataset',test_label]])
        
        #set num features based off of num columns in dataframe:
        bp.get_num_features(df_train)

        #form df into matrices:
        x_y_train = bp.get_x_y(df_train)
        x_y_val = bp.get_x_y(df_val)
        x_y_test = bp.get_x_y(df_test)
        
        
        ############ prep data for LSTM ############
        
        #first normalize data    
        X_train = bp.normalize_data(x_y_train[0])
        X_val = bp.normalize_data(x_y_val[0])
        X_test = bp.normalize_data(x_y_test[0])
        
        #feature scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_val = sc.transform(X_val)
        X_test = sc.transform(X_test)

        #For LSTM, make 3d (numrows,batch_size(i.e. num of sequences bp.batch_size),num_features)
        X_train = bp.make2d_3d(x_y_train[0])
        y_train = bp.make2d_3d(x_y_train[1])
        X_val = bp.make2d_3d(x_y_val[0])
        y_val = bp.make2d_3d(x_y_val[1])
        X_test = bp.make2d_3d(x_y_test[0])
        y_test = bp.make2d_3d(x_y_test[1])
        

        ############ Build the model ############
        
        
        #Apply the data in batches (of 20)
        train_data_generator = KerasBatchGenerator(X_train,y_train,num_steps=bp.batch_size,batch_size_model=bp.batch_size,num_features=bp.num_features,num_output_labels=y_train.shape[2],skip_step=1)
        
        val_data_generator = KerasBatchGenerator(X_val,y_val,num_steps=bp.batch_size,batch_size_model=bp.batch_size,num_features=bp.num_features,num_output_labels=y_train.shape[2],skip_step=1)
        

        #hidden layer: 40 features * 2 (I saw something like this somewhere..)
        model = Sequential()

        model.add(LSTM(80,return_sequences=True,input_shape=(bp.batch_size,bp.num_features)))
        model.add(Dropout(0.2))
        
        model.add(LSTM(80,return_sequences=True))
        model.add(Dropout(0.2))
        
        model.add(TimeDistributed(Dense(bp.num_classes_total)))
        model.add(Activation('softmax'))
        
        # when not assigning one-hot-encoded values to y_train multi-class data, use 'sparse_categorical_crossentropy' (as long as labels are integers)
        model.compile(loss='sparse_categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
        
        epochs = 5
        ##for when enough data is available (I think validation data is not sufficient)
        #model.fit_generator(train_data_generator.generate(),len(X_train)//(train_data_generator.batch_size_model*train_data_generator.skip_step),epochs, validation_data=val_data_generator.generate(),validation_steps=len(X_val)//(val_data_generator.batch_size_model*val_data_generator.skip_step))
        model.fit_generator(train_data_generator.generate(),len(X_train)//(train_data_generator.batch_size_model*train_data_generator.skip_step),epochs)
        
        
        ############ Evaluate the model ############
        
        score = model.evaluate(X_test,y_test,verbose=0)
        acc = "%s: %.2f%%" % (model.metrics_names[1], score[1]*100)
        print("Model Accuracy:")
        print(acc)
        logging.info("Model Accuracy: {}".format(acc))
        
        
        ############ Save the model ############
        
        date = get_date()
        model_name = 'engspeech2ipa_{}'.format(date)
        logging.info('Saving Model ')
        model_json = model.to_json()
        with open(model_name+'.json','w') as json_file:
            json_file.write(model_json)
        model.save_weights(model_name+'.h5')
        sv_model_msg = "Model saved as as {}".format(model_name)
        logging.info(sv_model_msg)
        print(sv_model_msg)
        
        
        ############ Close up shop ############
        
        elapsed_time_hours = (time.time()-prog_start)/3600
        timepassed_message = 'Elapsed time in hours: {}'.format(elapsed_time_hours)
        logging.info(timepassed_message)
        print(timepassed_message)
        

    except DatabaseLimitError as e:
        logging.error("DatabaseLimitError: {}".format(e))
    except ValidateDataRequiresTestDataError as e:
        logging.error("ValidateDataRequiresTestDataError: {}".format(e))
    except ShiftLargerThanWindowError as e:
        logging.error("ShiftLargerThanWindowError: {}".format(e))
    except TrainDataMustBeSetError as e:
        logging.error("TrainDataMustBeSetError: {}".format(e))
    except EmptyDataSetError as e:
        logging.error("EmptyDataSetError: {}".format(e))
    except KeyError as e:
        logging.error("KeyError: {}".format(e))
    except MFCCdataNotFoundError as e:
        logging.error("MFCCdataNotFoundError: {}".format(e))
    except Error as e:
        logging.error("Database error: {}".format(e))
    except SystemExit:
        logging.error("SystemExit: Had to exit program early.")
    #Close database connections:
    finally:
        db.close_conn()
        logging.info("database {} successfully closed.".format(database))
