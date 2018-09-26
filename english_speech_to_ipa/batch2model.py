import os

import numpy as np
import pandas as pd
from sqlite3 import Error

from sql_data import Connect_db
from batch_prep import Batch_Data

from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten

from Errors import Error, DatabaseLimitError, ValidateDataRequiresTestDataError, ShiftLargerThanWindowError, TrainDataMustBeSetError, EmptyDataSetError, MFCCdataNotFoundError


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
    #table where combined datasets will be saved
    table_final = 'english_40mfcc_ipawindow3_ipashift3_datasets20batches'
    db = Connect_db(database,table_ipa,table_mfcc,table_final)

    logging.info("Database where data is pulled from: {}".format(database))
    logging.info("Table where data is pulled from: {}".format(table_final))

    try: 
        #The following commands allow me to know how many classes of IPA combinations I have and such.
        data_ipa = db.sqldata2df(db.table_ipa,limit=1000000)
        logging.info("Loaded data from table {}".format(table_ipa))
        x_ipa = data_ipa.values
        batch_prep = Batch_Data(data_ipa = x_ipa)
        #setup dataset dict (which number is associated w train/validate/test)
        batch_prep.train_val_test()
        train_label = batch_prep.get_dataset_value(batch_prep.str_train)
        val_label = batch_prep.get_dataset_value(batch_prep.str_val)
        test_label = batch_prep.get_dataset_value(batch_prep.str_test)
        
        #get IPA values (how many classes I have, ultimately)
        ipa_window = 3
        _,num_classes = batch_prep.all_ipa_present(ipa_window)
        #define batches... just to be sure
        batch_size = 20
        window_shift = 3
        batch_prep.def_batch(batch_size,window_shift)
        
        #get train data:
        df_train = db.sqldata2df(table_final,column_value_list=[['dataset',train_label]])
        df_val = db.sqldata2df(table_final,column_value_list=[['dataset',val_label]])
        df_test = db.sqldata2df(table_final,column_value_list=[['dataset',test_label]])
        
        #set num features based off of num columns in dataframe:
        batch_prep.get_num_features(df_train)

        #form df into matrices:
        x_y_train = batch_prep.get_x_y(df_train)
        x_y_val = batch_prep.get_x_y(df_val)
        x_y_test = batch_prep.get_x_y(df_test)
        
        #prep data for LSTM
        
        #first normalize data    
        X_train = batch_prep.normalize_data(x_y_train[0])
        y_train = batch_prep.normalize_data(x_y_train[1])
        X_val = batch_prep.normalize_data(x_y_val[0])
        y_val = batch_prep.normalize_data(x_y_val[1])
        X_test = batch_prep.normalize_data(x_y_test[0])
        y_test = batch_prep.normalize_data(x_y_test[1])
    
    
        #feature scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_val = sc.transform(X_val)
        X_test = sc.transform(X_test)

            
        X_train = batch_prep.make2d_3d(x_y_train[0])
        y_train = batch_prep.make2d_3d(x_y_train[1])
        X_val = batch_prep.make2d_3d(x_y_val[0])
        y_val = batch_prep.make2d_3d(x_y_val[1])
        X_test = batch_prep.make2d_3d(x_y_test[0])
        y_test = batch_prep.make2d_3d(x_y_test[1])


        #categorical data for multiple classes
        #num classes based on number of possible 3-letter combinations of all ipa characters
        y_train = keras.utils.to_categorical(y_train, num_classes=batch_prep.num_classes)
        y_val = keras.utils.to_categorical(y_val, num_classes=batch_prep.num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes=batch_prep.num_classes)
        
        input_dim = X_train.shape[2]
        
        #Build Model:
        model = Sequential()
        model.add(LSTM(100, return_sequences=True,input_shape=(batch_prep.batch_size,input_dim)))
        model.add(LSTM(100, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(batch_prep.num_classes,activation='softmax'))
        #model.add(TimeDistributed(Dense(batch_prep.num_classes)))
        #model.add(Activation('softmax'))
        
        model.compile(loss='categorical_crossentropy',optimizer='adam',metics=['categorical_accuracy'])
        
        #numbers: batchsize and epochs
        model.fit(X_train,y_train,10,50)
        
        
        score = model.evaluate(X_test,y_test,verbose=0)
        acc = "%s: %.2f%%" % (model.metrics_names[1], score[1]*100)
        print("Model Accuracy:")
        print(acc)
        logging.info("Model Accuracy: {}".format(acc))
        
        print('Saving Model')
        date = get_date()
        model_name = 'engspeech2ipa_{}'.format(date)
        model_json = model.to_json()
        with open(modelname+'.json','w') as json_file:
            json_file.write(model_json)
        model.save_weights(modelname+'.h5')
        print('Done!')
        elapsed_time_hours = (time.time()-prog_start)/3600
        timepassed_message = 'Elapsed time in hours: {}'.format(elapsed_time_hours)
        

    except DatabaseLimitError as e:
        logging.error("Error occurred: {}".format(e))
    except ValidateDataRequiresTestDataError as e:
        logging.error("Error occurred: {}".format(e))
    except ShiftLargerThanWindowError as e:
        logging.error("Error occurred: {}".format(e))
    except TrainDataMustBeSetError as e:
        logging.error("Error occurred: {}".format(e))
    except EmptyDataSetError as e:
        logging.error("Error occurred: {}".format(e))
    except KeyError as e:
        logging.error("Error occurred: {}".format(e))
    except MFCCdataNotFoundError as e:
        logging.error("Error occurred: {}".format(e))
    except Error as e:
        logging.error("Database error: {}".format(e))
    except SystemExit:
        logging.error("Had to exit program early.")
    #Close database connections:
    finally:
        db.close_conn()
        logging.info("database {} successfully closed.".format(database))

