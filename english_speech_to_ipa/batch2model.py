import os

import numpy as np
import pandas as pd
from sqlite3 import Error

from sql_data import Connect_db
from batch_prep import Batch_Data

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
        
        dataset_matrices = [x_y_train,x_y_val,x_y_test]
        for dataset in dataset_matrices:
            print("Shape of input 2d: {}".format(dataset[0].shape))
            #print(dataset[0])
            print("Shape of output 2d: {}".format(dataset[1].shape))
            #print(dataset[1])
            
        #prep data for LSTM
        #make 3d
        for dataset in dataset_matrices:
            input3d = batch_prep.make2d_3d(dataset[0])
            output3d = batch_prep.make2d_3d(dataset[1])
            print("Shape of input 3d: {}".format(input3d.shape))
            print("Shape of output 3d: {}".format(output3d.shape))
    
        prog_end = time.time()
        logging.info("Program ended: {}".format(prog_end))
        elapsed_time_hours = (time.time()-prog_start)/3600
        timepassed_message = 'Elapsed time in hours: {}'.format(elapsed_time_hours)
        logging.info(timepassed_message)

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

