import os

import numpy as np
import pandas as pd
from sqlite3 import Error

from sql_data import Connect_db
from batch_prep import Batch_Data

from Errors import Error, DatabaseLimitError, ValidateDataRequiresTestDataError, ShiftLargerThanWindowError, TrainDataMustBeSetError, EmptyDataSetError


#needed for logging 
import time
import logging.handlers
from my_logger import start_logging, get_date
#for logging:
script_purpose = 'speechwnoise_ipa_mfcc_combine_data' #will name logfile 
current_filename = os.path.basename(__file__)
session_name = get_date() #make sure this session has a unique identifier - link to model name and logging information


if __name__=="__main__":
    start_logging(script_purpose)
    prog_start = time.time()
    logging.info(prog_start)
    
    database_ipa = 'speech2IPA3.db'
    table_ipa = 'speech2ipa'
    ipa = Connect_db(database_ipa,table_ipa)

    database_mfcc = 'sp_mfcc_IPA3.db'
    table_mfcc = 'mfcc_40'
    mfcc = Connect_db(database_mfcc, table_mfcc)

    #to save the collected datasets:
    database_final = 'batchdata_mfcc_ipa_datasets5.db'
    table_final = 'english'
    final = Connect_db(database_final,table_final)

    logging.info("Database for IPA Annotations: {}".format(database_ipa))
    logging.info("Database for MFCC features: {}".format(database_mfcc))
    logging.info("Database where new dataset is saved: {}".format(database_final))
    try:
        data_ipa = ipa.sqldata2df(limit=1000000)
        logging.info("Loaded data from {}".format(database_ipa))
        data_mfcc = mfcc.sqldata2df(limit=1000000)
        logging.info("Loaded data from {}".format(database_mfcc))

        x_ipa = data_ipa.values
        x_mfcc = data_mfcc.values

        batch_prep = Batch_Data(x_ipa,x_mfcc)
        ipa_list, num_classes = batch_prep.all_ipa_present(ipa_window=3)
        logging.info("\n\nIPA characters existent in dataset: \n{}\n\n".format(ipa_list))
        logging.info("Number of total classes: {}".format(num_classes))

        #set up train,validate,test data
        #default settings result in data categorized so: 60% train, 20% validate, 20% train
        #also sets dict of key and value pairs: train = 1, val = 2, test = 3, for database
        batch_prep.train_val_test()
        #the ipa_train will control the data sets; the mfcc data will rely on the ipa data
        #Note: because each row of IPA data might be different lengths in MFCC data, 
        #the sets won't 100% correspond to their designated sizes. HOWEVER, it is more 
        #important (for now) to keep as much speaker between group mixing. That is most easily 
        #achieved with the IPA data
        ipa_datasets = batch_prep.get_datasets()
        #define perameters for the batches - will be defined to all batches made
        #in this class instance
        batch_prep.def_batch(batch_size=20,ipa_shift=3)
        #save the batches to sql database
        count = 0
        for batch_item in ipa_datasets:
            count+=1
            update = "Completing dataset {} out of {}".format(count,len(ipa_datasets))
            print(update)
            logging.info(update)
            key_value = "Dataset key:value pair = {}:{}".format(batch_item[1],batch_prep.get_dataset_value(batch_item[1]))
            print(key_value)
            logging.info(key_value)
            count2 = 0
            for row in batch_item[0]:
                count2 += 1
                update = "Completing row {} out of {}".format(count2,len(batch_item[0]))
                batch_row, total_row_batches = batch_prep.generate_batch(row,batch_item[1])
                logging.info("Completed batch of row {}".format(count2))
                logging.info("Now saving data to database.")
                final.databatch2sql(batch_row)
                percent_thru = "Data successfully saved. {}% Through dataset {} of {}".format(count2/len(batch_item[0])*100,count,len(ipa_datasets))
                print(percent_thru)
                logging.info(percent_thru)
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
    except Error as e:
        logging.error("Database error: {}".format(e))
    #Close database connections:
    finally:
        ipa.close_conn()
        mfcc.close_conn()
        final.close_conn()
        logging.info("databases {}, {}, {} successfully closed.".format(database_ipa,database_mfcc,database_mfcc))


'''
Question I have: would it make a difference if ipa stress markers were included? 
'''
