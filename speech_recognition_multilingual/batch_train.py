import os

import numpy as np
import pandas as pd
from sqlite3 import Error

from sql_data import Connect_db
from batch_prep import Batch_Data

from Errors import Error, DatabaseLimitError, ValidateDataRequiresTestDataError, ShiftLargerThanWindowError, TrainDataMustBeSetError, EmptyDataSetError

#data_ipa = sqldata2df()
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

try:
    data_ipa = ipa.sqldata2df(limit=1000000)
    data_mfcc = mfcc.sqldata2df(limit=1000000)

    x_ipa = data_ipa.values
    x_mfcc = data_mfcc.values

    batch_prep = Batch_Data(x_ipa,x_mfcc)
    ipa_list, num_classes = batch_prep.all_ipa_present(ipa_window=3)
    print("\n\nIPA characters existent in dataset: \n{}\n\n".format(ipa_list))
    print("Number of total classes: {}".format(num_classes))

    #set up train,validate,test data
    #default settings result in data categorized so: 60% train, 20% validate, 20% train
    #also sets dict of key and value pairs: train = 1, val = 2, test = 3, for database
    batch_prep.train_val_test()
    #the ipa_train will control the data sets; the mfcc data will rely on the ipa data
    #Note: because each row of IPA data might be different lengths in MFCC data, 
    #the sets won't 100% correspond to their designated sizes. HOWEVER, it is more 
    #important (for now) to keep as much speaker between group mixing. That is most easily 
    #achieved with the IPA data
    ipa_train, ipa_val, ipa_test = batch_prep.get_datasets()
    ipa_train_label = ipa_train[1]
    ipa_train_data = ipa_train[0]
    print("\n\nTrain Data (rows = {}): \n{}".format(len(ipa_train_data),ipa_train_data))
    #print("\n\nValidation Data (rows = {}): \n{}".format(len(ipa_val),ipa_val))
    #print("\n\nTest Data (rows = {}): \n{}".format(len(ipa_test),ipa_test))

    #define perameters for the batches - will be defined to all batches made
    #in this class instance
    batch_prep.def_batch(batch_size=20,ipa_shift=3)
    
    for row in ipa_train_data:
        batch_row_train, total_row_batches = batch_prep.generate_batch(row,ipa_train_label)
        print(batch_row_train.shape)
        final.databatch2sql(batch_row_train)


except DatabaseLimitError as dle:
    print(dle)
except ValidateDataRequiresTestDataError as vde:
    print(vde)
except ShiftLargerThanWindowError as slw:
    print(slw)
except TrainDataMustBeSetError as tds:
    print(tds)
except EmptyDataSetError as eds:
    print(eds)
except KeyError as ke:
    print(ke)
except Error as e:
    print("Database error: {}".format(e))
#Close database connections:
finally:
    ipa.close_conn()
    mfcc.close_conn()
    final.close_conn()


'''
Next steps:

1) Form X and y data, after one-hot-encoding the y data with keras.utils.to_categorical(y, num_classes = batch_prep.num_classes).

Question I have: would it make a difference if ipa stress markers were included? 
'''
