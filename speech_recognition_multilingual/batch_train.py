import os

import numpy as np
import pandas as pd
from sqlite3 import Error

from sql_data import Connect_db
from batch_prep import Batch_Data

#data_ipa = sqldata2df()
database_ipa = 'speech2IPA3.db'
table_ipa = 'speech2ipa'
ipa = Connect_db(database_ipa,table_ipa)

database_mfcc = 'sp_mfcc_IPA3.db'
table_mfcc = 'mfcc_40'
mfcc = Connect_db(database_mfcc, table_mfcc)

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
batch_prep.train_val_test(train=0.8,test=0.2)
#the ipa_train will control the data sets; the mfcc data will rely on the ipa data
#Note: because each row of IPA data might be different lengths in MFCC data, 
#the sets won't 100% correspond to their designated sizes. HOWEVER, it is more 
#important (for now) to keep as much speaker between group mixing. That is most easily 
#achieved with the IPA data
ipa_train, ipa_val, ipa_test = batch_prep.get_datasets()
print("\n\nTrain Data (rows = {}): \n{}".format(len(ipa_train),ipa_train))
print("\n\nValidation Data (rows = {}): \n{}".format(len(ipa_val),ipa_val))
print("\n\nTest Data (rows = {}): \n{}".format(len(ipa_test),ipa_test))


#batch_mfcc,total_batches = batch_prep.generate_batch(batch_size=18,ipa_window=3,ipa_shift=3)

batch_train, total_batches = batch_prep.generate_batch(ipa_train,batch_size=18,ipa_window=3,ipa_shift=3)
print(total_batches)
print(batch_train)

batch_val, total_val_batches = batch_prep.generate_batch(ipa_val,batch_size=18,ipa_window=3,ipa_shift=3)
print(total_val_batches)
print(batch_val) 

#for i in range(total_batches):
    #print("\nBatch {}:".format(i+1))
    #len_batches = len(batch_mfcc[i])
    #ipa_vals = batch_mfcc[i][0][40:]
    #ipa_vals = [int(val) for val in ipa_vals]
    #print("IPA values are: {}".format(ipa_vals))
    #for x in ipa_vals:
        #print(x, list(batch_prep.dict_ipa.keys())[list(batch_prep.dict_ipa.values()).index(x)])

'''
Next steps:

2) Form X and y data, after one-hot-encoding the y data with keras.utils.to_categorical(y, num_classes = batch_prep.num_classes).

Question I have: would it make a difference if ipa stress markers were included? 
'''
