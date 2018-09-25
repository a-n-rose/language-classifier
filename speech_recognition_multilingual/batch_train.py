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

batch_mfcc,total_batches = batch_prep.generate_batch(batch_size=18,ipa_window=3,ipa_shift=3)

for i in range(total_batches):
    print("\nBatch {}:".format(i+1))
    len_batches = len(batch_mfcc[i])
    ipa_vals = batch_mfcc[i][0][40:]
    ipa_vals = [int(val) for val in ipa_vals]
    print("IPA values are: {}".format(ipa_vals))
    for x in ipa_vals:
        print(x, list(batch_prep.dict_ipa.keys())[list(batch_prep.dict_ipa.values()).index(x)])

'''
Next steps:

1) Form Train, Validate, Test Datasets (do this manually... keep from same speakers to get mixed up too much) -- do this w ipa data
2) Form X and y data, after one-hot-encoding the y data with keras.utils.to_categorical(y, num_classes = batch_prep.num_classes).

Question I have: would it make a difference if ipa stress markers were included? 
'''
