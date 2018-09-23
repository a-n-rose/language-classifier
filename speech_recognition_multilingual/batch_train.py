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
#print(x_mfcc[:,40])

batch_prep = Batch_Data(x_ipa,x_mfcc)
batch1 = batch_prep.generate_batch(18,3,1)
print(batch1)

'''
Got this working, but need to figure out how to label the batches... how to connect them to the ipa characters

Next step: one-hot-encode the ipa characters?

Also, had a problem if batch_size was 20 (vs. 18) Perhaps different way of adding data to the numpy array? So there are zeros where there aren't any data added?

'''
