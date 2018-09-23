'''
ToDo:
1) find max num MFCCs per IPA character (probably 6-7) 
'''


import numpy as np
import pandas as pd
import config
from pathlib import Path


class Batch_Data:
    def __init__(self,data_ipa,data_mfcc):
        self.ipa = data_ipa
        self.mfcc = data_mfcc
        self.data_index = 0

    def remove_spaces(self,string):
        newstring = string.replace(" ","")
        return newstring

    def get_tgz_name(self,path,wav):
        path_split = Path(path).name
        tgz_name = "{}.tgz_{}".format(path_split,wav)
        return tgz_name

    #for each row in data_ipa
    def generate_batch(self,batch_size,ipa_window,ipa_shift):
        if self.data_index is not None:
            data_index = self.data_index
        else:
            print("All IPA data prepped for network.")
            return None
        print(data_index)
        #get annotation data for output label
        ipa = self.ipa[data_index]
        recording_session = ipa[0]
        wavefile = ipa[1]
        annotation_ipa = self.remove_spaces(ipa[3])
        num_ipa = len(annotation_ipa)
        mfcc_id = self.get_tgz_name(recording_session,wavefile)
        
        #get mfcc data, and align w ipa data
        mfcc_indices = np.where(self.mfcc[:,40]==mfcc_id)
        mfcc = self.mfcc[mfcc_indices,:40]
        
        mfcc = mfcc.reshape(mfcc.shape[1],mfcc.shape[2])
        #print(mfcc)
        print(mfcc.shape)
        num_mfcc = mfcc.shape[0]
        print("num mfcc samples: {}".format(num_mfcc))
        num_features = mfcc.shape[1]
        print("num_features: {}".format(num_features))
        num_mfcc_per_ipa = num_mfcc//num_ipa
        batch_mfcc = num_mfcc_per_ipa*3
        assert batch_mfcc <= batch_size
        
        
        #figure out how many batches of MFCC data I have for the total number of IPA chars
        #do I want to overlap? Yes I think so.. 
        
        #num batches = num_ipa - 2
        total_batches = num_ipa - (ipa_window - 1)
        
        #create skeleton for where batches will be collected
        batch = np.zeros(shape=(total_batches,batch_size,num_features))
        
        for batch_iter in range(total_batches):
            start = batch_iter * (num_mfcc_per_ipa * ipa_shift) #shifting at indicated shift length (e.g. if ipa_shift = 1, then shift 1 letter at a time)
            end = start + batch_mfcc #window of 3 letters
            assert end < len(mfcc) 
            batch[batch_iter]=(mfcc[start:end,:])
        
        if data_index < len(ipa):
            self.data_index += 1
        else:
            self.data_index = None
            print("Through all of IPA data")
        return batch
