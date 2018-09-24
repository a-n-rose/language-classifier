'''
ToDo:
1) find max num MFCCs per IPA character (probably 6-7) 
'''


import numpy as np
import pandas as pd
import config
from pathlib import Path
import time


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

    def build_ipa_dict(self,ipa_list):
        dict_ipa = dict()
        for num in range(len(ipa_list)):
            dict_ipa[ipa_list[num]] = num
        print(dict_ipa)
        self.dict_ipa = dict_ipa
        return self

    def all_ipa_present(self):
        try:
            start = time.time()
            ipa_chars = []
            count = 0
            for annotation in self.ipa[:,3]:
                for char in annotation:
                    count += 1
                    if char in ipa_chars:
                        pass
                    else:
                        ipa_chars.append(char)
            self.build_ipa_dict(ipa_chars)
            end = time.time()
            total_time = end - start
            return ipa_chars, total_time, count
        except Exception as e:
            print(e)
            
    def retrieve_ipa_key(self,ipa_list):
        ipa_keys = []
        for char in ipa_list:
            ipa_keys.append(self.dict_ipa[char])
        return ipa_keys

    #for each row in data_ipa
    def generate_batch(self,batch_size,ipa_window,ipa_shift):
        if ipa_shift > ipa_window:
            raise ShiftLargerThanWindowError("The shift cannot exceed the size of the window of IPA characters.")
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
        #do I want to overlap? Not right now..
        overlap = False
        #make sure there is a total of 3 IPA characters (or the size of the window) per input (don't end up with with only 1 IPA character as classification sequence)
        if overlap == False:   
            num_ipa_diff = num_ipa % ipa_window
            num_ipa -= num_ipa_diff
            annotation_ipa = annotation_ipa[:num_ipa]
            
        total_batches = int(num_ipa/ipa_shift - (ipa_window - ipa_shift))
        #create skeleton for where batches will be collected
        batch = np.zeros(shape=(total_batches,batch_size,num_features+ipa_window))
        
        for batch_iter in range(total_batches):
            start = batch_iter * (num_mfcc_per_ipa * ipa_shift) #shifting at indicated shift length (e.g. if ipa_shift = 1, then shift 1 letter at a time)
            end = start + batch_mfcc #window of 3 letters
            assert end < len(mfcc) 
            index_ipa = batch_iter * ipa_shift
            ipa_label = annotation_ipa[index_ipa:index_ipa+ipa_window]
            ipa_ints = self.retrieve_ipa_key(ipa_label)
            batch_input = mfcc[start:end,:]
            len_mfccs = len(batch_input)
            add_ints = np.repeat([ipa_ints],len_mfccs,axis=0)
            batch_input = np.c_[batch_input,add_ints]
            batch[batch_iter]=batch_input
        
        if data_index < len(ipa):
            self.data_index += 1
        else:
            self.data_index = None
            print("Through all of IPA data")
        return batch, total_batches