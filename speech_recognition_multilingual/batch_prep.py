'''
ToDo:
1) find max num MFCCs per IPA character (probably 6-7) 
'''


import numpy as np
import pandas as pd
import config
from pathlib import Path
import time
import itertools


class Batch_Data:
    def __init__(self,data_ipa,data_mfcc):
        self.ipa = data_ipa
        self.mfcc = data_mfcc
        self.data_index = 0
        
    def train_val_test(self,train=None,validate=None,test=None):
        if train and validate and test == None:
            self.perc_train = 0.6
            self.perc_val = 0.2
            self.perc_test = 0.2
        elif train and test != None and validate == None:
            self.perc_train = train
            self.perc_validate = 0
            self.perc_test = test
        elif train and validate and test != None:
            self.perc_train = train
            self.perc_validate = validate
            self.perc_test = test
        elif train != None and validate and test == None:
            self.perc_train = train
            self.validate = 0
            self.test = round(1.-train,1)
            print("No validation set created. Only test set based on percentage alloted to train data.")
        elif train and validate != None and test == None:
            raise ValidateDataRequiresTestDataError("In order to set aside validation data, please enter percentage for test data. Otherwise, remove all settings.")
        return self
    
    def get_dataset(self):
        data_total_rows = len(self.ipa)
        self.rows_train = int(data_total_rows * self.perc_train)
        self.rows_val = int(data_total_rows * self.perc_val)
        self.rows_test = int(data_total_rows * self.perc_test)


    def remove_spaces_endofline(self,list_or_string):
        if isinstance(list_or_string,str):
            newstring = list_or_string.replace(" ","")
            newstring = newstring.replace("\n","")
        elif isinstance(list_or_string,list):
            newstring = list_or_string.copy()
            newstring.remove(" ")
            newstring.remove("\n")
        return newstring

    def get_tgz_name(self,path,wav):
        path_split = Path(path).name
        tgz_name = "{}.tgz_{}".format(path_split,wav)
        return tgz_name

    def build_ipa_dict(self,ipa_list):
        dict_ipa = dict()
        for num in range(len(ipa_list)):
            dict_ipa[ipa_list[num]] = num
        self.dict_ipa = dict_ipa
        return self

    def get_num_classes(self,ipa_list,ipa_window):
        self.poss_combinations = itertools.combinations(ipa_list, ipa_window)
        count = 0
        for i in self.poss_combinations:
            count += 1
        self.num_classes = count
        return self
    
    def all_ipa_present(self,ipa_window):
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
            ipa_chars = self.remove_spaces_endofline(ipa_chars)
            self.build_ipa_dict(ipa_chars)
            self.get_num_classes(ipa_chars,ipa_window)
            end = time.time()
            total_time = end - start
            return ipa_chars, self.num_classes
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
        #get annotation data for output label
        ipa = self.ipa[data_index]
        recording_session = ipa[0]
        wavefile = ipa[1]
        annotation_ipa = self.remove_spaces_endofline(ipa[3])
        num_ipa = len(annotation_ipa)
        mfcc_id = self.get_tgz_name(recording_session,wavefile)
        
        #get mfcc data, and align w ipa data
        mfcc_indices = np.where(self.mfcc[:,40]==mfcc_id)
        mfcc = self.mfcc[mfcc_indices,:40]
        
        mfcc = mfcc.reshape(mfcc.shape[1],mfcc.shape[2])
        num_mfcc = mfcc.shape[0]
        print("num mfcc samples for this recording: {}".format(num_mfcc))
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
        #ipa window is added here for the classification info (i.e. how many ipa ids will be located here)
        batch = np.zeros(shape=(total_batches,batch_size,num_features+ipa_window))
        
        for batch_iter in range(total_batches):
            start = batch_iter * (num_mfcc_per_ipa * ipa_shift) #shifting at indicated shift length (e.g. if ipa_shift = 1, then shift 1 letter at a time)
            end = start + batch_mfcc #window of __ letters
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
