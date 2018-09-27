'''
ToDo:
1) find max num MFCCs per IPA character (probably 6-7) 
'''


import numpy as np
import pandas as pd
from pathlib import Path
import time
import itertools

from Errors import Error, ValidateDataRequiresTestDataError, ShiftLargerThanWindowError, TrainDataMustBeSetError, EmptyDataSetError, MFCCdataNotFoundError

class Batch_Data:
    def __init__(self,data_ipa=None,data_mfcc=None):
        if data_ipa is not None:
            self.ipa = data_ipa
        if data_mfcc is not None:
            self.mfcc = data_mfcc
        
    def train_val_test(self,train=None,validate=None,test=None):
        if train == None and validate == None and test == None:
            self.perc_train = 0.6
            self.perc_val = 0.2
            self.perc_test = 0.2
        elif train != None and test != None and validate == None:
            self.perc_train = train
            self.perc_val = 0
            self.perc_test = test
        elif train != None and validate != None and test != None:
            self.perc_train = train
            self.perc_val = validate
            self.perc_test = test
        elif train != None and validate == None and test == None:
            self.perc_train = train
            self.perc_val = 0
            self.perc_test = round(1.-train,1)
            print("No validation set created. Only test set based on percentage alloted to train data.")
        elif train != None and validate != None and test == None:
            raise ValidateDataRequiresTestDataError("In order to set aside validation data, please enter percentage for test data. Otherwise, remove all settings.")
        elif train == None:
            raise TrainDataMustBeSetError("Train percentage must be set for the validation or test data to be prepared.")
        self.dict_trainvaltest = {'train':1,'validate':2,'test':3}
        self.str_train = 'train'
        self.str_val = 'validate'
        self.str_test = 'test'
        return self
    
    def get_datasets(self):
        data_total_rows = len(self.ipa)
        self.rows_train = int(data_total_rows * self.perc_train)
        self.rows_val = int(data_total_rows * self.perc_val)
        self.rows_test = int(data_total_rows * self.perc_test)
        self.train_ipa = self.ipa[:self.rows_train]
        self.val_ipa = self.ipa[self.rows_train:self.rows_train+self.rows_val]
        self.test_ipa = self.ipa[self.rows_train+self.rows_val:self.rows_train+self.rows_val+self.rows_test]
        return (self.train_ipa,self.str_train), (self.val_ipa,self.str_val), (self.test_ipa,self.str_test)


    def remove_spaces_endofline(self,list_or_string):
        if isinstance(list_or_string,str):
            newstring = list_or_string.replace(" ","")
            newstring = newstring.replace("\n","")
        elif isinstance(list_or_string,list):
            newstring = list_or_string.copy()
            newstring.remove(" ")
            newstring.remove("\n")
        return newstring

    def build_ipa_dict(self,ipa_list):
        dict_ipa = dict()
        for num in range(len(ipa_list)):
            dict_ipa[ipa_list[num]] = num+1
        #create entry for zero classification (zero padded entries)
        dict_ipa[""] = 0
        self.dict_ipa = dict_ipa
        return self

    #NOT EFFECTIVE!!!!
    #def get_num_classes(self,ipa_list):
        #self.poss_combinations = itertools.combinations(ipa_list, self.ipa_window)
        #count = 0
        #for i in self.poss_combinations:
            #count += 1
        #self.num_classes = count +1 #Final +1 accounts for the classification of [0,0,0], i.e. zero padded entries
        #return self
  
    def retrieve_ipa_vals(self,ipa_list):
        ipa_keys = []
        for char in ipa_list:
            ipa_keys.append(self.dict_ipa[char])
        return ipa_keys
  
    def list2int(self,ipa_list):
        ipa_vals = self.retrieve_ipa_vals(ipa_list)
        num_str = ""
        for val in ipa_vals:
            num_str+=str(val)
        num_int = int(num_str)
        return num_int

    def doc_ipa_present(self,ipa_window,ipa_shift):
        self.ipa_window = ipa_window
        self.ipa_shift = ipa_shift
        ipa_chars = []
        ipa_classes = []
        count = 0
        for annotation in self.ipa[:,3]: #3 refers to the column w ipa annotations
            annotation = self.remove_spaces_endofline(annotation)
            for char_idx in range(len(annotation)):
                count += 1
                if annotation[char_idx] in ipa_chars:
                    pass
                else:
                    ipa_chars.append(annotation[char_idx])
                if char_idx + ipa_shift < len(annotation):
                    ipa_label = annotation[char_idx:char_idx+ipa_shift]
                    #ipa_label = self.list2int(ipa_label)
                    if ipa_label in ipa_classes:
                        pass
                    else:
                        ipa_classes.append(ipa_label)
        #NEED TO REMOVE UNWANTED CHARACTERS EARLIER TO IDENTIFY TOTAL CLASSES
        #ipa_chars = self.remove_spaces_endofline(ipa_chars)
        self.build_ipa_dict(ipa_chars)
        self.num_classes = len(ipa_classes)
        self.classes = ipa_classes
        return ipa_chars, self.num_classes
            
        
    def def_batch(self,batch_size):
        self.batch_size = batch_size
        return self
    
    def get_dataset_value(self,label_str):
        label_int = self.dict_trainvaltest[label_str]
        return label_int

    def get_tgz_name(self,path,wav):
        #path = path[1:-1]
        #path_split = Path(path).name
        tgz_name = "{}_{}".format(path,wav)
        return tgz_name

    def generate_batch(self,ipa_dataset_row,dataset_label):
        if len(ipa_dataset_row)<1:
            raise EmptyDataSetError("The provided dataset row is empty.")
        if self.ipa_shift > self.ipa_window:
            raise ShiftLargerThanWindowError("The shift cannot exceed the size of the window of IPA characters.")
        #get dataset value to apply to data
        dataset_label_int = self.get_dataset_value(dataset_label)
        #get annotation data for output label
        ipa = ipa_dataset_row
        recording_session = ipa[0]
        wavefile = ipa[1]
        annotation_ipa = self.remove_spaces_endofline(ipa[3])
        num_ipa = len(annotation_ipa)
        mfcc_id = self.get_tgz_name(recording_session,wavefile)
        #get mfcc data, and align w ipa data
        mfcc_indices = np.where(self.mfcc[:,40]==mfcc_id)
        if len(mfcc_indices[0]) == 0:
            raise MFCCdataNotFoundError("No MFCC data found that matches the session ID.")
        mfcc = self.mfcc[mfcc_indices,:40]
        
        mfcc = mfcc.reshape(mfcc.shape[1],mfcc.shape[2])
        num_mfcc = mfcc.shape[0]
        num_features = mfcc.shape[1]
        num_mfcc_per_ipa = num_mfcc//num_ipa
        batch_mfcc = num_mfcc_per_ipa*3
    
        #figure out how many batches of MFCC data I have for the total number of IPA chars
        #do I want to overlap? Not right now..
        overlap = False
        #make sure there is a total of 3 IPA characters (or the size of the window) per input (don't end up with with only 1 IPA character as classification sequence)
        if overlap == False:   
            num_ipa_diff = num_ipa % self.ipa_window
            num_ipa -= num_ipa_diff
            annotation_ipa = annotation_ipa[:num_ipa]
            
        total_batches = int(num_ipa/self.ipa_shift - (self.ipa_window - self.ipa_shift))
        #create skeleton for where batches will be collected
        #1 is added for label column; another for dataset column
        batch = np.zeros(shape=(total_batches,self.batch_size,num_features+1+1))
        
        for batch_iter in range(total_batches):
            start = batch_iter * (num_mfcc_per_ipa * self.ipa_shift) #shifting at indicated shift length (e.g. if ipa_shift = 1, then shift 1 letter at a time)
            #Ensure the input batchsizes match what is expected:
            if batch_mfcc <= self.batch_size:
                end = start + batch_mfcc #window of __ letters
            else:
                over = batch_mfcc - self.batch_size
                end = start + batch_mfcc - over
            if end > len(mfcc):
                end = len(mfcc)
            index_ipa = batch_iter * self.ipa_shift
            ipa_label = annotation_ipa[index_ipa:index_ipa+self.ipa_window]
            ipa_ints = self.list2int(ipa_label)
            batch_input = mfcc[start:end,:]
            len_mfccs = len(batch_input)
            add_ints = np.repeat([ipa_ints],len_mfccs,axis=0)
            if batch_input.shape[0] < self.batch_size:
                diff = self.batch_size - batch_input.shape[0]
                pad_zeros = np.zeros(shape=(diff,batch_input.shape[1]))
                batch_input = np.r_[batch_input,pad_zeros]
                add_zeros = np.repeat([0],diff,axis=0)
                add_ints = np.r_[add_ints,add_zeros]
            add_dataset_label = np.repeat([dataset_label_int],len(batch_input),axis=0)
            batch_input = np.c_[add_dataset_label,batch_input,add_ints]
            #batch_input = np.c_[batch_input,add_ints]
            batch[batch_iter]=batch_input
        
        return batch, total_batches
    
    def retrieve_ipa_keys(self,ipa_list):
        ipa_keys = []
        for x in ipa_list:
            ipa_keys.append(list(self.dict_ipa.keys())[list(self.dict_ipa.values()).index(x)])
        return ipa_keys
    
    def get_num_features(self,df):
        num_cols = len(df.columns)
        # -1 stands for the 'dataset' column, another 1 stands for the label column
        num_features = num_cols - 1 - 1
        self.num_features = num_features
        return self
    
    def get_x_y(self,df):
        #plus 1 because of 'dataset' column at beginning
        x = df.iloc[:,1:self.num_features+1].values
        y = df.iloc[:,self.num_features+1:].values
        return x,y
    
    def make2d_3d(self,matrix):
        new_rows = len(matrix)//self.batch_size
        cols = matrix.shape[1]
        data_3d = matrix.reshape(new_rows,self.batch_size,cols)
        return data_3d
    
    def normalize_data(self,matrix):
        mean = np.mean(matrix, axis=0)
        std = np.std(matrix,axis=0)
        matrix = (matrix-mean)/std
        return matrix
    
