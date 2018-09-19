'''
ToDo:
1) find max num MFCCs per IPA character (probably 6-7) 
'''


import numpy as np
import pandas as pd


def remove_spaces(string):
    newstring = string.replace(" ","")
    return newstring

def get_tgz_name(path,wav):
    path_split = Path(path).name
    tgz_name = "{}.tgz_{}".format(path_split,wav)
    return tgz_name

data_ipa = get_ipa_annotations()
data_mfcc = get_mfcc_features()


#for each row in data_ipa
def generate_batch(data_ipa,data_mfcc,batch_size,ipa_window,ipa_shift):
    global data_index
    
    #get annotation data for output label
    ipa = data_ipa[data_index]
    recording_session = ipa[0]
    wavefile = ipa[1]
    annotation_ipa = remove_spaces(ipa[3])
    num_ipa = len(ipa_annotation)
    mfcc_id = get_tgz_name(recording_session,wavefile)
    
    #get mfcc data, and align w ipa data
    mfcc = data_mfcc where data_mfcc[0]==mfcc_id
    num_mfcc = len(mfcc)
    num_features = mfcc.shape[1]
    num_mfcc_per_ipa = num_mfcc//num_ipa
    batch_mfcc = num_mfcc_per_ipa*3
    assert batch_mfcc <= batch_size
    
    
    #figure out how many batches of MFCC data I have for the total number of IPA chars
    #do I want to overlap? Yes I think so.. 
    
    #num batches = num_ipa - 2
    total_batches = num_ipa - (ipa_window - 1)
    
    #create skeleton for where batches will be collected
    batch = np.ndarray(shape=(total_batches,batch_size,num_features), dtype=np.int32)
    
    for batch_iter in range(total_batches):
        start = batch_iter * (num_mfcc_per_ipa * ipa_shift) #shifting at indicated shift length (e.g. if ipa_shift = 1, then shift 1 letter at a time)
        end = start + batch_mfcc #window of 3 letters
        assert end < len(mfcc_data) 
        batch[batch_iter]=(batch_iter,mfcc_data[start:end])
    
    if global_index < len(data_ipa):
        global_index += 1

    return batch
    
    
    
    
    
