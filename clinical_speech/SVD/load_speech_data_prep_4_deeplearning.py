import sqlite3
import numpy as np
import pandas as pd
import random
import traceback
import math

#import data
#separate into male and female (similar to the study I'm comparing findings with)


def get_speech_data(gender):
    '''
    Use * here due to high number of features: over 120 columns, with speaker_id, sex, and label (healthy/clinical) at the end
    '''
    msg = ''' SELECT * FROM speaker_data WHERE sex=? '''
    t = (gender,)
    c.execute(msg,t)
    data = pd.DataFrame(c.fetchall())
    return data

def split_train_val_test(speaker_ids,perc_train=None,perc_val=None,perc_test=None):
    '''
    Splits speakers into training, validation, and test
    default: 80-10-10 ratio
    
    should put in 'random seed' functionality..
    '''
    if perc_train is None:
        perc_train = 0.8
        perc_val = 0.1
        perc_test = 0.1
        
    num_speakers = len(speaker_ids)
    num_train = int(num_speakers * perc_train)
    num_val = int(num_speakers * perc_val)
    num_test = int(num_speakers * perc_test)
    
    train = [0] * num_train
    val = [1] * num_val
    test = [2] * num_test
    
    randomly_assigned_conditions = np.concatenate((train,val,test))
    random.shuffle(randomly_assigned_conditions)
    
    train_speakers = []
    val_speakers = []
    test_speakers = []
    
    #the number of assigned conditions might be slightly less than num_speakers
    #using int() above does not round up, only down
    if len(randomly_assigned_conditions) < num_speakers:
        diff = num_speakers - len(randomly_assigned_conditions)
        for j in range(diff):
            rand_choice = np.random.choice([0,1,2],p=[0.8,0.1,0.1])
            randomly_assigned_conditions=np.append(randomly_assigned_conditions,rand_choice) 
    for i in range(num_speakers):

        if randomly_assigned_conditions[i] == 0:
            train_speakers.append(speaker_ids[i])
        elif randomly_assigned_conditions[i] == 1:
            val_speakers.append(speaker_ids[i])
        elif randomly_assigned_conditions[i] == 2:
            test_speakers.append(speaker_ids[i])
    return train_speakers, val_speakers, test_speakers


def find_new_matrix_length(dataframe,col_id,speaker_ids,context_window_size):
    total_samples_sum = 0
    for speaker in speaker_ids:
        num_samples = sum(dataframe[col_id]==speaker)
        num_patches = num_samples//(context_window_size*2+1)
        total_samples_needed = (context_window_size*2+1)*num_patches
        total_samples_sum += total_samples_needed
    return total_samples_sum


def limit_samples_window_size(dataframe,col_id,id_list,context_window_size):
    #need to ensure each speaker has patches sized 19
    matrix_numrows = find_new_matrix_length(dataframe,col_id,id_list,context_window_size)
    
    spect_patches = np.empty((matrix_numrows,121)) #120 = num features + 1 label column
    
    #add only necessary samples to train matrix 
    #remove samples that do not fit in 19 set frames
    feature_cols = list(range(1,121))
    features = dataframe.loc[:,feature_cols].values
    id_cols = [121,123]
    ids_labels = dataframe.loc[:,id_cols].values
    row_id = 0
    for speaker in id_list:
        equal_speaker = ids_labels[:,0] == speaker
        indices = np.where(equal_speaker)[0]
        
        num_samples = len(indices)
        max_sets = num_samples//(context_window_size*2+1)
        tot_samples_needed = max_sets * (context_window_size*2+1)
        
        #get label for speaker
        label = ids_labels[indices[0],1]
        
        #keep track of sample number per speaker (not more than can create full 19 frames)
        local_count = 0
        
        for index in indices:
            local_count+=1
            #add label to feature array - keep track of it
            new_row = np.append(features[index],label)
            spect_patches[row_id] = new_row
            if local_count == tot_samples_needed:
                break
    return spect_patches

def prep_data_4_model(sex,context_window_size = None):
    if context_window_size is None:
        context_window_size = 9

    data = get_speech_data(sex)
    cols = data.columns
    col_id = cols[-3]
    ids = data[col_id]
    ids = ids.unique()
    train_ids, val_ids, test_ids = split_train_val_test(ids)
    
    data["dataset"] = data[col_id].apply(lambda x: 0 if x in train_ids else(1 if x in val_ids else 2))
    
    train = data[data["dataset"]==0]
    val = data[data["dataset"]==1]
    test = data[data["dataset"]==2]
    
    X_y_train = limit_samples_window_size(train,col_id,train_ids,context_window_size)
    X_y_val = limit_samples_window_size(val,col_id,val_ids,context_window_size)
    X_y_test = limit_samples_window_size(test,col_id,test_ids,context_window_size)

    return X_y_train, X_y_val, X_y_test


def shape4convnet(data_features_labels_matrix,context_window_size=None):
    if context_window_size is None:
        context_window_size = 9
    #prepping shape for ConvNet. 
    #shape = (num_spectrogram_sets/images, 19, 120, 1)
    #num of images --> len(data_labels)
    #size of the "image": 19 X 120  (19 frames in total, 120 features)
    #1 --> grayscale (important for ConvNet to include)
    

    #separate features from labels:
    features = data_features_labels_matrix[:,:-1]
    labels = data_features_labels_matrix[:,-1]
    
    num_frame_sets = len(data_features_labels_matrix)//(context_window_size*2+1)
    
    #make sure only number of samples are included to make up complete context window frames of 19 frames (if context window frame == 9, 9 before and 9 after a central frame, so 9 * 2 + 1)
    check = len(data_features_labels_matrix)/float(context_window_size*2+1)
    if math.modf(check)[0] != 0.0:
        print("Extra Samples not properly removed")
    else:
        print("No extra samples found")
        
    X = features.reshape(num_frame_sets,context_window_size*2+1,features.shape[1],1)
    y = labels.reshape(num_frame_sets,context_window_size*2+1)
    y = y[:,0]
    return X, y
    

if __name__=="__main__":
    try:
        conn = sqlite3.connect("healthy_dysphonia_speech.db")
        c = conn.cursor()

        train_f, val_f, test_f = prep_data_4_model("female")
        train_m, val_m, test_m = prep_data_4_model("male")
        
        #reshape data to fit model
        X_f_train, y_f_train = shape4convnet(train_f)
        X_f_val, y_f_val = shape4convnet(val_f)
        X_f_test, y_f_test = shape4convnet(test_f)

    except Exception as e:
        #I know I need to fix this... 
        traceback.print_exception(e)
    finally:
        if conn:
            conn.close()
