'''
Implementing the ConvNet and LSTM methods from the paper:

Dysarthric Speech Recognition Using Convolutional LSTM Neural Network

Kim, M., Cao, B., An, K., & Wang, J. (2018)


'''


import sqlite3
import numpy as np
import pandas as pd
import random
import traceback
import math

#for the model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, LSTM, MaxPooling2D, Dropout, SimpleRNN, Reshape, TimeDistributed, Activation, ZeroPadding2D

#to catch errors I think might happen
from errors import TotalSamplesNotAlignedSpeakerSamples

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


def collect_sample_sizes(dataframe,col_id,speaker_ids,context_window_size):
    total_samples_sum = 0
    samples_list = []
    for speaker in speaker_ids:
        num_samples = sum(dataframe[col_id]==speaker)
        num_patches = num_samples//(context_window_size*2+1)
        total_samples_needed = (context_window_size*2+1)*num_patches
        samples_list.append(total_samples_needed)
    return samples_list

def fill_matrix_speaker_samples_zero_padded(matrix2fill, row_id, data_supplied, indices, speaker_label, len_samps_per_id, label_for_zeropadded_rows,context_window):
    '''
    This function fills a matrix full of zeros with the same number of rows dedicated to 
    each speaker. 
    
    If the speaker has too many samples, not all will be included. 
    If the speaker has too few samples, only the samples that will complete a full window will
    be included; the rest will be replaced with zeros/zero padded.
    
    
    1) I need the len of matrix, to be fully divisible by len_samps_per_id 
    
    2) len_samps_per_id needs to be divisible by context_window_total (i.e. context_window * 2 + 1)
    
    2) label column assumed to be last column of matrix2fill
    
    #mini test scenario... need to put this into unittests
    empty_matrix = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    
    #each id has 3 rows
    data_supplied = np.array([[2,2,3],[2,2,3],[2,2,3],[2,2,3],[2,2,3],[2,2,3],[4,4,5],[4,4,5],[4,4,5],[4,4,5],[4,4,5],[4,4,5],[4,4,5],[4,4,5],[4,4,5],[4,4,5],[4,4,5],[4,4,5],[4,4,5],[4,4,5],[1,1,7],[1,1,7],[1,1,7],[1,1,7],[1,1,7],[1,1,7],[1,1,7],[1,1,7],[1,1,7],[1,1,7]])
    
    indices_too_few = [0,1,2,3,4,5] #too few samples (6/10)  (total_window_size = 5) 
    
    label_too_few = 1
    
    indices_too_many = [6,7,8,9,10,11,12,13,14,15,16,17,18,19] #too many (14/10) (total_window_size = 5) 
    
    label_too_many = 0
    
    indices_just_right = [20,21,22,23,24,25,26,27,28,29] #10/10 (total_window_size = 5) 
    
    label_just_right = 1
    
    len_samps_per_id = 10
    
    label_for_zeropadded_rows = 2
    
    empty_matrices should be:
    
    row_id = 0 --> row_id = 10
    
    matrix_too_few = np.array([[2,2,3,1],[2,2,3,1],[2,2,3,1],[2,2,3,1],[2,2,3,1],[0,0,0,2],[0,0,0,2],[0,0,0,2],[0,0,0,2],[0,0,0,2],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    
    row_id = 10 --> row_id = 20
    matrix_too_many = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[4,4,5,0],[4,4,5,0],[4,4,5,0],[4,4,5,0],[4,4,5,0],[4,4,5,0],[4,4,5,0],[4,4,5,0],[4,4,5,0],[4,4,5,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    
    row_id = 20 --> row_id = 30
    matrix_too_many = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,7,1],[1,1,7,1],[1,1,7,1],[1,1,7,1],[1,1,7,1],[1,1,7,1],[1,1,7,1],[1,1,7,1],[1,1,7,1],[1,1,7,1]])
    '''
    try:
        context_window_total = context_window*2+1
        
        tot_samps_speaker = len(indices)
        tot_samps_sets = tot_samps_speaker//context_window_total
        tot_samps_possible = tot_samps_sets * context_window_total
        
        if tot_samps_possible > len_samps_per_id:
            tot_samps_possible = len_samps_per_id
            indices = indices[:tot_samps_possible]
        
        #keep track of the samples put into the new matrix
        #don't want samples to exceed amount set by variable 'len_samps_per_id'
        samp_count = 0
        
        for index in indices:
            
            #samples only get added to matrix if fewer than max number
            if samp_count < len_samps_per_id and row_id < len(matrix2fill):
                new_row = np.append(data_supplied[index],speaker_label)
                matrix2fill[row_id] = new_row
                samp_count += 1
                row_id += 1
            else:
                if row_id >= len(matrix2fill):
                    raise TotalSamplesNotAlignedSpeakerSamples("Row id exceeds length of matrix to fill.")
            # if all user samples used, but fewer samples put in matrix than max amount, zero pad
            if samp_count < len_samps_per_id and samp_count == tot_samps_possible:
                zero_padded = len_samps_per_id - samp_count
                
                if np.modf(zero_padded/context_window_total)[0] != 0.0:
                    raise TotalSamplesNotAlignedSpeakerSamples("Zero padded rows don't match window frame size")
                
                for row in range(zero_padded):
                    #leave zeros, just change label
                    matrix2fill[row_id][-1] = label_for_zeropadded_rows
                    row_id += 1
                    samp_count += 1
            
            #once all necessary samples put into matrix, leave loop and continue w next speaker 
            elif samp_count == len_samps_per_id:
                break
            
            #samp_count should not be greater than len_samps_per_id... if it is, something went wrong.
            elif samp_count > len_samps_per_id:
                raise TotalSamplesNotAlignedSpeakerSamples("More samples collected than max amount")

    except TotalSamplesNotAlignedSpeakerSamples as e:
        print(e)
        row_id = False
    
    return matrix2fill, row_id

def limit_samples_window_size(dataframe,col_id,id_list,context_window_size):
    #need to ensure each speaker has patches w sizes according to the context_window_size (i.e. context_window_size*2+1) -- the context window are the frames surrounding the sample of a particular class. If context window = 9, the whole frame would be 19.
    samples_list = collect_sample_sizes(dataframe,col_id,id_list,context_window_size)
    print("max len samples")
    print(max(samples_list))
    
    
    '''
    make every sample set (from each speaker) same number (based off of largest sample size)
    zero pad if shorter
    ignore samples if longer
    
    zero padded values given label == 2 so that real labels (i.e. 0 = healthy, 1 = clinical) are not affected.
    '''
    
    #setting the sample set number:
    #608 seemed around the max num samples available from speakers, from the training set.
    # Note: this must be divisible by the number of frames per set (i.e. if 19, 608/19 --> 32)
    # This division is necessary when reshaping the data for the ConvNet+LSTM (see shape4convnet_LSTM() below)
    reference_samp_num = 608 
    
    #set len of new matrix
    matrix_numrows = reference_samp_num * len(samples_list)

    #new matrix that will hold all the (necessary) data for training the models
    spect_patches = np.zeros((matrix_numrows,121)) #120 = num features + 1 label column

    #add only necessary samples to matrix 
    #remove samples that do not fully fill in the frames (don't want frames getting mixed with zero-padded labels; full-frame with values or full-frame with zeros)
    feature_cols = list(range(1,121))
    features = dataframe.loc[:,feature_cols].values
    id_cols = [121,123] #121 has ids; 123 has labels
    ids_labels = dataframe.loc[:,id_cols].values
    row_id = 0

    label_for_zeropadded_rows = 2
    
    try:
        if np.modf(matrix_numrows/reference_samp_num)[0] != 0.0:
            raise TotalSamplesNotAlignedSpeakerSamples("Length of matrix does not align with total samples for each speaker")
        
        for speaker in id_list:
            equal_speaker = ids_labels[:,0] == speaker
            indices = np.where(equal_speaker)[0]
            
        
            #get label for speaker
            label = ids_labels[indices[0],1]
            
            spect_patches, row_id = fill_matrix_speaker_samples_zero_padded(spect_patches,row_id,features,indices,label,reference_samp_num,label_for_zeropadded_rows,context_window_size)

    except TotalSamplesNotAlignedSpeakerSamples as e:
        print(e)
        spect_patches = None
    return spect_patches

def prep_data_4_model(sex,context_window_size = None):
    if context_window_size is None:
        context_window_size = 9

    data = get_speech_data(sex)
    cols = data.columns
    col_id = cols[-3]
    ids = data[col_id]
    ids = ids.unique()
    
    #see how many samples each speaker has:
    samps_list = []
    for speaker in ids:
        samps_list.append(sum(data[col_id]==speaker))
    
    #not necessary - just as fyi
    max_samp = max(samps_list)
    print("the max samples from the {} speakers: {}".format(sex,max_samp))
    
    #cannot have speakers mixed in train/val/test datasets. Separate based on speaker.
    train_ids, val_ids, test_ids = split_train_val_test(ids)
    
    #not necessary - just a little check-in
    num_train_speakers = len(train_ids)
    print("Number of speakers in training set {}: {}".format(sex,num_train_speakers))
    
    #assign corresponding speech data to the train/validation/test set
    data["dataset"] = data[col_id].apply(lambda x: 0 if x in train_ids else(1 if x in val_ids else 2))
    
    train = data[data["dataset"]==0]
    val = data[data["dataset"]==1]
    test = data[data["dataset"]==2]
    
    X_y_train = limit_samples_window_size(train,col_id,train_ids,context_window_size)
    X_y_val = limit_samples_window_size(val,col_id,val_ids,context_window_size)
    X_y_test = limit_samples_window_size(test,col_id,test_ids,context_window_size)

    return X_y_train, X_y_val, X_y_test

def shape4convnet_LSTM(data_features_labels_matrix,num_samples_per_speaker, context_window_size=None):
    '''
    prep data shape for ConvNet+LSTM:
    shape = (num_speakers, num_sets_per_speaker; num_frames_per_set; num_features_per_frame; grayscale)
    
    If ConvNet and LSTM put together --> (66,32,19,120,1) if 66 speakers
    - ConvNet needs grayscale 
    - LSTM needs num_sets_per_speaker 
    
    If separate:
    - Convent needs grayscale (19,120,1)
    - LSTM needs number features in a series, i.e. 19 (19,120)
    '''
    
    if context_window_size is None:
        context_window_size = 9
    
    #separate features from labels:
    features = data_features_labels_matrix[:,:-1]
    labels = data_features_labels_matrix[:,-1]
    num_frame_sets = num_samples_per_speaker//(context_window_size*2+1)
    
    num_sets_samples = len(data_features_labels_matrix)//num_frame_sets
    
    num_speakers = len(data_features_labels_matrix)//num_samples_per_speaker
    
    #make sure only number of samples are included to make up complete context window frames of e.g. 19 frames (if context window frame == 9, 9 before and 9 after a central frame, so 9 * 2 + 1)
    check = len(data_features_labels_matrix)//num_frame_sets
    if math.modf(check)[0] != 0.0:
        print("Extra Samples not properly removed")
    else:
        print("No extra samples found")
    
    #reshaping data to suit ConvNet + LSTM model training. 
    #see notes at top of function definition
    X = features.reshape(len(data_features_labels_matrix)//num_samples_per_speaker,num_samples_per_speaker//(context_window_size*2+1),context_window_size*2+1,features.shape[1],1)
    y_indices = list(range(0,len(labels),608))
    y = labels[y_indices]
    return X, y
    

if __name__=="__main__":
    try:
        conn = sqlite3.connect("healthy_dysphonia_speech.db")
        c = conn.cursor()
        

        train_f, val_f, test_f  = prep_data_4_model("female")
        train_m, val_m, test_m = prep_data_4_model("male")
        
        #done collecting data, close database
        conn.close()
        
        if train_f is None:
            raise TotalSamplesNotAlignedSpeakerSamples("Something went wrong..")
        
        #reshape data to fit model(s)
        X_f_train, y_f_train = shape4convnet_LSTM(train_f,608)
        X_f_val, y_f_val = shape4convnet_LSTM(val_f,608)
        X_f_test, y_f_test = shape4convnet_LSTM(test_f,608)
        
        X_m_train, y_m_train = shape4convnet_LSTM(train_m,608)
        X_m_val, y_m_val = shape4convnet_LSTM(val_m,608)
        X_m_test, y_m_test = shape4convnet_LSTM(test_m,608)


        #now create models from paper:
        
        #TIME-FREQUENCY CONVNET w LSTM
        ###################################################################################################
        '''
        helpful resource:
        https://machinelearningmastery.com/cnn-long-short-term-memory-networks/
        '''

        #TIME-FREQUENCY CONVNET
        tfcnn = Sequential()
        # feature maps = 40
        # 8x4 time-frequency filter (goes along both time and frequency axes)
        input_size = (19,120,1)
        tfcnn.add(Conv2D(40, kernel_size=(8,4), activation='relu'))
        #non-overlapping pool_size 3x3
        tfcnn.add(MaxPooling2D(pool_size=(3,3)))
        tfcnn.add(Dropout(0.25))
        tfcnn.add(Flatten())
        
        #prepare LSTM
        tfcnn_lstm = Sequential()
        # this is where it would be good for all speakers to have same length... 
        # time-steps (?), image shape (19,120), grayscale (1)
        tfcnn_lstm.add(TimeDistributed(tfcnn,input_shape=(32,19,120,1)))
        tfcnn_lstm.add(LSTM(32)) #num timesteps
        tfcnn_lstm.add(Dense(1,activation="sigmoid"))
        
        
        print(tfcnn_lstm.summary())
        
        
        #compile model
        tfcnn_lstm.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
        #train model
        tfcnn_lstm.fit(X_f_train, y_f_train, validation_data=(X_f_val,y_f_val),epochs=100)
        
        #predict test data
        pred = tfcnn_lstm.predict(X_f_test)
        pred = pred >0.5
        pred = pred.astype(float)
        
        #see how many were correct
        correct = 0
        for i, item in enumerate(y_f_test):
            if item == pred[i]:
                correct += 1
        score = round(correct/float(len(y_f_test)) * 100, 2)
        print("model earned a score of {}%  for the test data.".format(score))


    except TotalSamplesNotAlignedSpeakerSamples as e:
        print(e)
    except Exception as e:
        #I know I need to fix this... 
        traceback.print_exception(e)
    finally:
        if conn:
            conn.close()
