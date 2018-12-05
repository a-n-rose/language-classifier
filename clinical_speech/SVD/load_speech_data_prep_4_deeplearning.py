import sqlite3
import numpy as np
import pandas as pd
import random
import traceback


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
    print(randomly_assigned_conditions)
    
    train_speakers = []
    val_speakers = []
    test_speakers = []
    
    #the number of assigned conditions might be slightly less than num_speakers
    #using int() above does not round up, only down
    if len(randomly_assigned_conditions) < num_speakers:
        diff = num_speakers - len(randomly_assigned_conditions)
        for j in range(diff):
            rand_choice = np.random.choice([0,1,2],p=[0.8,0.1,0.1])
            print(randomly_assigned_conditions)
            randomly_assigned_conditions=np.append(randomly_assigned_conditions,rand_choice) 
            print(rand_choice)
            print(randomly_assigned_conditions)
    for i in range(num_speakers):

        if randomly_assigned_conditions[i] == 0:
            train_speakers.append(speaker_ids[i])
        elif randomly_assigned_conditions[i] == 1:
            val_speakers.append(speaker_ids[i])
        elif randomly_assigned_conditions[i] == 2:
            test_speakers.append(speaker_ids[i])
    return train_speakers, val_speakers, test_speakers


def prep_data_4_model(sex):

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
    
    X_train = train.iloc[:,1:121].values
    y_train = train.iloc[:,-2].values
    
    X_val = val.iloc[:,1:121].values
    y_val = val.iloc[:,-2].values
    
    X_test = test.iloc[:,1:121].values
    y_test = test.iloc[:,-2].values
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    
if __name__=="__main__":
    try:
        conn = sqlite3.connect("healthy_dysphonia_speech.db")
        c = conn.cursor()

        train_f, val_f, test_f = prep_data_4_model("female")
        train_m, val_m, test_m = prep_data_4_model("male")
        
        '''
        Next: feed deep neural networks 9 context window framed data
        First to convnet then to LSTM... but perhaps each individually first.
        '''

    except Exception as e:
        #I know I need to fix this... 
        traceback.print_exception(e)
    finally:
        if conn:
            conn.close()
