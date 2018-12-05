'''
Collects male and female speech of healthy and dysphonic speakers
Calculates 40 log mel-filterbanks, their delta (rate of change) and delta delta (acceleration_of_change) 
Puts these as features (total 120) into database

Speaker IDs, sex, and healthy or dysphonia is also saved 
'''

import sqlite3
import glob
import numpy as np
import pandas as pd
import librosa




def create_data_table(num_features):
    cols = []
    for i in range(num_features):
        cols.append("'{}' REAL".format(i))
    cols_str = ", ".join(cols)
    
    msg = '''CREATE TABLE IF NOT EXISTS speaker_data(sample_id INTEGER PRIMARY KEY,  %s,speaker_id INT, sex TEXT,label INT)''' % cols_str
    
    c.execute(msg)
    conn.commit()
    return None


def save_features_sql(prepped_features,num_features):
    cols = ""
    for i in range(num_features):
        cols += " ?,"
    
    msg = ''' INSERT INTO speaker_data VALUES(NULL,%s ?, ?, ?) ''' % cols
    c.executemany(msg,prepped_features)
    conn.commit()
    return None


def get_filenames(group,gender):
    filenames = []
    for wav in glob.glob("./speech_data/dataset/{}/{}/sentences/*.wav".format(group,gender)):
        filenames.append(wav)
    return filenames

def match_condition_lengths(clinical,healthy):
    if len(healthy) > len(clinical):
        healthy = healthy[:len(clinical)]
    elif len(clinical) > len(healthy):
        clinical = clinical[:len(healthy)]
    return clinical, healthy

def get_mel_spectrogram_derivatives(filename,num_mels):
    '''
    get mel spectrogram at windows of 25 ms and shifts of 10 ms
    '''
    y, sr = librosa.load(filename, sr=1600)
    spect = librosa.feature.melspectrogram(y,sr=sr,hop_length=int(0.010*sr),n_fft=int(0.025*sr),n_mels=num_mels)
    rate_of_change = librosa.feature.delta(spect)
    acceleration_of_change = librosa.feature.delta(spect,order=2)
    
    #transpose so features are columns and rows are frames
    spect = spect.transpose()
    rate_of_change = rate_of_change.transpose()
    acceleration_of_change = acceleration_of_change.transpose()
    
    return spect, rate_of_change, acceleration_of_change

def get_all_features(filename,num_mels):
    spect, rate_of_change, acceleration_of_change = get_mel_spectrogram_derivatives(filename,num_mels=num_mels)
    len_values = len(spect)
    speaker_all_features = np.empty((len_values,num_mels*3))
    for i in range(len_values):                
        speaker_all_features[i] = np.concatenate((spect[i],rate_of_change[i],acceleration_of_change[i]))
    return speaker_all_features

def prep_features_sql(filename,group,gender):
    speaker_id = filename.split("-")[0]
    speaker_id = speaker_id.split("/")[-1]
    features = get_all_features(filename,num_mels = 40)
    df = pd.DataFrame(features)
    df["speaker_id"] = speaker_id
    df["sex"] = gender
    df["label"] = group
    vals = df.to_dict(orient="index")
    prepped_data = []
    for key, value in vals.items():
        prepped_data.append(tuple(value.values()))
    return prepped_data

def collect_features_save_sql(filename_list):

    for filename in filename_list:
        if "healthy" in filename:
            group = 0
        else:
            group = 1
        if "female" in filename:
            gender = "female"
        else:
            gender = "male"
        prepped_features = prep_features_sql(filename,group,gender)
        save_features_sql(prepped_features,num_features = 120)
    return None



if __name__ == "__main__":
    
    conn = sqlite3.connect("healthy_dysphonia_speech.db")
    c = conn.cursor()

    healhy_women = get_filenames("healthy","female")
    dysphonia_women = get_filenames("dysphonia","female")

    healthy_men = get_filenames("healthy","male")
    dysphonia_men = get_filenames("dysphonia","male")

    #want to randomize this in the future
    #but for now, simply matches length of healthy and dysphonia speech recordings
    dysphonia_women, healthy_women = match_condition_lengths(dysphonia_women,healhy_women)

    dysphonia_men, healthy_men = match_condition_lengths(dysphonia_men,healthy_men)

    create_data_table(num_features = 120) #40 mel-filterbanks * 3 (1st and 2nd derivatives are included as features)

    try:
        collect_features_save_sql(dysphonia_women)
        collect_features_save_sql(healthy_women)
        collect_features_save_sql(dysphonia_men)
        collect_features_save_sql(healthy_men)
    except Exception as e:
        print(e)
    finally:
        conn.close()
