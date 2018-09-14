'''
Speech data collected from:
 "The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)" by Livingstone & Russo is licensed under CC BY-NA-SC 4.0.


To label the speech data, I used this labeling chart, available from the website. 

https://zenodo.org/record/1188976

Filename identifiers 

    Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
    Vocal channel (01 = speech, 02 = song).
    Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
    Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
    Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
    Repetition (01 = 1st repetition, 02 = 2nd repetition).
    Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

'''

import os, tarfile
import numpy as np
import pandas as pd
import librosa
import glob
import shutil
import sqlite3
from sqlite3 import Error
from pathlib import Path
import time
import random

import prep_noise as prep_data

from pympler import tracker

import logging.handlers
from my_logger import start_logging, get_date
#for logging:
script_purpose = 'MFCC_extraction_emotion_VAD' #will name logfile 
current_filename = os.path.basename(__file__)
session_name = get_date() #make sure this session has a unique identifier - link to model name and logging information

 


#global variables
database = 'sp_mfcc_emotions_VAD.db'
noisegroup = 'none' # Options: 'matched' 'none' 'random'
#if no noise, environment_noise = None; otherwise, put name of wavefile here
environment_noise = None #Options: None or wavefile i.e. 'background_noise.wav'  
#specify number of mfccs --> reflects the number of columns
#this needs to match the others in the database, therefore should be changed with caution
num_mfcc = 40


def parser(wavefile,num_mfcc,env_noise=None):
    try:
        y, sr = librosa.load(wavefile, res_type= 'kaiser_fast')
        y = prep_data.normalize(y)
        #remove the silence at beginning and end of recording
        y = prep_data.get_speech_samples(y,sr)
        
        rand_scale = 0.0
        #randomly assigns speaker data to 1 (train) 2 (validation) or 3 (test)
        if env_noise is not None:
            #at random apply varying amounts of environment noise
            rand_scale = random.choice([0.0,0.25,0.5,0.75,1.0,1.25])
            logging.info("Scale of noise applied: {}".format(rand_scale))
            if rand_scale:
                #apply *known* environemt noise to signal
                total_length = len(y)/sr
                envnoise_normalized = prep_data.normalize(env_noise)
                envnoise_scaled = prep_data.scale_noise(envnoise_normalized,rand_scale)
                envnoise_matched = prep_data.match_length(envnoise_scaled,sr,total_length)
                if len(envnoise_matched) != len(y):
                    diff = int(len(y) - len(envnoise_matched))
                    if diff < 0:
                        envnoise_matched = envnoise_matched[:diff]
                    else:
                        envnoise_matched = np.append(envnoise_matched,np.zeros(diff,))
                y += envnoise_matched
        mfccs = librosa.feature.mfcc(y, sr, n_mfcc=num_mfcc,hop_length=int(0.010*sr),n_fft=int(0.025*sr))
        return mfccs, sr, rand_scale
    except EOFError as error:
        logging.exception('def parser() resulted in {} for the file: {}'.format(error,wavefile))
    except ValueError as ve:
        logging.exception("Error occured ({}) with the file {}".format(ve,wavefile))
    
    return None, None, None


def insert_data(filename,feature, sr, noise_scale,dataset_group,label=None):
    if sr:
        parts = Path(filename).parts
        #collect label information from filename:
        labels = parts[1]
        modality = int(labels[:2])
        voice_channel = int(labels[3:5])
        emotion = int(labels[6:8])
        intensity = int(labels[9:11])
        statement = int(labels[12:14])
        repetition = int(labels[15:17])
        speaker = int(labels[18:20])
        gender = speaker%2  #0 = female, 1 = male
        
        columns = list((range(0,num_mfcc)))
        column_str = []
        for i in columns:
            column_str.append(str(i))
        feature_df = pd.DataFrame(feature)
        curr_df = pd.DataFrame.transpose(feature_df)
        curr_df.columns = column_str
        #add additional columns with helpful info such as filename,noise info, label
        curr_df["filename"] = filename
        curr_df["noisegroup"] = noisegroup
        curr_df["noiselevel"] = noise_scale 
        curr_df["dataset"] = dataset_group
        curr_df["speaker"] = speaker
        curr_df["intensity"] = intensity
        curr_df["statement"] = statement
        curr_df["repetition"] = intensity
        curr_df["speaker_sex"] = gender
        curr_df["label"] = emotion
        
        
        x = curr_df.values
        num_cols = num_mfcc + len(['filename','noisegroup','noiselevel','dataset','speaker','intensity','statement','repetition','speaker_sex','label'])
        col_var = ""
        for i in range(num_cols):
            if i < num_cols-1:
                col_var+=' ?,'
            else:
                col_var+=' ?'
        c.executemany(' INSERT INTO mfcc_40 VALUES (%s) ' % col_var,x)
        conn.commit()
    else:
        logging.exception("Failed MFCC extraction: {} in the directory: {}".format(filename,label))
    
    return None

if __name__ == '__main__':
    try:
        tr_tot = tracker.SummaryTracker()
        start_logging(script_purpose)
        prog_start = time.time()
        logging.info(prog_start)

        #initialize database
        conn = sqlite3.connect(database)
        c = conn.cursor()
        
        
        print("Database will be saved as: {}".format(database))
        print("Noisegroup of collected MFCCs: {}".format(noisegroup))
        print("Noise wavefile: {}".format(environment_noise))
        print("Number of MFCCs to be extracted: {}".format(num_mfcc))
        
        check_variables = input("\nIMPORTANT!!!!\nAre the items listed above correct? (Y or N): ")
        if 'y' in check_variables.lower():

            #load environment noise to be added to training data
            if environment_noise: 
                try:
                    env_noise = librosa.load(environment_noise)[0]
                except FileNotFoundError as fnf:
                    print("\nCannot find {} in cwd.\n".format(environment_noise))
                    raise fnf
            else:
                env_noise = None

            columns = list((range(0,num_mfcc)))
            column_type = []
            for i in columns:
                column_type.append('"'+str(i)+'" REAL')


            c.execute(''' CREATE TABLE IF NOT EXISTS mfcc_40(%s,filename  TEXT, noisegroup TEXT, noiselevel REAL, dataset INT, speaker INT, intensity INT, statement INT, repetition INT, speaker_sex INT, label INT) ''' % ", ".join(column_type))
            conn.commit()



            #collect label information
            waves = []
            for wave in glob.glob('**/*.wav'):
                waves.append(wave)
                
            for wav in waves:

                #80% training, 20% test
                dataset_group = random.choice([1,1,1,1,1,1,1,1,3,3])
                print("Extracting MFCC features...")
                feature,sr,noise_scale = parser(wav, num_mfcc,env_noise)
                print("MFCC data has been extracted. Now saving to database")
                insert_data(wav,feature, sr, noise_scale,dataset_group,label=None)
                conn.commit()
                print("MFCC data from {} has been saved!".format(wav))
                
                    
            conn.commit()
            print_message = '\nData has been committed to database'
            print(print_message)
            logging.info(print_message)
            conn.close()
            print_message = "Database has been closed"
            print(print_message)
            logging.info(print_message)
            print("\nMFCC data has been successfully saved!")
            print("All audio files have been processed")
            elapsed_time = time.time()-prog_start
            logging.info("Elapsed time in hours: {}".format(elapsed_time/3600))
            print("Elapsed time in hours: ", elapsed_time/3600)
            tr_tot.print_diff()
        else:
            print_message = "\nRun the script after you correct the global variables within the script."
            print(print_message)
            logging.info(print_message)
    except sqlite3.Error as e:
        logging.exception("Database error: %s" % e)
    except Exception as e:
        logging.exception("Error occurred: %s" % e)
    finally:
        if conn:
            conn.close()
