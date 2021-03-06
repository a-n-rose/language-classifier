'''
Run this script from a directory with subdirectory(ies) containing .tgz files (that contain wavefiles) or .wav files

The subdirectories should be labeled by their language - the subdirectory names will be used to categorize the data when training the algorithm. 

IMPORTANT:
This randomly assigns speakers to train, validation, and test sets (speakers should not mix in these sets). Therefore, if you cannot ensure each speaker has only one wavfile or tgz file, or that each tgz file is dedicated to only 1 speaker, please create 'English_train', 'English_test', 'English_validate' subdirectories in the cwd (instead of just 'English') with sufficient audio files in each. You can then ignore the 'dataset' variable. 

Global variables such as database name and type of background noise group (added to training data) need to be defined before running the script. 

This script allows you to see how far along the program is in each directory

This calculates 40 coefficiens (rather than 13) and will therefore create a database with 45 columns (40 coefficiencts with 
filename, noise labels, and categorical label)
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
import random

import prep_noise as prep_data

import time
import logging.handlers
from my_logger import start_logging, get_date
#for logging:
script_purpose = 'MFCC_extraction_' #will name logfile 
current_filename = os.path.basename(__file__)
session_name = get_date() #make sure this session has a unique identifier - link to model name and logging information



#global variables
database = 'sp_mfcc_VAD.db'
noisegroup = 'matched_VAD' # Options: 'matched' 'none' 'random'
#if no noise, environment_noise = None; otherwise, put name of wavefile here
environment_noise = 'background_noise_poor_recording.wav' #Options: None or wavefile i.e. 'background_noise.wav'  
#specify number of mfccs --> reflects the number of columns
#this needs to match the others in the database, therefore should be changed with caution
num_mfcc = 40


#'.' below means current directory
def extract(tar_url, extract_path='.'):
    tar = tarfile.open(tar_url, 'r')
    for item in tar:
        tar.extract(item, extract_path)
        if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
            extract(item.name, "./" + item.name[:item.name.rfind('/')])
            
def parser(wavefile,num_mfcc,env_noise=None):
    try:
        y, sr = librosa.load(wavefile, res_type= 'kaiser_fast')
        y = prep_data.normalize(y)
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


def insert_data(filename,feature, sr, noise_scale,dataset_group,label):
    if sr:
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
        curr_df["label"] = label
        
        x = curr_df.values
        num_cols = num_mfcc + len(['filename','noisegroup','noiselevel','dataset','label'])
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


            c.execute(''' CREATE TABLE IF NOT EXISTS mfcc_40(%s,filename  TEXT, noisegroup TEXT, noiselevel REAL, dataset INT,label TEXT) ''' % ", ".join(column_type))
            conn.commit()

                
            #collect directory names:
            dir_list = []
            for label in glob.glob('*/'):
                dir_list.append(label)
            if len(dir_list) > 0:
                print("The directories found include: ", dir_list)
            else:
                print("No directories found")
            
            for j in range(len(dir_list)):
                directory = dir_list[j]
                os.chdir(directory)
                label = directory[:-1]
                print("Now processing the directory: "+label)
                
                #check for all wave files in each subdirectory:
                wavefiles = []
                for wav in glob.glob('**/*.wav',recursive = True):
                    wavefiles.append(wav)
                if len(wavefiles) > 0:
                    for v in range(len(wavefiles)):
                        #assigns wavefile to train, validate, or train dataset (1,2,3 respectively)
                        #does expect each speaker to have only 1 wavefile
                        dataset_group = random.choice([1,1,1,1,1,1,1,2,2,2,3,3,3])
                        wav = wavefiles[v]
                        feature,sr,noise_scale = parser(wav, num_mfcc,env_noise)
                        filename = Path(wav).name
                        insert_data(wav,feature, sr, noise_scale,dataset_group,label)
                        conn.commit()
                        
                        update = "\nProgress: \nwavefile {} ({} out of {})".format(filename,v+1,len(wavefiles))
                        dir_percentage = "Appx. {}% through directory {}".format(((v+1)/(len(wavefiles)))*100,label)
                        total_percentage = "Appx. {}% through all directories".format(((j+1)/(len(dir_list)))*100)
                    
                        logging.info(update)
                        print(update)
                        print(dir_percentage)
                        print(total_percentage)

                                
                else:
                    print_message = "No wave files found in directory: {}".format(label)
                    print(print_message)
                    logging.info(print_message)
                
            
                #check for all .tgz files (and the wave files within them)
                tgz_list = []
                for tgz in glob.glob('**/*.tgz',recursive=True):
                    tgz_list.append(tgz)
                if len(tgz_list) > 0:
                    for t in range(len(tgz_list)):
                        #assigns zipfile to train, validate, or train dataset (1,2,3 respectively)
                        #does expect each speaker to have only 1 zipfile/an entire zipfile to be dedicated to only 1 speaker
                        tgz_name = tgz_list[t]
                        dataset_group = random.choice([1,1,1,1,1,1,1,2,2,2,3,3,3])
                        extract(tgz_list[t], extract_path = '/tmp/audio')
                        filename = os.path.splitext(tgz_list[t])[0]
                        waves_list = []
                        for w in glob.glob('/tmp/audio/**/*.wav',recursive=True):
                            waves_list.append(w)
                        if len(waves_list) > 0:
                            for k in range(len(waves_list)):
                                wav = waves_list[k]
                                feature,sr,noise_scale = parser(wav, num_mfcc,env_noise)
                                wav_name = str(Path(wav).name)
                                insert_data(tgz_name+'_'+wav_name,feature, sr, noise_scale,dataset_group,label)
                                conn.commit()
                                
                                update = "\nProgress: \nwavefile {} ({} out of {})\ntgz file {} ({} out of {})".format(wav_name,k+1,len(waves_list),filename,t+1,len(tgz_list))
                                percentage = "Appx. {}% through file {}".format(((k+1)/(len(waves_list)))*100,filename)
                                dir_percentage = "Appx. {}% through directory {}".format(((t+1)/(len(tgz_list)))*100,label)
                                total_percentage = "Appx. {}% through all directories".format(((j+1)/(len(dir_list)))*100)
                            
                                logging.info(update)
                                print(update)
                                print(percentage)
                                print(dir_percentage)
                                print(total_percentage)

                        else:
                            update_nowave_inzip = "No .wav files found in zipfile: {}".format(tgz_list[t])
                            logging.info(update_nowave_inzip)
                            print(update_nowave_inzip)
                        tgz_filename = str(Path(filename).name)
                        shutil.rmtree('/tmp/audio/'+tgz_filename)
                else:
                    print_message = "No tgz files found in directory: {}".format(label)
                    print(print_message)
                    logging.info(print_message)
                
                print("\nFinished processing directory {}\n".format(label))
                os.chdir("..")
                    
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
    except Error as e:
        logging.exception("Database error: %s" % e)
    except Exception as e:
        logging.exception("Error occurred: %s" % e)
    finally:
        if conn:
            conn.close()
