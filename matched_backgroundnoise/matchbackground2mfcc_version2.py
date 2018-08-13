'''
Run this script from a directory with subdirectory(ies) containing .tgz files (that contain wavefiles) or .wav files

The subdirectories should be labeled by their language - the subdirectory names will be used to categorize the data when training the algorithm

Global variables such as database name and type of background noise group (added to training data) need to be defined before running the script.

This script allows you to see how far along the program is in each directory

This calculates 40 coefficiens (rather than 13) and will therefore create a database with 45 columns (40 coefficiencts with 
filename, directory, noise labels, and categorical label)
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
import datetime
import random
import logging
import logging.handlers
logger = logging.getLogger(__name__)
from pympler import tracker

import prep_noise as prep_data



#global variables
database = 'sp_mfcc.db'
noisegroup = 'matched' #other groups: 'none' and 'random'
environment_noise = 'background_noise.wav'
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
        
        rand_scale = 0.0
        if env_noise:
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
    
    return None, None


def insert_data(filename,feature, sr, noise_scale,label):
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
        curr_df["label"] = label
        
        x = curr_df.as_matrix()
        num_cols = num_mfcc + len(['filename','noisegroup','noiselevel','label'])
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
        
        #default format: severity:logger name:message
        #documentation: https://docs.python.org/3.6/library/logging.html#logrecord-attributes 
        log_formatterstr='%(levelname)s , %(asctime)s, "%(message)s", %(name)s , %(threadName)s'
        log_formatter = logging.Formatter(log_formatterstr)
        logging.root.setLevel(logging.DEBUG)
        #logging.basicConfig(format=log_formatterstr,
        #                    filename='/tmp/tradinglog.csv',
        #                    level=logging.INFO)
        #for logging infos:
        file_handler_info = logging.handlers.RotatingFileHandler('mfccloginfo.csv',
                                                                  mode='a',
                                                                  maxBytes=1.0 * 1e6,
                                                                  backupCount=200)
        #file_handler_debug = logging.FileHandler('/tmp/tradinglogdbugger.csv', mode='w')
        file_handler_info.setFormatter(log_formatter)
        file_handler_info.setLevel(logging.INFO)
        logging.root.addHandler(file_handler_info)
        
        
        #https://docs.python.org/3/library/logging.handlers.html
        #for logging errors:
        file_handler_error = logging.handlers.RotatingFileHandler('mfcclogerror.csv', mode='a',
                                                                  maxBytes=1.0 * 1e6,
                                                                  backupCount=200)
        file_handler_error.setFormatter(log_formatter)
        file_handler_error.setLevel(logging.ERROR)
        logging.root.addHandler(file_handler_error)
        
        #for logging infos:
        file_handler_debug = logging.handlers.RotatingFileHandler('mfcclogdbugger.csv',
                                                                  mode='a',
                                                                  maxBytes=2.0 * 1e6,
                                                                  backupCount=200)
        #file_handler_debug = logging.FileHandler('/tmp/tradinglogdbugger.csv', mode='w')
        file_handler_debug.setFormatter(log_formatter)
        file_handler_debug.setLevel(logging.DEBUG)
        logging.root.addHandler(file_handler_debug)
        
    
        #initialize database
        conn = sqlite3.connect(database)
        c = conn.cursor()

        #load environment noise to be added to training data
        if environment_noise: 
            env_noise = librosa.load(environment_noise)[0]
        else:
            env_noise = None

        prog_start = time.time()
        logging.info(prog_start)
        columns = list((range(0,num_mfcc)))
        column_type = []
        for i in columns:
            column_type.append('"'+str(i)+'" REAL')


        c.execute(''' CREATE TABLE IF NOT EXISTS mfcc_40(%s,filename  TEXT, noisegroup TEXT, noiselevel REAL, label TEXT) ''' % ", ".join(column_type))
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
                    wav = wavefiles[i]
                    feature,sr,noise_scale = parser(wav, num_mfcc,env_noise)
                    filename = Path(wav).name
                    insert_data(filename,feature, sr, noise_scale,label)
                    progress_message = "Progress: \nfinished processing {} \n{} % wavefiles completed. \nDirectories processed: {}/{}".format(filename,(v+1/len(wavefiles)*100),j+1,len(dir_list))
                    print(progress_message)
                    logging.info(progress_message)
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
                    extract(tgz_list[t], extract_path = '/tmp/audio')
                    filename = os.path.splitext(tgz_list[t])[0]
                    waves_list = []
                    for wav in glob.glob('/tmp/audio/**/*.wav',recursive=True):
                        waves_list.append(wav)
                    if len(waves_list) > 0:
                        for k in range(len(waves_list)):
                            feature,sr,noise_scale = parser(wav, num_mfcc,env_noise)
                            wav_name = str(Path(wav).name)
                            insert_data(filename+'_'+wav_name,feature, sr, noise_scale,label)
                            update = "Progress: \nwavefile {} ({}) out of {}\ntgz file {} ({}) out of {}".format(k+1,str(Path(waves_list[k]).name),len(waves_list),i+1,filename,len(files_list))
                            percentage = "{}% through tgz file {}".format(((k+1)/(len(waves_list)))*100,filename)
                        
                            logging.info(update)
                            print(percentage)
                            print(update)
                    else:
                        update_nowave_inzip = "No .wav files found in zipfile: {}".format(files_list[i])
                        logging.info(update_nowave_inzip)
                        print(update_nowave_inzip)
                    shutil.rmtree('/tmp/audio/'+filename)
            else:
                print_message = "No tgz files found in directory: {}".format(label)
                print(print_message)
                logging.info(print_message)
            
            print("Finished processing directory ",label)
            os.chdir("..")
                
        conn.commit()
        conn.close()
        print("MFCC data has been successfully saved!")
        print("All audio files have been processed")
        elapsed_time = time.time()-prog_start
        logging.info("Elapsed time in hours: {}".format(elapsed_time/3600))
        print("Elapsed time in hours: ", elapsed_time/3600)
        tr_tot.print_diff()
    except sqlite3.Error as e:
        logging.exception("Database error: %s" % e)
    except Exception as e:
        logging.exception("Error occurred: %s" % e)
    finally:
        if conn:
            conn.close()

            
            
