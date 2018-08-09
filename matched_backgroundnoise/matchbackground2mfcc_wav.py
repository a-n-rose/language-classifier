'''
This script is suited to exctract features from .wav files that are in one or more folders within the cwd.  

40 MFCCs will be extracted and saved via SQLite3. One can use fewer in training as desired.

Options for MFCC extraction: the current script allows you to supply a recording of background noise; . This would ideally match the background levels new recordings will be made, i.e. for the classifier to classify.  This noise is then added (at varying levels) to the training wavefiles, which are then processed for MFCC extraction.

This may take several hours (possibly around 10 hours) 

Logging will be saved in a .csv in the cwd.

'''
import os 
import numpy as np
import pandas as pd
import librosa
import glob
import sqlite3
import random
import time
import logging
import logging.handlers
logger = logging.getLogger(__name__)
from pympler import tracker


import prep_noise



def parser(wavefile,num_mfcc,env_noise):
    try:
        y, sr = librosa.load(wavefile, res_type= 'kaiser_fast')
        y = prep_noise.normalize(y)
        
        #at random apply varying amounts of environment noise
        rand_scale = random.choice([0.0,0.25,0.5,0.75,1.0,1.25])
        logging.info("Scale of noise applied: {}".format(rand_scale))
        if rand_scale:
            #apply *matched* environemt noise to signal
            total_length = len(y)/sr
            envnoise_normalized = prep_noise.normalize(env_noise)
            envnoise_scaled = prep_noise.scale_noise(envnoise_normalized,rand_scale)
            envnoise_matched = prep_noise.match_length(envnoise_scaled,sr,total_length)
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

    
def get_save_mfcc(tgz_file,label,dirname,num_mfcc,env_noise):
    noisegroup = 'matched'#other groups: 'none' and 'random'
    filename = os.path.splitext(tgz_file)[0]
    feature, sr, noise_scale = parser(filename+".wav",num_mfcc,env_noise)
    if sr:
        columns = list((range(0,num_mfcc)))
        #turn coefficient number into column names
        column_str = []
        for i in columns:
            column_str.append(str(i))
        feature_df = pd.DataFrame(feature)
        curr_df = pd.DataFrame.transpose(feature_df)
        #apply column names to dataframe
        curr_df.columns = column_str
        #add additional columns with helpful info such as filenames/directory name/noise info
        curr_df["filename"] = filename
        curr_df["directory"] = dirname
        curr_df["noisegroup"] = noisegroup
        curr_df["noiselevel"] = noise_scale 
        curr_df["label"] = label
         
        x = curr_df.as_matrix()
        num_cols = num_mfcc + len(['filename','directory','noisegroup','noiselevel','label'])
        col_var = ""
        for i in range(num_cols):
            if i < num_cols-1:
                col_var+=' ?,'
            else:
                col_var+=' ?'
        c.executemany(' INSERT INTO mfcc_40 VALUES (%s) ' % col_var,x)
        conn.commit()
    else:
        logging.exception("Failed MFCC extraction: {} in the directory: {}".format(tgz_file,dirname))
    return None


if __name__ == '__main__':
    try:
        
        
        #------------------------------- Logger ---------------------------#
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
        
        
        
        
        
        
        
        #-----------------------------------------------------------------#
        
        #initialize database
        conn = sqlite3.connect('sp_mfcc.db')
        c = conn.cursor()

        #load environment noise to be added to training data
        env_noise = librosa.load('background_noise.wav')[0]

        #label = input("Which category is this speech? ")
        label = 'Test'
        prog_start = time.time()
        logging.info(label)
        logging.info(prog_start)
        #specify number of mfccs --> reflects the number of columns
        num_mfcc = 40
        columns = list((range(0,num_mfcc)))
        column_type = []
        for i in columns:
            column_type.append('"'+str(i)+'" REAL')


        c.execute(''' CREATE TABLE IF NOT EXISTS mfcc_40(%s,filename  TEXT, directory TEXT, noisegroup TEXT, noiselevel REAL, label TEXT) ''' % ", ".join(column_type))
        conn.commit()


        #collect directory names:
        dir_list = []
        for dirname in glob.glob('*/'):
            dir_list.append(dirname)
        if len(dir_list) > 0:
            print("The directories found include: ", dir_list)
        else:
            print("No directories found")


        for j in range(len(dir_list)):
        #for j in range(1):
            directory = dir_list[j]
            os.chdir(directory)
            dirname = directory[:-1]
            logging.info(dirname)
            print("Now processing the directory: "+dirname)
            files_list = []
            for wav in glob.glob('*.wav'):
                files_list.append(wav)
            if len(files_list) != 0:
                for i in range(len(files_list)):
                    logging.info(files_list[i])
                    get_save_mfcc(files_list[i],label,dirname,num_mfcc,env_noise)
                    logging.info("Successfully processed {} from the directory {}".format(files_list[i],dirname))
                    logging.info("Progress: \nwavefile {} out of {}\nindex = {} \ndirectory {} out of {} \nindex = {}".format(i+1,len(files_list),i,j+1,len(dir_list),j))
                    print("Progress: ", ((i+1)/(len(files_list)))*100,"%  (",dirname,": ",j+1,"/",len(dir_list)," directories)")
            else:
                print("No wave files found in ", dirname)
            os.chdir("..")
            print("The wave files in the "+ dirname + " directory have been processed successfully")

        conn.commit()
        conn.close()
        print("MFCC data has been successfully saved!")
        print("All wave files have been processed")
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
