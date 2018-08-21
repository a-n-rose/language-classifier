 
import numpy as np
import datetime
import os
import sqlite3
from sqlite3 import Error
from pathlib import Path
import glob
import keras
from keras.models import model_from_json
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from preprocess_speech import reduce_noise, get_date
from id_ur_speech_func import ID_UR_Speech


def get_date():
    time = datetime.datetime.now()
    time_str = "{}y{}m{}d{}h{}m{}s".format(time.year,time.month,time.day,time.hour,time.minute,time.second)
    return(time_str)

def parser(wavefile,num_mfcc,env_noise=None):
    try:
        y, sr = librosa.load(wavefile, res_type= 'kaiser_fast')
        y = prep_data.normalize(y)
        
        rand_scale = 0.0
        #randomly assigns speaker data to 1 (train) 2 (validation) or 3 (test)
        if env_noise is not None:
            #at random apply varying amounts of environment noise
            rand_scale = random.choice([0.0,0.25,0.5,0.75,1.0,1.25])
            #logging.info("Scale of noise applied: {}".format(rand_scale))
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
        #logging.exception('def parser() resulted in {} for the file: {}'.format(error,wavefile))
        print('def parser() resulted in {} for the file: {}'.format(error,wavefile))
        print(error)
    except ValueError as ve:
        #logging.exception("Error occured ({}) with the file {}".format(ve,wavefile))
        print("Error occured ({}) with the file {}".format(ve,wavefile))
        print(ve)
    
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
        #logging.exception("Failed MFCC extraction: {} in the directory: {}".format(filename,label))
        print("Failed MFCC extraction: {} in the directory: {}".format(filename,label))
    
    return None




if __name__ == '__main__':
    try:
        date = get_date()
        database = 'sp_mfcc_{}'.format(date)
        
        conn = sqlite3.connect(database)
        c = conn.cursor
        noisegroup = 'none'
        curr_speech = ID_UR_Speech(date)
        directory_user = './user_recordings/'
        directory_processed_speech = './processed_recordings/'
        if not os.path.exists(directory_user):
            os.makedirs(directory_user)
        if not os.path.exists(directory_processed_speech):
            os.makedirs(directory_processed_speech)
        test_mic = curr_speech.start_action('test your mic')
        sec = 5
        if test_mic:
            print("Now recording. Please stay quiet as we measure the background noise.")
        mictest = curr_speech.test_mic(sec)
        if mictest == False:
            print("We couldn't test your mic..")
            print("Please check your settings and connections.")
            curr_speech.cont = False
        while curr_speech.cont == True:
            print("For model testing purposes, which langauge do/will you speak?")
            curr_speech.language = input()
            curr_speech.cont = curr_speech.start_action('test me on your language')
            if curr_speech.cont:
                curr_speech.play_go()
                user_speech = curr_speech.record_user(60)
                time_str = curr_speech.get_date()
                user_recording_filename = '{}_{}.wav'.format(directory_user,time_str)
                curr_speech.save_rec(user_recording_filename,user_speech,fs=22050)
                #subtract noise
                if reduce_noise(directory_user+user_recording_filename,directory_user+'background_{}.wav'.format(date)):
                    #save speech to MFCCs 
                    env_noise = None
                    num_mfcc = 40
                    wav = directory_processed_speech+'rednoise_{}.wav'.format(date)
                    feature,sr,noise_scale = parser(wav, num_mfcc,env_noise)
                    #prepare database to save data

                    columns = list((range(0,num_mfcc)))
                    column_type = []
                    for i in columns:
                        column_type.append('"'+str(i)+'" REAL')

                    
                    c.execute(''' CREATE TABLE IF NOT EXISTS mfcc_40_user(%s,filename  TEXT, noisegroup TEXT, noiselevel REAL, dataset INT,label TEXT) ''' % ", ".join(column_type))
                    conn.commit()
                    
                    name = Path(wav).name
                    insert_data(name,feature,sr,noise_scale=0,dataset_group=0,label=curr_speech.language)
                    conn.commit()
                    
                    #prepare data
                    df = pd.DataFrame(feature)
                    X = df.values
                    
                    #normalize
                    mean = np.mean(X,axis=0)
                    std = np.std(X,axis=0)
                    X=(X-mean)/std
                    
                    #feature scaling
                    sc = StandardScaler()
                    X = sc.fit_transform(X)
                    
                    for model in glob.glob('./models/*.json'):
                        model_name = os.path.splitext(model)[0]
                        json_file = open(model_name+'.json', 'r')
                        loaded_model_json = json_file.read()
                        json_file.close()
                        loaded_model = model_from_json(loaded_model_json)
                        # load weights into new model
                        loaded_model.load_weights(model_name+".h5")
                        print("Loaded model from disk")

                        # evaluate loaded model on new data
                        try:
                            loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                            classify = loaded_model.predict(X)
                            classify = (classify > 0.5)
                            english = sum(classify == 0)
                            german = sum(classify == 1)
                            max_index = np.argmax([english,german])
                            classification = max_index
                            if classification == 0:
                                prediction = 'English'
                            elif classification == 1:
                                prediction = 'German'
                            else:
                                print("Error ocurred - no language predicted")
                            print("The model '{}' \n\npredicted your language to be: \n\n{}".format(model_name,prediction))
                        except Exception as e:
                            print(e)
                    
    except Exception as e:
        print (e)
    finally:
        if conn:
            conn.close()
                
