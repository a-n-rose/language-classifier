import os, tarfile
import glob
from pathlib import Path
from subprocess import check_output
import random

import prep_noise as prep_data


class Speech_Data():
    def __init__(self,database,num_hours = None, window_sec = None,window_shift_sec = None, sr = None, num_mfcc = None, noise = None):
        self.database = database
        self.conn = sqlite3.connect(database)
        self.c = self.conn.cursor()
        if num_hours is None:
            self.num_hours = 10
        else:
            self.num_hours = num_hours
        if window_sec is None:
            self.window_sec = 0.025
        else:
            self.window_sec = window_sec
        if window_shift_sec is None:
            self.window_shift_sec = 0.01
        else:
            self.window_shift_sec = window_shift_sec
        mfcc_per_hour = window_shift * 3600000
        self.num_mfcc_rows = self.num_hours * mfcc_per_hour
        if sr is None:
            self.sr = 22500
        else:
            self.sr = sr
        if num_mfcc is None:
            self.num_mfcc = 40
        else:
            self.num_mfcc = num_mfcc
        self.env_noise = noise

    def prep_ipa_cols(self,tablename,columns_list):
        msg = ''' CREATE TABLE IF NOT EXISTS {}(annotation_id int primary key, %s) '''.format(tablename) % ", ".join(columns_list)
        return msg
    
    def prep_mfcc_cols(self,tablename,columns_list):
        mfcc_columns = list(range(self.num_mfcc))
        mfcc_column_type = []
        for i in mfcc_columns:
            mfcc_column_type.append('"'+str(i)+'" real')
        msg = ''' CREATE TABLE IF NOT EXISTS {}(mfcc_id int primary key, %s,%s) '''.format(tablename) % (", ".join(mfcc_column_type),", ".join(columns_list))
        return msg

    def create_sql_table(self,msg):
        self.c.execute(msg)
        self.conn.commit()
        return None
        
        
        
        
        
        
        
        
    def parser(self,wavefile):
        try:
            y, sr = librosa.load(wavefile, sr = self.sr,res_type= 'kaiser_fast')
            y = prep_data.normalize(y)
            y = prep_data.get_speech_samples(y,sr)
            rand_scale = 0.0
            #randomly assigns speaker data to 1 (train) 2 (validation) or 3 (test)
            if self.env_noise is not None:
                #at random apply varying amounts of environment noise
                rand_scale = random.choice([0.0,0.25,0.5,0.75,1.0,1.25])
                if rand_scale > 0.0:
                    #apply *known* environemt noise to signal
                    total_length = len(y)/sr
                    envnoise_normalized = prep_data.normalize(self.env_noise)
                    envnoise_scaled = prep_data.scale_noise(envnoise_normalized,rand_scale)
                    envnoise_matched = prep_data.match_length(envnoise_scaled,sr,total_length)
                    if len(envnoise_matched) != len(y):
                        diff = int(len(y) - len(envnoise_matched))
                        if diff < 0:
                            envnoise_matched = envnoise_matched[:diff]
                        else:
                            envnoise_matched = np.append(envnoise_matched,np.zeros(diff,))
                    y += envnoise_matched
            mfccs = librosa.feature.mfcc(y, sr, n_mfcc=self.num_mfcc,hop_length=int(self.window_shift_sec*sr),n_fft=int(self.window_sec*sr))
            return mfccs, sr, rand_scale
        except EOFError as error:
            logging.exception('def parser() resulted in {} for the file: {}'.format(error,wavefile))
        except ValueError as ve:
            logging.exception("Error occured ({}) with the file {}".format(ve,wavefile))
        
        return None, None, None

    def insert_data(self,filename,feature, sr, noise_scale,dataset_group,label):
        if sr:
            columns = list((range(0,self.num_mfcc)))
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
            self.c.executemany(' INSERT INTO mfcc_40 VALUES (%s) ' % col_var,x)
            self.conn.commit()
            return True,"MFCCs successfully saved to database."
        
        return False,"Error occurred in saving MFCCs to database."

    def annotations2IPA(c,conn,tablename,path,language,session):
        annotation_file = path + 'etc/PROMPTS'
        annotations = open(annotation_file,'r').readlines()
        for annotation in annotations:
            annotation_list = annotation.split(' ')
            wavename = Path(annotation_list[0]).name+'.wav'
            words = annotation_list[1:]
            words_str = ' '.join(words)
            print("Annotation of wavefile {} is: {}".format(wavename,words_str))
            ipa = check_output(["espeak", "-q", "--ipa", "-v", "en-us", words_str]).decode("utf-8")
            print("IPA translation: {}".format(ipa))
            cmd = '''INSERT INTO {} VALUES (?,?,?,?,?)'''.format(tablename)
            c.execute(cmd, (session,wavename,words_str,ipa,language))
            conn.commit()
            print("Annotation saved successfully.")
        return None

    def wav2MFCC(c,conn,tablename_MFCC,newpath,language,session_id):
        dataset_group = random.choice([1,1,1,1,1,1,1,2,2,2,3,3,3])
        waves_list = []
        for w in glob.glob(newpath+'**/*.wav',recursive=True):
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
        
    def collect_tgzfiles():
        tgz_list = []
        #get tgz files even if they are several sudirectories deep - recursive=True
        for tgz in glob.glob('**/*.tgz',recursive=True):
            tgz_list.append(tgz)
        return tgz_list

    def extract(tar_url, extract_path='.'):
        tar = tarfile.open(tar_url, 'r')
        for item in tar:
            tar.extract(item, extract_path)
            if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
                extract(item.name, "./" + item.name[:item.name.rfind('/')])
        return None

    def tgz_2_IPA_MFCC(tgz_list,tablename_IPA,tablename_MFCC):
        if len(tgz_list)>0:
            tmp_path = '/tmp/annotations'
            for t in range(len(tgz_list)):
                extract(tgz_list[t],extract_path = tmp_path)
                session_info = Path(tgz_list[t])
                #print("Path: {}".format(session_info.parts))
                language = session_info.parts[0]
                #print("Language label: {}".format(language))
                session_id = os.path.splitext(session_info.parts[1])[0]
                #print("session ID: {}".format(session_id))
                newpath = "{}/{}/".format(tmp_path,session_id)
                annotations2IPA(tablename_IPA,newpath,language,session_id)
                wav2MFCC(tablename_MFCC,newpath,language,session_id)
                return True
        return False
