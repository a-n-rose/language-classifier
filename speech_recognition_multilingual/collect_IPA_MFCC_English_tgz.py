#interacting with paths/folders
import os

#handling, organizing, and saving data
import sqlite3
from sqlite3 import Error

#IPA extraction (international phonetic alphabet)
from collect_speechdata import Speech_Data

#needed for logging 
import time
import logging.handlers
from my_logger import start_logging, get_date
#for logging:
script_purpose = 'speech_ipa_mfcc_data_collection' #will name logfile 
current_filename = os.path.basename(__file__)
session_name = get_date() #make sure this session has a unique identifier - link to model name and logging information



#global variables:
database='speech_wnoise_ipa_mfcc.db'
noisefile = 'background_noise_poor_recording.wav'#options: None 'somewavefile.wav'
tablename_annotations = 'speech_ipa'
tablename_mfcc = 'speech_mfcc'
tablename_list = [tablename_annotations,tablename_mfcc]

# parent table
tablecols_annotations = ['session text','wave text','path text','original text','ipa text','label text']

#child table (don't need to repeat data)... but I don't see how this would work..
tablecols_mfcc = ['noisegroup text','noiselevel real','dataset int','label text','annotation_id int','foreign key (annotation_id) references ipa_annotations(annotation_id)']

tablecols_list = [tablecols_annotations,tablecols_mfcc]

if __name__ == '__main__':
    try:
        #check variables:
        print("The purpose of this script is: {}".format(script_purpose))
        print("The database name used is: {}".format(database))
        print("The tablename(s) is/are: {}".format(", ".join(tablename_list)))
        print("The columns include: {}".format(tablecols_list))
        print("Background noise: {}".format(noisefile))
        print("Are the above variables correct? (Please type 'Y' or 'N')")
        check_variables = input()
        if 'y' not in check_variables.lower():
            print("\nPlease correct those variables and then rerun the script.\n")
            raise SystemExit
        
        
        start_logging(script_purpose)
        prog_start = time.time()
        logging.info(prog_start)
        
        spdata = Speech_Data(database,num_hours=10,noise=noisefile)
        
        #create tables if not exist for IPA annotations and MFCC data:
        #the annotation table is the parent table
        msg_ipa_table = spdata.prep_ipa_cols(tablename_annotations,tablecols_annotations)
        spdata.create_sql_table(msg_ipa_table)
        
        msg_mfcc_table = spdata.prep_mfcc_cols(tablename_mfcc,tablecols_mfcc)
        spdata.create_sql_table(msg_mfcc_table)
        
        
        #start collecting data in tgz files
        #first extracting tgz in tmp directory
        
        tgz_list = collect_tgzfiles()

        #collect annotations and save to database
        collected = spdata.tgz_2_IPA_MFCC(tgz_list,tablename_annotations,tablename_mfcc)
        
        if collected:
            print("Annotations have been collected.")
                
        
    except Exception as e:
        logging.error("Error occurred: {}".format(e))
    except Error as e:
        logging.error("Database error occurred: {}".format(e))
    finally:
        if speech_data.conn:
            speech_data.conn.close()
            print("Database has been closed.")
