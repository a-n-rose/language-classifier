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
num_mfcc=13
noisefile = 'background_noise_poor_recording.wav' #options: None 'somewavefile.wav' i.e.: 'background_noise_poor_recording.wav'
noise_doc = bool(noisefile)
tablename_annotations = 'speech_as_ipa'
tablename_mfcc = 'speech_as_mfcc'
tablename_list = [tablename_annotations,tablename_mfcc]
database='speech_noise_{}_ipa_{}mfcc.db'.format(noise_doc, num_mfcc)

# parent table
tablecols_annotations = ['session_ID text','filename text','annotation_original','annotation_IPA','label text']

#child table (don't need to repeat data)... but I don't see how this would work..
tablecols_mfcc = ['filename text','noisegroup text','noiselevel real','dataset int','label text']

tablecols_list = [tablecols_annotations,tablecols_mfcc]

if __name__ == '__main__':
    try:
            
        start_logging(script_purpose)
        prog_start = time.time()
        logging.info(prog_start)
        
        spdata = Speech_Data(database,num_hours=10,num_mfcc = num_mfcc,noise=noisefile)
        #check variables:
        print("The purpose of this script is: {}".format(script_purpose))
        print("The database name to save data is: {}".format(database))
        print("The tablename(s) is/are: {}".format(", ".join(tablename_list)))
        print("The columns include: {}".format(tablecols_list))
        print("Background noise: {}".format(noisefile))
        print("Are the above variables correct? (Please type 'Y' or 'N')")
        check_variables = input()
        if 'y' not in check_variables.lower():
            raise SystemExit("Please correct those variables and then rerun the script.")
        
        #create tables if not exist for IPA annotations and MFCC data:
        #the annotation table is the parent table
        msg_ipa_table = spdata.prep_ipa_cols(tablename_annotations,tablecols_annotations)
        spdata.create_sql_table(msg_ipa_table)
        
        msg_mfcc_table = spdata.prep_mfcc_cols(tablename_mfcc,tablecols_mfcc)
        spdata.create_sql_table(msg_mfcc_table)
        
        #start collecting data in tgz files
        #first extracting mtgz in tmp directory
        
        tgz_list = spdata.collect_tgzfiles()

        #collect annotations and save to database
        collected = spdata.tgz_2_IPA_MFCC(tgz_list,tablename_annotations,tablename_mfcc)
        
        if collected:
            print("Annotations and MFCCs have been collected.")
                    
        
    except Exception as e:
        logging.error("Error occurred: {}".format(e))
    except Error as e:
        logging.error("Database error occurred: {}".format(e))
    except SystemExit as e:
        logging.error("SystemExit: {}".format(e))
    finally:
        if spdata.conn:
            spdata.conn.close()
            print("Database has been closed.")
