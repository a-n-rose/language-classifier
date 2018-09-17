#interacting with paths/folders
import os

#handling, organizing, and saving data
import sqlite3
from sqlite3 import Error

#IPA extraction (international phonetic alphabet)
from collect_speech_data import collect_tgzfiles, tgz_2_annotations

#needed for logging 
import time
import logging.handlers
from my_logger import start_logging, get_date
#for logging:
script_purpose = 'translate_speech2IPA' #will name logfile 
current_filename = os.path.basename(__file__)
session_name = get_date() #make sure this session has a unique identifier - link to model name and logging information



#global variables:
database='speech2IPA.db'
tablename = 'speech2IPA'


if __name__ == '__main__':
    try:
        start_logging(script_purpose)
        prog_start = time.time()
        logging.info(prog_start)

        conn = sqlite3.connect(database)
        c = conn.cursor()


        #check variables:
        print("The purpose of this script is: {}".format(script_purpose))
        print("The database name used is: {}".format(database))
        print("The tablename is: {}".format(tablename))
        
        
        print("Are the above variables correct? (Please type 'Y' or 'N')")
        check_variables = input()
        if 'y' in check_variables.lower():

            c.execute("CREATE TABLE IF NOT EXISTS {}(session_ID TEXT, filename  TEXT, annotation_original TEXT, annotation_IPA TEXT, label TEXT) ".format(tablename))
            conn.commit()
            
            tgz_list = collect_tgzfiles()
        
            #collect annotations and save to database
            collected = tgz_2_annotations(tgz_list,c,conn,tablename)
            
            if collected:
                print("Annotations have been collected.")
        else:
            print("Please correct those variables and then rerun the script.")
        
    except Exception as e:
        logging.error("Error occurred: {}".format(e))
    except Error as e:
        logging.error("Database error occurred: {}".format(e))
    finally:
        if conn:
            conn.close()
            print("Database has been closed.")
