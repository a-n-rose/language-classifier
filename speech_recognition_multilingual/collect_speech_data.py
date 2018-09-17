import os, tarfile
import glob
from pathlib import Path
from subprocess import check_output

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

def tgz_2_annotations(tgz_list,c,conn,tablename):
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
            annotations2IPA(c,conn,tablename,newpath,language,session_id)
            return True
    return False
            
            
