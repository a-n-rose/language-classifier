# Language Classifier
A language classifier model built via deep learning techniques. Currently, the classifer is an ANN. Future classifiers will include CNN, among others.

## Getting Started

1) For speech file collection, voxforge.org is a great option. Many languages are available. For building the classifer, only English and German were used (additional languages are in the process of being added to the model). To download the English and German speech files, below are instructions, respectively. Note: each of these steps take several hours (download files; exctract MFCC data; train models with MFCC data)

* English:
Enter the following command into your commandline, in the directory where you'd like the speech files to be saved. This will download the files into a zipfile:
```
$ wget -r -A.tgz http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Original/48kHz_16bit/s
```
Note: there are several folders with speech data: the above folder is "48kHz_16bit" and has the most speech data. Other folders with varying amounts of speech data include "16kHz_16bit/","32kHz_16bit/","44.1kHz_16bit/","8kHz_16bit/". Just exchange the text and you will get the files in the other folders as well.

* German:
Download German speech from: http://www.voxforge.org/home/forums/other-languages/german/open-speech-data-corpus-for-german
This has been structured so that if the zipfile is extracted, tons of memory will be used up. Unzip file somewhere with sufficient memory. I think it needs a total of 20GiB.

2) Structure the speech files so that in your cwd all the English (or whatever language files) are located in a subdirectory called 'English' and German files in a 'German' subdirectory. The name of each subdirectory will be used as the categorical label for the speech files within them. (It doesn't matter if the wave/zip files are in additional subdirectories within the language (i.e. 'English', 'German') folder, just as long as they are within their corresponding language directory.)

Note: when extracting MFCCs, this script expects a one-to-one ratio of speaker and wavefile/tgzfile. It assigns each wavefile and tgzfile to either the train, validate, or test data sets. This is to avoid mixing a speaker's data between groups. If this doesn't work with your data, ignore the column 'dataset'. Another way you could create train, validate, and test datasets is to create them as subdirectories of the cwd (i.e. 'English_train', 'English_test') and place the corresponding audio files into those subdirectories.

3) *Before* running the script 'speech2mfcc_w_wo_noise_wav_tgz.py':
* Check the global variables, i.e. database name, the noise group label, etc.
* If you want to apply noise, have a noise wavefile in the cwd and input that into the script
* Depending on how the script is set up, it expects a background noise wavefile to apply background noise to the training data. For now it just uses one file from the cwd and pulls random snippets of the noise to apply to the speech data.
* For tgz zip files, the script unzips them in /tmp/audio and deletes the extracted files immediately after collecting the MFCCs. Just FYI.

4) Run  in the script in cwd. The MFCCs (if with noise, at varying levels) will be saved in a database. 

With a lot of speech data, this will take several hours.

5) *Before* running the script 'train_ann_mfcc_basic.py':
* Check global variables, i.e. database name, table name, batchsize, epochs, and the name the model should be saved under.

6) run 'train_ann_mfcc_basic.py' in same directory as the database

### Prerequisites

You need a machine with at least 20GiB of free memory (I would aim for more). Make sure you can let this machine run for several hours for each step for each language (i.e. 1) download 2) extract MFCCs 3) train models)

For required installations (i.e. versions used building this), please refer to the installations.md file.

### Installing

To start a virtual environment:
```
$ python3 -m venv env
$ source env/bin/activate
```

Then install dependencies via pip:
```
(env)...$ pip install numpy, pandas, librosa, pympler, keras, tensorflow
```

After checking the global variables, run speech2mfcc_w_wo_noise_wav_tgz.py in cwd where subdirectories of wave and zip files are located
```
(env)...$ python3 speech2mfcc_w_wo_noise_wav_tgz.py
```

After checking the global variables, run train_ann_mfcc_basic.py in cwd where mfcc database is located
```
(env)...$ python3 train_ann_mfcc_basic.py
```

To deactivate the environment:
```
(env)...$ deactivate
```

## ToDo
* Use MFCCs to train algorithms
* limit number of rows pulled out for training --> set up batch training 
* print "are these variables correct?" (global variables) when asking for input (checking whether or not someone has put in the right global variables)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* VoxForge, although I do hope for more speech databases that offer a lot of clinical speech data as well as even ratios of male and female voices.
