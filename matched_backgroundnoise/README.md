# Language Classifier
A language classifier model built via deep learning techniques

## Getting Started

For speech file collection, voxforge.org is a great option. Many languages are available. For building the classifer, only English and German were used (additional languages are in the process of being added to the model). To download the English and German speech files, below are instructions, respectively. Note: each of these steps take several hours (download files; exctract MFCC data; train models with MFCC data)

* English:
1) Enter the following command into your commandline, in the directory where you'd like the speech files to be saved. This will download the files into a zipfile:
```
wget -r -A.tgz http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Original/48kHz_16bit/s
```
Note: there are several folders with speech data: the above folder is "48kHz_16bit" and has the most speech data. Other folders with varying amounts of speech data include "16kHz_16bit/","32kHz_16bit/","44.1kHz_16bit/","8kHz_16bit/". Just exchange the text and you will get the files in the other folders as well.
2)run "MFCC13_zip_wav_sqlite3.py" in the parent directory of where the "48kHz_16bit" etc. folders are located (e.g. www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Original). This will save the wave files' MFCCs (13, at 25ms windows w 10ms shifts) to a database "sp_mfcc" in the table "mfcc_13" in the current directory, via sqlite3. Note: you will be asked for which language category the speech is. Type in "English" or whatever label you want to use for the data. This label will be combined with the wave file's directory label to keep track of which group of files it belongs to (i.e. "48kHz_16bit" vs "8kHz_16bit" in case that is relevant to your purpose). This can be used as the dependent variable/category for training a neural network.

* German:
1) Downloaded German speech from: http://www.voxforge.org/home/forums/other-languages/german/open-speech-data-corpus-for-german
This has been structured so that if the zipfile is extracted, tons of memory will be used up. Unzip file somewhere with sufficient memory. I think it needs a total of 20GiB.
2) run "matchbackground2mfcc_wav.py" script in the directory where the folders "test", "dev", and "train" are located. The wave files in each directory will be processed and their MFCCs (40) will be saved to a database "sp_mfcc", in the table "mfcc_40" via sqlite3. Note: Be aware of how the speech is labeled, i.e. which language. The filename, directory, label, noise group (whether or not noise was added) and the amount of noise will be saved in the database as well.

### Prerequisites

You need a machine with at least 20GiB of free memory (I would aim for more). Also, a GPU is ideal. (I haven't tried with a CPU and wouldn't advise it). Make sure you can let this machine run for several hours for each step for each language (i.e. 1) download 2) extract MFCCs 3) train models)

For required installations, please refer to the installations.md file.
Currently, the installations.md file contains what is necessary for the processing of audiofiles

### Installing

To start a virtual environment:
```
python3 -m venv env
source env/bin/activate
```

Then install dependencies via pip:
```
pip install _________
```

Run matchbackground2mfcc_wav.py in cwd where subdirectories of wave files are located
```
python3 matchbackground2mfcc_wav.py
```

To deactivate the environment:
```
deactivate
```

## ToDo
* fix language labeling - right now input() is unreliable, and the way the speech files are structured make using directory names as language labels difficult. Currently, one has to remember to change language label in the code... 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* VoxForge, although I do hope for more speech databases that offer a lot of clinical speech data as well as even ratios of male and female voices.
