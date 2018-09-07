# Language Classifier
A language classifier model built via deep learning techniques. Currently I have trained a simple artificial neural network (ANN) and a long short term memory (LSTM) recurrent neural network (RNN). For reference, <a href="https://a-n-rose.github.io/2018/08/22/language-classifier.html">here</a> is my blog post documenting the development of the ANN classifier and <a href="https://a-n-rose.github.io/2018/09/07/language-classifier-LSTM.html">here</a> is one on the LSTM.

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

* Russian:
Same as English
```
$ wget -r -A.tgz http://www.repository.voxforge1.org/downloads/Russian/Trunk/Audio/Original/48kHz_16bit/
```

2) Structure the speech files so that in your cwd all the English (or whatever language files) are located in a subdirectory called 'English' and German files in a 'German' subdirectory. The name of each subdirectory will be used as the categorical label for the speech files within them. (It doesn't matter if the wave/zip files are in additional subdirectories within the language (i.e. 'English', 'German') folder, just as long as they are within their corresponding language directory.)

Note: when extracting MFCCs, this script expects a one-to-one ratio of speaker and wavefile/tgzfile. It assigns each wavefile and tgzfile to either the train, validate, or test data sets. This is to avoid mixing a speaker's data between groups. While this might not work 100%, it would at least help avoiding too much mixing (something I plan on comparing down-the-line). If this doesn't work with your data, ignore the column 'dataset'. Another way you could create train, validate, and test datasets is to create them as subdirectories of the cwd (i.e. 'English_train', 'English_test') and place the corresponding audio files into those subdirectories. 

3) *Before* running the script 'speech2mfcc_w_wo_noise_wav_tgz.py' (see folder 'extract_mfcc'):
* Check the global variables, i.e. database name, the noise group label, etc.
* If you want to apply noise, have a noise wavefile in the cwd and input that into the script
* Depending on how the script is set up, it expects a background noise wavefile to apply background noise to the training data. For now it just uses one file from the cwd and pulls random snippets of the noise to apply to the speech data.
* For tgz zip files, the script unzips them in /tmp/audio and deletes the extracted files immediately after collecting the MFCCs. Just FYI.

4) Run  in the script in cwd. The MFCCs (if with noise, at varying levels) will be saved in a database. 

With a lot of speech data, this will take several hours.

5) *Before* running the script 'train_lstm_ann_languageclassifier_simple.py' (see folder 'build_classifier'):
* Check global variables, i.e. type of neural network ('ANN' vs 'LSTM'), database name, table name, batchsize, epochs, and the name the model should be saved under.

6) run 'train_lstm_ann_languageclassifier_simple.py' in same directory as the database.

### Prerequisites

These scripts were written on a Linux machine and were not tested on others. For example, Glob is used for collection of filenames; I doubt this would work on Macs or Winows.

You need a machine with around 60GiB of free memory (I would aim for more). Make sure you can let this machine run for several hours for each step for each language (i.e. 1) download 2) extract MFCCs 3) train models). 

For required installations (i.e. versions used building this), please refer to the installations.md file.

### Installing

To start a virtual environment:
```
$ python3 -m venv env
$ source env/bin/activate
```

Then install dependencies via pip:
```
(env)...$ pip install numpy, pandas, librosa, keras, tensorflow
```

After checking the global variables, run speech2mfcc_w_wo_noise_wav_tgz.py in cwd where subdirectories of wave and zip files are located
```
(env)...$ python3 speech2mfcc_w_wo_noise_wav_tgz.py
```
I ran this script with the '48kHz_16bit' folder from the English dataset in a 'English' subdirectory and the 'train' folder from the German dataset in a 'German'subdirectory. It took this script appx. 16 hours to extract the MFCCs from these audio files.

After checking the global variables, run train_lstm_ann_languageclassifier_simple.py in cwd where mfcc database is located:
```
(env)...$ python3 train_lstm_ann_languageclassifier_simple.py
```

To deactivate the environment:
```
(env)...$ deactivate
```

## ToDo
* Set up batch training 
* Apply random noises to MFCC data (something like <a href="http://dcase.community/challenge2018/task-general-purpose-audio-tagging">this</a>?)
* Train other neural nets with data, i.e. CNN 
* Compare MFCC vs raw form vs logmelspectrum vs combination of these as training data
* Apply Keras TimeDistributed module to look at MFCC patterns over time.
* Compare Delta-Delta / differential and acceleration coefficients and Keras TimeDistributed module.
* Apply Voice Activy Detector to remove silences from speech.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* VoxForge
