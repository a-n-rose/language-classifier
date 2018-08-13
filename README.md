# Language Classifier
A language classifier model built via deep learning techniques

## Getting Started

1) For speech file collection, voxforge.org is a great option. Many languages are available. For building the classifer, only English and German were used (additional languages are in the process of being added to the model). To download the English and German speech files, below are instructions, respectively. Note: each of these steps take several hours (download files; exctract MFCC data; train models with MFCC data)

* English:
Enter the following command into your commandline, in the directory where you'd like the speech files to be saved. This will download the files into a zipfile:
```
wget -r -A.tgz http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Original/48kHz_16bit/s
```
Note: there are several folders with speech data: the above folder is "48kHz_16bit" and has the most speech data. Other folders with varying amounts of speech data include "16kHz_16bit/","32kHz_16bit/","44.1kHz_16bit/","8kHz_16bit/". Just exchange the text and you will get the files in the other folders as well.

* German:
Downloaded German speech from: http://www.voxforge.org/home/forums/other-languages/german/open-speech-data-corpus-for-german
This has been structured so that if the zipfile is extracted, tons of memory will be used up. Unzip file somewhere with sufficient memory. I think it needs a total of 20GiB.

2) Structure the speech files so that in your cwd all the English (or whatever language files) are located in a subdirectory called 'English' and German files in a 'German' subdirectory. The name of each subdirectory will be used as the categorical label for the speech files within them. 

3) *Before* running the script 'speech2mfcc_w_wo_noise_wav_tgz.py':
* Check the global variables, i.e. database name, the noise group label, etc.
* If you want to apply noise, have a noise wavefile in the cwd and input that into the script
* Depending on how the script is set up, it expects a background noise wavefile to apply background noise to the training data. For now it just uses one file from the cwd and pulls random snippets of the noise to apply to the speech data.
* For tgz zip files, the script unzips them in /tmp/audio and deletes the extracted files immediately after collecting the MFCCs. Just FYI.

4) Run  in the script in cwd. The MFCCs (if with noise, at varying levels) will be saved in a database. 

With a lot of speech data, this will take several hours.


### Prerequisites

You need a machine with at least 20GiB of free memory (I would aim for more). Make sure you can let this machine run for several hours for each step for each language (i.e. 1) download 2) extract MFCCs 3) train models)

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

Run matchbackground2mfcc.py in cwd where subdirectories of wave and zip files are located
```
python3 speech2mfcc_w_wo_noise_wav_tgz.py
```

To deactivate the environment:
```
deactivate
```

## ToDo
* Use MFCCs to train algorithms

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* VoxForge, although I do hope for more speech databases that offer a lot of clinical speech data as well as even ratios of male and female voices.
