## English Speech Recognition to International Phonetic Alphabet (IPA)

This is an experiment to see if one can create a simple speech recognition system using Voxforge speech and non-aligned annotation data. 

These scripts do the following:
1) collect the annotations of each wave file (within a tgz file)
2) translates those annotations to IPA
3) collect 40 MFCCs of each wave file (at windows of 25ms and shifts of 10ms; only for when speech is detected)
4) links the IPA characters to the MFCCs by dividing the length of MFCC samples by the total IPA characters (w irrelevant characters removed, i.e. spaces and '\n')
5) saves IPA and MFCC data in new table where each sample of 20 MFCCs is labeled by a series of 3 IPA characters (identified by ints)
6) prepares the data to be used for training: also calculates the number of classes for the model to use, i.e. all possible combinations of IPAs in sets of threes.

I hope this makes sense.. I will explain this in more detail in my <a href="">blog</a>.

## Requirements
* Espeak
* All other requirements for the language classifier (e.g. Numpy, Pandas, Keras, etc.)

## ToDo
* Improve progress documentation in commandline
* Improve logging and printing statements in general 
* Add embedding layer
* Apply models to new "real-world" speech (not from where original data was pulled)
