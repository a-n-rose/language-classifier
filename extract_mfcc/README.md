## Extract MFCCs

This prepares speech data to train classification models. MFCCs are extracted from audio files (in either .wav or .tgz format). For increased robustness, background noise(s) can be applied to the speech data. 

Additionally, each wavefile/tgzfile is assigned to a train, validate, or test 'dataset' (saved as 1,2,3 in 'dataset' column). This is to avoid too much mixing of single speakers in train, validate, and test datasets and exists only to provide additional options for data analysis, ideally making the creation of train/test or train/validat/test datasets easier down-the-road. Note: these assignments are made with 70-15-15 ratios, respectively (i.e. appx 70% files assigned to train, 15% to validate, 15% to test).

### Getting Started

If you want to include background noise, save a wavefile of the desired noise where you run the script. The longer and more diverse the noise, the higher the likelihood the model trained on the output will be robust. 

There are three 'noisegroup' variables:

1) 'none'

Enter None for the variable 'environment_noise'. No wavefile will be loaded and applied to your MFCC data.

2) 'matched'

Enter the wavefile name for the variable 'environment_noise'. This wavefile should contain noise that 'matches' the environment where additional recordings will be made that the classifier should classify.

3) 'random'

Enter the wavefile name for the variable 'environment_noise'. This wavefile should contain a series of random background noises. If you do not know the environment in which recordings will be made for the classifer to classify, this is the best option.

Run the script in the same directory as the wavefile (if you want noise) and subdirectories of the different classes of speech (e.g. languages, healthy vs clinical).

```
$ python3 speech2mfcc_w_wo_noise_wav_tgz.py
```

This will extract MFCCs and save them to a database called 'sp_mfcc' (or whatever name you'd like) in the table 'mfcc_40' (or whatever name you'd like).
