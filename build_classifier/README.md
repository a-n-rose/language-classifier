## Build Classifier

Train models on MFCC data to build classifier. Run the script in the same directory as the database. Be sure to check the global variables (especially the name the model will be saved under) before running. 

### Things to try out
* How many MFCCs work best?
Some researchers include all 40 MFCCs and others fewer: 20 or only 12-13. The first coefficient correllates with the amplitudes of the audio files - you can see if it's good to include that or not. The first 13 coefficients are most relevant for human hearing/processing. Does it help or hinder the model if more than 13 coefficients are used? Play around.

* Train-Validate-Test datasets
Does it help if the 'dataset' column is used to designate train and test datasets? Or is there no difference if you create them via sklearn?
