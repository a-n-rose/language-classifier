## Voice Activity Detection Scripts

These scripts allow you to check for voice activity. You should also be able to plot (by saving the plot as .png) the signal, energy, and power of an audio signal, both of the original signal and also of the signal with silences removed. To see my blog on this, click <a href="https://a-n-rose.github.io/2018/09/06/updating-VAD.html">here</a>.

As of now, this script provides a simple VAD and does not remove silences throughout a recording, just the beginning and ending silences. **Speech (or sound in general) is detected if the energy level of a sample is greater than the mean energy of the signal, for a consecutive three samples.**

The audio signal is plotted by loading the wav via librosa, and plotting the sample values. Power was calculated by first taking the absolute values of the stft, then squaring them. Those values were used for plotting the power. The energy values were calculated by taking the square root of the mean of power values:

<a href="https://www.codecogs.com/eqnedit.php?latex=energy&space;=&space;\sqrt(\sum(power_i)/&space;P)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?energy&space;=&space;\sqrt(\sum(power_i)/&space;P)" title="energy = \sqrt(\sum(power_i)/ P)" /></a>

Those values were used for plotting the energy. 

### Getting Started
Simply indicate the path to the wavefile you want to examine/apply speech detection to in the script 'remove_silence.py'

Make sure 'prep_speech.py' is in the same folder as 'remove_silence.py'

Then run it:

```
$ python3 remove_silence.py
```

It will save plots of the original wavefile as well as of the wave with the beginning and ending silences removed. If successful, the following will print on your screen:

```
Plotting of original and speech only data successfully completed and saved.
```

Then you can have a look at all the pretty plots.

### Prerequisites

* numpy
* matplotlib
* librosa

### Installing

To set up a virtual environment, in the directory you have your project in, type the following into your commandline:
```
python -m venv env
source env/bin/activate
```
Your virtual environment should be activated and you can install all the dependencies via:
```
pip install numpy, matplotlib, librosa
```
To deactivate the virtual environment, just type:
```
deactivate
```

## ToDo
* add functionality to remove silence throughout recording

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


