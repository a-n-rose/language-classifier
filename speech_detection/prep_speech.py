import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path

class Voice_Start_Finish():
    
    def __init__(self, wave_input_filename):
        self._read_wav(wave_input_filename)._convert_to_mono()
        
        self.sample_window = 0.025 #25 ms
        self.sample_overlap = 0.01 #10ms
        self.speech_window = 0.5 #half a second
        self.speech_energy_threshold = 0.6 #60% of energy in voice band
        self.get_stft()
        self.get_energy()
        self.get_energy_mean()
        self.get_power()
        self.get_power_mean()
        self.sound_index(start=True)
        self.sound_index(start=False)
        self.speech_start_time, self.speech_signal_start = self.index2time(self.speech_energy_start)
        self.speech_end_time, self.speech_signal_end = self.index2time(self.speech_energy_end)
        self.signal_length = len(self.data)/self.rate
        self.speech_data = self.data[self.speech_signal_start:self.speech_signal_end]
        self.speech_length = len(self.speech_data)/self.rate
        
        
    def _read_wav(self, wave_file):
        self.data, self.rate = librosa.load(wave_file,sr=None)
        self.channels = len(self.data.shape)
        self.filename_path = wave_file
        self.filename = Path(wave_file).name
        return self
    
    def _convert_to_mono(self):
        if self.channels == 2 :
            self.data = np.mean(self.data, axis=1, dtype=self.data.dtype)
            self.channels = 1
        return self
    
    def get_stft(self):
        stft = librosa.stft(self.data,hop_length=int(0.01*self.rate),n_fft=int(0.02*self.rate))
        self.stft = np.transpose(stft)
        return self

    def get_power(self):
        stft = self.stft.copy()
        self.power = np.abs(stft)**2
        return self

    def get_energy(self):
        #stft.shape[1] == bandwidths/frequencies
        #stft.shape[0] pertains to the time domain
        rms_list = [np.sqrt(sum(np.abs(self.stft[row])**2)/self.stft.shape[1]) for row in range(len(self.stft))]
        self.energy = rms_list
        return self

    def get_energy_mean(self):
        self.energy_mean = sum(self.energy)/len(self.energy)
        return self
    
    def get_power_mean(self):
        self.power_mean = sum(self.power)/len(self.power)
        return self

    def index2time(self,index):
        percentile = index/(len(self.energy))
        signal_index = int(percentile*len(self.data))
        time = (len(self.data)*percentile)/self.rate
        
        return time, signal_index
        
    def suspended_energy(self,row,start):
        if start == True:
            if self.energy[row+1] and self.energy[row+2] and self.energy[row+3] > self.energy_mean:
                return True
        else:
            if self.energy[row-1] and self.energy[row-2] and self.energy[row-3] > self.energy_mean:
                return True

    def sound_index(self,start = True):
        if start == True:
            side = 1
            beg = 0
            end = len(self.energy)
        else:
            side = -1
            beg = len(self.energy)-1
            end = -1
        for row in range(beg,end,side):
            if self.energy[row] > self.energy_mean:
                if self.suspended_energy(row,start=start):
                    if start==True:
                        #to catch plosive sounds
                        while row >= 0:
                            row -= 1
                            row -= 1
                            if row < 0:
                                row = 0
                            break
                        self.speech_energy_start = row
                        return self
                    else:
                        #to catch quiet consonant endings
                        while row <= len(self.energy):
                            row += 1
                            row += 1
                            if row > len(self.energy):
                                row = len(self.energy)
                            break
                        self.speech_energy_end = row
                        return self
        else:
            print("No speech detected.")
        return self
    
    def plot_original_signal(self):
        data = self.data
        length = self.signal_length
        plt.plot(data)
        plt.title("AUDIO ({})".format(self.filename))
        plt.xlabel("samples across time (0 - {} sec)".format(round(length,1)))
        plt.ylabel("amplitude")
        plt.savefig("signal_original_{}.png".format(self.filename))
        plt.gcf().clear()
        return None
    
    def plot_speechonly_signal(self):
        data = self.speech_data
        length = self.speech_length
        plt.plot(data)
        plt.title("AUDIO - beginning and ending silences removed ({})".format(self.filename))
        plt.xlabel("samples across time (0 - {} sec)".format(round(length,1)))
        plt.ylabel("amplitude")
        plt.savefig("signal_speechonly_{}.png".format(self.filename))
        plt.gcf().clear()
        return None
    
    def plot_energy(self):
        data = self.energy
        length = self.signal_length
        plt.plot(data)
        plt.title("ENERGY ({})".format(self.filename))
        plt.xlabel("samples across time (0 - {} sec)".format(round(length,1)))
        plt.ylabel("energy")
        plt.savefig("energy_{}.png".format(self.filename))
        plt.gcf().clear()
        return None
    
    def plot_speechonly_energy(self):
        data = self.energy[self.speech_energy_start:self.speech_energy_end]
        length = self.speech_length
        plt.plot(data)
        plt.title("ENERGY - beginning and ending silences removed ({})".format(self.filename))
        plt.xlabel("samples across time (0 - {} sec)".format(round(length,1)))
        plt.ylabel("energy")
        plt.savefig("energy_speechonly_{}.png".format(self.filename))
        plt.gcf().clear()
        return None
    
    def plot_power(self):
        data = self.power
        length = self.signal_length
        plt.plot(data)
        plt.title("POWER ({})".format(self.filename))
        plt.xlabel("samples across time (0 - {} sec)".format(round(length,1)))
        plt.ylabel("power")
        plt.savefig("power_{}.png".format(self.filename))
        plt.gcf().clear()
        return None
    
    def plot_speechonly_power(self):
        data = self.power[self.speech_energy_start:self.speech_energy_end]
        length = self.speech_length
        plt.plot(data)
        plt.title("POWER - beginning and ending silences removed ({})".format(self.filename.upper()))
        plt.xlabel("samples across time (0 - {} sec)".format(round(length,1)))
        plt.ylabel("power")
        plt.savefig("power_speechonly_{}.png".format(self.filename))
        plt.gcf().clear()
        return None
