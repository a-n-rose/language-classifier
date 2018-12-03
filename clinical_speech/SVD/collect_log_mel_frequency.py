 
import os
import glob
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


#first just working with healthy and dysphonic speech from  men
#Saarbrucken Voice Database

#total number: 300
healthy_data_location = "./speech_data/healthy_dysphonia/healthy/sentences/"
#total number: 29
dysphonia_data_location = "./speech_data/healthy_dysphonia/dysphonia/sentences/"

healthy_test = healthy_data_location+"61-phrase.wav"
dysphonia_test = dysphonia_data_location+"2323-phrase.wav"

[y_dysphonia, sr]  = librosa.load(dysphonia_test, sr = 16000)

[y_healthy, sr]  = librosa.load(healthy_test, sr = 16000)

#following settings from article:
#Dysarthric Speech Recognition Using Convolutional LSTM Neural Network
specto = librosa.feature.melspectrogram(y_healthy,sr=sr,hop_length=int(0.010*sr),n_fft=int(0.025*sr))

#need to take 1st and 2nd derivatives
#To calculate delta (first derivative) or delta delta (second derivative) you have to calculate the rate of change and acceleration changes basing on MFCC.

#first derivative = delta (rate of change)
specto_delta = librosa.feature.delta(specto)

#second derivative = delta delta (acceleration changes)
specto_delta_delta = librosa.feature.delta(specto,order=2)

log_specto = librosa.amplitude_to_db(specto)
log_specto_delta = librosa.amplitude_to_db(specto_delta)
log_specto_delta_delta = librosa.amplitude_to_db(specto_delta_delta)

plt.subplot(3,1,1)
librosa.display.specshow(log_specto,sr=sr,x_axis='time',y_axis='mel',hop_length=int(0.010*sr))

#plt.figure(figsize=(12,4))
plt.title("mel power spectrogram")
plt.colorbar(format="%+02.0f dB")

plt.subplot(3,1,2)
librosa.display.specshow(log_specto_delta,sr=sr,x_axis='time',y_axis='mel',hop_length=int(0.010*sr))
plt.title("delta spectrogram")
plt.colorbar(format="%+02.0f dB")

plt.subplot(3,1,3)
librosa.display.specshow(log_specto_delta_delta,sr=sr,x_axis='time',y_axis='mel',hop_length=int(0.010*sr))
plt.title("delta delta spectrogram")
plt.colorbar(format="%+02.0f dB")

plt.tight_layout()

plt.show()

#print(specto.shape)

#print(log_specto.shape)
