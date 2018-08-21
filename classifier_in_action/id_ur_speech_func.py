
import glob
import os
import datetime
import random
import numpy as np
import sounddevice as sd
import soundfile as sf
import pygame
import librosa



class ID_UR_Speech():
    def __init__(self,date):
        print(
            '''
            Welcome! 
            
            Let's see if I can guess which type of speak you spoke.
            '''
            )
        self.cont = True
        self.date = date
        self.language = None
        
    def start_action(self,action):
        print("Press ENTER to {} or type 'exit' to leave: ".format(action))
        user_ready = input()
        if user_ready == '':
            print("Great!")
            return True
        elif 'exit' in user_ready.lower():
            return False
        else:
            print("Sorry, I didn't understand that..")
            self.start_action(action)
            
    def record_user(self,duration):
        duration = duration
        fs = 22050
        user_rec = sd.rec(int(duration*fs),samplerate=fs,channels=1)
        sd.wait()   
        return(user_rec)
    
    def check_rec(self,user_rec):
        '''
        Need to see if recording worked and meausre the amount of background noise
        '''
        if user_rec.any():
            return True
        return False
    
    def play_rec(self,recording):
        fs = 22050
        sd.play(recording, fs)
        sd.wait()
        return None
    
    def test_record(self,sec):
        '''
        The user will need to do a test record to analyze natural voice.
        Perhaps read a sentence aloud?
        '''

        user_rec = self.record_user(sec)

        if self.check_rec(user_rec):
            filename = './user_recordings/background_{}.wav'.format(self.date)
            self.noisefile = filename
            self.save_rec(filename,user_rec,fs=22050)
            return user_rec
        else:
            print(
                    '''
                    Hmmmmmm there seems to be a problem.
                    Is your mic connected and/or activated?
                    
                    Sorry for the inconvenience.
                    '''
                    )
        
        return None
    
    
    def test_mic(self,sec):
        user_rec = self.test_record(sec)
        sd.wait()
        if user_rec.any():
            sd.wait()
            print("Thanks!")
            #self.play_rec(user_rec)
            #sd.wait()
            return None
        else:    
            print("Hmmmmm.. something went wrong. Check your mic and try again.")
            if self.start_action('test your mic'):
                self.test_mic(sec)
            else:
                return False
            
            
    def play_go(self):
        pygame.init()
        topic = random.choice(["animals you think are really cool","great color combinations","badass storms you've experienced","a time someone was really nice"])
        print("Start talking for one minute after the tone. \n\nOptional topic: {}.\n\n".format(topic))
        go_wave = '231277__steel2008__race-start-ready-go.wav'
        go_sound = pygame.mixer.Sound('soundfiles/{}'.format(go_wave))
        go_sound.play()
        while pygame.mixer.get_busy():
            pass
        print("And... talk away!")
        return None
    
    def save_rec(self,filename,rec,fs):
        sf.write(filename,rec,fs)
        return None
    
    
    def close_game(self):
        '''
        close and save anything that was open during the game
        '''
        pygame.quit()
            

    def wave2stft(self,wavefile):
        y, sr = librosa.load(wavefile)
        if len(y)%2 != 0:
            y = y[:-1]
        stft = librosa.stft(y)
        stft = np.transpose(stft)
        return stft, y, sr

    def stft2wave(self,stft,len_origsamp):
        istft = np.transpose(stft.copy())
        samples = librosa.istft(istft,length=len_origsamp)
        return samples

    def stft2power(self,stft_matrix):
        stft = stft_matrix.copy()
        power = np.abs(stft)**2
        return(power)

    def get_energy(self,stft_matrix):
        #stft.shape[1] == bandwidths/frequencies
        #stft.shape[0] pertains to the time domain
        rms_list = [np.sqrt(sum(np.abs(stft_matrix[row])**2)/stft_matrix.shape[1]) for row in range(len(stft_matrix))]
        return rms_list

    def get_energy_mean(self,energy_list):
        mean = sum(energy_list)/len(energy_list)
        return mean

    def get_pitch(self,wavefile):
        y, sr = librosa.load(wavefile)
        if len(y)%2 != 0:
            y = y[:-1]
        pitches,mag = librosa.piptrack(y=y,sr=sr)
        return pitches,mag

    def get_pitch2(self,y,sr):
        pitches,mag = librosa.piptrack(y=y,sr=sr)
        return pitches,mag

    def get_pitch_mean(self,matrix_pitches):
        p = matrix_pitches.copy()
        p_mean = [np.mean(p[:,time_unit]) for time_unit in range(p.shape[1])]
        p_mean = np.transpose(p_mean)
        #remove beginning artifacts:
        pmean = p_mean[int(len(p_mean)*0.07):]
        return pmean
                
    def pitch_sqrt(self,pitch_mean):
        psqrt = np.sqrt(pitch_mean)
        return psqrt
        
    def get_mean_bandwidths(self,matrix_bandwidths):
        bw = matrix_bandwidths.copy()
        bw_mean = [np.mean(bw[:,bandwidth]) for bandwidth in range(bw.shape[1])]
        return bw_mean

    def get_var_bandwidths(self,matrix_bandwidths):
        if len(matrix_bandwidths) > 0:
            bw = matrix_bandwidths.copy()
            bw_var = [np.var(bw[:,bandwidth]) for bandwidth in range(bw.shape[1])]
            return bw_var
        return None

    def rednoise(self,noise_powerspec_mean,noise_powerspec_variance, speech_powerspec_row,speech_stft_row):
        npm = noise_powerspec_mean
        npv = noise_powerspec_variance
        spr = speech_powerspec_row
        stft_r = speech_stft_row.copy()
        for i in range(len(spr)):
            if spr[i] <= npm[i] + npv[i]:
                stft_r[i] = 1e-3
        return stft_r


    def suspended_energy(self,rms_speech,row,rms_mean_noise,start):
        if start == True:
            if rms_speech[row+1] and rms_speech[row+2] > rms_mean_noise:
                return True
        else:
            if rms_speech[row-1] and rms_speech[row-2] > rms_mean_noise:
                return True


    def sound_index(self,rms_speech, start = True, rms_mean_noise = None):
        if rms_mean_noise == None:
            rms_mean_noise = 1
        if start == True:
            side = 1
            beg = 0
            end = len(rms_speech)
        else:
            side = -1
            beg = len(rms_speech)-1
            end = -1
        for row in range(beg,end,side):
            if rms_speech[row] > rms_mean_noise:
                if suspended_energy(rms_speech,row,rms_mean_noise,start=start):
                    if start==True:
                        #to catch plosive sounds
                        while row >= 0:
                            row -= 1
                            row -= 1
                            if row < 0:
                                row = 0
                            break
                        return row,True
                    else:
                        #to catch quiet consonant endings
                        while row <= len(rms_speech):
                            row += 1
                            row += 1
                            if row > len(rms_speech):
                                row = len(rms_speech)
                            break
                        return row,True
        else:
            print("No speech detected.")
        return beg,False

            
    def savewave(self,filename,samples,sr):
        librosa.output.write_wav(filename,samples,sr)
        print("File has been saved")
            
            
    def reduce_noise(self,wavefile,background_noise):
        y_stft, y, sr = self.wave2stft(wavefile)
        y_power = self.stft2power(y_stft)
        y_energy = self.get_energy(y_stft)
        n_stft, ny, nsr = self.wave2stft(noise)
        n_power = self.stft2power(n_stft)
        n_energy = self.get_energy(n_stft)
        n_energy_mean = self.get_energy_mean(n_energy)
        
        npow_mean = self.get_mean_bandwidths(n_power)
        npow_var = self.get_var_bandwidths(n_power)
        
        y_stftred = np.array([self.rednoise(npow_mean,npow_var,y_power[i],y_stft[i]) for i in range(y_stft.shape[0])])
        
        voice_start,voice = self.sound_index(y_energy,start=True,rms_mean_noise = n_energy_mean)
        if voice:
            print(voice_start)
            print(voice_start/len(y_energy))
            start = voice_start/len(y_energy)
            start_time = (len(y)*start)/sr
            print("Start time: {} sec".format(start_time))
            y_stftred = y_stftred[voice_start:]
            voicestart_samp = self.stft2wave(y_stftred,len(y))
            date = self.date
            self.savewave('./processed_recordings/rednoise_speechstart_{}.wav'.format(date),voicestart_samp,sr)
            print('Removed silence from beginning of recording. File saved.')
            
        else:
            #handle no speech in recording, or too much background noise
            print("No speech detected.")
            return False
        
        rednoise_samp = self.stft2wave(y_stftred,len(y))
        date = self.date
        self.savewave('./processed_recordings/rednoise_{}.wav'.format(self.date),rednoise_samp,sr)
        print('Background noise reduction complete. File saved.')
        return True
