'''
Helpful resource:
http://adventuresinmachinelearning.com/keras-lstm-tutorial/
'''

import numpy as np
from keras.utils import to_categorical

class KerasBatchGenerator(object):
    def __init__(self,data_x,data_y,num_steps,batch_size_model,num_features,num_output_labels,skip_step):
        self.data_x = data_x
        self.data_y = data_y
        self.num_steps = num_steps #number of sample sequences (i.e. 20 MFCC samples in a sequence --> 1 three-character-ipa label)
        self.batch_size_model = batch_size_model
        self.num_features = num_features
        self.num_output_labels = num_output_labels #right now: 1 label (a three character IPA label)
        self.current_idx = 0
        self.skip_step = skip_step
        
        
    def generate(self):
        x = np.zeros((self.batch_size_model,self.num_steps,self.num_features))
        y = np.zeros((self.batch_size_model,self.num_steps,self.num_output_labels))
        while True:
            for i in range(self.batch_size_model):
                if self.current_idx + self.batch_size_model >= len(self.data_x):
                    self.current_idx = 0
                x[i,:,:] = self.data_x[self.current_idx]
                y[i,:,:] = self.data_y[self.current_idx]
                self.current_idx += self.skip_step
            yield x,y
                
                

                            
