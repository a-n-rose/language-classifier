#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 18:28:19 2018

@author: airos
"""

import numpy as np
import librosa
import math
import random

    
def match_length(noise,sr,desired_length):
    noise2 = np.array([])
    final_noiselength = sr*desired_length
    original_noiselength = len(noise)
    frac, int_len = math.modf(final_noiselength/original_noiselength)
    for i in range(int(int_len)):
        noise2 = np.append(noise2,noise)
    if frac:
        max_index = int(original_noiselength*frac)
        end_index = len(noise) - max_index
        rand_start = random.randrange(0,end_index)
        noise2 = np.append(noise2,noise[rand_start:rand_start+max_index])
    if len(noise2) != final_noiselength:
        diff = int(final_noiselength - len(noise2))
        if diff < 0:
            noise2 = noise2[:diff]
        else:
            noise2 = np.append(noise2,np.zeros(diff,))
    return(noise2)

def normalize(array):
    max_abs = max(abs(array))
    if max_abs > 1:
        mult_var = 1.0/max_abs
        array_norm = array*mult_var
        return(array_norm)
    else:
        return(array)

def scale_noise(np_array,factor):
    '''
    If you want to reduce the amplitude by half, the factor should equal 0.5
    '''
    return(np_array*factor)
    
   
