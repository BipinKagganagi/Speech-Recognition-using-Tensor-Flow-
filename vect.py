
# coding: utf-8

# In[1]:


import os
import wavio
import tensorflow as tf
import numpy as np
import pandas as p
import matplotlib as mp
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import scipy.io.wavfile
from scipy.fftpack import dct


# In[2]:


def get_vect(file_name):
    sample_rate, signal = scipy.io.wavfile.read(file_name)
    #C:\Users\crameshb\Documents\Neural AI\Speech\train\audio\offprint(signal)
    signal = signal[0:int(1 * sample_rate)]




    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])





    frame_size = 0.025
    frame_stride = 0.01

    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]


    frames *= np.hamming(frame_length)

    N = 512
    mag_frames = np.absolute(np.fft.rfft(frames, N))
    pow_frames = ((1.0 / N) * ((mag_frames) ** 2))

    nfilt = 40
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((N + 1) * hz_points / sample_rate)
    fbank = np.zeros((nfilt, int(np.floor(N/2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
    num_ceps = 12
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
    filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    """
    x = mfcc
    x_flat = x.flatten(order = 'C')
    x_flat = x_flat.reshape((x_flat.shape[0], 1))
    x_T = x_flat.transpose()
    """
   # mn_pow_frames,mn_filter_banks,mn_mfcc = np.median(pow_frames),np.median(filter_banks),np.median(mfcc)
    return(mfcc)
    """
    x = mn_pow_frames,mn_filter_banks,mn_mfcc
    x = np.array(x)
    x = x.reshape((x.shape[0], 1))
    x = x.transpose()
    return(x)
"""
     


# In[5]:


#y = get_vect('C:/Users/crameshb/Documents/Neural AI/Speech/train/audio/bed/1b63157b_nohash_0.wav')


# In[21]:


#get_vect('C:/Users/crameshb/Documents/Neural AI/Speech/train/audio/sheila/0d53e045_nohash_0.wav')

