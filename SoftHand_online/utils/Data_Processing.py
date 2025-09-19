import numpy as np
from scipy.signal import hilbert
from scipy.signal import butter, lfilter, sosfiltfilt
from scipy.interpolate import interp1d


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter(order, [lowcut,highcut], fs=fs, btype='bandpass',output="sos")
    return sosfiltfilt(sos, data, axis=3)

def Time_Gain_Compensation(US, sp_freq, centre_freq , coef_att):
    Sequence_depth = np.arange(0, US.shape[3], 1) * (1/sp_freq) * (1e2 * 1540 / 2)
    US *= np.exp(coef_att * (centre_freq/1e6) * Sequence_depth)
    return US

def Arange_resolution(US, n):
    return np.mean(US[:,:,:,:].reshape(US.shape[0], US.shape[1], US.shape[2], -1, n),axis=4)

def Envelop(US):
    return np.absolute(hilbert(US, axis=3))

def LogCompress(US, a):
    return np.log10(1 + a * US) / np.log10(1 + a)

def LogCompress2(US):
    US = US / np.amax(US)
    return 20* np.log10(US)

def to_pixels(US):
    m = interp1d([np.amin(US),np.amax(US)], [0,255])
    US = m(US)
    return US

def mask(US):
    for x in range(US.shape[0]):
        start = np.argmax(US[x,10:] > 50)
        US[x,:10+start] = 0
    return US

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx