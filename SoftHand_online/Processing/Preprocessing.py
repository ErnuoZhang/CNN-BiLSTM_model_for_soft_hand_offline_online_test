import numpy as np
import numpy.typing as npt
from scipy.signal import hilbert, decimate
from scipy.signal import butter, lfilter, sosfiltfilt
from scipy.interpolate import interp1d

# Ultrasound preprocessing Utilities
def butter_bandpass_filter(data: npt.ArrayLike, lowcut: float, highcut: float, fs: float, order: int = 5) -> np.ndarray:
    """
    Simple Butterworth bandpass filter for Raw US data. RF line should be on axis 3
    """
    sos = butter(order, [lowcut,highcut], fs=fs, btype='bandpass', output="sos")
    return sosfiltfilt(sos, data, axis=3)

def arange_resolution(US: npt.ArrayLike, n: int) -> np.ndarray:
    """
    Downsamples Raw US RF line sampling rate by averaging n points in non overlaping windows.
    RF line should be on axis 3. n must be divisible by the initial sample size
    """
    assert(US.shape[2] % n == 0), " RF line sample size and n must be divisable"
    return np.mean(US[:,:,:].reshape(US.shape[0], US.shape[1], -1, n), axis=3)

def envelop(US: npt.ArrayLike, pad: bool = False) -> np.ndarray:
    """
    Hilbert envelop of Raw US RF data. RF line should be on axis 3.
    Can apply constant padding on the edges to eliminate enveloping artifact on the borders.
    """
    if pad:
        US = np.pad(US, [(0,0), (0,0), (pad,pad)], mode='constant', constant_values=0)
        return np.absolute(hilbert(US, axis=2))[:,:,pad:int(-pad)]
    else:
        return np.absolute(hilbert(US, axis=2))
    
def log_compress_toDb(US: npt.ArrayLike) -> np.ndarray:
    """
    Standart log compress. a is the compression factor
    """
    return 20* np.log10(US / np.amax(US))

def log_compress(US: npt.ArrayLike, a: float) -> np.ndarray:
    """
    Standart log compress. a is the compression factor
    """
    return np.log10(1 + a * US) / np.log10(1 + a)

def to_pixels(US: npt.ArrayLike) -> np.ndarray:
    """
    Interpolates data from original range and type to int8 for pixel intensity values [0,255]
    """
    m = interp1d([np.amin(US),np.amax(US)], [0,255])
    US = m(US)
    return US.astype(np.uint8)

# MOCAP data processing utilities
def interpolate_array(array: npt.ArrayLike, rate: int) -> np.ndarray:
    """
    linear interpolation
    """
    interp = []
    for x in range(array.shape[1]):
        print(np.linspace(0, array.shape[0], round(array.shape[0]/rate)).shape)
        print(np.arange(0, array.shape[0], 1).shape)
        print(array[:, x].shape)
        interp.append(np.interp(np.linspace(0, array.shape[0], round(array.shape[0]/rate)), np.linspace(0, array.shape[0], array.shape[0]), array[:, x]))

    return np.stack(interp, axis=0).reshape(-1, array.shape[1])

# Other utilities
def downsample_array(array: npt.ArrayLike, rate: int) -> np.ndarray:
    return array[::rate,...]

def butter_lowpass_filter(data: npt.ArrayLike, lowcut: float, fs: float, order: int = 5) -> np.ndarray:
    """
    Simple Butterworth lowpass filter for MOCAP angle data. Filtering applyed on axis 0
    """
    sos = butter(order, lowcut, fs=fs, btype='lowpass', output="sos")
    return sosfiltfilt(sos, data, axis=0)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx