from multiprocessing.dummy import Process
import numpy as np
import time
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert
from numpy.lib.stride_tricks import sliding_window_view
import multiprocessing as mp
from scipy.ndimage import convolve1d
from scipy.signal import butter, lfilter, sosfiltfilt

import Processing.Preprocessing


def time_gain_compensation(signal: np.ndarray, gain_factor: float = 3) -> np.ndarray:
    gain = np.linspace(1.0, gain_factor, signal.shape[-1])
    return signal * gain


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    x = np.arange(-size // 2 + 1, size // 2 + 1)
    kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel

def gaussian_smooth(signal: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    kernel = gaussian_kernel(size=31, sigma=sigma)
    return convolve1d(signal, kernel, axis=-1, mode='reflect')

def apply_hilbert(signal: np.ndarray, pad_len: int = 200) -> np.ndarray:
    signal_padded = np.pad(signal, [(0, 0), (0, 0), (pad_len, pad_len)], mode='reflect')
    envelope = np.abs(hilbert(signal_padded, axis=-1))
    return envelope[:, :, pad_len:-pad_len]

def apply_log_compression(signal: np.ndarray, a: float = 3000) -> np.ndarray:
    return np.log10(1 + a * signal) / np.log10(1 + a)


class USCruncher():
    def _Setup_USCruncher(self, nchanels, samples_per_line):
        self.us_diagonal = np.zeros((1, nchanels, samples_per_line))

 
    def Process_US(self):
        for c in range(16):
            self.us_diagonal[0, c, :] = self.US[c, c, :]

        self.us_diagonal = time_gain_compensation(self.us_diagonal)

        self.us_diagonal = gaussian_smooth(self.us_diagonal)

        self.us_diagonal = apply_hilbert(self.us_diagonal, pad_len=100)

        self.us_diagonal = apply_log_compression(self.us_diagonal)

        self.Features = self.us_diagonal[:, :, 200:-20]  # crop
        # print(self.Features[0,0,1000:1050])
        
