import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from scipy.ndimage import convolve1d
import wandb

def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, btype='low', analog=False)

def lowpass_filter(data, cutoff=4, fs=30.0, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data, axis=0)

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

def preprocess_ultrasound_pipeline(data_root, prefix, cutoff_freq=4):
    us_path = os.path.join(data_root, f"{prefix}_raw_US.npy")
    angles_path = os.path.join(data_root, f"{prefix}_leap_angles.npy")

    if not os.path.exists(us_path) or not os.path.exists(angles_path):
        raise FileNotFoundError(f" Missing files: {us_path} or {angles_path}")

    raw_us = np.load(us_path)
    angles = np.load(angles_path)

    if raw_us.ndim != 4 or raw_us.shape[1] != raw_us.shape[2]:
        raise ValueError(f" Invalid raw_us shape: {raw_us.shape}")

    frames = min(raw_us.shape[0], angles.shape[0])
    channels = raw_us.shape[1]
    time_samples = raw_us.shape[3]

    us_3d = np.zeros((frames, channels, time_samples))
    for c in range(channels):
        us_3d[:, c, :] = raw_us[:frames, c, c, :]

    # ===== Visualization (only once, frame 100, channel 4) =====
    if prefix.endswith("0"):  # ensure only visualize one trial to avoid redundancy
        frame_idx, channel_idx = 100, 4
        raw_signal = us_3d[frame_idx, channel_idx, :]

        step1 = time_gain_compensation(raw_signal[None, None, :])[0, 0]
        step2 = gaussian_smooth(step1[None, None, :])[0, 0]
        step3 = apply_hilbert(step2[None, None, :], pad_len=100)[0, 0]
        step4 = apply_log_compression(step3[None, None, :])[0, 0]
        step5 = step4[200:-20]

        plt.figure(figsize=(12, 10))
        plt.subplot(6, 1, 1)
        plt.plot(raw_signal)
        plt.title("Raw Signal (Channel 4, Frame 100)")

        plt.subplot(6, 1, 2)
        plt.plot(step1)
        plt.title("After Time Gain Compensation")

        plt.subplot(6, 1, 3)
        plt.plot(step2)
        plt.title("After Gaussian Smoothing")

        plt.subplot(6, 1, 4)
        plt.plot(step3)
        plt.title("After Hilbert Transform (Envelope)")

        plt.subplot(6, 1, 5)
        plt.plot(step4)
        plt.title("After Log Compression")

        plt.subplot(6, 1, 6)
        plt.plot(step5)
        plt.title("Final Cropped Signal")

        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        save_path = os.path.join("results", "preprocessing_pipeline_visualization.png")
        plt.savefig(save_path, dpi=300)
        wandb.log({"Preprocessing Visualization": wandb.Image(save_path)})
        plt.close()
        print(" Visualization saved and uploaded to WandB.")

    # ===== Full Pipeline =====
    us_3d = time_gain_compensation(us_3d)
    us_3d = gaussian_smooth(us_3d)
    us_3d = apply_hilbert(us_3d, pad_len=100)
    us_3d = apply_log_compression(us_3d)
    us_3d = us_3d[:, :, 200:-20]  # crop

    #  Lowpass filtering on joint angles only
    angles_filtered = lowpass_filter(angles[:frames, :2], cutoff=cutoff_freq, fs=30.0)

    print(f" Trial {prefix} preprocessed successfully.")
    return us_3d, angles_filtered
