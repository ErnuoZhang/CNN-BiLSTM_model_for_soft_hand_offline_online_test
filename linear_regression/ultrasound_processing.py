import os
import numpy as np
from scipy.signal import hilbert
from scipy.ndimage import convolve1d

def time_gain_compensation(signal: np.ndarray, gain_factor: float = 3) -> np.ndarray:
    """
    Apply time-based gain compensation along the last axis (time).
    signal: (B, C, T), returns float64
    """
    gain = np.linspace(1.0, gain_factor, signal.shape[-1])
    return signal * gain

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    x = np.arange(-size // 2 + 1, size // 2 + 1)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    return kernel / kernel.sum()

def gaussian_smooth(signal: np.ndarray, sigma: float = 1.0, size: int = 31) -> np.ndarray:
    """
    1D Gaussian smoothing along time axis.
    """
    kernel = gaussian_kernel(size, sigma)
    return convolve1d(signal, kernel, axis=-1, mode='reflect')

def apply_hilbert(signal: np.ndarray, pad_len: int = 100) -> np.ndarray:
    """
    Hilbert envelope along time; pad to reduce edge effects, then crop.
    signal: (B, C, T) -> (B, C, T)
    """
    signal_padded = np.pad(signal, [(0, 0), (0, 0), (pad_len, pad_len)], mode='reflect')
    envelope = np.abs(hilbert(signal_padded, axis=-1))
    return envelope[:, :, pad_len:-pad_len]

def apply_log_compression(signal: np.ndarray, a: float = 3000) -> np.ndarray:
    """
    Log compression (safe for positive Hilbert envelopes).
    """
    return np.log10(1 + a * signal) / np.log10(1 + a)

def linear_fit_features(signal: np.ndarray,
                        segment_len: int = 300,
                        crop_head: int = 200,
                        crop_tail: int = 20) -> np.ndarray:
    """
    From each (B, C, T) channel, crop head/tail to remove artifacts,
    slice into non-overlapping segments of length segment_len, and
    compute slope/intercept via least squares per segment.

    Returns:
        features: (B, C * num_segments * 2) as float32
    """
    B, C, T = signal.shape
    # Crop to avoid boundary artifacts
    end = max(crop_head, T - crop_tail)
    signal = signal[:, :, crop_head:end]
    T = signal.shape[-1]

    # Truncate to multiple of segment_len
    num_segments = T // segment_len
    if num_segments == 0:
        raise ValueError(f"segment_len={segment_len} is too large for current T={T}.")
    signal = signal[..., :num_segments * segment_len]
    signal = signal.reshape(B, C, num_segments, segment_len)

    # Design matrix for y ≈ m*x + b
    X_axis = np.arange(segment_len)
    X_design = np.vstack([X_axis, np.ones_like(X_axis)]).T  # (segment_len, 2)

    features = np.zeros((B, C, num_segments, 2), dtype=np.float64)
    for b in range(B):
        for c in range(C):
            y = signal[b, c]  # (num_segments, segment_len)
            for s in range(num_segments):
                coef, _, _, _ = np.linalg.lstsq(X_design, y[s], rcond=None)
                features[b, c, s] = coef  # [slope, intercept]

    feats = features.reshape(B, -1).astype(np.float32)

    # 保底：清理可能的 NaN/Inf
    if not np.all(np.isfinite(feats)):
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    return feats

def preprocess_ultrasound_pipeline(us_path,
                                   angles_path,
                                   hilbert_pad: int = 100,
                                   segment_len: int = 300,
                                   crop_head: int = 200,
                                   crop_tail: int = 20):
    """
    Load ultrasound (F, C, C, T) and angle data, build (frames, features) X and (frames, 2) Y.

    Returns:
        X: (frames, features)  float32
        Y: (frames, 2)        float32 (angles[:, [1, 3]])
    """
    # --- Load ---
    raw_us = np.load(us_path)    # shape: (F, C, C, T)
    angles = np.load(angles_path)

    frames = min(raw_us.shape[0], angles.shape[0])
    C = raw_us.shape[2]
    T = raw_us.shape[3]

    # Extract diagonal channels into (F, C, T)
    us_3d = np.zeros((frames, C, T), dtype=np.float64)
    for c in range(C):
        us_3d[:, c, :] = raw_us[:frames, c, c, :]

    # --- Processing chain ---
    us_3d = time_gain_compensation(us_3d)
    us_3d = gaussian_smooth(us_3d, sigma=1.0, size=31)
    us_3d = apply_hilbert(us_3d, pad_len=hilbert_pad)
    us_3d = apply_log_compression(us_3d, a=3000)

    # --- Features + labels ---
    X = linear_fit_features(us_3d, segment_len=segment_len,
                            crop_head=crop_head, crop_tail=crop_tail)  # (frames, features)

    Y = angles[:frames, [1, 3]]  # 取两列角度（不改数值）

    # 标签小数检测（避免误用整度标签）
    if np.allclose(Y, np.round(Y)):
        print(f"[Warn] Labels look integer-like at {angles_path} (dtype={angles.dtype}). "
              f"Consider using prefixed angle file if available.")
    else:
        print(f"[Info] Labels have decimals at {os.path.basename(angles_path)} (dtype={angles.dtype}).")

    # 确保下游 dtype 为 float，不改变数值
    Y = Y.astype(np.float32)
    return X.astype(np.float32), Y
