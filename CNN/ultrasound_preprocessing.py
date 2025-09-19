# ultrasound_preprocessing.py
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from scipy.ndimage import convolve1d
import wandb


def butter_lowpass(cutoff, fs, order=4):
    """Butterworth 低通滤波器系数。"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, btype='low', analog=False)


def lowpass_filter(data, cutoff=4.0, fs=30.0, order=4):
    """对二维角度数据做低通（axis=0）。"""
    if data.ndim != 2:
        data = np.reshape(data, (data.shape[0], -1))
    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data, axis=0)


def time_gain_compensation(signal: np.ndarray, gain_factor: float = 3.0) -> np.ndarray:
    """
    时间增益补偿（沿最后一维 T）。
    signal: (B, C, T) 或 (C, T)
    """
    if signal.ndim == 2:
        signal = signal[None, ...]
        squeeze_back = True
    else:
        squeeze_back = False

    gain = np.linspace(1.0, gain_factor, signal.shape[-1], dtype=np.float64)
    out = signal * gain

    if squeeze_back:
        out = out[0]
    return out


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    x = np.arange(-size // 2 + 1, size // 2 + 1)
    kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel.astype(np.float64)


def gaussian_smooth(signal: np.ndarray, sigma: float = 1.0, size: int = 31) -> np.ndarray:
    """
    沿时间轴进行 1D 高斯平滑。
    支持 (B, C, T) 或 (C, T)。
    """
    if signal.ndim == 2:
        signal = signal[None, ...]
        squeeze_back = True
    else:
        squeeze_back = False

    kernel = gaussian_kernel(size=size, sigma=sigma)
    out = convolve1d(signal, kernel, axis=-1, mode='reflect')

    if squeeze_back:
        out = out[0]
    return out


def apply_hilbert(signal: np.ndarray, pad_len: int = 200) -> np.ndarray:
    """
    包络提取（Hilbert），并使用反射填充避免边缘效应。
    支持 (B, C, T) 或 (C, T)。
    """
    if signal.ndim == 2:
        signal = signal[None, ...]
        squeeze_back = True
    else:
        squeeze_back = False

    pad = [(0, 0), (0, 0), (pad_len, pad_len)]
    sig_pad = np.pad(signal, pad, mode='reflect')
    env = np.abs(hilbert(sig_pad, axis=-1))
    env = env[:, :, pad_len:-pad_len]

    if squeeze_back:
        env = env[0]
    return env


def apply_log_compression(signal: np.ndarray, a: float = 3000.0) -> np.ndarray:
    """对数压缩，抑制动态范围。"""
    return np.log10(1.0 + a * signal) / np.log10(1.0 + a)


def _safe_wandb_log_image(tag: str, path: str):
    """W&B 安全上传图片，未初始化时跳过。"""
    try:
        if wandb.run is not None:
            wandb.log({tag: wandb.Image(path)})
    except Exception:
        pass


def preprocess_ultrasound_pipeline(
    data_root: str,
    prefix: str,
    cutoff_freq: float = 4.0,
    fs: float = 30.0,
    crop_left: int = 200,
    crop_right: int = 20,
    visualize: bool = False
):
    """
    完整的预处理流程：
      1) 从 .npy 读取 (frames, channels, shots, time) 的原始超声，以及角度数据
      2) 选取对角线通道（若 channels != shots，则退化为固定 shot=0）
      3) TGC -> 高斯平滑 -> Hilbert 包络 -> 对数压缩 -> 裁剪 (沿 time)
      4) 仅对角度做低通滤波（J1/J2），输出与 X 帧数对齐

    返回：
      X: (frames, channels, time_after_crop), float32
      Y: (frames, 2)  -> 低通后的 J1、J2
    """
    us_path = os.path.join(data_root, f"{prefix}_raw_US.npy")
    angles_path = os.path.join(data_root, f"{prefix}_leap_angles.npy")

    if not os.path.exists(us_path) or not os.path.exists(angles_path):
        raise FileNotFoundError(f"❌ Missing files: {us_path} or {angles_path}")

    raw_us = np.load(us_path)
    angles = np.load(angles_path)

    if raw_us.ndim != 4:
        raise ValueError(f"❌ Invalid raw_us ndim={raw_us.ndim}, expected 4-D (F,C,S,T). Got shape: {raw_us.shape}")

    F, C, S, T = raw_us.shape
    frames = min(F, angles.shape[0])
    channels = min(C, S) if C != S else C  # 若非方阵，取 min 并退化到 shot=0

    # 将 (F, C, S, T) 映射到 (frames, channels, T)
    us_3d = np.zeros((frames, channels, T), dtype=np.float64)
    if C == S:
        for c in range(channels):
            us_3d[:, c, :] = raw_us[:frames, c, c, :]
    else:
        # 非方阵情况，使用 shot=0 的各通道
        for c in range(channels):
            us_3d[:, c, :] = raw_us[:frames, c, 0, :]

    # ========== 可选可视化（只画一次，防止过多图片） ==========
    if visualize:
        frame_idx = min(100, frames - 1)
        channel_idx = min(4, channels - 1)
        raw_signal = us_3d[frame_idx, channel_idx, :]

        step1 = time_gain_compensation(raw_signal[None, None, :])[0, 0]
        step2 = gaussian_smooth(step1[None, None, :])[0, 0]
        step3 = apply_hilbert(step2[None, None, :], pad_len=100)[0, 0]
        step4 = apply_log_compression(step3[None, None, :])[0, 0]
        step5 = step4[crop_left: T - crop_right]

        os.makedirs("results", exist_ok=True)
        plt.figure(figsize=(12, 10))
        plt.subplot(6, 1, 1); plt.plot(raw_signal); plt.title(f"Raw (Ch {channel_idx}, Frame {frame_idx})")
        plt.subplot(6, 1, 2); plt.plot(step1);      plt.title("After Time Gain Compensation")
        plt.subplot(6, 1, 3); plt.plot(step2);      plt.title("After Gaussian Smoothing")
        plt.subplot(6, 1, 4); plt.plot(step3);      plt.title("After Hilbert (Envelope)")
        plt.subplot(6, 1, 5); plt.plot(step4);      plt.title("After Log Compression")
        plt.subplot(6, 1, 6); plt.plot(step5);      plt.title("Final Cropped Signal")
        plt.tight_layout()
        save_path = os.path.join("results", "preprocessing_pipeline_visualization.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        _safe_wandb_log_image("Preprocessing Visualization", save_path)

    # ========== 全流程 ==========
    X = time_gain_compensation(us_3d)             # (F, C, T)
    X = gaussian_smooth(X)                        # (F, C, T)
    X = apply_hilbert(X, pad_len=100)             # (F, C, T)
    X = apply_log_compression(X)                  # (F, C, T)
    X = X[:, :, crop_left: T - crop_right]        # 裁剪
    X = X.astype(np.float32, copy=False)

    # 只取 J1 / J2，并低通滤波
    Y = angles[:frames, :2].astype(np.float64, copy=False)
    Y = lowpass_filter(Y, cutoff=cutoff_freq, fs=fs)
    Y = Y.astype(np.float32, copy=False)

    print(f"✅ Trial {prefix} preprocessed successfully. X={X.shape}, Y={Y.shape}")
    return X, Y
