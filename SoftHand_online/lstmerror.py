import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===== 配置 =====
csv_path = r"C:\Users\11952\Desktop\softhand_results\sweep_3_best_mnodel\predictions.csv"
abs_error = True
bin_size = 50
smooth_win = 25
use_log = False
clip_percentile = 99.5
share_colorbar = True  # 这里保留逻辑，但我们用右侧独立轴放单个色条

# ===== 读数据并取误差（第5、6列） =====
df = pd.read_csv(csv_path)
e1 = df.iloc[:, 4].to_numpy()
e2 = df.iloc[:, 5].to_numpy()
if abs_error:
    e1 = np.abs(e1); e2 = np.abs(e2)

# ===== 平滑 =====
def moving_average(x, w):
    if w is None or w <= 1:
        return x
    pad = w // 2
    x_pad = np.pad(x, (pad, pad), mode='reflect')
    kernel = np.ones(w) / w
    y = np.convolve(x_pad, kernel, mode='valid')
    return y[:len(x)]

e1 = moving_average(e1, smooth_win)
e2 = moving_average(e2, smooth_win)

# ===== 分桶 =====
def bin_reduce(x, bin_size, reduce="mean"):
    if bin_size <= 1:
        return x[None, :]
    N = len(x)
    K = int(np.ceil(N / bin_size))
    N_fit = K * bin_size
    if N_fit != N:
        x = np.pad(x, (0, N_fit - N), mode='edge')
    X = x.reshape(K, bin_size)
    if reduce == "mean":
        v = X.mean(axis=1)
    elif reduce == "max":
        v = X.max(axis=1)
    else:
        raise ValueError("reduce must be 'mean' or 'max'")
    return v[None, :]

E1 = bin_reduce(e1, bin_size, reduce="mean")
E2 = bin_reduce(e2, bin_size, reduce="mean")
H = np.vstack([E1, E2])

# ===== 对数/色标范围 =====
if use_log:
    H_plot = np.log1p(H)
    cbar_label = "log(1 + Absolute Error)"
else:
    H_plot = H
    cbar_label = "Absolute Error"

vmax = np.percentile(H_plot, clip_percentile)
vmin = 0.0

# ===== 作图（右侧独立色条轴，避免重叠）=====
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(14, 4.8), constrained_layout=True)
gs = GridSpec(nrows=2, ncols=2, width_ratios=[1, 0.035], height_ratios=[1, 1], figure=fig)

ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
cax = fig.add_subplot(gs[:, 1])  # 右侧色条轴，占两行

cm = 'viridis'

im0 = ax0.imshow(H_plot[0:1, :], aspect='auto', cmap=cm,
                 interpolation='nearest', vmin=vmin, vmax=vmax)
ax0.set_yticks([0]); ax0.set_yticklabels(["Joint 1"])
ax0.set_title(f"Prediction Error Heatmap")

im1 = ax1.imshow(H_plot[1:2, :], aspect='auto', cmap=cm,
                 interpolation='nearest', vmin=vmin, vmax=vmax)
ax1.set_yticks([0]); ax1.set_yticklabels(["Joint 2"])
ax1.set_xlabel("Frame")

# 统一色条：放到右侧 cax，不会与图重叠
cb = fig.colorbar(im1, cax=cax)
cb.set_label(cbar_label)

# 原始帧编号刻度（按照分桶还原大致位置）
K = H.shape[1]
xticks = np.linspace(0, K-1, num=6, dtype=int)
ax1.set_xticks(xticks)
ax1.set_xticklabels((xticks + 1) * bin_size)

for ax in (ax0, ax1):
    ax.tick_params(axis='both', labelsize=10)

plt.show()
