import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== 可配置参数 =====
csv_path = r"C:\Users\11952\Desktop\zyyonline_metrics.csv"  # 你的在线实验数据
joint = 2   # 选择关节：1 或 2
use_inlier_only = False  # 是否仅在内点（基于真实角度 0~100°）上计算 DTW
downsample = 10         # 下采样步长，用于可视化与降低计算量（例如每 10 帧取 1 次）
window = 80             # Sakoe–Chiba 带宽（下采样后的样本数），控制时间轴允许的最大偏移
z_normalize = False     # 是否对两个序列做 z-score 归一化（仅影响 DTW 与绘图，不改变原数据）

# ===== 读取数据 =====
df = pd.read_csv(csv_path)
if joint == 1:
    pred = df["pred1"].to_numpy()
    true = df["true1"].to_numpy()
elif joint == 2:
    pred = df["pred2"].to_numpy()
    true = df["true2"].to_numpy()
else:
    raise ValueError("joint 必须为 1 或 2")

# ===== 按物理范围筛内点（仅基于真实值） =====
if use_inlier_only:
    mask = (true >= 0) & (true <= 100)
    pred = pred[mask]
    true = true[mask]

# ===== 下采样（为了可视化和 DTW 计算的可行性） =====
pred_ds = pred[::downsample]
true_ds = true[::downsample]

# ===== 可选：z-score 归一化（仅用于 DTW/绘图） =====
def zscore(x):
    m = np.mean(x)
    s = np.std(x) + 1e-12
    return (x - m) / s

pred_use = zscore(pred_ds) if z_normalize else pred_ds.copy()
true_use = zscore(true_ds) if z_normalize else true_ds.copy()

# ===== 实现带 Sakoe–Chiba 约束的 DTW =====
def dtw_sakoe_chiba(x, y, w):
    """
    x: (N,), y: (M,)
    w: 带宽（允许的时间偏移），单位：样本
    返回：总距离、累积代价矩阵（用于可视化）、对齐路径（索引对列表）
    """
    N, M = len(x), len(y)
    inf = 1e18
    D = np.full((N + 1, M + 1), inf, dtype=float)
    D[0, 0] = 0.0

    # 累积代价
    for i in range(1, N + 1):
        j_start = max(1, i - w)
        j_end = min(M, i + w)
        xi = x[i - 1]
        for j in range(j_start, j_end + 1):
            cost = abs(xi - y[j - 1])  # L1 距离，亦可改为 (xi - y[j-1])**2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    # 回溯路径
    i, j = N, M
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        steps = [(i - 1, j), (i, j - 1), (i - 1, j - 1)]
        i, j = min(steps, key=lambda ij: D[ij])

    path.reverse()
    return D[N, M], D[1:, 1:], path

# 计算 DTW
dtw_dist, acc_cost, path = dtw_sakoe_chiba(true_use, pred_use, window)

# ===== 估计整体时滞（延迟）：使用交叉相关在下采样后估计 =====
def estimate_lag(x, y, max_lag=None):
    """
    估计 y 相对 x 的滞后（正值表示 y 滞后于 x）
    """
    N = len(x)
    x0 = (x - np.mean(x)) / (np.std(x) + 1e-12)
    y0 = (y - np.mean(y)) / (np.std(y) + 1e-12)
    corr = np.correlate(y0, x0, mode="full")
    lags = np.arange(-N + 1, N)
    if max_lag is not None:
        mask = (lags >= -max_lag) & (lags <= max_lag)
        corr = corr[mask]
        lags = lags[mask]
    best_lag = lags[np.argmax(corr)]
    return best_lag, corr, lags

best_lag, corr, lags = estimate_lag(true_ds, pred_ds, max_lag=5 * window)

# ===== 图 1：真实 vs 预测（下采样后） =====
plt.figure(figsize=(10, 4))
plt.plot(true_ds, label="True (downsampled)")
plt.plot(pred_ds, label="Pred (downsampled)")
plt.title(f"Joint {joint} Time Series (downsample={downsample})\nDTW distance={dtw_dist:.2f}, Estimated lag={best_lag} (samples@ds)")
plt.xlabel("Frame (downsampled)")
plt.ylabel("Angle")
plt.legend()
plt.tight_layout()
plt.show()

# ===== 图 2：DTW 累积代价矩阵 + 对齐路径 =====
plt.figure(figsize=(6, 5))
plt.imshow(acc_cost, origin="lower", aspect="auto")
# 叠加路径（注意 path 是下采样后的索引）
pi = [p[0] for p in path]
pj = [p[1] for p in path]
plt.plot(pj, pi)  # 横轴为 pred 索引，纵轴为 true 索引
plt.title(f"Joint {joint} DTW Accumulated Cost (Sakoe–Chiba window={window})")
plt.xlabel("Pred index (downsampled)")
plt.ylabel("True index (downsampled)")
plt.tight_layout()
plt.show()

# ===== 图 3：对齐映射曲线（i -> j） =====
plt.figure(figsize=(6, 5))
plt.plot(pi, pj)
plt.title(f"Joint {joint} Alignment Mapping (True idx -> Pred idx)")
plt.xlabel("True index (downsampled)")
plt.ylabel("Pred index (downsampled)")
plt.tight_layout()
plt.show()

# ===== 打印关键数值 =====
print(f"[Joint {joint}] DTW distance (downsampled) = {dtw_dist:.3f}")
print(f"[Joint {joint}] Estimated lag via xcorr (downsampled samples) = {best_lag}")
print(f"[Joint {joint}] Inlier-only = {use_inlier_only}, Downsample={downsample}, Window={window}, z-norm={z_normalize}")
