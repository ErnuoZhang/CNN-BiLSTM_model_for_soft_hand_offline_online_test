import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# ========= 配置 =========
csv_path = r"C:\Users\11952\Desktop\softhand_results\sweep_3_best_mnodel\predictions.csv"
point_alpha = 0.25      # 大数据量时散点透明度
point_size  = 8         # 散点大小
subsample   = None      # 如需子采样可设为整数，例如 10000 表示随机抽样1万点
save_path   = None      # 如需保存图像，设置为路径，例如 r"C:\temp\pred_vs_true.png"

# ========= 读数据（前四列：J1真、J1预测、J2真、J2预测） =========
df = pd.read_csv(csv_path)
y1_true = df.iloc[:, 0].to_numpy()
y1_pred = df.iloc[:, 1].to_numpy()
y2_true = df.iloc[:, 2].to_numpy()
y2_pred = df.iloc[:, 3].to_numpy()

# ========= 可选：随机子采样，避免点太密 =========
if subsample is not None and subsample < len(df):
    idx = np.random.choice(len(df), size=subsample, replace=False)
    y1_true, y1_pred = y1_true[idx], y1_pred[idx]
    y2_true, y2_pred = y2_true[idx], y2_pred[idx]

# ========= 辅助函数：拟合并作图 =========
def plot_pred_vs_true(ax, y_true, y_pred, title):
    # y=x 理想线范围
    all_vals = np.concatenate([y_true, y_pred])
    vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
    margin = 0.02 * (vmax - vmin + 1e-12)
    vmin, vmax = vmin - margin, vmax + margin

    # 拟合回归直线 y = a*x + b
    X = y_true.reshape(-1, 1)
    reg = LinearRegression().fit(X, y_pred)
    y_fit = reg.predict(X)
    a, b = reg.coef_[0], reg.intercept_
    r2 = r2_score(y_pred, y_fit)  # 也等价于 reg.score(X, y_pred)
    r, pval = pearsonr(y_true, y_pred)

    # 绘图
    ax.scatter(y_true, y_pred, s=point_size, alpha=point_alpha, edgecolors='none')
    # y=x 理想线
    ax.plot([vmin, vmax], [vmin, vmax], linestyle='--', linewidth=1.5, label='Ideal: y = x')
    # 回归直线
    xx = np.linspace(vmin, vmax, 100)
    ax.plot(xx, a*xx + b, linewidth=2, label=f'Fit: y = {a:.3f}x + {b:.3f}')
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.4)

    ax.set_title(title)
    ax.set_xlabel('True Angle (deg)')
    ax.set_ylabel('Predicted Angle (deg)')

    # 文字标注（R² 与 r）
    text = f"r = {r:.4f}"
    ax.annotate(text, xy=(0.02, 0.98), xycoords='axes fraction',
                ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.9))

    ax.legend(loc='lower right', framealpha=0.9)

# ========= 画两关节子图 =========
plt.figure(figsize=(12, 5))
ax1 = plt.subplot(1, 2, 1)
plot_pred_vs_true(ax1, y1_true, y1_pred, 'Joint 1: True vs Predicted')

ax2 = plt.subplot(1, 2, 2)
plot_pred_vs_true(ax2, y2_true, y2_pred, 'Joint 2: True vs Predicted')

plt.tight_layout()
if save_path:
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
