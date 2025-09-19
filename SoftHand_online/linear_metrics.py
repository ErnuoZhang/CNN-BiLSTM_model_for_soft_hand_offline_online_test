import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 文件路径
file_path = r"C:\Users\11952\Desktop\softhand_results\linear regression\1 subject\ridge_y_scaled_prefixed_no_val_joint2_true_pred.csv"

# 跳过第一行文字，读取数据
data = pd.read_csv(file_path, skiprows=1, header=None).astype(float)

# 提取真实值和预测值
y_true = data.iloc[:, 1].values
y_pred = data.iloc[:, 2].values

# 计算误差
errors = y_pred - y_true

# 计算95%置信区间
mean_err = np.mean(errors)
sem_err = stats.sem(errors)  # 标准误差
ci95 = stats.t.interval(0.95, len(errors)-1, loc=mean_err, scale=sem_err)

print(f"Mean error: {mean_err:.4f}")
print(f"95% Confidence Interval: {ci95}")

# 绘制误差分布直方图
plt.figure(figsize=(8,5))
plt.hist(errors, bins=50, density=True, color='steelblue', edgecolor='black', alpha=0.6)

# KDE 曲线
kde = stats.gaussian_kde(errors)
x_vals = np.linspace(min(errors), max(errors), 500)
plt.plot(x_vals, kde(x_vals), color='darkred', linewidth=2, label="KDE Fit")

# 正态分布拟合曲线
std_err = np.std(errors)
pdf = stats.norm.pdf(x_vals, mean_err, std_err)
plt.plot(x_vals, pdf, color='orange', linestyle='--', linewidth=2, label="Gaussian Fit")

# 均值和置信区间
plt.axvline(mean_err, color='red', linestyle='--', linewidth=2, label=f'Mean={mean_err:.4f}')
plt.axvline(ci95[0], color='green', linestyle='--', linewidth=2, label=f'95% CI lower={ci95[0]:.4f}')
plt.axvline(ci95[1], color='green', linestyle='--', linewidth=2, label=f'95% CI upper={ci95[1]:.4f}')

plt.title("Error Distribution with KDE & Gaussian Fit and 95% CI")
plt.xlabel("Prediction Error")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
