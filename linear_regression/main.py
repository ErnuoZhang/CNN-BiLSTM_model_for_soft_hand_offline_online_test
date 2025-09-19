import os
import numpy as np
import pandas as pd
import wandb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler

from ultrasound_processing import preprocess_ultrasound_pipeline

# -----------------------------
# CONFIG
# -----------------------------
DATA_ROOT = r"C:\Users\Be.Neuro\Desktop\Data"
CACHE_FILE = "preprocessed_data_ridge_prefixed_no_val.npz"  # 新缓存名，避免读到旧策略的缓存
PROJECT = "ridge-ultrasound"
RUN_NAME = "ridge_y_scaled_prefixed_no_val"  # 标注：Y做归一化 + 用前缀角度 + 无验证集

# 受试者（若只跑一个：SUBJECTS = ["SH01"]）
SUBJECTS = [f"SH0{i}" for i in range(1, 2)]

# -----------------------------
# 工具：在一个 trial 目录中，优先找到前缀角度文件
# -----------------------------
def find_us_and_angles(trial_path: str):
    """
    在 trial 目录中查找 * _raw_US.npy 以确定 prefix，
    返回 (us_path, angles_path)：
      1) 优先 <prefix>_leap_angles.npy（浮点、小数）
      2) 回退 _leap_angles.npy（可能是 int16 整度，并打印警告）
    如果任何一个缺失则返回 (None, None)。
    """
    prefix = None
    for fn in os.listdir(trial_path):
        if fn.endswith("_raw_US.npy"):
            prefix = fn.replace("_raw_US.npy", "")
            break
    if prefix is None:
        return None, None

    us_path = os.path.join(trial_path, f"{prefix}_raw_US.npy")
    angles_pref = os.path.join(trial_path, f"{prefix}_leap_angles.npy")
    angles_nopref = os.path.join(trial_path, "_leap_angles.npy")

    if os.path.exists(angles_pref):
        angles_path = angles_pref
    elif os.path.exists(angles_nopref):
        angles_path = angles_nopref
        print(f"[Warn] Fallback to integer-likely angles: {angles_path} "
              f"(prefer {angles_pref} if available).")
    else:
        return None, None

    if not (os.path.exists(us_path) and os.path.exists(angles_path)):
        return None, None

    return us_path, angles_path

# -----------------------------
# Data loading / caching（无验证集）
# -----------------------------
def load_or_cache_data(data_root: str, cache_file: str = CACHE_FILE):
    """
    加载/缓存数据（无验证集）：
      - trial 以 3 个为一组（0/1 -> 训练，2 -> 测试），与 CNN 划分风格一致；
      - 角度优先用 <prefix>_leap_angles.npy，找不到回退 _leap_angles.npy；
      - 最终仅返回训练集与测试集。
    """
    if os.path.exists(cache_file):
        print(f"[Info] Loading cached data from {cache_file}")
        data = np.load(cache_file)
        return data["X_train"], data["Y_train"], data["X_test"], data["Y_test"]

    X_train, Y_train, X_test, Y_test = [], [], [], []

    for subject in SUBJECTS:
        subject_path = os.path.join(data_root, subject)
        if not os.path.isdir(subject_path):
            print(f"[Warn] Missing subject dir: {subject_path}")
            continue

        for i in range(0, 27, 3):           # 0..26，每3个一组
            for offset in [0, 1, 2]:
                trial_num = i + offset
                trial_path = os.path.join(subject_path, f"{trial_num}")
                if not os.path.isdir(trial_path):
                    continue

                us_path, angles_path = find_us_and_angles(trial_path)
                if not us_path or not angles_path:
                    continue

                try:
                    X, Y = preprocess_ultrasound_pipeline(us_path, angles_path)
                except Exception as e:
                    print(f"[Warn] Skipping {trial_path}: {e}")
                    continue

                if offset in [0, 1]:
                    X_train.append(X); Y_train.append(Y)
                else:  # offset == 2 -> 全部划到测试集（不再二分 val/test）
                    X_test.append(X);  Y_test.append(Y)

    if not X_train or not Y_train:
        raise ValueError("No training data collected.")
    if not X_test or not Y_test:
        raise ValueError("No testing data collected.")

    X_train = np.concatenate(X_train, axis=0)
    Y_train = np.concatenate(Y_train, axis=0)
    X_test  = np.concatenate(X_test,  axis=0)
    Y_test  = np.concatenate(Y_test,  axis=0)

    # 额外：检查标签是否被整数化
    def int_like(a): return np.allclose(a, np.round(a))
    if int_like(Y_train) or int_like(Y_test):
        print("[Warn] Some labels look integer-like. Make sure prefixed float angle files are available.")

    os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
    np.savez_compressed(cache_file,
                        X_train=X_train, Y_train=Y_train,
                        X_test=X_test,   Y_test=Y_test)
    print(f"[Info] Cached data -> {cache_file}")
    return X_train, Y_train, X_test, Y_test

# -----------------------------
# Scaling（仅用训练集拟合）
# -----------------------------
def scale_train_test(X_train, Y_train, X_test, Y_test):
    """
    - X：训练集拟合 MinMax，再变换到测试集
    - Y：训练集拟合 MinMax；训练目标是归一化后的 Y；测试集仅用于最终对比（真实单位）
    """
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_train_s = x_scaler.fit_transform(X_train)
    X_test_s  = x_scaler.transform(X_test)

    Y_train_s = y_scaler.fit_transform(Y_train)
    # 注意：Y_test 不进入拟合；保留真实单位
    return X_train_s, Y_train_s, X_test_s, Y_test, y_scaler

# -----------------------------
# Train（Ridge）
# -----------------------------
def train_model(X_scaled, Y_scaled, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(X_scaled, Y_scaled)
    return model

# -----------------------------
# Save predictions (two CSVs)
# -----------------------------
def save_test_predictions_per_joint(Y_true_real, Y_pred_real, run_name: str):
    os.makedirs("results", exist_ok=True)
    n = Y_true_real.shape[0]
    frames = np.arange(n)

    # Joint 1
    df1 = pd.DataFrame({"Frame": frames, "True": Y_true_real[:, 0], "Pred": Y_pred_real[:, 0]})
    path1 = f"results/{run_name}_joint1_true_pred.csv"
    df1.to_csv(path1, index=False, float_format="%.6f")
    print(f"[Info] Saved -> {path1}")

    # Joint 2
    df2 = pd.DataFrame({"Frame": frames, "True": Y_true_real[:, 1], "Pred": Y_pred_real[:, 1]})
    path2 = f"results/{run_name}_joint2_true_pred.csv"
    df2.to_csv(path2, index=False, float_format="%.6f")
    print(f"[Info] Saved -> {path2}")

# -----------------------------
# Main
# -----------------------------
def main():
    wandb.init(project=PROJECT, name=RUN_NAME, config={"model": "Ridge", "alpha": 1.0, "y_scaled": True})
    run_name = wandb.run.name

    # 1) 加载/缓存（无验证集）
    X_train, Y_train, X_test, Y_test = load_or_cache_data(DATA_ROOT, CACHE_FILE)

    # 2) 缩放（仅用训练集拟合）
    X_train_s, Y_train_s, X_test_s, Y_test_real, y_scaler = scale_train_test(
        X_train, Y_train, X_test, Y_test
    )

    # 3) 训练
    model = train_model(X_train_s, Y_train_s, alpha=1.0)

    # 4) 测试集预测并反缩放回真实角度
    Y_pred_s = model.predict(X_test_s)
    Y_pred_real = y_scaler.inverse_transform(Y_pred_s)

    # 5) 导出 CSV（两个关节各一份）
    save_test_predictions_per_joint(Y_test_real, Y_pred_real, run_name=run_name)

    wandb.finish()

if __name__ == "__main__":
    main()
