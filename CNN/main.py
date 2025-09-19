# main_cnn.py
import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import wandb
from ultrasound_preprocessing import preprocess_ultrasound_pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================== Dataset 类 ========================
class UltrasoundDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ======================== CNN 模型 ========================
class CNNRegressor(nn.Module):
    def __init__(self, channels, time_samples, conv1_filters=64, conv2_filters=128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=conv1_filters,
                               kernel_size=7, padding="same")
        self.pool1 = nn.MaxPool1d(kernel_size=4)
        self.conv2 = nn.Conv1d(in_channels=conv1_filters, out_channels=conv2_filters,
                               kernel_size=5, padding="same")
        self.pool2 = nn.MaxPool1d(kernel_size=4)

        with torch.no_grad():
            dummy = torch.zeros(1, channels, time_samples)
            x = self.pool1(torch.relu(self.conv1(dummy)))
            x = self.pool2(torch.relu(self.conv2(x)))
            self.flatten_dim = x.view(1, -1).shape[1]

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        return self.head(x)


# ======================== 数据加载与缓存 ========================
def load_or_cache_data(data_root,
                       cache_file="preprocessed_data_cnn.npz",
                       cutoff_freq=4.0,
                       fs=30.0,
                       visualize_once=True):
    """
    读取或预处理+缓存数据。沿用“0/1 -> 训练; 2 -> 验证+测试”的 trial 组织方式。
    """
    if os.path.exists(cache_file):
        data = np.load(cache_file)
        return data["X_train"], data["Y_train"], data["X_val"], data["Y_val"], data["X_test"], data["Y_test"]

    subjects = [f"SH0{i}" for i in range(1, 6)]
    X_train, Y_train, X_val, Y_val, X_test, Y_test = [], [], [], [], [], []

    first_visualized = not visualize_once  # 若 visualize_once=True，则第一次可视化；否则都不画

    for subject in subjects:
        subject_path = os.path.join(data_root, subject)
        if not os.path.isdir(subject_path):
            continue

        for i in range(0, 27, 3):         # 每 3 个 trial 为一组
            for offset in [0, 1, 2]:      # 0/1 -> 训练, 2 -> 验证+测试
                trial_num = i + offset
                trial_path = os.path.join(subject_path, f"{trial_num}")
                if not os.path.isdir(trial_path):
                    continue

                prefix = None
                for fn in os.listdir(trial_path):
                    if fn.endswith("_raw_US.npy"):
                        prefix = fn.replace("_raw_US.npy", "")
                        break
                if prefix is None:
                    continue

                us_path = os.path.join(trial_path, f"{prefix}_raw_US.npy")
                ang_path = os.path.join(trial_path, f"{prefix}_leap_angles.npy")
                if not (os.path.exists(us_path) and os.path.exists(ang_path)):
                    continue

                # 只对第一个 trial 可视化，避免上传太多图
                visualize_flag = (not first_visualized) if visualize_once else False

                X, Y = preprocess_ultrasound_pipeline(
                    trial_path, prefix,
                    cutoff_freq=cutoff_freq, fs=fs,
                    crop_left=200, crop_right=20,
                    visualize=visualize_flag
                )

                if visualize_flag:
                    first_visualized = True

                if offset in [0, 1]:
                    X_train.append(X); Y_train.append(Y)
                else:
                    mid = X.shape[0] // 2
                    X_val.append(X[:mid]);  Y_val.append(Y[:mid])
                    X_test.append(X[mid:]); Y_test.append(Y[mid:])

    X_train, Y_train = np.concatenate(X_train), np.concatenate(Y_train)
    X_val,   Y_val   = np.concatenate(X_val),   np.concatenate(Y_val)
    X_test,  Y_test  = np.concatenate(X_test),  np.concatenate(Y_test)

    np.savez(cache_file, X_train=X_train, Y_train=Y_train,
             X_val=X_val, Y_val=Y_val, X_test=X_test, Y_test=Y_test)
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


# ======================== 指标计算 ========================
def compute_metrics(Y_true, Y_pred):
    rmse = np.sqrt(np.mean((Y_pred - Y_true) ** 2, axis=0))
    nrmse = rmse / np.clip(np.ptp(Y_true, axis=0), 1e-8, None)
    mape = np.mean(np.abs((Y_true - Y_pred) / np.clip(Y_true, 1e-8, None)), axis=0) * 100
    r2 = 1 - np.sum((Y_true - Y_pred) ** 2, axis=0) / np.clip(
        np.sum((Y_true - np.mean(Y_true, axis=0)) ** 2, axis=0), 1e-8, None)
    return rmse, nrmse, mape, r2


# ======================== 主函数（固定参数） ========================
def main():
    # 固定参数配置（可自行修改）
    CFG = {
        "project": "ultrasound-cnn-fixed",
        "data_root": r"C:\Users\Be.Neuro\Desktop\Data",
        "cache_file": "preprocessed_data_cnn.npz",
        "batch_size": 32,
        "learning_rate": 5e-4,
        "epochs": 100,
        "patience": 20,
        "conv1_filters": 64,
        "conv2_filters": 128,
        "scheduler_step": 20,
        "scheduler_gamma": 0.9,
        "cutoff_freq": 4.0,   # 角度低通滤波截止频率（Hz）
        "fs": 30.0,           # 角度采样频率（Hz）
        "visualize_once": True
    }

    os.makedirs("results", exist_ok=True)
    writer = SummaryWriter("runs/cnn_fixed")

    # 初始化 wandb（固定配置）
    wandb.init(project=CFG["project"], config=CFG)
    config = wandb.config

    # 读缓存/预处理
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_or_cache_data(
        config.data_root, config.cache_file, cutoff_freq=config.cutoff_freq,
        fs=config.fs, visualize_once=config.visualize_once
    )

    # 仅缩放输出 Y
    scaler_y = MinMaxScaler()
    Y_train = scaler_y.fit_transform(Y_train)
    Y_val   = scaler_y.transform(Y_val)
    Y_test_scaled = scaler_y.transform(Y_test)  # 仅供 test_loader 使用

    # DataLoader
    train_loader = DataLoader(UltrasoundDataset(X_train, Y_train),
                              batch_size=config.batch_size, shuffle=True)
    val_loader   = DataLoader(UltrasoundDataset(X_val,   Y_val),
                              batch_size=config.batch_size)
    test_loader  = DataLoader(UltrasoundDataset(X_test,  Y_test_scaled),
                              batch_size=config.batch_size)

    # 模型
    model = CNNRegressor(
        channels=X_train.shape[1],
        time_samples=X_train.shape[2],
        conv1_filters=config.conv1_filters,
        conv2_filters=config.conv2_filters
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma
    )
    loss_fn = nn.MSELoss()

    best_loss, counter = float("inf"), 0
    logs = {"train": [], "val": [], "nrmse": [], "mape": [], "r2": []}

    # 训练
    for epoch in tqdm(range(config.epochs), desc="Training"):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_loss += loss_fn(out, yb).item() * xb.size(0)
                preds.append(out.cpu().numpy())
                trues.append(yb.cpu().numpy())
        val_loss /= len(val_loader.dataset)

        pred_val_scaled = np.concatenate(preds)
        true_val_scaled = np.concatenate(trues)
        pred_val = scaler_y.inverse_transform(pred_val_scaled)
        true_val = scaler_y.inverse_transform(true_val_scaled)
        _, nrmse, mape, r2 = compute_metrics(true_val, pred_val)

        logs["train"].append(train_loss)
        logs["val"].append(val_loss)
        logs["nrmse"].append(float(np.mean(nrmse)))
        logs["mape"].append(float(np.mean(mape)))
        logs["r2"].append(float(np.mean(r2)))

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        writer.add_scalars("Loss", {"Train": train_loss, "Val": val_loss}, epoch)
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_nRMSE": logs["nrmse"][-1],
            "val_MAPE": logs["mape"][-1],
            "val_R2":   logs["r2"][-1],
            "lr": current_lr,
        })

        # 保存最优权重
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "results/best_model_cnn.pth")
            wandb.run.summary["best_val_loss"] = float(best_loss)
            wandb.run.summary["best_model_path"] = os.path.abspath("results/best_model_cnn.pth")
            counter = 0
        else:
            counter += 1
            if counter >= config.patience:
                break

    # 保存训练日志表
    pd.DataFrame(logs).to_csv("results/training_metrics.csv", index=False)

    # ===== 测试评估 =====
    model.load_state_dict(torch.load("results/best_model_cnn.pth", map_location=device))
    model.eval()

    preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            preds.append(model(xb.to(device)).cpu().numpy())
    Y_pred_scaled = np.concatenate(preds)
    Y_pred = scaler_y.inverse_transform(Y_pred_scaled)
    Y_true = Y_test  # 与未缩放真值对比

    rmse, nrmse, mape, r2 = compute_metrics(Y_true, Y_pred)
    wandb.log({
        "Test RMSE_J1": float(rmse[0]), "Test RMSE_J2": float(rmse[1]),
        "Test NRMSE_J1": float(nrmse[0]), "Test NRMSE_J2": float(nrmse[1]),
        "Test MAPE_J1": float(mape[0]), "Test MAPE_J2": float(mape[1]),
        "Test R2_J1": float(r2[0]), "Test R2_J2": float(r2[1])
    })

    # 预测曲线
    pred_df = pd.DataFrame({
        "True_J1": Y_true[:, 0], "Pred_J1": Y_pred[:, 0],
        "True_J2": Y_true[:, 1], "Pred_J2": Y_pred[:, 1]
    })
    pred_df.to_csv("results/predictions.csv", index=False)

    for i in range(2):
        plt.figure()
        plt.plot(Y_true[:, i], label="True", color="black")
        plt.plot(Y_pred[:, i], label="Pred", color="blue")
        plt.title(f"Joint {i + 1} Prediction")
        plt.legend()
        path = f"results/joint{i + 1}.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        wandb.log({f"Joint {i + 1} Prediction": wandb.Image(path)})
        plt.close()

    writer.close()
    wandb.finish()
    print("\n✅ 所有结果保存在 results/ 文件夹中。")


if __name__ == "__main__":
    main()
