import os
import json
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import wandb
from ultrasound_preprocessing import preprocess_ultrasound_pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UltrasoundDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class CNNBiLSTMRegressor(nn.Module):
    def __init__(self, channels, time_samples, lstm_hidden_dim=128, conv1_filters=64, conv2_filters=128):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, conv1_filters, kernel_size=7, padding='same')
        self.pool1 = nn.MaxPool1d(kernel_size=4)
        self.conv2 = nn.Conv1d(conv1_filters, conv2_filters, kernel_size=5, padding='same')
        self.pool2 = nn.MaxPool1d(kernel_size=4)
        self.lstm1 = nn.LSTM(conv2_filters, lstm_hidden_dim, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(lstm_hidden_dim * 2, lstm_hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.permute(0, 2, 1)  # [B, T, C]
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return self.fc(x[:, -1, :])

def load_or_cache_data(data_root, cache_file="preprocessed_data.npz"):
    if os.path.exists(cache_file):
        data = np.load(cache_file)
        return data["X_train"], data["Y_train"], data["X_val"], data["Y_val"], data["X_test"], data["Y_test"]

    subjects = [f"SH0{i}" for i in range(1, 6)]
    X_train, Y_train, X_val, Y_val, X_test, Y_test = [], [], [], [], [], []
    for subject in subjects:
        subject_path = os.path.join(data_root, subject)
        for i in range(0, 27, 3):
            for offset in [0, 1, 2]:
                trial_num = i + offset
                trial_path = os.path.join(subject_path, f"{trial_num}")
                if not os.path.isdir(trial_path):
                    continue
                prefix = None
                for file in os.listdir(trial_path):
                    if file.endswith("_raw_US.npy"):
                        prefix = file.replace("_raw_US.npy", "")
                        break
                if prefix is None:
                    continue
                us_path = os.path.join(trial_path, f"{prefix}_raw_US.npy")
                angles_path = os.path.join(trial_path, f"{prefix}_leap_angles.npy")
                if not (os.path.exists(us_path) and os.path.exists(angles_path)):
                    continue

                X, Y = preprocess_ultrasound_pipeline(trial_path, prefix)
                if offset in [0, 1]:
                    X_train.append(X); Y_train.append(Y)
                elif offset == 2:
                    mid = X.shape[0] // 2
                    X_val.append(X[:mid]); Y_val.append(Y[:mid])
                    X_test.append(X[mid:]); Y_test.append(Y[mid:])

    X_train, Y_train = np.concatenate(X_train), np.concatenate(Y_train)
    X_val, Y_val = np.concatenate(X_val), np.concatenate(Y_val)
    X_test, Y_test = np.concatenate(X_test), np.concatenate(Y_test)

    np.savez(cache_file, X_train=X_train, Y_train=Y_train,
             X_val=X_val, Y_val=Y_val, X_test=X_test, Y_test=Y_test)
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def compute_metrics(Y_true, Y_pred):
    rmse = np.sqrt(np.mean((Y_pred - Y_true) ** 2, axis=0))
    nrmse = rmse / np.clip(np.ptp(Y_true, axis=0), 1e-8, None)
    mape = np.mean(np.abs((Y_true - Y_pred) / np.clip(Y_true, 1e-8, None)), axis=0) * 100
    r2 = 1 - np.sum((Y_true - Y_pred) ** 2, axis=0) / np.clip(
        np.sum((Y_true - np.mean(Y_true, axis=0)) ** 2, axis=0), 1e-8, None)
    return rmse, nrmse, mape, r2

def main():
    wandb.init()
    config = wandb.config
    run_name = wandb.run.name
    print("🌀 Sweep running:", dict(config))
    os.makedirs("results", exist_ok=True)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    all_best_dir = os.path.join(base_dir, "new sweep model")
    os.makedirs(all_best_dir, exist_ok=True)

    writer = SummaryWriter("runs/cnn_bilstm")
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_or_cache_data(r"C:\\Users\\Be.Neuro\\Desktop\\Data")

    scaler_y = MinMaxScaler()
    Y_train = scaler_y.fit_transform(Y_train)
    Y_val = scaler_y.transform(Y_val)
    Y_test_scaled = scaler_y.transform(Y_test)

    train_loader = DataLoader(UltrasoundDataset(X_train, Y_train), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(UltrasoundDataset(X_val, Y_val), batch_size=config.batch_size)
    test_loader = DataLoader(UltrasoundDataset(X_test, Y_test_scaled), batch_size=config.batch_size)

    model = CNNBiLSTMRegressor(
        X_train.shape[1], X_train.shape[2],
        config.lstm_hidden_dim, config.conv1_filters, config.conv2_filters
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)  # 每10个epoch乘0.8
    loss_fn = nn.MSELoss()

    best_loss, counter, patience = float("inf"), 0, 20
    logs = {"train": [], "val": [], "nrmse": [], "mape": [], "r2": []}

    for epoch in tqdm(range(config.epochs), desc="Training"):
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
        logs["nrmse"].append(np.mean(nrmse))
        logs["mape"].append(np.mean(mape))
        logs["r2"].append(np.mean(r2))

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_nRMSE": logs["nrmse"][-1],
            "val_MAPE": logs["mape"][-1],
            "val_R2": logs["r2"][-1],
            "lr": current_lr,
        })

        if val_loss < best_loss:
            best_loss = val_loss
            safe_run = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(run_name))
            model_filename = f"best_{safe_run}_epoch{epoch+1:03d}_valloss{best_loss:.6f}.pth"
            model_path = os.path.join(all_best_dir, model_filename)
            torch.save(model.state_dict(), model_path)
            latest_best_path = os.path.join("results", f"best_model_{safe_run}.pth")
            torch.save(model.state_dict(), latest_best_path)
            wandb.run.summary["best_val_loss"] = float(best_loss)
            wandb.run.summary["best_model_path"] = latest_best_path
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    pd.DataFrame(logs).to_csv("results/training_metrics.csv", index=False)

    safe_run = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(run_name))
    best_ckpt_path = os.path.join("results", f"best_model_{safe_run}.pth")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    model.eval()

    preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            preds.append(model(xb.to(device)).cpu().numpy())
    Y_pred_scaled = np.concatenate(preds)
    Y_pred = scaler_y.inverse_transform(Y_pred_scaled)
    Y_true = Y_test
    rmse, nrmse, mape, r2 = compute_metrics(Y_true, Y_pred)

    wandb.log({
        "Test RMSE_J1": float(rmse[0]), "Test RMSE_J2": float(rmse[1]),
        "Test NRMSE_J1": float(nrmse[0]), "Test NRMSE_J2": float(nrmse[1]),
        "Test MAPE_J1": float(mape[0]), "Test MAPE_J2": float(mape[1]),
        "Test R2_J1": float(r2[0]), "Test R2_J2": float(r2[1])
    })

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

if __name__ == "__main__":
    main()
