import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset


class UltrasoundDataset(Dataset):
    def __init__(self, X, Y):
        
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class CNNBiLSTMRegressor(nn.Module):
    def __init__(self, channels, time_samples, lstm_hidden_dim=64, conv1_filters=32, conv2_filters=64):
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
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return self.fc(x[:, -1, :])