"""RUL 회귀 모델 — 1D-CNN, LSTM. 입력 (B, window, F)."""
import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """센서 시계열을 채널로 보는 1D conv 회귀기."""
    def __init__(self, n_feat, window, ch=64, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_feat, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch * 2, 3, padding=1), nn.BatchNorm1d(ch * 2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(ch * 2, 64), nn.ReLU(),
                                  nn.Dropout(p), nn.Linear(64, 1))

    def forward(self, x):                # x: (B, W, F) -> (B, F, W)
        return self.head(self.net(x.transpose(1, 2))).squeeze(-1)


class LSTMReg(nn.Module):
    def __init__(self, n_feat, window, hid=96, layers=2, p=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hid, layers, batch_first=True,
                            dropout=p if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(hid, 64), nn.ReLU(),
                                  nn.Dropout(p), nn.Linear(64, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def build(name, n_feat, window):
    return {"cnn": CNN1D, "lstm": LSTMReg}[name](n_feat, window)
