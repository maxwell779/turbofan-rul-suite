"""RUL 회귀 모델 동물원 — 입력 (B, window, F) → 스칼라 RUL.
cnn · lstm · gru · bilstm · cnnlstm · tcn · dlinear. build(name,...,**hp)."""
import torch
import torch.nn as nn


class CNN1D(nn.Module):
    def __init__(self, n_feat, window, ch=64, k=5, pool="adaptive", p=0.3, act="relu"):
        super().__init__()
        A = nn.GELU if act == "gelu" else nn.ReLU
        pad = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(n_feat, ch, k, padding=pad), nn.BatchNorm1d(ch), A(),
            nn.Conv1d(ch, ch, k, padding=pad), nn.BatchNorm1d(ch), A(),
            nn.Conv1d(ch, ch * 2, 3, padding=1), nn.BatchNorm1d(ch * 2), A(),
            nn.AdaptiveMaxPool1d(1) if pool == "max" else nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(ch * 2, 64), A(), nn.Dropout(p), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.net(x.transpose(1, 2))).squeeze(-1)


class _RNN(nn.Module):
    def __init__(self, cell, n_feat, window, hid=96, layers=2, p=0.3, bi=False):
        super().__init__()
        self.rnn = cell(n_feat, hid, layers, batch_first=True, dropout=p if layers > 1 else 0.0, bidirectional=bi)
        d = hid * (2 if bi else 1)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(d, 64), nn.ReLU(), nn.Dropout(p), nn.Linear(64, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def LSTMReg(n_feat, window, hid=96, layers=2, p=0.3): return _RNN(nn.LSTM, n_feat, window, hid, layers, p, False)
def GRUReg(n_feat, window, hid=96, layers=2, p=0.3): return _RNN(nn.GRU, n_feat, window, hid, layers, p, False)
def BiLSTM(n_feat, window, hid=96, layers=2, p=0.3): return _RNN(nn.LSTM, n_feat, window, hid, layers, p, True)


class CNNLSTM(nn.Module):
    def __init__(self, n_feat, window, ch=64, hid=96, layers=1, p=0.3, k=5):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(n_feat, ch, k, padding=k // 2), nn.BatchNorm1d(ch), nn.ReLU(),
                                  nn.Conv1d(ch, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.ReLU())
        self.lstm = nn.LSTM(ch, hid, layers, batch_first=True, dropout=p if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(hid, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.lstm(h)
        return self.head(out[:, -1, :]).squeeze(-1)


class _TCNBlock(nn.Module):
    def __init__(self, ci, co, k, d, p):
        super().__init__()
        pad = (k - 1) * d
        self.c1 = nn.Conv1d(ci, co, k, padding=pad, dilation=d)
        self.c2 = nn.Conv1d(co, co, k, padding=pad, dilation=d)
        self.bn1, self.bn2 = nn.BatchNorm1d(co), nn.BatchNorm1d(co)
        self.drop = nn.Dropout(p); self.act = nn.ReLU()
        self.down = nn.Conv1d(ci, co, 1) if ci != co else None
        self.pad = pad

    def _crop(self, y):
        return y[:, :, :-self.pad] if self.pad else y

    def forward(self, x):
        y = self.drop(self.act(self.bn1(self._crop(self.c1(x)))))
        y = self.drop(self.act(self.bn2(self._crop(self.c2(y)))))
        res = x if self.down is None else self.down(x)
        return self.act(y + res)


class TCN(nn.Module):
    def __init__(self, n_feat, window, ch=64, levels=4, k=3, p=0.2):
        super().__init__()
        blocks, ci = [], n_feat
        for i in range(levels):
            blocks.append(_TCNBlock(ci, ch, k, 2 ** i, p)); ci = ch
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(ch, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        y = self.tcn(x.transpose(1, 2))
        return self.head(y[:, :, -1]).squeeze(-1)


class DLinear(nn.Module):
    """시리즈 분해(이동평균 추세+나머지) → 채널별 선형 → 스칼라. 강baseline(DLinear 논쟁)."""
    def __init__(self, n_feat, window, kernel=25, p=0.0):
        super().__init__()
        self.k = min(kernel, window if window % 2 else window - 1)
        if self.k % 2 == 0: self.k += 1
        self.lin_t = nn.Linear(window * n_feat, 64)
        self.lin_s = nn.Linear(window * n_feat, 64)
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):                       # x: (B,W,F)
        xt = x.transpose(1, 2)
        trend = torch.nn.functional.avg_pool1d(xt, self.k, stride=1, padding=self.k // 2)
        trend = trend[:, :, :xt.size(2)]
        seas = (xt - trend)
        t = self.lin_t(trend.flatten(1)); s = self.lin_s(seas.flatten(1))
        return self.head(t + s).squeeze(-1)


class RevIN(nn.Module):
    """Reversible Instance Norm — 윈도별(인스턴스) 평균/분산 정규화 + 학습 affine.
    RUL 회귀는 출력이 스칼라라 denorm 불필요 → 입력 분포shift 완화용 정규화로 사용.
    출처: Kim et al. ICLR2022."""
    def __init__(self, n_feat, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(n_feat)); self.beta = nn.Parameter(torch.zeros(n_feat))

    def forward(self, x):                       # x: (B,W,F)
        mu = x.mean(1, keepdim=True); sd = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + self.eps)
        return (x - mu) / sd * self.gamma + self.beta


class PatchTST(nn.Module):
    """PatchTST식: patching + 채널독립 Transformer 인코더 → 회귀헤드. (+RevIN 옵션)
    출처: Nie et al. ICLR2023. 고정윈도→스칼라 RUL로 적응(decoder 대신 회귀헤드)."""
    def __init__(self, n_feat, window, patch=10, stride=5, d_model=64, nhead=4, layers=3, p=0.2, revin=True):
        super().__init__()
        self.F = n_feat; self.patch = patch; self.stride = stride
        self.revin = RevIN(n_feat) if revin else None
        self.npatch = (window - patch) // stride + 1
        self.embed = nn.Linear(patch, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.npatch, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2, p, batch_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(p), nn.Linear(n_feat * d_model, 128), nn.GELU(),
                                  nn.Dropout(p), nn.Linear(128, 1))

    def forward(self, x):                       # x: (B,W,F)
        if self.revin is not None: x = self.revin(x)
        B = x.size(0)
        xc = x.transpose(1, 2)                  # (B,F,W)
        patches = xc.unfold(2, self.patch, self.stride)   # (B,F,npatch,patch)
        z = self.embed(patches) + self.pos.unsqueeze(1)   # (B,F,npatch,d)
        z = z.reshape(B * self.F, self.npatch, -1)
        z = self.tr(z).mean(1)                  # (B*F, d)
        z = z.reshape(B, self.F, -1)            # (B,F,d)
        return self.head(z).squeeze(-1)


_REG = {"cnn": CNN1D, "lstm": LSTMReg, "gru": GRUReg, "bilstm": BiLSTM,
        "cnnlstm": CNNLSTM, "tcn": TCN, "dlinear": DLinear, "patchtst": PatchTST}


def build(name, n_feat, window, **hp):
    return _REG[name](n_feat, window, **hp)
