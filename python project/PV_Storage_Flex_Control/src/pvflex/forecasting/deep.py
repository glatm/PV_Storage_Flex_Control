# =============================================================================
# forecasting/deep.py  ——  SOTA 深度预测基线（LSTM / Transformer / N-BEATS）
# -----------------------------------------------------------------------------
# 【这个文件解决什么问题】
#   论文里不能只说"我的方法误差 13%"，还得证明"它比当前最好水平（SOTA）好"。
#   本文件实现三种主流的深度学习预测模型，作为【对手】来和本项目的方法比较。
#   赢过它们，方法的贡献才站得住脚。
#
# 【三个模型分别是什么（大白话）】
#   · LSTM        长短期记忆网络。擅长记"前几天的走势"，像有记忆的曲线追踪器。
#   · Transformer 靠"注意力机制"工作的模型。能自动找出"历史上哪些天和今天像"，
#                 再参考那些天来预测（GPT 也是这个家族）。
#   · N-BEATS     专为时间序列设计的纯全连接结构，简单但在很多数据集上意外能打。
#
# 【设计要点（为什么这么做才对论文有利）】
#   · 预测粒度 = 日总量（PV 日发电量 / 负荷日用电量）。
#     ⚠️ 为什么不用小时级对比：负荷的小时级数据是"日总量 × 形状"摊出来的，
#        本身就带失真。若用小时级比，所有模型都在拟合同一个失真的目标，
#        比出来的差异可能只是"谁更会拟合形状"，没有意义。
#        用日总量则是【真实目标】（PV 由小时聚合而来、负荷本就是日合计），
#        不受摊分失真影响，对比最公平。
#   · 输入 = [过去 L 天的日总量窗口] + [目标日的日历特征（月份/星期/是否周末/年内相位）]
#     —— 前者让模型看"最近趋势"，后者让模型知道"这是几月、周几"（季节性与工作日效应）
#   · 输出 = 下一天的日总量（一步前向预测）
#   · 滚动起点(walk-forward)交叉验证在 benchmark.py 里做；本文件只负责"单模型能拟合"
#
# ⚠️ 全部 CPU 运行（本机暂无 GPU）；数据量 ~365 天、模型很小，训练很快。
#    随机种子固定，保证结果可复现（这是 SCI 论文的硬要求之一）。
# =============================================================================

import numpy as np
import torch
import torch.nn as nn

try:
    torch.set_num_threads(max(1, torch.get_num_threads()))  # 不抢占全部核
except Exception:
    pass


def set_seed(seed: int = 0):
    """固定随机种子，保证可复现。"""
    torch.manual_seed(seed)
    np.random.seed(seed)


# -----------------------------------------------------------------------------
# 1) LSTM 预测器
# -----------------------------------------------------------------------------
class LSTNet(nn.Module):
    def __init__(self, lookback: int, exo_dim: int, hidden: int = 32,
                 layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.in_dim = 1 + exo_dim
        self.lstm = nn.LSTM(self.in_dim, hidden, layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, seq, exo):
        # seq: (B, L) -> (B, L, 1)；exo: (B, exo_dim)
        x = seq.unsqueeze(-1)
        if self.in_dim > 1:
            exo_rep = exo.unsqueeze(1).repeat(1, x.size(1), 1)
            x = torch.cat([x, exo_rep], dim=-1)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


# -----------------------------------------------------------------------------
# 2) Transformer 预测器（轻量 encoder + 末端池化）
# -----------------------------------------------------------------------------
class TransformerNet(nn.Module):
    def __init__(self, lookback: int, exo_dim: int, d_model: int = 32,
                 nhead: int = 4, nlayers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.in_dim = 1 + exo_dim
        self.embed = nn.Linear(self.in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=64, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, nlayers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, seq, exo):
        x = seq.unsqueeze(-1)
        if self.in_dim > 1:
            exo_rep = exo.unsqueeze(1).repeat(1, x.size(1), 1)
            x = torch.cat([x, exo_rep], dim=-1)
        x = self.embed(x)
        x = self.enc(x)
        return self.head(x[:, -1, :])


# -----------------------------------------------------------------------------
# 3) N-BEATS 预测器（标准 residual + basis 结构；可选外生分支）
# -----------------------------------------------------------------------------
class _NBeatsBlock(nn.Module):
    def __init__(self, width: int, layers: int, backcast_length: int):
        super().__init__()
        self.backcast_length = backcast_length
        self.proj = nn.Linear(backcast_length, width)  # 输入 (B,L) -> (B,width)
        self.fc = nn.ModuleList([nn.Linear(width, width) for _ in range(layers)])
        self.drop = nn.Dropout(0.1)
        self.basis = nn.Linear(width, backcast_length + 1)  # backcast(L) + forecast(1)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.act(self.proj(x))
        for f in self.fc:
            h = self.drop(self.act(f(h)))
        theta = self.basis(h)                       # (B, L+1)
        backcast = theta[:, :self.backcast_length]  # (B, L)
        forecast = theta[:, self.backcast_length:]  # (B, 1)
        return backcast, forecast


class NBEATSNet(nn.Module):
    def __init__(self, lookback: int, exo_dim: int, nblocks: int = 3,
                 width: int = 32, layers: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList(
            [_NBeatsBlock(width, layers, lookback) for _ in range(nblocks)])
        self.exo_lin = nn.Linear(exo_dim, 1) if exo_dim > 0 else None

    def forward(self, seq, exo):
        # seq: (B, L)
        residual = seq
        forecast_sum = torch.zeros(seq.size(0), 1, device=seq.device)
        for b in self.blocks:
            backcast, forecast = b(residual)
            residual = residual - backcast
            forecast_sum = forecast_sum + forecast
        if self.exo_lin is not None and exo is not None:
            forecast_sum = forecast_sum + self.exo_lin(exo)
        return forecast_sum


# -----------------------------------------------------------------------------
# 统一训练/预测包装
# -----------------------------------------------------------------------------
class DeepForecaster:
    """
    对一个“日总量序列”做一步前向预测的深度学习模型包装。

    参数
    ----
    kind      : 'lstm' / 'transformer' / 'nbeats'
    lookback  : 回看窗口天数（默认 14）
    exo_dim   : 外生特征维度；use_exo=False 时传 0
    use_exo   : 是否使用日历外生特征（消融用）
    epochs    : 训练轮数
    lr        : 学习率
    batch     : 批大小（小数据用全批）
    seed      : 随机种子
    """

    def __init__(self, kind: str, lookback: int = 14, exo_dim: int = 0,
                 use_exo: bool = True, epochs: int = 60, lr: float = 1e-3,
                 batch: int = 64, seed: int = 0):
        set_seed(seed)
        self.kind = kind.lower()
        self.lookback = lookback
        self.use_exo = use_exo
        self.exo_dim = exo_dim if use_exo else 0
        self.epochs = epochs
        self.lr = lr
        self.batch = batch
        self.seed = seed
        self._build()

    def _build(self):
        if self.kind == 'lstm':
            self.model = LSTNet(self.lookback, self.exo_dim)
        elif self.kind == 'transformer':
            self.model = TransformerNet(self.lookback, self.exo_dim)
        elif self.kind == 'nbeats':
            self.model = NBEATSNet(self.lookback, self.exo_dim)
        else:
            raise ValueError(f'未知深度模型: {self.kind}')
        self.device = torch.device('cpu')
        self.model.to(self.device)
        self.model.train()

    # ---- 标准化（仅用训练集统计量，避免泄露）----
    def _standardize(self, arr, mean, std):
        std = std if std > 1e-9 else 1.0
        return (arr - mean) / std

    def fit(self, seqs: np.ndarray, exos: np.ndarray, y: np.ndarray, verbose: bool = False):
        """
        训练。seqs: (N, L)；exos: (N, exo_dim) 或 None；y: (N,)。
        """
        seqs = np.asarray(seqs, float)
        y = np.asarray(y, float)
        # 统计量只在训练集上算
        self.seq_mean = float(seqs.mean())
        self.seq_std = float(seqs.std())
        self.y_mean = float(y.mean())
        self.y_std = float(y.std())
        s_seq = torch.tensor(self._standardize(seqs, self.seq_mean, self.seq_std),
                             dtype=torch.float32, device=self.device)
        s_y = torch.tensor(self._standardize(y, self.y_mean, self.y_std),
                           dtype=torch.float32, device=self.device).unsqueeze(-1)
        if self.exo_dim > 0 and exos is not None:
            x_exo = torch.tensor(np.asarray(exos, float), dtype=torch.float32,
                                 device=self.device)
        else:
            x_exo = None

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        n = s_seq.size(0)
        self.model.train()
        for ep in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]
                s_in = s_seq[idx]
                y_in = s_y[idx]
                exo_in = x_exo[idx] if x_exo is not None else None
                opt.zero_grad()
                pred = self.model(s_in, exo_in)
                loss = loss_fn(pred, y_in)
                loss.backward()
                opt.step()
        if verbose:
            print(f'  [{self.kind}{"+exo" if self.use_exo else ""}] 训练完成'
                  f'（样本 {n}，窗口 {self.lookback}）')
        return self

    def predict(self, seqs: np.ndarray, exos: np.ndarray = None) -> np.ndarray:
        """预测，返回反标准化后的日总量（≥0 裁剪，物理约束）。"""
        seqs = np.asarray(seqs, float)
        s_seq = torch.tensor(self._standardize(seqs, self.seq_mean, self.seq_std),
                             dtype=torch.float32, device=self.device)
        if self.exo_dim > 0 and exos is not None:
            x_exo = torch.tensor(np.asarray(exos, float), dtype=torch.float32,
                                 device=self.device)
        else:
            x_exo = None
        self.model.eval()
        with torch.no_grad():
            s_pred = self.model(s_seq, x_exo).cpu().numpy().ravel()
        pred = s_pred * self.y_std + self.y_mean
        return np.maximum(pred, 0.0)


_MODELS = {
    'LSTM': 'lstm',
    'Transformer': 'transformer',
    'N-BEATS': 'nbeats',
}


def available_deep_models():
    return list(_MODELS.keys())
