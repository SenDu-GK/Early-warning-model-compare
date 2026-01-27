from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn


def _ensure_dilations(dilations: Iterable[int] | None) -> list[int]:
    if dilations is None:
        return [1, 2, 4, 8]
    out = [int(d) for d in dilations]
    if not out:
        raise ValueError("DILATIONS must contain at least one value.")
    return out


def _pool_last(feats: torch.Tensor) -> torch.Tensor:
    # feats: [B, C, L]
    return feats[..., -1]


def _pool_mean(feats: torch.Tensor) -> torch.Tensor:
    # feats: [B, C, L]
    return feats.mean(dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None

    def _chomp(self, x: torch.Tensor, padding: int) -> torch.Tensor:
        if padding == 0:
            return x
        return x[..., :-padding]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad1 = self.conv1.padding[0]
        out = self.conv1(x)
        out = self._chomp(out, pad1)
        out = self.relu(out)
        out = self.dropout(out)

        pad2 = self.conv2.padding[0]
        out = self.conv2(out)
        out = self._chomp(out, pad2)
        out = self.relu(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        res = res[..., -out.shape[-1] :]
        return self.relu(out + res)


class TCNEncoder(nn.Module):
    def __init__(
        self,
        in_ch: int,
        hidden_size: int,
        kernel_size: int,
        dilations: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        ch_in = in_ch
        for d in dilations:
            layers.append(ResidualBlock(ch_in, hidden_size, dilation=int(d), kernel_size=kernel_size, dropout=dropout))
            ch_in = hidden_size
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class GRUEncoder(nn.Module):
    def __init__(self, in_ch: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_ch,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L] -> [B, L, C]
        x_seq = x.transpose(1, 2)
        out, _ = self.gru(x_seq)
        # return as [B, H, L] to match TCN-style pooling.
        return out.transpose(1, 2)


class TCNClassifier(nn.Module):
    def __init__(
        self,
        in_ch: int,
        hidden_size: int = 64,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4, 8),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = TCNEncoder(
            in_ch=in_ch,
            hidden_size=hidden_size,
            kernel_size=kernel_size,
            dilations=_ensure_dilations(dilations),
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        pooled = _pool_mean(feats)
        logits = self.head(pooled)
        return logits.squeeze(-1)


class GRUClassifier(nn.Module):
    def __init__(
        self,
        in_ch: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = GRUEncoder(in_ch=in_ch, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        pooled = _pool_last(feats)
        logits = self.head(pooled)
        return logits.squeeze(-1)


# Survival (hazard) models -----------------------------------------------------


class SurvivalTCN(nn.Module):
    def __init__(
        self,
        in_ch: int,
        hidden_size: int = 64,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4, 8),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = TCNEncoder(
            in_ch=in_ch,
            hidden_size=hidden_size,
            kernel_size=kernel_size,
            dilations=_ensure_dilations(dilations),
            dropout=dropout,
        )
        self.hazard_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        pooled = _pool_mean(feats)
        logits = self.hazard_head(pooled)
        return logits.squeeze(-1)


class SurvivalGRU(nn.Module):
    def __init__(
        self,
        in_ch: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = GRUEncoder(in_ch=in_ch, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.hazard_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        pooled = _pool_last(feats)
        logits = self.hazard_head(pooled)
        return logits.squeeze(-1)


# Forecasting models -----------------------------------------------------------


class ForecastTCN(nn.Module):
    def __init__(
        self,
        in_ch: int,
        horizon_f: int,
        hidden_size: int = 64,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4, 8),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.horizon_f = int(horizon_f)
        self.encoder = TCNEncoder(
            in_ch=in_ch,
            hidden_size=hidden_size,
            kernel_size=kernel_size,
            dilations=_ensure_dilations(dilations),
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.horizon_f),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        pooled = _pool_mean(feats)
        out = self.head(pooled)
        return out


class ForecastGRU(nn.Module):
    def __init__(
        self,
        in_ch: int,
        horizon_f: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.horizon_f = int(horizon_f)
        self.encoder = GRUEncoder(in_ch=in_ch, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.horizon_f),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        pooled = _pool_last(feats)
        out = self.head(pooled)
        return out


def build_model(cfg: dict, in_ch: int) -> nn.Module:
    task = str(cfg.get("TASK", "leadtime_cls")).lower()
    model_cfg = cfg["MODEL"]
    name = str(model_cfg.get("NAME", "tcn")).lower()

    tcn_cfg = model_cfg.get("TCN", {})
    gru_cfg = model_cfg.get("GRU", {})

    if task == "leadtime_cls":
        if name == "tcn":
            return TCNClassifier(
                in_ch=in_ch,
                hidden_size=int(tcn_cfg.get("HIDDEN_SIZE", 64)),
                kernel_size=int(tcn_cfg.get("KERNEL_SIZE", 3)),
                dilations=tcn_cfg.get("DILATIONS", [1, 2, 4, 8]),
                dropout=float(tcn_cfg.get("DROPOUT", 0.1)),
            )
        if name == "gru":
            return GRUClassifier(
                in_ch=in_ch,
                hidden_size=int(gru_cfg.get("HIDDEN_SIZE", 64)),
                num_layers=int(gru_cfg.get("NUM_LAYERS", 1)),
                dropout=float(gru_cfg.get("DROPOUT", 0.1)),
            )

    if task == "survival_discrete":
        if name == "tcn":
            return SurvivalTCN(
                in_ch=in_ch,
                hidden_size=int(tcn_cfg.get("HIDDEN_SIZE", 64)),
                kernel_size=int(tcn_cfg.get("KERNEL_SIZE", 3)),
                dilations=tcn_cfg.get("DILATIONS", [1, 2, 4, 8]),
                dropout=float(tcn_cfg.get("DROPOUT", 0.1)),
            )
        if name == "gru":
            return SurvivalGRU(
                in_ch=in_ch,
                hidden_size=int(gru_cfg.get("HIDDEN_SIZE", 64)),
                num_layers=int(gru_cfg.get("NUM_LAYERS", 1)),
                dropout=float(gru_cfg.get("DROPOUT", 0.1)),
            )

    if task == "forecast_anom":
        horizon_f = int(cfg.get("FORECAST", {}).get("HORIZON_F", 5))
        if name == "tcn":
            return ForecastTCN(
                in_ch=in_ch,
                horizon_f=horizon_f,
                hidden_size=int(tcn_cfg.get("HIDDEN_SIZE", 64)),
                kernel_size=int(tcn_cfg.get("KERNEL_SIZE", 3)),
                dilations=tcn_cfg.get("DILATIONS", [1, 2, 4, 8]),
                dropout=float(tcn_cfg.get("DROPOUT", 0.1)),
            )
        if name == "gru":
            return ForecastGRU(
                in_ch=in_ch,
                horizon_f=horizon_f,
                hidden_size=int(gru_cfg.get("HIDDEN_SIZE", 64)),
                num_layers=int(gru_cfg.get("NUM_LAYERS", 1)),
                dropout=float(gru_cfg.get("DROPOUT", 0.1)),
            )

    raise ValueError(f"Unknown task/model combination: task={task}, model={name}")
