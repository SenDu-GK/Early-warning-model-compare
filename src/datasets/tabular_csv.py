from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


Task = Literal["classification", "regression"]


class TabularDataset(Dataset):
    """Torch dataset that supports both classification and regression labels."""

    def __init__(self, X: np.ndarray, y: np.ndarray, task: Task):
        self.X = torch.tensor(X, dtype=torch.float32)
        if task == "classification":
            self.y = torch.tensor(y, dtype=torch.long)
        else:
            self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    in_dim: int
    out_dim: int
    task: Task
    classes: list[str] | None = None


def _normalize_drop_cols(df: pd.DataFrame, drop_cols: Iterable[str] | None) -> Sequence[str]:
    if not drop_cols:
        return []
    missing = [c for c in drop_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns to drop not found: {missing}")
    return list(drop_cols)


def make_loaders_from_csv(
    csv_path: str,
    label_col: str,
    drop_cols: Iterable[str] | None = None,
    task: Task = "regression",
    batch_size: int = 64,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> DataBundle:
    """Create train/val loaders from a CSV with scaling and optional column dropping."""

    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise ValueError(f"label_col='{label_col}' not found in columns: {list(df.columns)}")

    drop_cols_list = _normalize_drop_cols(df, drop_cols)
    feature_df = df.drop(columns=[label_col] + drop_cols_list)

    if task == "classification":
        y, classes = pd.factorize(df[label_col])
        classes = [str(c) for c in classes]
        out_dim = len(classes)
    else:
        y = df[label_col].to_numpy(dtype=np.float32)
        classes = None
        out_dim = 1

    X = feature_df.to_numpy(dtype=np.float32)

    stratify = y if task == "classification" else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_ratio, random_state=seed, stratify=stratify
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    train_ds = TabularDataset(X_train, y_train, task)
    val_ds = TabularDataset(X_val, y_val, task)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    in_dim = X_train.shape[1]

    return DataBundle(train_loader, val_loader, in_dim, out_dim, task, classes)
