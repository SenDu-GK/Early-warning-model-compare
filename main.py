from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from geopandas import GeoDataFrame
from shapely.geometry import Polygon
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import build_model

warnings.filterwarnings("ignore", category=UserWarning)


DEFAULT_CONFIG_PATH = Path("configs/leadtime_tcn.yaml")
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_SPLITS_DIR = DEFAULT_OUTPUT_DIR / "splits"


# -----------------------------------------------------------------------------
# Config + setup
# -----------------------------------------------------------------------------


def ensure_run_dirs(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "risk_maps_csv").mkdir(parents=True, exist_ok=True)
    (run_dir / "risk_maps_png_latlon").mkdir(parents=True, exist_ok=True)
    (run_dir / "risk_maps_png_utm").mkdir(parents=True, exist_ok=True)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping/dictionary.")
    return cfg


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Data + geometry
# -----------------------------------------------------------------------------


def load_points_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"ID", "Fi", "Lambda"}
    if not required.issubset(df.columns):
        raise ValueError("Input CSV must contain columns: ID, Fi, Lambda")
    return df


def detect_and_sort_epochs(df: pd.DataFrame) -> Tuple[List[str], List[pd.Timestamp]]:
    epoch_cols = [c for c in df.columns if re.match(r"^D\d{8}$", c)]
    if not epoch_cols:
        raise ValueError("No displacement columns detected with pattern ^DYYYYMMDD$")
    epoch_dates = [pd.to_datetime(c[1:], format="%Y%m%d") for c in epoch_cols]
    sorted_pairs = sorted(zip(epoch_cols, epoch_dates), key=lambda x: x[1])
    sorted_cols, sorted_dates = zip(*sorted_pairs)
    return list(sorted_cols), list(sorted_dates)


def _parse_polygon_from_kml(kml_path: str) -> Polygon:
    tree = ET.parse(kml_path)
    root = tree.getroot()
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    coords_el = root.find(
        ".//kml:Polygon/kml:outerBoundaryIs/kml:LinearRing/kml:coordinates", ns
    )
    if coords_el is None or not coords_el.text:
        raise ValueError("Failed to find Polygon coordinates in KML.")
    coords_text = coords_el.text.strip()
    coords: list[tuple[float, float]] = []
    for token in coords_text.replace("\n", " ").split():
        lon_lat_alt = token.split(",")
        if len(lon_lat_alt) < 2:
            continue
        lon, lat = map(float, lon_lat_alt[:2])
        coords.append((lon, lat))
    if len(coords) < 3:
        raise ValueError("Polygon requires at least 3 coordinate pairs.")
    return Polygon(coords)


def load_polygon_kml(kml_path: str) -> Polygon:
    try:
        gdf = gpd.read_file(kml_path, driver="KML")
        if gdf.empty:
            raise ValueError("Empty KML file.")
        geom = gdf.geometry.iloc[0]
        if geom.geom_type == "MultiPolygon":
            geom = max(geom.geoms, key=lambda g: g.area)
        return Polygon(geom.exterior.coords)
    except Exception:
        return _parse_polygon_from_kml(kml_path)


def compute_utm_epsg(polygon: Polygon) -> int:
    centroid = polygon.centroid
    lon, lat = centroid.x, centroid.y
    zone = int((lon + 180) // 6) + 1
    if lat >= 0:
        return 32600 + zone
    return 32700 + zone


def label_points(
    df: pd.DataFrame,
    polygon: Polygon,
    epsg_utm: int,
    buffer_m: float,
    near_min_m: float,
    near_max_m: float,
    far_min_m: float,
) -> Tuple[GeoDataFrame, GeoDataFrame, Polygon, dict[str, int]]:
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["Lambda"], df["Fi"]),
        crs="EPSG:4326",
    )
    poly_gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")

    gdf_utm = gdf.to_crs(epsg=epsg_utm)
    poly_utm = poly_gdf.to_crs(epsg=epsg_utm)
    poly_shape = poly_utm.geometry.iloc[0]

    inside = gdf_utm.within(poly_shape)
    dist = gdf_utm.distance(poly_shape)

    labels = np.full(len(gdf), -2, dtype=int)  # -2 ignored, 2 buffer, 3 near, 0 far, 1 core
    labels[inside.to_numpy()] = 1

    buffer_mask = (~inside.to_numpy()) & (dist <= buffer_m)
    labels[buffer_mask] = 2

    near_lower = max(buffer_m, near_min_m)
    near_mask = (~inside.to_numpy()) & (dist > near_lower) & (dist <= near_max_m)
    labels[near_mask] = 3

    far_mask = (~inside.to_numpy()) & (dist >= far_min_m)
    labels[far_mask] = 0

    label_text = np.full(len(gdf), "ignored", dtype=object)
    label_text[labels == 1] = "core"
    label_text[labels == 3] = "near-field"
    label_text[labels == 2] = "buffer"
    label_text[labels == 0] = "far-field"

    gdf_utm["distance_m"] = dist.to_numpy()
    gdf_utm["label"] = labels
    gdf_utm["label_text"] = label_text
    gdf_utm["E_utm"] = gdf_utm.geometry.x
    gdf_utm["N_utm"] = gdf_utm.geometry.y

    gdf["distance_m"] = gdf_utm["distance_m"].to_numpy()
    gdf["label"] = labels
    gdf["label_text"] = label_text

    total = len(gdf)
    core = int((labels == 1).sum())
    near = int((labels == 3).sum())
    far = int((labels == 0).sum())
    buffer_n = int((labels == 2).sum())
    ignored = total - core - near - far - buffer_n
    print(
        "Label counts: "
        f"total={total}, core={core}, near-field={near}, far-field={far}, buffer={buffer_n}, ignored={ignored}"
    )

    label_map = {"far-field": 0, "core": 1, "buffer": 2, "near-field": 3, "ignored": -2}
    return gdf, gdf_utm, poly_shape, label_map


def get_index_for_date(epoch_dates: Sequence[pd.Timestamp], date_str: str) -> int:
    target = pd.Timestamp(date_str)
    diffs = [abs((d - target).days) for d in epoch_dates]
    idx = int(np.argmin(diffs))
    if epoch_dates[idx] != target:
        print(
            "Warning: exact date not found in epochs. "
            f"Using nearest epoch {epoch_dates[idx].date()} for requested {target.date()}"
        )
    return idx


# -----------------------------------------------------------------------------
# Feature construction (vectorized where it matters)
# -----------------------------------------------------------------------------


def build_feature_channels_batch(window_vals: np.ndarray, input_mode: str) -> np.ndarray:
    """Vectorized feature construction.

    window_vals: [N, L] with NaNs allowed.
    Returns: [N, C, L]
    """
    if window_vals.ndim != 2:
        raise ValueError(f"window_vals must be [N, L], got shape={window_vals.shape}")

    mask = ~np.isnan(window_vals)
    any_valid = mask.any(axis=1)
    first_idx = mask.argmax(axis=1)

    # Gather first valid values; when none valid, baseline=0.
    gathered = np.take_along_axis(window_vals, first_idx[:, None], axis=1).squeeze(1)
    baseline = np.where(any_valid, gathered, 0.0).astype(np.float32)

    vals = np.where(np.isnan(window_vals), baseline[:, None], window_vals).astype(np.float32)
    x_rel = vals - baseline[:, None]
    mask_f = mask.astype(np.float32)

    if input_mode == "2ch":
        return np.stack([x_rel, mask_f], axis=1)

    if input_mode != "4ch":
        raise ValueError(f"Unsupported INPUT_MODE: {input_mode}")

    dx = np.zeros_like(x_rel)
    dx[:, 1:] = x_rel[:, 1:] - x_rel[:, :-1]

    d2x = np.zeros_like(x_rel)
    d2x[:, 1:] = dx[:, 1:] - dx[:, :-1]

    return np.stack([x_rel, dx, d2x, mask_f], axis=1)


def build_infer_windows_batch(values: np.ndarray, k: int, window_length: int, input_mode: str) -> np.ndarray:
    start = k - window_length + 1
    window_vals = values[:, start : k + 1]
    return build_feature_channels_batch(window_vals, input_mode)


def near_event_range(t_last: int, horizon_h: int) -> range:
    start = max(0, t_last - horizon_h)
    end = max(0, t_last)
    return range(start, end)


# -----------------------------------------------------------------------------
# Splits (shared across all tasks/runs)
# -----------------------------------------------------------------------------


def split_point_ids(point_ids: Iterable[str], point_is_core: Dict[str, int], split_cfg: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.array(sorted(set(point_ids)))
    seed = int(split_cfg["SEED"])
    train_frac = float(split_cfg["TRAIN_FRAC"])
    val_frac = float(split_cfg["VAL_FRAC"])
    test_frac = float(split_cfg["TEST_FRAC"])

    frac_sum = train_frac + val_frac + test_frac
    if not np.isclose(frac_sum, 1.0):
        raise ValueError(f"Split fractions must sum to 1.0, got {frac_sum}")

    stratify = np.array([point_is_core.get(pid, 0) for pid in ids])
    stratify_arg = stratify if len(np.unique(stratify)) > 1 else None

    try:
        train_ids, temp_ids = train_test_split(
            ids,
            test_size=(1.0 - train_frac),
            random_state=seed,
            stratify=stratify_arg,
        )
    except ValueError:
        train_ids, temp_ids = train_test_split(
            ids,
            test_size=(1.0 - train_frac),
            random_state=seed,
            stratify=None,
        )

    rel_test = test_frac / (val_frac + test_frac)
    temp_stratify = np.array([point_is_core.get(pid, 0) for pid in temp_ids])
    temp_stratify_arg = temp_stratify if len(np.unique(temp_stratify)) > 1 else None

    try:
        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=rel_test,
            random_state=seed,
            stratify=temp_stratify_arg,
        )
    except ValueError:
        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=rel_test,
            random_state=seed,
            stratify=None,
        )

    return np.array(train_ids), np.array(val_ids), np.array(test_ids)


def save_point_split(split_path: Path, train_ids: np.ndarray, val_ids: np.ndarray, test_ids: np.ndarray) -> None:
    split_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_ids": [str(x) for x in train_ids.tolist()],
        "val_ids": [str(x) for x in val_ids.tolist()],
        "test_ids": [str(x) for x in test_ids.tolist()],
    }
    split_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_point_split(split_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    return (
        np.array(payload["train_ids"], dtype=object),
        np.array(payload["val_ids"], dtype=object),
        np.array(payload["test_ids"], dtype=object),
    )


def resolve_shared_split(
    split_path: Path,
    all_point_ids: Sequence[str],
    point_is_core: Dict[str, int],
    split_cfg: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if split_path.exists():
        train_ids, val_ids, test_ids = load_point_split(split_path)
        print(f"Loaded shared point split: {split_path}")
        return train_ids, val_ids, test_ids

    train_ids, val_ids, test_ids = split_point_ids(all_point_ids, point_is_core, split_cfg)
    save_point_split(split_path, train_ids, val_ids, test_ids)
    print(f"Saved shared point split: {split_path}")
    return train_ids, val_ids, test_ids


# -----------------------------------------------------------------------------
# Lead-time classification task
# -----------------------------------------------------------------------------


def label_for_sample_leadtime(
    label_text: str,
    k: int,
    t_last: int,
    horizon_h: int,
    neg_mode: str,
) -> int:
    near_rng = near_event_range(t_last, horizon_h)
    if label_text == "core":
        if k in near_rng:
            return 1
        if k < (t_last - horizon_h):
            return 0
        return -1
    if label_text == "near-field":
        return 0
    if label_text == "far-field" and neg_mode == "near_plus_far":
        return 0
    return -1


def build_training_windows_leadtime(
    values: np.ndarray,
    ids: np.ndarray,
    label_texts: np.ndarray,
    epoch_cols: Sequence[str],
    epoch_dates: Sequence[pd.Timestamp],
    t_last: int,
    window_length: int,
    horizon_h: int,
    input_mode: str,
    neg_mode: str,
) -> Tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], pd.DataFrame]:
    n_points, n_epochs = values.shape
    start_k = window_length - 1
    end_k = min(t_last - 1, n_epochs - 1)

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    meta_rows: list[dict] = []

    point_to_indices: dict[str, list[int]] = {}

    for i in range(n_points):
        pid = str(ids[i])
        p_label = str(label_texts[i])
        point_to_indices.setdefault(pid, [])

        for k in range(start_k, end_k + 1):
            y = label_for_sample_leadtime(p_label, k, t_last, horizon_h, neg_mode)
            if y < 0:
                continue

            start = k - window_length + 1
            window_vals = values[i, start : k + 1][None, :]
            feats = build_feature_channels_batch(window_vals, input_mode)[0]
            X_list.append(feats)
            y_list.append(y)

            idx = len(y_list) - 1
            point_to_indices[pid].append(idx)

            meta_rows.append(
                {
                    "sample_idx": idx,
                    "point_id": pid,
                    "point_label": p_label,
                    "k": k,
                    "epoch_col": epoch_cols[k],
                    "epoch_date": epoch_dates[k],
                    "y": y,
                }
            )

    if not X_list:
        raise ValueError("No labeled samples were created. Check labeling and horizon settings.")

    X = np.stack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    meta = pd.DataFrame(meta_rows)

    point_to_idx_arr = {pid: np.asarray(idxs, dtype=np.int64) for pid, idxs in point_to_indices.items()}
    return X, y, point_to_idx_arr, meta


def indices_for_points(point_ids: Sequence[str], point_to_indices: dict[str, np.ndarray]) -> np.ndarray:
    parts: list[np.ndarray] = []
    for pid in point_ids:
        idxs = point_to_indices.get(str(pid))
        if idxs is not None and len(idxs) > 0:
            parts.append(idxs)
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(parts)


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def _compute_pos_weight(y: np.ndarray) -> torch.Tensor:
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(max(neg / pos, 1.0), dtype=torch.float32)


def train_model_binary(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, cfg: dict, run_dir: Path) -> Path:
    if len(train_loader.dataset) == 0:
        raise ValueError("Training dataset is empty. Check labels and splits.")

    y_train = train_loader.dataset.y.numpy()
    pos_weight = _compute_pos_weight(y_train).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_cfg = cfg["TRAINING"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["LR"]),
        weight_decay=float(train_cfg["WEIGHT_DECAY"]),
    )

    max_epochs = int(train_cfg["MAX_EPOCHS"])
    patience = int(train_cfg["PATIENCE"])

    best_val = float("inf")
    best_path = run_dir / "best_model.pt"
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses: list[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                val_losses.append(float(loss_fn(logits, yb).item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience}).")
                break

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    return best_path


def evaluate_auc_binary(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    if len(loader.dataset) == 0:
        return float("nan")
    model.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            scores.append(probs)
            labels.append(yb.numpy())
    y_true = np.concatenate(labels)
    y_score = np.concatenate(scores)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def compute_risk_cube_leadtime(
    model: nn.Module,
    values: np.ndarray,
    k_indices: Sequence[int],
    window_length: int,
    input_mode: str,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    risk_cube = np.full((len(k_indices), values.shape[0]), np.nan, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for j, k in enumerate(k_indices):
            Xk = build_infer_windows_batch(values, k, window_length, input_mode)
            risks = batched_predict_sigmoid(model, Xk, device=device, batch_size=batch_size)
            risk_cube[j] = risks
    return risk_cube


# -----------------------------------------------------------------------------
# Discrete-time survival task
# -----------------------------------------------------------------------------


class SurvivalPointDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, event_flag: np.ndarray, point_ids: np.ndarray) -> None:
        # X_seq: [N, T, C, L]
        self.X_seq = torch.from_numpy(X_seq)
        self.event_flag = torch.from_numpy(event_flag.astype(np.float32))
        self.point_ids = point_ids

    def __len__(self) -> int:
        return int(self.X_seq.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X_seq[idx], self.event_flag[idx]


def precompute_seq_windows(values: np.ndarray, k_indices: Sequence[int], window_length: int, input_mode: str) -> np.ndarray:
    n_points = values.shape[0]
    n_steps = len(k_indices)
    n_ch = 4 if input_mode == "4ch" else 2
    X_seq = np.empty((n_points, n_steps, n_ch, window_length), dtype=np.float32)
    for j, k in enumerate(k_indices):
        X_seq[:, j] = build_infer_windows_batch(values, k, window_length, input_mode)
    return X_seq


def survival_nll_loss(h: torch.Tensor, event_flag: torch.Tensor, pos_weight: float) -> torch.Tensor:
    """Discrete-time survival NLL.

    h: [B, T] hazards in (0,1)
    event_flag: [B] (1 for core event at last step, 0 for censored)
    pos_weight: weighting multiplier for event points
    """
    eps = 1e-6
    h = h.clamp(min=eps, max=1.0 - eps)
    log_surv = torch.log1p(-h)

    t_event = h.shape[1] - 1
    surv_before = log_surv[:, :t_event].sum(dim=1)
    log_h_event = torch.log(h[:, t_event])

    nll_event = -(surv_before + log_h_event)
    nll_cens = -(log_surv.sum(dim=1))

    weights = torch.where(event_flag > 0.5, torch.full_like(event_flag, float(pos_weight)), torch.ones_like(event_flag))
    losses = torch.where(event_flag > 0.5, nll_event, nll_cens) * weights
    return losses.mean()


def train_model_survival(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: dict,
    run_dir: Path,
) -> Path:
    train_cfg = cfg["TRAINING"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["LR"]),
        weight_decay=float(train_cfg["WEIGHT_DECAY"]),
    )

    max_epochs = int(train_cfg["MAX_EPOCHS"])
    patience = int(train_cfg["PATIENCE"])

    y_train = train_loader.dataset.event_flag.numpy()
    n_event = float((y_train > 0.5).sum())
    n_cens = float((y_train <= 0.5).sum())
    pos_weight = max(n_cens / max(n_event, 1.0), 1.0)
    print(f"Survival pos_weight (event vs censored): {pos_weight:.3f}")

    best_val = float("inf")
    best_path = run_dir / "best_model.pt"
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses: list[float] = []
        for xb_seq, event_b in train_loader:
            xb_seq = xb_seq.to(device)  # [B, T, C, L]
            event_b = event_b.to(device)  # [B]

            B, T, C, L = xb_seq.shape
            xb_flat = xb_seq.view(B * T, C, L)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb_flat).view(B, T)
            h = torch.sigmoid(logits)
            loss = survival_nll_loss(h, event_b, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for xb_seq, event_b in val_loader:
                xb_seq = xb_seq.to(device)
                event_b = event_b.to(device)
                B, T, C, L = xb_seq.shape
                xb_flat = xb_seq.view(B * T, C, L)
                logits = model(xb_flat).view(B, T)
                h = torch.sigmoid(logits)
                val_losses.append(float(survival_nll_loss(h, event_b, pos_weight=pos_weight).item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience}).")
                break

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    return best_path


def compute_risk_cube_survival(
    model: nn.Module,
    X_seq: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Compute hazards and cumulative risk R_k for all points/epochs.

    X_seq: [N, T, C, L]
    Returns risk_cube: [T, N]
    """
    model.eval()
    N, T, C, L = X_seq.shape
    hazards = np.empty((N, T), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, N, batch_size):
            xb = torch.from_numpy(X_seq[i : i + batch_size]).to(device)
            B = xb.shape[0]
            xb_flat = xb.view(B * T, C, L)
            logits = model(xb_flat).view(B, T)
            h = torch.sigmoid(logits).detach().cpu().numpy()
            hazards[i : i + batch_size] = h

    hazards = np.clip(hazards, 1e-6, 1.0 - 1e-6)
    surv = np.cumprod(1.0 - hazards, axis=1)
    risk = 1.0 - surv
    return risk.T.astype(np.float32)


# -----------------------------------------------------------------------------
# Forecasting + anomaly risk task
# -----------------------------------------------------------------------------


class ForecastDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, mask: np.ndarray) -> None:
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.mask = torch.from_numpy(mask)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx], self.mask[idx]


def build_future_targets(values: np.ndarray, k: int, horizon_f: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return future values and mask with padding to horizon_f.

    values: [N, E]
    Returns y_true, mask: [N, Hf]
    """
    N, E = values.shape
    y = np.full((N, horizon_f), np.nan, dtype=np.float32)
    start = k + 1
    end = min(E, start + horizon_f)
    if start < end:
        y[:, : end - start] = values[:, start:end]
    mask = ~np.isnan(y)
    return y, mask.astype(np.float32)


def build_forecast_samples(
    values: np.ndarray,
    row_indices: np.ndarray,
    k_indices: Sequence[int],
    window_length: int,
    horizon_f: int,
    input_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    m_list: list[np.ndarray] = []

    for k in k_indices:
        Xk_all = build_infer_windows_batch(values, k, window_length, input_mode)
        y_all, m_all = build_future_targets(values, k, horizon_f)

        Xk = Xk_all[row_indices]
        yk = y_all[row_indices]
        mk = m_all[row_indices]

        # Keep only samples with at least one observed future step.
        keep = mk.sum(axis=1) > 0
        if not np.any(keep):
            continue
        X_list.append(Xk[keep])
        y_list.append(yk[keep])
        m_list.append(mk[keep])

    if not X_list:
        raise ValueError("No forecast samples created. Check horizon and k range.")

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.float32)
    m = np.concatenate(m_list, axis=0).astype(np.float32)
    return X, y, m


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    target_filled = torch.nan_to_num(target, nan=0.0)
    err = torch.abs(pred - target_filled) * mask
    denom = torch.clamp(mask.sum(), min=1.0)
    return err.sum() / denom


def train_model_forecast(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: dict,
    run_dir: Path,
) -> Path:
    train_cfg = cfg["TRAINING"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["LR"]),
        weight_decay=float(train_cfg["WEIGHT_DECAY"]),
    )

    max_epochs = int(train_cfg["MAX_EPOCHS"])
    patience = int(train_cfg["PATIENCE"])

    best_val = float("inf")
    best_path = run_dir / "best_model.pt"
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses: list[float] = []
        for xb, yb, mb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = masked_mae(pred, yb, mb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for xb, yb, mb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                mb = mb.to(device)
                pred = model(xb)
                val_losses.append(float(masked_mae(pred, yb, mb).item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience}).")
                break

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    return best_path


def batched_predict_sigmoid(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    out = np.empty(X.shape[0], dtype=np.float32)
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).to(device)
            logits = model(xb)
            out[i : i + batch_size] = torch.sigmoid(logits).detach().cpu().numpy()
    return out


def batched_predict_forecast(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    out = np.empty((X.shape[0], model.horizon_f), dtype=np.float32)  # type: ignore[attr-defined]
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).to(device)
            pred = model(xb).detach().cpu().numpy()
            out[i : i + batch_size] = pred
    return out


def compute_anomaly_cube_forecast(
    model: nn.Module,
    values: np.ndarray,
    k_indices: Sequence[int],
    window_length: int,
    horizon_f: int,
    input_mode: str,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    anom_cube = np.full((len(k_indices), values.shape[0]), np.nan, dtype=np.float32)

    for j, k in enumerate(k_indices):
        Xk = build_infer_windows_batch(values, k, window_length, input_mode)
        pred = batched_predict_forecast(model, Xk, device=device, batch_size=batch_size)

        y_true, mask_true = build_future_targets(values, k, horizon_f)
        err = np.abs(pred - np.nan_to_num(y_true, nan=0.0)) * mask_true
        denom = mask_true.sum(axis=1)
        valid = denom > 0
        anom = np.full(values.shape[0], np.nan, dtype=np.float32)
        anom[valid] = (err[valid].sum(axis=1) / denom[valid]).astype(np.float32)
        anom_cube[j] = anom

    return anom_cube


def calibrate_anomaly_to_risk(
    anom_cube: np.ndarray,
    k_indices: Sequence[int],
    near_idx: np.ndarray,
    calib_end_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    early_mask = np.array([k <= calib_end_k for k in k_indices])
    baseline = anom_cube[early_mask][:, near_idx].ravel() if len(near_idx) else np.array([])
    baseline = baseline[~np.isnan(baseline)]
    if len(baseline) == 0:
        raise ValueError("No baseline anomalies available for calibration.")

    baseline_sorted = np.sort(baseline.astype(np.float32))
    denom = float(len(baseline_sorted))

    risk_cube = np.full_like(anom_cube, np.nan, dtype=np.float32)
    flat = anom_cube.ravel()
    valid = ~np.isnan(flat)
    ranks = np.searchsorted(baseline_sorted, flat[valid], side="right").astype(np.float32)
    risk_vals = ranks / denom
    risk_cube.ravel()[valid] = risk_vals
    return risk_cube.astype(np.float32), baseline_sorted


# -----------------------------------------------------------------------------
# Shared evaluation + outputs (risk_cube-driven)
# -----------------------------------------------------------------------------


def bootstrap_auc_ci(pos: np.ndarray, neg: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    for _ in range(n_boot):
        pos_s = pos[rng.integers(0, len(pos), len(pos))]
        neg_s = neg[rng.integers(0, len(neg), len(neg))]
        y = np.concatenate([np.ones_like(pos_s), np.zeros_like(neg_s)])
        s = np.concatenate([pos_s, neg_s])
        if len(np.unique(y)) < 2:
            continue
        try:
            aucs.append(float(roc_auc_score(y, s)))
        except ValueError:
            continue
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def bootstrap_gap_ci(pos: np.ndarray, neg: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    gaps: list[float] = []
    for _ in range(n_boot):
        pos_s = pos[rng.integers(0, len(pos), len(pos))]
        neg_s = neg[rng.integers(0, len(neg), len(neg))]
        gaps.append(float(np.nanmean(pos_s) - np.nanmean(neg_s)))
    return float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))


def permutation_p_value_auc(pos: np.ndarray, neg: np.ndarray, n_perm: int, seed: int) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    y = np.concatenate([np.ones(len(pos), dtype=np.int32), np.zeros(len(neg), dtype=np.int32)])
    s = np.concatenate([pos, neg])
    if len(np.unique(y)) < 2:
        return float("nan")
    observed = float(roc_auc_score(y, s))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        try:
            auc = float(roc_auc_score(yp, s))
        except ValueError:
            continue
        if auc >= observed:
            count += 1
    return float((count + 1) / (n_perm + 1))


def earliest_consecutive_exceedance(k_indices: Sequence[int], values: Sequence[float], tau: float, m: int) -> int | None:
    run = 0
    for k, v in zip(k_indices, values):
        if np.isnan(v):
            run = 0
            continue
        if v > tau:
            run += 1
            if run >= m:
                return k
        else:
            run = 0
    return None


def quantile_higher(x: np.ndarray, q: float) -> float:
    try:
        return float(np.quantile(x, q, method="higher"))
    except TypeError:
        return float(np.quantile(x, q, interpolation="higher"))


def compute_trend_pvalues(mean_df: pd.DataFrame, auc_df: pd.DataFrame) -> dict:
    def _spearman_p(x: np.ndarray, y: np.ndarray) -> float:
        mask = (~np.isnan(x)) & (~np.isnan(y))
        if mask.sum() < 3:
            return float("nan")
        xr = pd.Series(x[mask]).rank().to_numpy(dtype=float)
        yr = pd.Series(y[mask]).rank().to_numpy(dtype=float)
        r = np.corrcoef(xr, yr)[0, 1]
        n = mask.sum()
        if not np.isfinite(r) or n < 3:
            return float("nan")
        t = r * np.sqrt((n - 2) / max(1e-12, 1 - r * r))
        p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0))))
        return float(p)

    x = pd.to_datetime(mean_df["epoch_date"]).map(pd.Timestamp.toordinal).to_numpy(dtype=float)
    core = mean_df["core_mean_risk"].to_numpy(dtype=float)
    auc = auc_df["auc"].to_numpy(dtype=float) if not auc_df.empty else np.array([])

    core_p = _spearman_p(x, core)
    auc_p = _spearman_p(x[: len(auc)], auc) if len(auc) else float("nan")
    return {"core_mean_risk_p": core_p, "auc_core_vs_near_p": auc_p}


def evaluate_and_save_outputs(
    risk_cube: np.ndarray,
    k_indices: Sequence[int],
    values: np.ndarray,
    df_latlon: GeoDataFrame,
    df_utm: GeoDataFrame,
    polygon_latlon: Polygon,
    polygon_utm: Polygon,
    epoch_cols: Sequence[str],
    epoch_dates: Sequence[pd.Timestamp],
    t_last: int,
    cfg: dict,
    run_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    label_cfg = cfg["LABELING"]
    infer_cfg = cfg["INFERENCE"]
    stats_cfg = cfg["STATS"]
    thresh_cfg = cfg["THRESHOLDING"]
    plot_cfg = cfg["PLOTTING"]

    horizon_h = int(label_cfg["HORIZON_H"])
    neg_mode = str(label_cfg["NEG_MODE"]).lower()

    top_percent = float(infer_cfg["TOP_PERCENT"])
    dpi = int(plot_cfg["DPI"])

    label_texts = df_utm["label_text"].to_numpy()
    core_idx = np.where(label_texts == "core")[0]
    near_idx = np.where(label_texts == "near-field")[0]
    far_idx = np.where(label_texts == "far-field")[0]

    near_rng = set(near_event_range(t_last, horizon_h))

    risk_csv_dir = run_dir / "risk_maps_csv"
    png_latlon_dir = run_dir / "risk_maps_png_latlon"
    png_utm_dir = run_dir / "risk_maps_png_utm"

    # Risk maps.
    for j, k in enumerate(k_indices):
        risks = risk_cube[j]
        epoch_label = epoch_cols[k]

        thresh = float(np.nanpercentile(risks, 100.0 * (1.0 - top_percent)))
        top_mask = risks >= thresh

        risk_df = pd.DataFrame(
            {
                "ID": df_latlon["ID"].astype(str),
                "Fi": df_latlon["Fi"],
                "Lambda": df_latlon["Lambda"],
                "E_utm": df_utm["E_utm"],
                "N_utm": df_utm["N_utm"],
                "distance_m": df_utm["distance_m"],
                "label_text": df_utm["label_text"],
                "epoch_col": epoch_label,
                "epoch_date": epoch_dates[k].date(),
                "risk_pk": risks,
                f"top{int(top_percent*100)}": top_mask.astype(int),
            }
        )
        risk_df.to_csv(risk_csv_dir / f"risk_{epoch_label}.csv", index=False)

        poly_coords = np.array(polygon_latlon.exterior.coords)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(poly_coords[:, 0], poly_coords[:, 1], color="black", linewidth=1.5)
        sc = ax.scatter(
            df_latlon.loc[top_mask, "Lambda"],
            df_latlon.loc[top_mask, "Fi"],
            c=risks[top_mask],
            s=6,
            cmap="viridis",
        )
        ax.set_title(f"{epoch_label} top{int(top_percent*100)}% risk (lat/lon)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Risk")
        fig.tight_layout()
        fig.savefig(png_latlon_dir / f"risk_top_latlon_{epoch_label}.png", dpi=dpi)
        plt.close(fig)

        poly_utm_coords = np.array(polygon_utm.exterior.coords)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(poly_utm_coords[:, 0], poly_utm_coords[:, 1], color="black", linewidth=1.5)
        sc = ax.scatter(
            df_utm.loc[top_mask, "E_utm"],
            df_utm.loc[top_mask, "N_utm"],
            c=risks[top_mask],
            s=6,
            cmap="viridis",
        )
        ax.set_title(f"{epoch_label} top{int(top_percent*100)}% risk (UTM)")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Risk")
        fig.tight_layout()
        fig.savefig(png_utm_dir / f"risk_top_utm_{epoch_label}.png", dpi=dpi)
        plt.close(fig)

    # Time-series metrics.
    bootstrap_n = int(stats_cfg["BOOTSTRAP_N"])
    perm_n = int(stats_cfg["PERM_N"])
    permute_near_only = bool(stats_cfg.get("PERMUTE_FROM_NEAR_EVENT_ONLY", True))

    mean_rows: list[dict] = []
    auc_rows: list[dict] = []
    auc_far_rows: list[dict] = []

    for j, k in enumerate(k_indices):
        risks = risk_cube[j]
        epoch_label = epoch_cols[k]
        epoch_date = epoch_dates[k]

        core_r = risks[core_idx] if len(core_idx) else np.array([])
        near_r = risks[near_idx] if len(near_idx) else np.array([])
        far_r = risks[far_idx] if len(far_idx) else np.array([])

        core_mean = float(np.nanmean(core_r)) if len(core_r) else float("nan")
        near_mean = float(np.nanmean(near_r)) if len(near_r) else float("nan")
        far_mean = float(np.nanmean(far_r)) if len(far_r) else float("nan")
        gap_mean = core_mean - near_mean if not (np.isnan(core_mean) or np.isnan(near_mean)) else float("nan")

        gap_lo, gap_hi = bootstrap_gap_ci(core_r, near_r, bootstrap_n, seed=1000 + k)

        mean_rows.append(
            {
                "k": k,
                "epoch_col": epoch_label,
                "epoch_date": epoch_date.date(),
                "core_mean_risk": core_mean,
                "near_mean_risk": near_mean,
                "far_mean_risk": far_mean,
                "gap_core_minus_near": gap_mean,
                "gap_ci_lo": gap_lo,
                "gap_ci_hi": gap_hi,
                "is_near_event_window": int(k in near_rng),
            }
        )

        def _auc_and_ci(pos: np.ndarray, neg: np.ndarray, seed: int) -> Tuple[float, float, float]:
            if len(pos) == 0 or len(neg) == 0:
                return float("nan"), float("nan"), float("nan")
            y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
            s = np.concatenate([pos, neg])
            if len(np.unique(y)) < 2:
                return float("nan"), float("nan"), float("nan")
            auc = float(roc_auc_score(y, s))
            lo, hi = bootstrap_auc_ci(pos, neg, bootstrap_n, seed=seed)
            return auc, lo, hi

        auc, auc_lo, auc_hi = _auc_and_ci(core_r, near_r, seed=2000 + k)
        do_perm = (k in near_rng) if permute_near_only else True
        perm_p = permutation_p_value_auc(core_r, near_r, perm_n, seed=3000 + k) if do_perm else float("nan")

        auc_rows.append(
            {
                "k": k,
                "epoch_col": epoch_label,
                "epoch_date": epoch_date.date(),
                "auc": auc,
                "auc_ci_lo": auc_lo,
                "auc_ci_hi": auc_hi,
                "perm_p_value": perm_p,
                "is_near_event_window": int(k in near_rng),
            }
        )

        if neg_mode == "near_plus_far":
            auc_far, auc_far_lo, auc_far_hi = _auc_and_ci(core_r, far_r, seed=4000 + k)
            auc_far_rows.append(
                {
                    "k": k,
                    "epoch_col": epoch_label,
                    "epoch_date": epoch_date.date(),
                    "auc": auc_far,
                    "auc_ci_lo": auc_far_lo,
                    "auc_ci_hi": auc_far_hi,
                    "is_near_event_window": int(k in near_rng),
                }
            )

    mean_df = pd.DataFrame(mean_rows)
    auc_df = pd.DataFrame(auc_rows)
    auc_far_df = pd.DataFrame(auc_far_rows) if auc_far_rows else pd.DataFrame()

    mean_df.to_csv(run_dir / "mean_risk_over_time_by_class.csv", index=False)
    auc_df.to_csv(run_dir / "auc_over_time_core_vs_near.csv", index=False)
    if not auc_far_df.empty:
        auc_far_df.to_csv(run_dir / "auc_over_time_core_vs_far.csv", index=False)

    # Lead-time metric calibration.
    far_target = float(thresh_cfg["FAR_TARGET"])
    consecutive_m = int(thresh_cfg["CONSECUTIVE_M"])
    calib_offset = int(thresh_cfg["FAR_CALIB_END_OFFSET_FROM_LAST"])

    start_k = int(min(k_indices))
    calib_end_k = max(start_k, t_last - calib_offset)
    early_mask = np.array([k <= calib_end_k for k in k_indices])

    near_early = risk_cube[early_mask][:, near_idx].ravel() if len(near_idx) else np.array([])
    near_early = near_early[~np.isnan(near_early)]

    if len(near_early) == 0:
        tau = float("nan")
        far_actual = float("nan")
    else:
        tau = quantile_higher(near_early, 1.0 - far_target)
        far_actual = float(np.mean(near_early > tau))

    core_means = mean_df["core_mean_risk"].to_numpy()
    detect_k = earliest_consecutive_exceedance(k_indices, core_means, tau, consecutive_m) if not np.isnan(tau) else None

    event_date = pd.Timestamp(cfg["DATA"]["EVENT_DATE"])
    if detect_k is None:
        detect_date = pd.NaT
        lead_days = float("nan")
    else:
        detect_date = epoch_dates[int(detect_k)]
        lead_days = float((event_date - detect_date).days)

    near_rng_idx = list(near_event_range(t_last, horizon_h))
    near_rng_mask = np.isin(k_indices, near_rng_idx)
    core_near = risk_cube[near_rng_mask][:, core_idx].ravel() if len(core_idx) else np.array([])
    core_near = core_near[~np.isnan(core_near)]
    tpr = float(np.mean(core_near > tau)) if (len(core_near) and not np.isnan(tau)) else float("nan")

    lead_summary = {
        "tau": tau,
        "far_target": far_target,
        "far_actual": far_actual,
        "calib_end_k": int(calib_end_k),
        "calib_end_epoch": epoch_cols[calib_end_k],
        "detect_k": None if detect_k is None else int(detect_k),
        "detect_epoch": "" if detect_k is None else epoch_cols[int(detect_k)],
        "detect_date": "" if detect_k is None else str(detect_date.date()),
        "event_date": str(event_date.date()),
        "lead_time_days": lead_days,
        "consecutive_m": consecutive_m,
        "tpr_near_event": tpr,
    }
    (run_dir / "lead_time_summary.json").write_text(json.dumps(lead_summary, indent=2), encoding="utf-8")

    # Plots.
    def _plot_with_ci(df: pd.DataFrame, y: str, lo: str, hi: str, title: str, out_path: Path, ylabel: str) -> None:
        if df.empty:
            return
        x = pd.to_datetime(df["epoch_date"]).to_numpy()
        yv = df[y].to_numpy(dtype=float)
        lov = df[lo].to_numpy(dtype=float)
        hiv = df[hi].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(x, yv, marker="o", label=ylabel)
        ax.fill_between(x, lov, hiv, alpha=0.2)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)

    _plot_with_ci(
        auc_df,
        y="auc",
        lo="auc_ci_lo",
        hi="auc_ci_hi",
        title="AUC over time: core vs near-field",
        out_path=run_dir / "auc_over_time_core_vs_near.png",
        ylabel="AUC",
    )

    if not auc_far_df.empty:
        _plot_with_ci(
            auc_far_df,
            y="auc",
            lo="auc_ci_lo",
            hi="auc_ci_hi",
            title="AUC over time: core vs far-field",
            out_path=run_dir / "auc_over_time_core_vs_far.png",
            ylabel="AUC",
        )

    fig, ax = plt.subplots(figsize=(9, 4))
    x = pd.to_datetime(mean_df["epoch_date"]).to_numpy()
    ax.plot(x, mean_df["core_mean_risk"], marker="o", label="core")
    ax.plot(x, mean_df["near_mean_risk"], marker="o", label="near-field")
    if neg_mode == "near_plus_far":
        ax.plot(x, mean_df["far_mean_risk"], marker="o", label="far-field")
    if not np.isnan(tau):
        ax.axhline(tau, color="red", linestyle="--", linewidth=1.5, label=f"tau={tau:.3f}")
    ax.set_title("Mean risk over time by class")
    ax.set_ylabel("Mean risk")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(run_dir / "mean_risk_over_time_by_class.png", dpi=dpi)
    plt.close(fig)

    trend_p = compute_trend_pvalues(mean_df, auc_df)
    trend_lines = [
        "Trend p-values (Spearman approximation):",
        f"- core_mean_risk_p: {trend_p['core_mean_risk_p']}",
        f"- auc_core_vs_near_p: {trend_p['auc_core_vs_near_p']}",
    ]
    (run_dir / "trend_summary.txt").write_text("\n".join(trend_lines), encoding="utf-8")

    return mean_df, auc_df, auc_far_df, {**lead_summary, **trend_p}


def write_results_readme(cfg: dict, t_last: int, epoch_cols: Sequence[str], epoch_dates: Sequence[pd.Timestamp], run_dir: Path) -> None:
    label_cfg = cfg["LABELING"]
    thresh_cfg = cfg["THRESHOLDING"]
    event_date = cfg["DATA"]["EVENT_DATE"]
    last_obs = cfg["DATA"]["LAST_OBS_DATE"]
    horizon_h = int(label_cfg["HORIZON_H"])
    near_start = max(0, t_last - horizon_h)
    task = str(cfg.get("TASK", "leadtime_cls"))

    lines = [
        "# Results Guide",
        "",
        "## Run Command",
        "",
        "```bash",
        "export MPLBACKEND=Agg && python -u main.py | tee outputs/run_server.log",
        "```",
        "",
        "## Key Settings",
        "",
        f"- Task: {task}",
        f"- Event date: {event_date}",
        f"- Last observed InSAR epoch: {last_obs} ({epoch_cols[t_last]})",
        f"- Near-event window: last H={horizon_h} epochs before last observation",
        f"- Near-event epoch range: {epoch_cols[near_start]} to {epoch_cols[t_last-1]}",
        f"- FAR target for tau calibration: {thresh_cfg['FAR_TARGET']}",
        "",
        "## Outputs",
        "",
        "- Rolling risk maps (CSV): risk_maps_csv",
        "- Rolling risk maps (PNG lat/lon): risk_maps_png_latlon",
        "- Rolling risk maps (PNG UTM): risk_maps_png_utm",
        "- Mean risk over time: mean_risk_over_time_by_class.csv and .png",
        "- AUC over time (core vs near-field): auc_over_time_core_vs_near.csv and .png",
        "- Trend summary: trend_summary.txt",
        "- Lead-time summary: lead_time_summary.json",
        "",
        "## How To Interpret",
        "",
        "- Risk is standardized as risk_pk across tasks to enable shared evaluation/plots.",
        "- A good early-warning signal should show rising core mean risk toward the last observed epochs.",
        "- AUC(k) core vs near-field should improve near the end if the objective is working.",
        "- The lead-time summary reports the first sustained exceedance of tau by the core mean risk.",
        "",
        "## Notes",
        "",
        f"- Last observed epoch date in the data: {epoch_dates[t_last].date()}",
        "- Landslide date is later than the last observation, so lead time is measured relative to the event date.",
    ]

    out_path = run_dir / "README_results.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Task orchestration
# -----------------------------------------------------------------------------


def default_split_path_for_cfg(cfg: dict) -> Path:
    seed = int(cfg["SPLIT"]["SEED"])
    return DEFAULT_SPLITS_DIR / f"split_seed{seed}.json"


def summarize_split_counts_leadtime(y: np.ndarray, idx: np.ndarray, ids_split: np.ndarray, point_is_core: Dict[str, int], name: str) -> None:
    ys = y[idx] if len(idx) else np.array([])
    pos = int((ys == 1).sum()) if len(ys) else 0
    neg = int((ys == 0).sum()) if len(ys) else 0
    core_points = int(sum(point_is_core.get(pid, 0) for pid in ids_split))
    print(f"{name} points={len(ids_split)} (core_points={core_points}) | samples pos={pos}, neg={neg}")


def summarize_split_counts_survival(event_flag: np.ndarray, ids_split: np.ndarray, id_to_row: Dict[str, int], name: str) -> None:
    rows = [id_to_row[pid] for pid in ids_split if pid in id_to_row]
    ev = event_flag[rows] if rows else np.array([])
    n_event = int((ev > 0.5).sum()) if len(ev) else 0
    n_cens = int((ev <= 0.5).sum()) if len(ev) else 0
    print(f"{name} points={len(ids_split)} | events={n_event}, censored={n_cens}")


def summarize_split_counts_forecast(n_train: int, n_val: int, n_test: int) -> None:
    print(f"Forecast samples | train={n_train}, val={n_val}, test={n_test}")


def select_row_indices(ids: np.ndarray, allowed_ids: np.ndarray) -> np.ndarray:
    allowed = set(map(str, allowed_ids.tolist()))
    mask = np.array([str(pid) in allowed for pid in ids], dtype=bool)
    return np.where(mask)[0]


def run_experiment(cfg: dict, run_dir: Path, split_path: Path | None = None) -> dict:
    ensure_run_dirs(run_dir)

    seed = int(cfg["SPLIT"]["SEED"])
    set_seeds(seed)

    task = str(cfg.get("TASK", "leadtime_cls")).lower()

    data_cfg = cfg["DATA"]
    label_cfg = cfg["LABELING"]

    df = load_points_csv(str(data_cfg["CSV_PATH"]))
    epoch_cols, epoch_dates = detect_and_sort_epochs(df)

    last_obs_idx = get_index_for_date(epoch_dates, str(data_cfg["LAST_OBS_DATE"]))
    if last_obs_idx != len(epoch_cols) - 1:
        print(
            "Warning: last observation date is not the final epoch column. "
            f"Using index {last_obs_idx} ({epoch_cols[last_obs_idx]})."
        )
    t_last = last_obs_idx

    polygon = load_polygon_kml(str(data_cfg["KML_PATH"]))
    epsg_utm = compute_utm_epsg(polygon)

    df_latlon, df_utm, polygon_utm, _ = label_points(
        df=df,
        polygon=polygon,
        epsg_utm=epsg_utm,
        buffer_m=float(label_cfg["BUFFER_M"]),
        near_min_m=float(label_cfg["NEAR_MIN_M"]),
        near_max_m=float(label_cfg["NEAR_MAX_M"]),
        far_min_m=float(label_cfg["FAR_MIN_M"]),
    )

    values = df_latlon[epoch_cols].to_numpy(dtype=np.float32)
    ids = df_latlon["ID"].astype(str).to_numpy()
    label_texts = df_latlon["label_text"].astype(str).to_numpy()

    input_mode = str(label_cfg["INPUT_MODE"]).lower()
    neg_mode = str(label_cfg["NEG_MODE"]).lower()
    window_length = int(label_cfg["WINDOW_LENGTH"])
    horizon_h = int(label_cfg["HORIZON_H"])

    # Shared split file across all tasks/runs.
    if split_path is None:
        split_path = default_split_path_for_cfg(cfg)

    all_point_ids = (
        df_latlon.loc[df_latlon["label_text"].isin(["core", "near-field", "far-field"]), "ID"].astype(str).tolist()
    )
    point_is_core: dict[str, int] = {
        str(pid): int(lbl == "core")
        for pid, lbl in zip(df_latlon["ID"].astype(str), df_latlon["label_text"].astype(str))
    }
    train_ids, val_ids, test_ids = resolve_shared_split(split_path, all_point_ids, point_is_core, cfg["SPLIT"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"Using device: {device} ({device_name})")
    else:
        print(f"Using device: {device}")

    # Task-specific training/inference to produce risk_cube and k_indices.
    batch_size_train = int(cfg["TRAINING"]["BATCH_SIZE"])
    batch_size_infer = int(cfg["INFERENCE"]["BATCH_SIZE"])

    test_auc_proxy = float("nan")

    if task == "leadtime_cls":
        X, y, point_to_indices, _ = build_training_windows_leadtime(
            values=values,
            ids=ids,
            label_texts=label_texts,
            epoch_cols=epoch_cols,
            epoch_dates=epoch_dates,
            t_last=t_last,
            window_length=window_length,
            horizon_h=horizon_h,
            input_mode=input_mode,
            neg_mode=neg_mode,
        )

        train_idx = indices_for_points(train_ids, point_to_indices)
        val_idx = indices_for_points(val_ids, point_to_indices)
        test_idx = indices_for_points(test_ids, point_to_indices)

        print("Point split counts (grouped by point ID):")
        summarize_split_counts_leadtime(y, train_idx, train_ids, point_is_core, "Train")
        summarize_split_counts_leadtime(y, val_idx, val_ids, point_is_core, "Val")
        summarize_split_counts_leadtime(y, test_idx, test_ids, point_is_core, "Test")

        if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
            raise ValueError("One of the splits has zero samples. Check neg_mode and labeling.")

        in_ch = X.shape[1]
        train_loader = DataLoader(WindowDataset(X[train_idx], y[train_idx]), batch_size=batch_size_train, shuffle=True)
        val_loader = DataLoader(WindowDataset(X[val_idx], y[val_idx]), batch_size=batch_size_train, shuffle=False)
        test_loader = DataLoader(WindowDataset(X[test_idx], y[test_idx]), batch_size=batch_size_train, shuffle=False)

        model = build_model(cfg, in_ch=in_ch).to(device)
        best_path = train_model_binary(model, train_loader, val_loader, device=device, cfg=cfg, run_dir=run_dir)
        test_auc_proxy = evaluate_auc_binary(model, test_loader, device=device)
        print(f"Test AUC (group split, time-aware labels): {test_auc_proxy:.4f}")
        print(f"Best model saved to: {best_path}")

        start_k = window_length - 1
        end_k = min(t_last, len(epoch_cols) - 1)
        k_indices = list(range(start_k, end_k + 1))
        risk_cube = compute_risk_cube_leadtime(
            model,
            values=values,
            k_indices=k_indices,
            window_length=window_length,
            input_mode=input_mode,
            device=device,
            batch_size=batch_size_infer,
        )

    elif task == "survival_discrete":
        start_k = window_length - 1
        end_k = min(t_last, len(epoch_cols) - 1)
        k_indices = list(range(start_k, end_k + 1))

        X_seq_all = precompute_seq_windows(values, k_indices, window_length=window_length, input_mode=input_mode)

        # Event at t_last for core points; censored for others.
        event_flag_all = (label_texts == "core").astype(np.float32)

        # Neg mode controls which non-core points contribute to training.
        if neg_mode == "near_only":
            allowed_mask = np.isin(label_texts, ["core", "near-field"])
        else:
            allowed_mask = np.isin(label_texts, ["core", "near-field", "far-field"])

        ids_allowed = ids[allowed_mask]
        X_seq_allowed = X_seq_all[allowed_mask]
        event_allowed = event_flag_all[allowed_mask]

        id_to_row = {pid: i for i, pid in enumerate(ids_allowed.tolist())}

        train_rows = np.array([id_to_row[pid] for pid in train_ids if pid in id_to_row], dtype=np.int64)
        val_rows = np.array([id_to_row[pid] for pid in val_ids if pid in id_to_row], dtype=np.int64)
        test_rows = np.array([id_to_row[pid] for pid in test_ids if pid in id_to_row], dtype=np.int64)

        print("Point split counts (grouped by point ID):")
        summarize_split_counts_survival(event_allowed, train_ids, id_to_row, "Train")
        summarize_split_counts_survival(event_allowed, val_ids, id_to_row, "Val")
        summarize_split_counts_survival(event_allowed, test_ids, id_to_row, "Test")

        if len(train_rows) == 0 or len(val_rows) == 0 or len(test_rows) == 0:
            raise ValueError("One of the survival splits has zero points after filtering.")

        in_ch = X_seq_allowed.shape[2]
        train_loader = DataLoader(
            SurvivalPointDataset(X_seq_allowed[train_rows], event_allowed[train_rows], ids_allowed[train_rows]),
            batch_size=batch_size_train,
            shuffle=True,
        )
        val_loader = DataLoader(
            SurvivalPointDataset(X_seq_allowed[val_rows], event_allowed[val_rows], ids_allowed[val_rows]),
            batch_size=batch_size_train,
            shuffle=False,
        )

        model = build_model(cfg, in_ch=in_ch).to(device)
        best_path = train_model_survival(model, train_loader, val_loader, device=device, cfg=cfg, run_dir=run_dir)
        print(f"Best model saved to: {best_path}")

        # Risk for all points (including far-field) using cumulative hazard.
        risk_cube = compute_risk_cube_survival(
            model,
            X_seq=X_seq_all,
            device=device,
            batch_size=batch_size_infer,
        )

    elif task == "forecast_anom":
        forecast_cfg = cfg.get("FORECAST", {})
        horizon_f = int(forecast_cfg.get("HORIZON_F", 5))
        train_mode = str(forecast_cfg.get("TRAIN_MODE", "all")).lower()

        # Forecast indices must allow at least one future observation.
        start_k = window_length - 1
        end_k_forecast = min(t_last - 1, len(epoch_cols) - 1)
        k_indices_all = list(range(start_k, end_k_forecast + 1))

        # Optional strict training range.
        if train_mode == "early":
            calib_offset = int(cfg["THRESHOLDING"]["FAR_CALIB_END_OFFSET_FROM_LAST"])
            calib_end_k = max(start_k, t_last - calib_offset)
            k_indices_train = [k for k in k_indices_all if k <= calib_end_k]
        else:
            k_indices_train = k_indices_all

        train_rows = select_row_indices(ids, train_ids)
        val_rows = select_row_indices(ids, val_ids)
        test_rows = select_row_indices(ids, test_ids)

        X_train, y_train, m_train = build_forecast_samples(
            values, train_rows, k_indices_train, window_length, horizon_f, input_mode
        )
        X_val, y_val, m_val = build_forecast_samples(
            values, val_rows, k_indices_train, window_length, horizon_f, input_mode
        )
        X_test, y_test, m_test = build_forecast_samples(
            values, test_rows, k_indices_train, window_length, horizon_f, input_mode
        )

        summarize_split_counts_forecast(len(X_train), len(X_val), len(X_test))

        in_ch = X_train.shape[1]
        train_loader = DataLoader(ForecastDataset(X_train, y_train, m_train), batch_size=batch_size_train, shuffle=True)
        val_loader = DataLoader(ForecastDataset(X_val, y_val, m_val), batch_size=batch_size_train, shuffle=False)

        model = build_model(cfg, in_ch=in_ch).to(device)
        best_path = train_model_forecast(model, train_loader, val_loader, device=device, cfg=cfg, run_dir=run_dir)
        print(f"Best model saved to: {best_path}")

        # Inference indices use all epochs with a future horizon.
        k_indices = k_indices_all
        anom_cube = compute_anomaly_cube_forecast(
            model,
            values=values,
            k_indices=k_indices,
            window_length=window_length,
            horizon_f=horizon_f,
            input_mode=input_mode,
            device=device,
            batch_size=batch_size_infer,
        )

        # Calibrate anomaly -> risk using near-field early epochs.
        near_idx = np.where(label_texts == "near-field")[0]
        calib_offset = int(cfg["THRESHOLDING"]["FAR_CALIB_END_OFFSET_FROM_LAST"])
        calib_end_k = max(start_k, t_last - calib_offset)
        risk_cube, _ = calibrate_anomaly_to_risk(anom_cube, k_indices, near_idx=near_idx, calib_end_k=calib_end_k)

    else:
        raise ValueError(f"Unknown TASK: {task}")

    mean_df, auc_df, auc_far_df, lead_and_trend = evaluate_and_save_outputs(
        risk_cube=risk_cube,
        k_indices=k_indices,
        values=values,
        df_latlon=df_latlon,
        df_utm=df_utm,
        polygon_latlon=polygon,
        polygon_utm=polygon_utm,
        epoch_cols=epoch_cols,
        epoch_dates=epoch_dates,
        t_last=t_last,
        cfg=cfg,
        run_dir=run_dir,
    )

    write_results_readme(cfg, t_last=t_last, epoch_cols=epoch_cols, epoch_dates=epoch_dates, run_dir=run_dir)

    near_rng = list(near_event_range(t_last, horizon_h))
    near_epochs = [epoch_cols[k] for k in near_rng]
    print(
        "Near-event window epochs: "
        f"{near_epochs[0]} .. {near_epochs[-1]} (H={label_cfg['HORIZON_H']})"
    )
    print(f"Saved outputs under: {run_dir}")

    last_n = int(cfg["STATS"].get("LAST_N", horizon_h))
    auc_tail = auc_df.tail(last_n)["auc"].to_numpy(dtype=float)
    auc_tail_mean = float(np.nanmean(auc_tail)) if len(auc_tail) else float("nan")

    summary = {
        "run_dir": str(run_dir),
        "task": task,
        "model": str(cfg["MODEL"]["NAME"]).lower(),
        "input_mode": input_mode,
        "neg_mode": neg_mode,
        "H": horizon_h,
        "Hf": int(cfg.get("FORECAST", {}).get("HORIZON_F", 0)),
        "test_auc": test_auc_proxy,
        "auc_near_lastN_mean": auc_tail_mean,
        "lead_time_days": float(lead_and_trend.get("lead_time_days", float("nan"))),
        "far": float(lead_and_trend.get("far_actual", float("nan"))),
        "tpr": float(lead_and_trend.get("tpr_near_event", float("nan"))),
        "trend_pvalues": json.dumps(
            {
                "core_mean_risk_p": lead_and_trend.get("core_mean_risk_p"),
                "auc_core_vs_near_p": lead_and_trend.get("auc_core_vs_near_p"),
            }
        ),
        "split_path": str(split_path),
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Time-aware early-warning training/inference.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--run-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--split-file", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    run_dir = Path(args.run_dir)
    split_file = Path(args.split_file) if args.split_file else None

    cfg = load_config(cfg_path)
    run_experiment(cfg, run_dir=run_dir, split_path=split_file)


if __name__ == "__main__":
    main()
