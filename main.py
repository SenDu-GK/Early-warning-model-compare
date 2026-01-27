import math
import os
import random
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Polygon
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class Config:
    WINDOW_LENGTH: int = 12
    BUFFER: float = 100.0
    DIST_NEAR_MIN: float = 100.0
    DIST_NEAR_MAX: float = 500.0
    DIST_FAR: float = 500.0
    NEGATIVE_MODE: str = "mix"  # "far", "near", "mix"
    TRAIN_END_IDX: Optional[int] = None
    VAL_END_IDX: Optional[int] = None
    TEST_START_IDX: Optional[int] = None
    MAX_EPOCHS: int = 50
    BATCH_SIZE: int = 128
    INFER_BATCH_SIZE: int = 2048
    BOOTSTRAP_N: int = 300
    PERM_N: int = 200
    TREND_LAST_N: int = 10
    SEED: int = 42
    DROPOUT: float = 0.1
    HIDDEN: int = 64
    PATIENCE: int = 6
    LR: float = 1e-3
    WEIGHT_DECAY: float = 1e-4


EVENT_DATE = pd.Timestamp("2018-03-09")
OUTPUT_DIR = Path("outputs")


def _ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "risk_maps_csv").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "risk_maps_png_latlon").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "risk_maps_png_utm").mkdir(parents=True, exist_ok=True)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_points_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "ID" not in df.columns or "Fi" not in df.columns or "Lambda" not in df.columns:
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
    coords = []
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
        if not isinstance(geom, Polygon):
            geom = geom.unary_union
            if not isinstance(geom, Polygon):
                raise ValueError("KML does not contain a Polygon geometry.")
        return geom
    except Exception:
        return _parse_polygon_from_kml(kml_path)


def compute_utm_epsg(polygon: Polygon) -> int:
    centroid = polygon.centroid
    lon, lat = centroid.x, centroid.y
    zone = math.floor((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return epsg


def label_points(
    df: pd.DataFrame,
    polygon: Polygon,
    epsg_utm: int,
    buffer: float,
    dist_near_min: float,
    dist_near_max: float,
    dist_far: float,
) -> Tuple[GeoDataFrame, GeoDataFrame, Dict[str, str]]:
    gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df["Lambda"], df["Fi"]), crs="EPSG:4326")
    poly_gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
    gdf_utm = gdf.to_crs(epsg=epsg_utm)
    poly_utm = poly_gdf.to_crs(epsg=epsg_utm)

    poly_shape = poly_utm.geometry.iloc[0]
    inside = gdf_utm.within(poly_shape)
    dist = gdf_utm.distance(poly_shape)

    labels = np.full(len(gdf), -2, dtype=int)  # -2 ignored, 2 buffer, 3 near, 0 far, 1 core
    labels[inside.to_numpy()] = 1

    buffer_mask = (~inside.to_numpy()) & (dist <= buffer)
    labels[buffer_mask] = 2

    near_lower = max(buffer, dist_near_min)
    near_mask = (~inside.to_numpy()) & (dist > near_lower) & (dist <= dist_near_max)
    labels[near_mask] = 3

    far_mask = dist > dist_far
    labels[far_mask] = 0

    label_text = np.array(["ignored"] * len(labels), dtype=object)
    label_text[labels == 1] = "core"
    label_text[labels == 0] = "far-field"
    label_text[labels == 3] = "near-field"
    label_text[labels == 2] = "buffer"

    gdf["label"] = labels
    gdf["label_text"] = label_text

    gdf_utm["label"] = labels
    gdf_utm["label_text"] = label_text
    gdf_utm["E_utm"] = gdf_utm.geometry.x
    gdf_utm["N_utm"] = gdf_utm.geometry.y

    label_map = {str(row.ID): str(lbl) for row, lbl in zip(df.itertuples(index=False), label_text)}

    total = len(labels)
    core = int((labels == 1).sum())
    near = int((labels == 3).sum())
    far = int((labels == 0).sum())
    buffer_n = int((labels == 2).sum())
    ignored = total - core - far - near - buffer_n
    print(
        f"Label counts: total={total}, core={core}, near-field={near}, far-field={far}, buffer={buffer_n}, ignored={ignored}"
    )
    return gdf, gdf_utm, label_map


def select_negative_ids(
    point_ids: np.ndarray,
    label_text: np.ndarray,
    mode: str,
    seed: int,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    near_ids = point_ids[label_text == "near-field"]
    far_ids = point_ids[label_text == "far-field"]

    if mode == "near":
        return near_ids
    if mode == "far":
        return far_ids
    if mode != "mix":
        raise ValueError("NEGATIVE_MODE must be one of: far, near, mix")

    if len(near_ids) == 0 or len(far_ids) == 0:
        print("Warning: one negative class empty, falling back to available negatives.")
        return np.concatenate([near_ids, far_ids])

    n = min(len(near_ids), len(far_ids))
    near_sel = rng.choice(near_ids, size=n, replace=False)
    far_sel = rng.choice(far_ids, size=n, replace=False)
    return np.concatenate([near_sel, far_sel])


def split_point_ids(
    point_ids: np.ndarray,
    label_text: np.ndarray,
    negative_mode: str,
    seed: int,
) -> Tuple[List[str], List[str], List[str], Dict[str, int]]:
    core_ids = point_ids[label_text == "core"]
    neg_ids = select_negative_ids(point_ids, label_text, negative_mode, seed)

    selected_ids = np.concatenate([core_ids, neg_ids])
    if len(selected_ids) == 0:
        raise ValueError("No points selected for training. Check labels and NEGATIVE_MODE.")

    label_by_id = {pid: (1 if pid in set(core_ids) else 0) for pid in selected_ids}
    id_list = list(selected_ids)
    id_label_list = [label_by_id[pid] for pid in id_list]

    try:
        train_ids, temp_ids = train_test_split(
            id_list, test_size=0.3, random_state=seed, stratify=id_label_list
        )
    except ValueError:
        train_ids, temp_ids = train_test_split(id_list, test_size=0.3, random_state=seed, stratify=None)

    temp_labels = [label_by_id[pid] for pid in temp_ids]
    try:
        val_ids, test_ids = train_test_split(
            temp_ids, test_size=0.5, random_state=seed, stratify=temp_labels
        )
    except ValueError:
        val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=seed, stratify=None)

    return train_ids, val_ids, test_ids, label_by_id


def resolve_time_splits(n_epochs: int, cfg: Config) -> Tuple[int, int, int, List[int], List[int], List[int]]:
    min_idx = cfg.WINDOW_LENGTH - 1
    train_end = cfg.TRAIN_END_IDX
    if train_end is None:
        train_end = max(min_idx, n_epochs // 2 - 1)

    val_end = cfg.VAL_END_IDX
    if val_end is None:
        val_end = max(train_end, int(n_epochs * 0.75) - 1)

    test_start = cfg.TEST_START_IDX
    if test_start is None:
        test_start = val_end + 1

    train_end = max(min_idx, min(train_end, n_epochs - 1))
    val_end = max(train_end, min(val_end, n_epochs - 1))
    test_start = max(val_end + 1, min(test_start, n_epochs))

    train_indices = list(range(min_idx, train_end + 1))
    val_indices = list(range(train_end + 1, val_end + 1)) if val_end > train_end else []
    test_indices = list(range(test_start, n_epochs)) if test_start < n_epochs else []

    return train_end, val_end, test_start, train_indices, val_indices, test_indices


def build_windows_for_ids(
    disp_mat: np.ndarray,
    epoch_cols: Sequence[str],
    point_ids: np.ndarray,
    selected_ids: List[str],
    label_by_id: Dict[str, int],
    t_indices: List[int],
    window: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
    sequences: List[np.ndarray] = []
    labels: List[int] = []
    pid_list: List[str] = []
    time_list: List[int] = []

    id_to_idx = {pid: idx for idx, pid in enumerate(point_ids)}

    for pid in selected_ids:
        if pid not in id_to_idx:
            continue
        row_idx = id_to_idx[pid]
        disp = disp_mat[row_idx]
        y_val = label_by_id[pid]
        for t in t_indices:
            window_vals = disp[t - window + 1 : t + 1]
            mask = np.isfinite(window_vals).astype(np.float32)
            window_vals = np.nan_to_num(window_vals, nan=0.0)
            seq = np.stack([window_vals, mask], axis=0)
            sequences.append(seq)
            labels.append(y_val)
            pid_list.append(pid)
            time_list.append(t)

    if not sequences:
        return np.empty((0, 2, window), dtype=np.float32), np.empty((0,), dtype=np.float32), [], []

    X = np.stack(sequences, axis=0).astype(np.float32)
    y = np.array(labels, dtype=np.float32)
    return X, y, pid_list, time_list


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=padding, dilation=dilation)
        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.relu(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out = out + residual
        out = self.relu(out)
        return out


class TCN(nn.Module):
    def __init__(self, in_ch: int = 2, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.block1 = ResidualBlock(in_ch, hidden, dilation=1, dropout=dropout)
        self.block2 = ResidualBlock(hidden, hidden, dilation=2, dropout=dropout)
        self.block3 = ResidualBlock(hidden, hidden, dilation=4, dropout=dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.global_pool(out).squeeze(-1)
        logits = self.head(out).squeeze(-1)
        return logits


def _compute_pos_weight(y: np.ndarray) -> torch.Tensor:
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    if pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(max(neg / max(pos, 1), 1.0), dtype=torch.float32)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    out_path: Path,
    cfg: Config,
):
    if len(train_loader.dataset) == 0:
        raise ValueError("Training dataset is empty. Check time split and labels.")

    y_train = torch.cat([batch_y for _, batch_y in train_loader]).cpu().numpy()
    pos_weight = _compute_pos_weight(y_train).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    best_auc = -np.inf
    bad_epochs = 0
    best_state = None

    for epoch in range(1, cfg.MAX_EPOCHS + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_logits = []
        val_targets = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_logits.append(logits.cpu().numpy())
                val_targets.append(yb.cpu().numpy())
        if val_logits:
            val_logits = np.concatenate(val_logits)
            val_targets = np.concatenate(val_targets)
            try:
                val_auc = roc_auc_score(val_targets, torch.sigmoid(torch.tensor(val_logits)).numpy())
            except ValueError:
                val_auc = np.nan
        else:
            val_auc = np.nan

        avg_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        print(f"Epoch {epoch:02d}: train_loss={avg_train_loss:.4f}, val_auc={val_auc:.4f}")

        improved = not np.isnan(val_auc) and val_auc > best_auc
        if improved:
            best_auc = val_auc
            best_state = model.state_dict()
            bad_epochs = 0
            torch.save(best_state, out_path)
        else:
            bad_epochs += 1
        if bad_epochs >= cfg.PATIENCE:
            print("Early stopping triggered.")
            break

    if best_state is None:
        best_state = model.state_dict()
        torch.save(best_state, out_path)
    model.load_state_dict(best_state)


def evaluate_auc(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    logits_list = []
    targets_list = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            logits_list.append(logits.cpu().numpy())
            targets_list.append(yb.cpu().numpy())
    if not logits_list:
        return float("nan")
    preds = torch.sigmoid(torch.tensor(np.concatenate(logits_list))).numpy()
    targets = np.concatenate(targets_list)
    try:
        return roc_auc_score(targets, preds)
    except ValueError:
        return float("nan")


def bootstrap_auc(
    risks: np.ndarray,
    core_idx: np.ndarray,
    neg_idx: np.ndarray,
    n_boot: int,
    seed: int,
) -> Tuple[float, float]:
    rng = np.random.RandomState(seed)
    if len(core_idx) == 0 or len(neg_idx) == 0:
        return np.nan, np.nan
    aucs = []
    for _ in range(n_boot):
        c = rng.choice(core_idx, size=len(core_idx), replace=True)
        n = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        labels = np.concatenate([np.ones(len(c)), np.zeros(len(n))])
        scores = np.concatenate([risks[c], risks[n]])
        try:
            aucs.append(roc_auc_score(labels, scores))
        except ValueError:
            continue
    if not aucs:
        return np.nan, np.nan
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def bootstrap_gap_ci(
    risks: np.ndarray,
    core_idx: np.ndarray,
    neg_idx: np.ndarray,
    n_boot: int,
    seed: int,
) -> Tuple[float, float]:
    rng = np.random.RandomState(seed)
    if len(core_idx) == 0 or len(neg_idx) == 0:
        return np.nan, np.nan
    gaps = []
    for _ in range(n_boot):
        c = rng.choice(core_idx, size=len(core_idx), replace=True)
        n = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        gaps.append(float(np.nanmean(risks[c]) - np.nanmean(risks[n])))
    return float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))


def permutation_test_auc(
    risks: np.ndarray,
    labels: np.ndarray,
    n_perm: int,
    seed: int,
) -> float:
    rng = np.random.RandomState(seed)
    try:
        observed = roc_auc_score(labels, risks)
    except ValueError:
        return np.nan
    count = 0
    for _ in range(n_perm):
        perm_labels = rng.permutation(labels)
        try:
            auc = roc_auc_score(perm_labels, risks)
        except ValueError:
            continue
        if auc >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


def infer_risks_for_epoch(
    model: nn.Module,
    disp_mat: np.ndarray,
    t_idx: int,
    window: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    n_points = disp_mat.shape[0]
    start = t_idx - window + 1
    window_vals = disp_mat[:, start : t_idx + 1]
    risks = np.empty(n_points, dtype=np.float32)

    for i in range(0, n_points, batch_size):
        w = window_vals[i : i + batch_size]
        mask = np.isfinite(w).astype(np.float32)
        w = np.nan_to_num(w, nan=0.0).astype(np.float32)
        xb = np.stack([w, mask], axis=1)
        xb_t = torch.from_numpy(xb).to(device)
        with torch.no_grad():
            logits = model(xb_t)
            risks[i : i + batch_size] = torch.sigmoid(logits).cpu().numpy()
    return risks


def rolling_inference_and_save_outputs(
    model: nn.Module,
    df: pd.DataFrame,
    gdf_latlon: GeoDataFrame,
    gdf_utm: GeoDataFrame,
    polygon: Polygon,
    polygon_utm: Polygon,
    epoch_cols: Sequence[str],
    disp_mat: np.ndarray,
    device: torch.device,
    cfg: Config,
):
    epochs = list(epoch_cols)
    top_csv_dir = OUTPUT_DIR / "risk_maps_csv"
    png_latlon_dir = OUTPUT_DIR / "risk_maps_png_latlon"
    png_utm_dir = OUTPUT_DIR / "risk_maps_png_utm"

    label_text = gdf_latlon["label_text"].to_numpy()
    core_idx = np.where(label_text == "core")[0]
    near_idx = np.where(label_text == "near-field")[0]
    far_idx = np.where(label_text == "far-field")[0]

    mean_rows = []
    gap_ci_rows = []
    auc_rows_far = []
    auc_rows_near = []
    auc_rows_all = []

    for t_idx in range(cfg.WINDOW_LENGTH - 1, len(epochs)):
        epoch_label = epochs[t_idx]
        risks = infer_risks_for_epoch(
            model,
            disp_mat,
            t_idx,
            cfg.WINDOW_LENGTH,
            device,
            cfg.INFER_BATCH_SIZE,
        )

        risk_df = pd.DataFrame(
            {
                "ID": df["ID"].astype(str),
                "Fi": df["Fi"],
                "Lambda": df["Lambda"],
                "E": df["E"] if "E" in df.columns else np.nan,
                "N": df["N"] if "N" in df.columns else np.nan,
                "epoch_label": epoch_label,
                "risk_pk": risks,
            }
        )
        risk_df.to_csv(top_csv_dir / f"risk_{epoch_label}.csv", index=False)

        thresh = np.nanpercentile(risks, 80)
        top_mask = risks >= thresh

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set_title(f"{epoch_label} top20% risk")
        poly_coords = np.array(polygon.exterior.coords)
        ax.plot(poly_coords[:, 0], poly_coords[:, 1], color="black", linewidth=1.5, label="Polygon")
        sc = ax.scatter(
            gdf_latlon.loc[top_mask, "Lambda"],
            gdf_latlon.loc[top_mask, "Fi"],
            c=risks[top_mask],
            cmap="inferno",
            s=12,
            alpha=0.9,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.colorbar(sc, ax=ax, label="Risk")
        ax.legend()
        fig.tight_layout()
        fig.savefig(png_latlon_dir / f"risk_top20_latlon_{epoch_label}.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set_title(f"{epoch_label} top20% risk (UTM)")
        poly_utm_coords = np.array(polygon_utm.exterior.coords)
        ax.plot(poly_utm_coords[:, 0], poly_utm_coords[:, 1], color="black", linewidth=1.5, label="Polygon")
        sc = ax.scatter(
            gdf_utm.loc[top_mask, "E_utm"],
            gdf_utm.loc[top_mask, "N_utm"],
            c=risks[top_mask],
            cmap="inferno",
            s=12,
            alpha=0.9,
        )
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        plt.colorbar(sc, ax=ax, label="Risk")
        ax.legend()
        fig.tight_layout()
        fig.savefig(png_utm_dir / f"risk_top20_utm_{epoch_label}.png", dpi=200)
        plt.close(fig)

        core_mean = float(np.nanmean(risks[core_idx])) if len(core_idx) else np.nan
        near_mean = float(np.nanmean(risks[near_idx])) if len(near_idx) else np.nan
        far_mean = float(np.nanmean(risks[far_idx])) if len(far_idx) else np.nan
        gap_core_near = core_mean - near_mean if np.isfinite(core_mean) and np.isfinite(near_mean) else np.nan
        gap_core_far = core_mean - far_mean if np.isfinite(core_mean) and np.isfinite(far_mean) else np.nan

        mean_rows.append(
            {
                "epoch_label": epoch_label,
                "core_mean_risk": core_mean,
                "near_mean_risk": near_mean,
                "far_mean_risk": far_mean,
                "gap_core_near": gap_core_near,
                "gap_core_far": gap_core_far,
            }
        )

        core_far_mask = np.concatenate([core_idx, far_idx])
        core_near_mask = np.concatenate([core_idx, near_idx])
        core_all_mask = np.concatenate([core_idx, near_idx, far_idx])

        def _auc_for_mask(mask: np.ndarray, core_idx_local: np.ndarray) -> float:
            if len(core_idx_local) == 0:
                return np.nan
            neg_idx_local = np.setdiff1d(mask, core_idx_local)
            if len(neg_idx_local) == 0:
                return np.nan
            labels = np.concatenate([np.ones(len(core_idx_local)), np.zeros(len(neg_idx_local))])
            scores = np.concatenate([risks[core_idx_local], risks[neg_idx_local]])
            try:
                return roc_auc_score(labels, scores)
            except ValueError:
                return np.nan

        auc_far = _auc_for_mask(core_far_mask, core_idx)
        auc_near = _auc_for_mask(core_near_mask, core_idx)
        auc_all = _auc_for_mask(core_all_mask, core_idx)

        auc_far_ci = bootstrap_auc(risks, core_idx, far_idx, cfg.BOOTSTRAP_N, cfg.SEED + t_idx)
        auc_near_ci = bootstrap_auc(risks, core_idx, near_idx, cfg.BOOTSTRAP_N, cfg.SEED + t_idx + 101)
        auc_all_ci = bootstrap_auc(risks, core_idx, np.concatenate([near_idx, far_idx]), cfg.BOOTSTRAP_N, cfg.SEED + t_idx + 202)

        p_far = permutation_test_auc(
            np.concatenate([risks[core_idx], risks[far_idx]]),
            np.concatenate([np.ones(len(core_idx)), np.zeros(len(far_idx))]),
            cfg.PERM_N,
            cfg.SEED + t_idx + 303,
        ) if len(core_idx) and len(far_idx) else np.nan

        p_near = permutation_test_auc(
            np.concatenate([risks[core_idx], risks[near_idx]]),
            np.concatenate([np.ones(len(core_idx)), np.zeros(len(near_idx))]),
            cfg.PERM_N,
            cfg.SEED + t_idx + 404,
        ) if len(core_idx) and len(near_idx) else np.nan

        auc_rows_far.append(
            {
                "epoch_label": epoch_label,
                "auc": auc_far,
                "auc_ci_low": auc_far_ci[0],
                "auc_ci_high": auc_far_ci[1],
                "p_value": p_far,
            }
        )
        auc_rows_near.append(
            {
                "epoch_label": epoch_label,
                "auc": auc_near,
                "auc_ci_low": auc_near_ci[0],
                "auc_ci_high": auc_near_ci[1],
                "p_value": p_near,
            }
        )
        auc_rows_all.append(
            {
                "epoch_label": epoch_label,
                "auc": auc_all,
                "auc_ci_low": auc_all_ci[0],
                "auc_ci_high": auc_all_ci[1],
            }
        )

        gap_near_ci = bootstrap_gap_ci(risks, core_idx, near_idx, cfg.BOOTSTRAP_N, cfg.SEED + t_idx + 505)
        gap_far_ci = bootstrap_gap_ci(risks, core_idx, far_idx, cfg.BOOTSTRAP_N, cfg.SEED + t_idx + 606)
        gap_ci_rows.append(
            {
                "epoch_label": epoch_label,
                "gap_core_near_ci_low": gap_near_ci[0],
                "gap_core_near_ci_high": gap_near_ci[1],
                "gap_core_far_ci_low": gap_far_ci[0],
                "gap_core_far_ci_high": gap_far_ci[1],
            }
        )

    mean_df = pd.DataFrame(mean_rows)
    mean_df.to_csv(OUTPUT_DIR / "mean_risk_over_time_by_class.csv", index=False)

    gap_ci_df = pd.DataFrame(gap_ci_rows)
    gap_ci_df.to_csv(OUTPUT_DIR / "gap_ci_over_time.csv", index=False)

    auc_far_df = pd.DataFrame(auc_rows_far)
    auc_near_df = pd.DataFrame(auc_rows_near)
    auc_all_df = pd.DataFrame(auc_rows_all)
    auc_far_df.to_csv(OUTPUT_DIR / "auc_over_time_core_vs_far.csv", index=False)
    auc_near_df.to_csv(OUTPUT_DIR / "auc_over_time_core_vs_near.csv", index=False)
    auc_all_df.to_csv(OUTPUT_DIR / "auc_over_time_core_vs_all.csv", index=False)

    def _plot_auc(df: pd.DataFrame, out_path: Path, title: str):
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(df))
        ax.plot(x, df["auc"], marker="o", label="AUC")
        if "auc_ci_low" in df.columns:
            ax.fill_between(x, df["auc_ci_low"], df["auc_ci_high"], alpha=0.2, label="95% CI")
        ax.set_title(title)
        ax.set_ylabel("AUC")
        ax.set_xlabel("Epoch")
        ax.set_xticks(x)
        ax.set_xticklabels(df["epoch_label"], rotation=45, ha="right", fontsize=7)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    _plot_auc(auc_far_df, OUTPUT_DIR / "auc_over_time_core_vs_far.png", "AUC: core vs far-field")
    _plot_auc(auc_near_df, OUTPUT_DIR / "auc_over_time_core_vs_near.png", "AUC: core vs near-field")

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(mean_df))
    ax.plot(x, mean_df["core_mean_risk"], marker="o", label="core")
    ax.plot(x, mean_df["near_mean_risk"], marker="o", label="near-field")
    ax.plot(x, mean_df["far_mean_risk"], marker="o", label="far-field")
    ax.set_ylabel("Mean risk")
    ax.set_xlabel("Epoch")
    ax.set_xticks(x)
    ax.set_xticklabels(mean_df["epoch_label"], rotation=45, ha="right", fontsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "mean_risk_over_time_by_class.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, mean_df["gap_core_near"], marker="o", label="core - near")
    ax.plot(x, mean_df["gap_core_far"], marker="o", label="core - far")
    ax.set_ylabel("Risk gap")
    ax.set_xlabel("Epoch")
    ax.set_xticks(x)
    ax.set_xticklabels(mean_df["epoch_label"], rotation=45, ha="right", fontsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "gap_over_time.png", dpi=200)
    plt.close(fig)

    return mean_df, auc_far_df, auc_near_df, auc_all_df


def write_trend_summary(
    mean_df: pd.DataFrame,
    auc_far_df: pd.DataFrame,
    auc_near_df: pd.DataFrame,
    auc_all_df: pd.DataFrame,
    cfg: Config,
) -> None:
    def _trend(series: pd.Series) -> Tuple[float, float]:
        if series.isna().all():
            return np.nan, np.nan
        y = series.dropna().to_numpy()
        if len(y) < 2:
            return np.nan, np.nan
        x = np.arange(len(y))
        corr, p = spearmanr(x, y)
        return float(corr), float(p)

    tail_n = cfg.TREND_LAST_N
    core_tail = mean_df["core_mean_risk"].tail(tail_n)
    auc_far_tail = auc_far_df["auc"].tail(tail_n)
    auc_near_tail = auc_near_df["auc"].tail(tail_n)
    auc_all_tail = auc_all_df["auc"].tail(tail_n)

    core_corr, core_p = _trend(core_tail)
    far_corr, far_p = _trend(auc_far_tail)
    near_corr, near_p = _trend(auc_near_tail)
    all_corr, all_p = _trend(auc_all_tail)

    summary = [
        f"Trend window: last {tail_n} epochs",
        f"core_mean_risk Spearman r={core_corr:.4f}, p={core_p:.4f}",
        f"AUC core vs far Spearman r={far_corr:.4f}, p={far_p:.4f}",
        f"AUC core vs near Spearman r={near_corr:.4f}, p={near_p:.4f}",
        f"AUC core vs all Spearman r={all_corr:.4f}, p={all_p:.4f}",
    ]

    (OUTPUT_DIR / "trend_summary.txt").write_text("\n".join(summary))


def main():
    cfg = Config()
    cfg.MAX_EPOCHS = int(os.getenv("MAX_EPOCHS", str(cfg.MAX_EPOCHS)))
    _ensure_dirs()
    set_seeds(cfg.SEED)

    df = load_points_csv("Cadia_PSI.csv")
    epoch_cols, epoch_dates = detect_and_sort_epochs(df)
    polygon = load_polygon_kml("landslide area.kml")
    epsg_utm = compute_utm_epsg(polygon)

    gdf_latlon, gdf_utm, label_map = label_points(
        df,
        polygon,
        epsg_utm,
        buffer=cfg.BUFFER,
        dist_near_min=cfg.DIST_NEAR_MIN,
        dist_near_max=cfg.DIST_NEAR_MAX,
        dist_far=cfg.DIST_FAR,
    )

    disp_mat = df[epoch_cols].to_numpy(dtype=np.float32)
    point_ids = df["ID"].astype(str).to_numpy()
    label_text = gdf_latlon["label_text"].to_numpy()

    train_ids, val_ids, test_ids, label_by_id = split_point_ids(
        point_ids, label_text, cfg.NEGATIVE_MODE, cfg.SEED
    )

    train_end, val_end, test_start, train_t, val_t, test_t = resolve_time_splits(
        len(epoch_cols), cfg
    )

    print(
        "Time-aware split: "
        f"train <= idx {train_end} ({epoch_cols[train_end]}), "
        f"val <= idx {val_end} ({epoch_cols[val_end]}), "
        f"test >= idx {test_start} ({epoch_cols[test_start] if test_start < len(epoch_cols) else 'N/A'})"
    )
    print(
        f"Train epochs: {epoch_cols[train_t[0]]} to {epoch_cols[train_t[-1]]}"
        if train_t
        else "Train epochs: none"
    )
    print(
        f"Val epochs: {epoch_cols[val_t[0]]} to {epoch_cols[val_t[-1]]}"
        if val_t
        else "Val epochs: none"
    )
    print(
        f"Test epochs: {epoch_cols[test_t[0]]} to {epoch_cols[test_t[-1]]}"
        if test_t
        else "Test epochs: none"
    )

    X_train, y_train, _, _ = build_windows_for_ids(
        disp_mat, epoch_cols, point_ids, train_ids, label_by_id, train_t, cfg.WINDOW_LENGTH
    )
    X_val, y_val, _, _ = build_windows_for_ids(
        disp_mat, epoch_cols, point_ids, val_ids, label_by_id, val_t, cfg.WINDOW_LENGTH
    )
    X_test, y_test, _, _ = build_windows_for_ids(
        disp_mat, epoch_cols, point_ids, test_ids, label_by_id, test_t, cfg.WINDOW_LENGTH
    )

    train_loader = DataLoader(WindowDataset(X_train, y_train), batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(WindowDataset(X_val, y_val), batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(WindowDataset(X_test, y_test), batch_size=cfg.BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TCN(in_ch=2, hidden=cfg.HIDDEN, dropout=cfg.DROPOUT).to(device)
    best_model_path = OUTPUT_DIR / "best_model.pt"

    train_model(model, train_loader, val_loader, device, best_model_path, cfg)
    test_auc = evaluate_auc(model, test_loader, device)
    print(f"Test AUC (point/time split): {test_auc:.4f}")

    mean_df, auc_far_df, auc_near_df, auc_all_df = rolling_inference_and_save_outputs(
        model=model,
        df=df,
        gdf_latlon=gdf_latlon,
        gdf_utm=gdf_utm,
        polygon=polygon,
        polygon_utm=gpd.GeoSeries([polygon], crs="EPSG:4326").to_crs(epsg=epsg_utm).geometry.iloc[0],
        epoch_cols=epoch_cols,
        disp_mat=disp_mat,
        device=device,
        cfg=cfg,
    )

    write_trend_summary(mean_df, auc_far_df, auc_near_df, auc_all_df, cfg)


if __name__ == "__main__":
    main()
