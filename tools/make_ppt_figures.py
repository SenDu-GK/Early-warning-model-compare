import json
import os
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "outputs" / "experiments" / "summary.csv"
OUT_DIR = ROOT / "outputs" / "ppt_figs"

FIG_DPI = 220


def _ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_summary() -> pd.DataFrame:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"summary.csv not found: {SUMMARY_PATH}")
    df = pd.read_csv(SUMMARY_PATH)
    if "trend_pvalues" in df.columns:
        core_p = []
        auc_p = []
        for raw in df["trend_pvalues"].fillna("{}").astype(str):
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                obj = {}
            core_p.append(obj.get("core_mean_risk_p", np.nan))
            auc_p.append(obj.get("auc_core_vs_near_p", np.nan))
        df["core_mean_risk_p"] = core_p
        df["auc_core_vs_near_p"] = auc_p
    return df


def _variant_name(row: pd.Series) -> str:
    return f"{row['model_name']}_{row['input_mode']}_{row['neg_mode']}"


def _save_fig(fig: plt.Figure, name: str) -> Path:
    out = OUT_DIR / name
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_a_auc_heatmap(df: pd.DataFrame) -> Path:
    df = df.copy()
    df["variant"] = df.apply(_variant_name, axis=1)
    tasks = sorted(df["task"].unique())
    variants = sorted(df["variant"].unique())

    mat = np.full((len(tasks), len(variants)), np.nan, dtype=float)
    for i, task in enumerate(tasks):
        for j, var in enumerate(variants):
            sel = df[(df["task"] == task) & (df["variant"] == var)]
            if len(sel) > 0:
                mat[i, j] = float(sel.iloc[0]["AUC_near_lastN_mean"])

    fig, ax = plt.subplots(figsize=(max(8, len(variants) * 0.8), 3 + len(tasks) * 0.6))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=45, ha="right")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks)
    ax.set_title("AUC_near_lastN_mean by task and run variant")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)

    fig.colorbar(im, ax=ax, label="AUC_near_lastN_mean")
    fig.tight_layout()
    return _save_fig(fig, "FigA_task_auc_heatmap.png")


def fig_b_tpr_bar(df: pd.DataFrame) -> Path:
    df = df.copy()
    df["model_input"] = df["model_name"].astype(str) + "_" + df["input_mode"].astype(str)

    tasks = sorted(df["task"].unique())
    model_inputs = sorted(df["model_input"].unique())
    neg_modes = sorted(df["neg_mode"].unique())

    width = 0.8 / max(1, len(neg_modes))
    x = np.arange(len(tasks) * len(model_inputs))

    fig, ax = plt.subplots(figsize=(max(9, len(x) * 0.4), 4.5))

    for idx, neg in enumerate(neg_modes):
        vals = []
        for task in tasks:
            for mi in model_inputs:
                sel = df[(df["task"] == task) & (df["model_input"] == mi) & (df["neg_mode"] == neg)]
                vals.append(float(sel.iloc[0]["TPR"]) if len(sel) else np.nan)
        offset = (idx - (len(neg_modes) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, label=neg)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n{mi}" for t in tasks for mi in model_inputs], rotation=45, ha="right")
    ax.set_ylabel("TPR")
    ax.set_title("TPR by task and model/input (grouped by neg_mode)")
    ax.legend(title="neg_mode")
    fig.tight_layout()
    return _save_fig(fig, "FigB_task_tpr_bar.png")


def fig_c_leadtime_tradeoff(df: pd.DataFrame) -> Path:
    lead = df[df["task"] == "leadtime_cls"].copy()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    markers = {"tcn": "o", "gru": "s"}
    colors = {"2ch": "tab:blue", "4ch": "tab:orange"}

    for _, row in lead.iterrows():
        x = row["lead_time_days"]
        y = row["AUC_near_lastN_mean"]
        ax.scatter(
            x,
            y,
            marker=markers.get(row["model_name"], "o"),
            color=colors.get(row["input_mode"], "gray"),
            s=70,
            edgecolor="black",
        )

    best = select_best_run(lead)
    if best is not None:
        ax.annotate(
            best["run_name"],
            xy=(best["lead_time_days"], best["AUC_near_lastN_mean"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel("AUC_near_lastN_mean")
    ax.set_title("Leadtime trade-off (leadtime_cls)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_fig(fig, "FigC_leadtime_tradeoff.png")


def fig_d_trend_sig(df: pd.DataFrame) -> Path:
    df = df.copy()
    df["core_mean_risk_p"] = df["core_mean_risk_p"].clip(lower=1e-300)
    df["auc_core_vs_near_p"] = df["auc_core_vs_near_p"].clip(lower=1e-300)

    grouped = df.groupby("task")[["core_mean_risk_p", "auc_core_vs_near_p"]].mean().reset_index()
    x = np.arange(len(grouped))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(x - width / 2, -np.log10(grouped["core_mean_risk_p"]), width, label="core_mean_risk_p")
    ax.bar(x + width / 2, -np.log10(grouped["auc_core_vs_near_p"]), width, label="auc_core_vs_near_p")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["task"].tolist())
    ax.set_ylabel("-log10(p)")
    ax.set_title("Trend significance by task")
    ax.legend()
    fig.tight_layout()
    return _save_fig(fig, "FigD_trend_sig.png")


def select_best_run(df: pd.DataFrame) -> Optional[pd.Series]:
    if df.empty:
        return None
    eligible = df[df["FAR"] <= 0.01].copy()
    if eligible.empty:
        return None
    eligible = eligible.sort_values(["lead_time_days", "AUC_near_lastN_mean"], ascending=[False, False])
    return eligible.iloc[0]


def fig_e_best_run_curves(best_run: pd.Series) -> Path:
    run_dir = ROOT / best_run["run_dir"]
    mean_png = run_dir / "mean_risk_over_time_by_class.png"
    auc_png = run_dir / "auc_over_time_core_vs_near.png"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, img_path, title in [
        (axes[0], mean_png, "Mean Risk Over Time"),
        (axes[1], auc_png, "AUC Over Time (core vs near)"),
    ]:
        if img_path.exists():
            img = plt.imread(img_path)
            ax.imshow(img)
            ax.set_title(title)
        else:
            ax.text(0.5, 0.5, f"Missing: {img_path.name}", ha="center", va="center")
        ax.axis("off")

    fig.suptitle(f"Best run curves: {best_run['run_name']}")
    fig.tight_layout()
    return _save_fig(fig, "FigE_best_run_curves.png")


def _pick_map_images(run_dir: Path) -> List[Path]:
    latlon_dir = run_dir / "risk_maps_png_latlon"
    utm_dir = run_dir / "risk_maps_png_utm"

    candidates = sorted(latlon_dir.glob("*.png"))
    if not candidates:
        candidates = sorted(utm_dir.glob("*.png"))

    if not candidates:
        return []

    idxs = [0, len(candidates) // 2, len(candidates) - 1]
    return [candidates[i] for i in idxs]


def fig_f_best_run_maps(best_run: pd.Series) -> Path:
    run_dir = ROOT / best_run["run_dir"]
    maps = _pick_map_images(run_dir)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    if not maps:
        for ax in axes:
            ax.text(0.5, 0.5, "No map images found", ha="center", va="center")
            ax.axis("off")
        fig.tight_layout()
        return _save_fig(fig, "FigF_best_run_maps.png")

    for ax, path in zip(axes, maps):
        img = plt.imread(path)
        ax.imshow(img)
        ax.set_title(path.name)
        ax.axis("off")

    fig.suptitle(f"Best run maps: {best_run['run_name']}")
    fig.tight_layout()
    return _save_fig(fig, "FigF_best_run_maps.png")


def save_topline_table(df: pd.DataFrame) -> Path:
    cols = [
        "task",
        "model_name",
        "input_mode",
        "neg_mode",
        "AUC_near_lastN_mean",
        "lead_time_days",
        "FAR",
        "TPR",
        "core_mean_risk_p",
        "auc_core_vs_near_p",
    ]
    out = OUT_DIR / "topline_table.csv"
    df[cols].to_csv(out, index=False)
    return out


def write_slide_plan() -> Path:
    path = OUT_DIR / "slide_plan.md"
    content = """# Slide Plan (8 slides)

## Slide 1: Title + Objective
Talk track: Introduce the early-warning task, location, and high-level goal of risk rising near failure.

## Slide 2: Data & Spatial Labeling
Talk track: Describe InSAR points, landslide polygon, and near/far negative definitions.

## Slide 3: Task Definitions
Talk track: Summarize three paradigms (leadtime classification, survival hazard, forecast anomaly) and shared risk_pk outputs.

## Slide 4: Training/Validation Design
Talk track: Emphasize point-id splits, shared split file, and no leakage across runs.

## Slide 5: Metrics Overview
Talk track: Explain AUC-over-time, lead-time thresholding, FAR calibration, and trend statistics.

## Slide 6: Task Comparison (FigA + FigC)
Talk track: Compare AUC by task/variant; highlight leadtime trade-offs and where best leadtime run sits.

## Slide 7: Best Run Curves + Trend Significance (FigE + FigD)
Talk track: Show mean-risk and AUC curves for the best run; interpret trend p-values for temporal strengthening.

## Slide 8: Risk Map Montage (FigF)
Talk track: Show early/mid/late risk maps for the best run; discuss spatial concentration near failure.
"""
    path.write_text(content, encoding="utf-8")
    return path


def main() -> None:
    _ensure_out_dir()
    df = _load_summary()

    fig_paths = []
    fig_paths.append(fig_a_auc_heatmap(df))
    fig_paths.append(fig_b_tpr_bar(df))
    fig_paths.append(fig_c_leadtime_tradeoff(df))
    fig_paths.append(fig_d_trend_sig(df))

    best = select_best_run(df[df["task"] == "leadtime_cls"])
    if best is None:
        raise RuntimeError("No best run found for leadtime_cls with FAR<=0.01")

    fig_paths.append(fig_e_best_run_curves(best))
    fig_paths.append(fig_f_best_run_maps(best))

    table_path = save_topline_table(df)
    slide_plan = write_slide_plan()

    print(f"Best run: {best['run_name']}")
    for p in fig_paths:
        print(f"FIG: {p}")
    print(f"TABLE: {table_path}")
    print(f"SLIDES: {slide_plan}")


if __name__ == "__main__":
    main()
