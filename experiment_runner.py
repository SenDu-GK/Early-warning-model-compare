from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

import main as ew_main


EXPERIMENTS_DIR = Path("outputs/experiments")
SPLITS_DIR = Path("outputs/splits")

TASK_BASE_CONFIGS = {
    "leadtime_cls": Path("configs/leadtime_base.yaml"),
    "survival_discrete": Path("configs/survival_base.yaml"),
    "forecast_anom": Path("configs/forecast_base.yaml"),
}


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping/dictionary: {path}")
    return cfg


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def run_name(task: str, model: str, input_mode: str, neg_mode: str) -> str:
    return f"{task}_{model}_in{input_mode}_{neg_mode}"


def select_best(df: pd.DataFrame) -> pd.Series | None:
    if df.empty:
        return None
    eligible = df[df["FAR"] <= 0.01].copy()
    if eligible.empty:
        return None
    eligible = eligible.sort_values(["lead_time_days", "AUC_near_lastN_mean"], ascending=[False, False])
    return eligible.iloc[0]


def shared_split_path(seed: int) -> Path:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    return SPLITS_DIR / f"split_seed{seed}.json"


def main() -> None:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Use the leadtime base config as the source of truth for the shared split seed.
    lead_cfg = load_config(TASK_BASE_CONFIGS["leadtime_cls"])
    split_seed = int(lead_cfg["SPLIT"]["SEED"])
    split_path = shared_split_path(split_seed)
    print(f"Shared split file: {split_path}")

    models = ["tcn", "gru"]
    input_modes = ["2ch", "4ch"]
    neg_modes = ["near_only", "near_plus_far"]

    summaries: list[dict] = []

    for task, base_path in TASK_BASE_CONFIGS.items():
        base_cfg = load_config(base_path)
        base_cfg["TASK"] = task

        for model, input_mode, neg_mode in itertools.product(models, input_modes, neg_modes):
            name = run_name(task, model, input_mode, neg_mode)
            run_dir = EXPERIMENTS_DIR / name
            print("=" * 80)
            print(f"Running experiment: {name}")

            overrides = {
                "MODEL": {"NAME": model},
                "LABELING": {
                    "INPUT_MODE": input_mode,
                    "NEG_MODE": neg_mode,
                },
            }
            cfg = deep_update(base_cfg, overrides)

            # Save the exact config used for this run.
            run_dir.mkdir(parents=True, exist_ok=True)
            with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)

            summary = ew_main.run_experiment(cfg, run_dir=run_dir, split_path=split_path)
            summaries.append(
                {
                    "run_name": name,
                    "task": summary["task"],
                    "model_name": summary["model"],
                    "input_mode": summary["input_mode"],
                    "neg_mode": summary["neg_mode"],
                    "H": summary["H"],
                    "Hf": summary["Hf"],
                    "AUC_near_lastN_mean": summary["auc_near_lastN_mean"],
                    "lead_time_days": summary["lead_time_days"],
                    "FAR": summary["far"],
                    "TPR": summary["tpr"],
                    "trend_pvalues": summary["trend_pvalues"],
                    "test_auc": summary["test_auc"],
                    "run_dir": summary["run_dir"],
                    "split_path": summary["split_path"],
                }
            )

    summary_df = pd.DataFrame(summaries)
    summary_path = EXPERIMENTS_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    best = select_best(summary_df)
    print("=" * 80)
    print(f"Summary table: {summary_path}")
    if best is None:
        print("No run satisfied FAR<=1%. See summary.csv for details.")
    else:
        print(
            "Best run (FAR<=1%) by lead_time_days then AUC_near_lastN_mean: "
            f"{best['run_name']} | lead_time_days={best['lead_time_days']} | "
            f"AUC_near_lastN_mean={best['AUC_near_lastN_mean']} | FAR={best['FAR']}"
        )


if __name__ == "__main__":
    main()
