# Early-warning-model-compare

Lightweight PyTorch project for training and comparing tabular deep-learning baselines on the Cadia Des geocoded points dataset (and other CSVs).

## Repository layout
- `data/raw/cadia_des.csv` — provided 5 MB CSV (public) with geocoded point measurements.
- `src/datasets/tabular_csv.py` — loader that scales features, supports classification/regression, and handles column dropping.
- `src/models/mlp.py` — configurable multilayer perceptron; `--model linear` gives a no-hidden-layer baseline.
- `src/training/trainer.py` — training/eval loops with accuracy (cls) or MAE (reg).
- `scripts/train.py` — CLI entrypoint to run experiments; saves artifacts to `outputs/runs/<timestamp>/`.
- `requirements.txt` — core dependencies.

## Quickstart
```bash
cd Early-warning-model-compare
python -m venv .venv && source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
pip install -r requirements.txt

# Example: regression on the provided dataset (drop ID string column)
python scripts/train.py --csv data/raw/cadia_des.csv --label-col Velocity_mm_yr --task regression --drop-cols ID

# Example: classification (if your label column is categorical)
python scripts/train.py --csv data/raw/your.csv --label-col class_name --task classification --model mlp --hidden-dims 256 128 64
```

Key flags:
- `--label-col` choose the target column; default is `Velocity_mm_yr`.
- `--drop-cols` drop identifiers or non-feature columns (default: `ID`).
- `--task` switch between `regression` and `classification`.
- `--model` choose `mlp` (default) or `linear` baseline.
- `--hidden-dims`, `--dropout`, `--epochs`, `--lr`, `--val-ratio`, `--batch-size` tune training.

Outputs:
- Best checkpoint: `outputs/runs/<timestamp>/best_model.pt` (state dict + config + history).
- Logs: `config.json`, `history.json` in the same run folder.

## Next steps
- Add more model families (e.g., temporal CNN/transformer over date columns, gradient boosting baselines).
- Add k-fold cross-validation utility for robust comparison.
- Create a small EDA notebook to inspect feature distributions and label balance.
