import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch import nn, optim

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.tabular_csv import make_loaders_from_csv  # noqa: E402
from src.models.mlp import MLP, MLPConfig  # noqa: E402
from src.training.trainer import evaluate, train_one_epoch  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train and compare simple tabular models.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "cadia_des.csv",
        help="Path to the training CSV (features + label column).",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="Velocity_mm_yr",
        help="Name of the label column in the CSV.",
    )
    parser.add_argument(
        "--drop-cols",
        nargs="*",
        default=["ID"],
        help="Optional columns to drop before training (e.g., identifiers).",
    )
    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        default="regression",
        help="Choose loss/metric behavior.",
    )
    parser.add_argument(
        "--model",
        choices=["mlp", "linear"],
        default="mlp",
        help="Model family to train. Linear uses no hidden layers.",
    )
    parser.add_argument(
        "--hidden-dims",
        nargs="+",
        type=int,
        default=[128, 64],
        help="Hidden layer sizes for the MLP.",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device string understood by torch (e.g., cuda, cpu).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "runs",
        help="Where to store checkpoints and metrics.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    bundle = make_loaders_from_csv(
        csv_path=str(args.csv),
        label_col=args.label_col,
        drop_cols=args.drop_cols,
        task=args.task,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    if args.model == "mlp":
        model = MLP(
            MLPConfig(
                in_dim=bundle.in_dim,
                out_dim=bundle.out_dim,
                hidden_dims=tuple(args.hidden_dims),
                dropout=args.dropout,
            )
        )
    else:
        model = nn.Linear(bundle.in_dim, bundle.out_dim)

    model = model.to(args.device)

    loss_fn = nn.CrossEntropyLoss() if args.task == "classification" else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_val_loss = float("inf")
    metric_name = "acc" if args.task == "classification" else "mae"
    config_to_save = {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "classes": bundle.classes,
    }

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metric = train_one_epoch(
            model, bundle.train_loader, optimizer, loss_fn, args.device, args.task
        )
        val_loss, val_metric = evaluate(
            model, bundle.val_loader, loss_fn, args.device, args.task
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                f"train_{metric_name}": train_metric,
                "val_loss": val_loss,
                f"val_{metric_name}": val_metric,
            }
        )

        print(
            f"[{epoch:03d}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_{metric_name}={train_metric:.4f} val_{metric_name}={val_metric:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"model_state": model.state_dict(), "config": config_to_save, "history": history},
                run_dir / "best_model.pt",
            )

    # Persist run metadata for quick inspection.
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_to_save, f, indent=2)
    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Finished training. Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
