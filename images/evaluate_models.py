from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path
    preprocess: callable
    cm_out: Path


def _load_ds(data_dir: Path, image_size=(224, 224), batch_size: int = 32):
    # shuffle=False to keep deterministic ordering for y_true extraction.
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        shuffle=False,
        image_size=image_size,
        batch_size=batch_size,
    )
    class_names = list(ds.class_names)
    return ds, class_names


def _collect_labels(ds) -> np.ndarray:
    y = []
    for _, labels in ds:
        y.append(labels.numpy())
    return np.concatenate(y, axis=0)


def _predict(model, ds, preprocess) -> np.ndarray:
    def _pp(images, labels):
        images = tf.cast(images, tf.float32)
        images = preprocess(images)
        return images, labels

    pp_ds = ds.map(_pp, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    probs = model.predict(pp_ds, verbose=0)
    if probs.ndim != 2:
        raise ValueError(f"Unexpected model output shape: {probs.shape}")
    return probs


def _per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        average=None,
        zero_division=0,
    )
    df = pd.DataFrame(
        {
            "class": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )
    return df


def _save_cm_figure(cm: np.ndarray, labels: list[str], out_path: Path, title: str):
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=200)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)

    # Keep the plot compact; LaTeX already provides captions.
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate .h5 scene models on seg_test.")
    parser.add_argument(
        "--only",
        choices=["all", "mobilenet", "efficientnet"],
        default="all",
        help="Evaluate only one model (default: all).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main():
    args = _parse_args()
    root = Path(__file__).resolve().parent
    data_dir = root / "seg_test"
    if not data_dir.exists():
        raise SystemExit(f"Missing test directory: {data_dir}")

    ds, class_names = _load_ds(
        data_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
    )
    y_true = _collect_labels(ds)

    specs_all = [
        ModelSpec(
            name="MobileNetV2",
            path=root / "scene_mobilenetv2.h5",
            preprocess=tf.keras.applications.mobilenet_v2.preprocess_input,
            cm_out=root / "images" / "cm_mobilenetv2.png",
        ),
        ModelSpec(
            name="EfficientNetB0",
            path=root / "scene_efficientnetb0.h5",
            preprocess=tf.keras.applications.efficientnet.preprocess_input,
            cm_out=root / "images" / "cm_efficientnetb0.png",
        ),
    ]

    if args.only == "mobilenet":
        specs = [specs_all[0]]
    elif args.only == "efficientnet":
        specs = [specs_all[1]]
    else:
        specs = specs_all

    results = {}

    for spec in specs:
        if not spec.path.exists():
            raise SystemExit(f"Missing model: {spec.path}")

        print(f"\n=== Evaluating {spec.name} ===")
        model = tf.keras.models.load_model(spec.path)
        probs = _predict(model, ds, spec.preprocess)
        y_pred = probs.argmax(axis=1)

        acc = float(accuracy_score(y_true, y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        _save_cm_figure(cm, class_names, spec.cm_out, f"Confusion Matrix - {spec.name}")

        per_cls = _per_class_metrics(y_true, y_pred, class_names)
        macro = per_cls[["precision", "recall", "f1"]].mean().to_dict()
        weighted = (
            (per_cls[["precision", "recall", "f1"]].T * per_cls["support"].to_numpy()).T.sum()
            / per_cls["support"].sum()
        ).to_dict()

        results[spec.name] = {
            "accuracy": acc,
            "cm_path": str(spec.cm_out.relative_to(root)).replace("\\\\", "/"),
            "per_class": per_cls.to_dict(orient="records"),
            "macro_avg": {k: float(v) for k, v in macro.items()},
            "weighted_avg": {k: float(v) for k, v in weighted.items()},
        }

        print(f"Accuracy: {acc:.4f}")
        print(per_cls[["class", "precision", "recall", "f1", "support"]].to_string(index=False))
        print(f"Saved confusion matrix: {spec.cm_out}")

    out_json = root / "metrics_seg_test.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote metrics: {out_json}")


if __name__ == "__main__":
    main()
