"""
eval.py — Evaluate saved models and save all metrics/plots to results/

Usage:
    python eval.py --framework pytorch
    python eval.py --framework tensorflow
    python eval.py --framework both
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score
)

CLASSES     = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
RESULTS_DIR = 'results'
PTH_PATH    = os.path.join('models', 'rosly_mamekem_model.pth')
KERAS_PATH  = os.path.join('models', 'rosly_mamekem_model.keras')


def mkdir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, framework):
    cm   = confusion_matrix(y_true, y_pred)
    norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, data, title, fmt in zip(
        axes,
        [cm,    norm],
        ['Confusion Matrix (counts)', 'Confusion Matrix (normalized)'],
        ['d',   '.2f'],
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=CLASSES, yticklabels=CLASSES,
                    linewidths=0.5, ax=ax)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.tick_params(axis='x', rotation=45)

    fig.suptitle(f'{framework.upper()} – Confusion Matrix', fontsize=14, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, f'confusion_matrix_{framework}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {path}")


def plot_classification_report(y_true, y_pred, framework):
    report = classification_report(y_true, y_pred,
                                   target_names=CLASSES, output_dict=True)
    metrics = ['precision', 'recall', 'f1-score']
    data    = np.array([[report[c][m] for m in metrics] for c in CLASSES])

    fig, ax = plt.subplots(figsize=(10, 5))
    x      = np.arange(len(CLASSES))
    width  = 0.25
    colors = ['#4C72B0', '#DD8452', '#55A868']

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        bars = ax.bar(x + i * width, data[:, i], width,
                      label=metric.capitalize(), color=color)
        ax.bar_label(bars, fmt='%.2f', padding=2, fontsize=7)

    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASSES, rotation=30, ha='right')
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score')
    ax.set_title(f'{framework.upper()} – Precision / Recall / F1 per Class',
                 fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, f'classification_report_{framework}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {path}")


def plot_per_class_accuracy(y_true, y_pred, framework):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accs   = [(y_pred[y_true == i] == i).mean() * 100
              for i in range(len(CLASSES))]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(CLASSES, accs, color='steelblue', edgecolor='white')
    ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'{framework.upper()} – Per-Class Accuracy', fontweight='bold')
    ax.tick_params(axis='x', rotation=30)
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, f'per_class_accuracy_{framework}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {path}")


def save_metrics_txt(y_true, y_pred, accuracy, framework):
    report = classification_report(y_true, y_pred, target_names=CLASSES)
    path   = os.path.join(RESULTS_DIR, f'metrics_{framework}.txt')
    with open(path, 'w') as f:
        f.write(f"Framework  : {framework}\n")
        f.write(f"Test Accuracy : {accuracy:.2f}%\n\n")
        f.write(report)
    print(f"  Saved → {path}")


def save_all_plots(y_true, y_pred, accuracy, framework):
    mkdir()
    print(f"\n[{framework.upper()}] Generating results...")
    plot_confusion_matrix(y_true, y_pred, framework)
    plot_classification_report(y_true, y_pred, framework)
    plot_per_class_accuracy(y_true, y_pred, framework)
    save_metrics_txt(y_true, y_pred, accuracy, framework)


# ── PyTorch evaluation ────────────────────────────────────────────────────────

def evaluate_pytorch():
    import torch
    from tqdm import tqdm
    from utils.prep import get_data_pytorch
    from models.cnn_with_Pytorch import CNN1

    assert os.path.isfile(PTH_PATH), f"Model not found: {PTH_PATH}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[PyTorch] Loading {PTH_PATH}  |  device: {device}")

    _, test_loader = get_data_pytorch()

    model = CNN1(num_classes=6).to(device)
    model.load_state_dict(torch.load(PTH_PATH, map_location=device))
    model.eval()

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds) * 100
    print(f"[PyTorch] Test Accuracy: {accuracy:.2f}%")

    save_all_plots(all_labels, all_preds, accuracy, 'pytorch')


# ── TensorFlow evaluation ─────────────────────────────────────────────────────

def evaluate_tensorflow():
    import tensorflow as tf
    from utils.prep import get_data_tensorflow

    assert os.path.isfile(KERAS_PATH), f"Model not found: {KERAS_PATH}"

    print(f"\n[TensorFlow] Loading {KERAS_PATH}")

    _, test_gen = get_data_tensorflow()

    model = tf.keras.models.load_model(KERAS_PATH)
    model.summary()

    test_gen.reset()
    y_pred_prob = model.predict(test_gen, verbose=1)
    y_pred      = np.argmax(y_pred_prob, axis=1)
    y_true      = test_gen.classes

    accuracy = accuracy_score(y_true, y_pred) * 100
    print(f"[TensorFlow] Test Accuracy: {accuracy:.2f}%")

    save_all_plots(y_true, y_pred, accuracy, 'tensorflow')


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved models locally")
    parser.add_argument('--framework', type=str,
                        choices=['pytorch', 'tensorflow', 'both'],
                        default='both',
                        help="Which model to evaluate (default: both)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.framework in ('pytorch', 'both'):
        evaluate_pytorch()

    if args.framework in ('tensorflow', 'both'):
        evaluate_tensorflow()

    print(f"\nAll results saved in '{RESULTS_DIR}/'")
    print("Files:", os.listdir(RESULTS_DIR))


if __name__ == '__main__':
    main()
