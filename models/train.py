import os
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

CLASSES     = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
RESULTS_DIR = 'results'


def _mkdir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


# ── PyTorch Trainer ───────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, model, train_dataloader, test_dataloader,
                 lr, wd, epochs, patience, device):
        self.epochs           = epochs
        self.patience         = patience
        self.model            = model
        self.train_dataloader = train_dataloader
        self.test_dataloader  = test_dataloader
        self.device           = device
        self.optimizer        = torch.optim.Adam(model.parameters(),
                                                  lr=lr, weight_decay=wd)
        self.criterion        = nn.CrossEntropyLoss()

    def train(self, save=False, plot=False):
        self.model.train()
        self.train_acc  = []
        self.train_loss = []
        self.val_acc    = []
        self.val_loss   = []

        best_val_loss   = float('inf')
        no_improve      = 0
        best_state      = None

        for epoch in range(self.epochs):
            # ── Training loop ──────────────────────────────────────────────
            self.model.train()
            total_loss    = 0
            total_correct = 0
            total_samples = 0

            progress_bar = tqdm(self.train_dataloader,
                                desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)

            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss    = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                _, preds   = outputs.max(1)
                correct    = (preds == labels).sum().item()
                total      = labels.size(0)
                total_correct += correct
                total_samples += total
                total_loss    += loss.item()

                progress_bar.set_postfix({
                    'Acc':  f'{100.*total_correct/total_samples:.2f}%',
                    'Loss': f'{total_loss/total_samples:.4f}'
                })

            train_epoch_acc  = 100. * total_correct / total_samples
            train_epoch_loss = total_loss / total_samples
            self.train_acc.append(train_epoch_acc)
            self.train_loss.append(train_epoch_loss)

            # ── Validation loop ────────────────────────────────────────────
            val_acc, val_loss = self._validate()
            self.val_acc.append(val_acc)
            self.val_loss.append(val_loss)

            print(f"Epoch {epoch+1:>3}/{self.epochs}  "
                  f"Train Loss: {train_epoch_loss:.4f}  Train Acc: {train_epoch_acc:.2f}%  |  "
                  f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%")

            # ── Early stopping ────────────────────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.cpu().clone()
                                 for k, v in self.model.state_dict().items()}
                no_improve    = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch+1} "
                          f"(no improvement for {self.patience} epochs)")
                    break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()}
            )

        if save:
            torch.save(self.model.state_dict(), "rosly_mamekem_model.pth")
            print("Model saved → rosly_mamekem_model.pth")
        if plot:
            self.plot_training_history()

    @torch.no_grad()
    def _validate(self):
        """Run one pass over test_dataloader, return (accuracy, loss)."""
        self.model.eval()
        total_loss    = 0
        total_correct = 0
        total_samples = 0

        for inputs, labels in self.test_dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs  = self.model(inputs)
            loss     = self.criterion(outputs, labels)
            _, preds = outputs.max(1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            total_loss    += loss.item() * labels.size(0)

        return (100. * total_correct / total_samples,
                total_loss / total_samples)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss    = 0
        total_correct = 0
        total_samples = 0
        all_preds     = []
        all_labels    = []

        for inputs, labels in tqdm(self.test_dataloader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs  = self.model(inputs)
            loss     = self.criterion(outputs, labels)
            _, preds = outputs.max(1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            total_loss    += loss.item() * labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / total_samples
        accuracy = 100. * total_correct / total_samples
        print(f"\nTest Accuracy: {accuracy:.2f}%  |  Test Loss: {avg_loss:.4f}")
        self.save_results(all_labels, all_preds, accuracy, avg_loss, 'pytorch')
        return accuracy, avg_loss

    # ── Plots ──────────────────────────────────────────────────────────────────

    def plot_training_history(self):
        _mkdir()
        epochs = range(1, len(self.train_loss) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss — train (blue solid) vs val (orange dashed)
        ax1.plot(epochs, self.train_loss, color='#2196F3',
                 marker='o', markersize=3, label='Train Loss')
        ax1.plot(epochs, self.val_loss,   color='#FF9800',
                 marker='o', markersize=3, linestyle='--', label='Val Loss')
        ax1.set_title('Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy — train (green solid) vs val (red dashed)
        ax2.plot(epochs, self.train_acc, color='#4CAF50',
                 marker='o', markersize=3, label='Train Accuracy')
        ax2.plot(epochs, self.val_acc,   color='#F44336',
                 marker='o', markersize=3, linestyle='--', label='Val Accuracy')
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle('PyTorch – Training vs Validation', fontsize=14, fontweight='bold')
        fig.tight_layout()
        path = os.path.join(RESULTS_DIR, 'training_with_pytorch.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Saved → {path}")

    def save_results(self, y_true, y_pred, accuracy, loss, framework):
        _mkdir()
        self._plot_confusion_matrix(y_true, y_pred, framework)
        self._plot_classification_report(y_true, y_pred, framework)
        self._plot_per_class_accuracy(y_true, y_pred, framework)
        self._save_metrics_txt(y_true, y_pred, accuracy, loss, framework)

    def _plot_confusion_matrix(self, y_true, y_pred, framework):
        cm   = confusion_matrix(y_true, y_pred)
        norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for ax, data, title, fmt in zip(
            axes,
            [cm,   norm],
            ['Confusion Matrix (counts)', 'Confusion Matrix (normalized)'],
            ['d',  '.2f'],
        ):
            sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                        xticklabels=CLASSES, yticklabels=CLASSES,
                        linewidths=0.5, ax=ax)
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.tick_params(axis='x', rotation=45)

        fig.suptitle(f'{framework.upper()} – Confusion Matrix',
                     fontsize=14, fontweight='bold')
        fig.tight_layout()
        path = os.path.join(RESULTS_DIR, f'confusion_matrix_{framework}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Saved → {path}")

    def _plot_classification_report(self, y_true, y_pred, framework):
        report = classification_report(y_true, y_pred,
                                       target_names=CLASSES, output_dict=True)
        metrics = ['precision', 'recall', 'f1-score']
        data    = np.array([[report[c][m] for m in metrics] for c in CLASSES])

        fig, ax = plt.subplots(figsize=(10, 5))
        x      = np.arange(len(CLASSES))
        width  = 0.25
        colors = ['#4C72B0', '#DD8452', '#55A868']

        for i, (metric, color) in enumerate(zip(metrics, colors)):
            ax.bar(x + i * width, data[:, i], width,
                   label=metric.capitalize(), color=color)

        ax.set_xticks(x + width)
        ax.set_xticklabels(CLASSES, rotation=30, ha='right')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Score')
        ax.set_title(f'{framework.upper()} – Precision / Recall / F1 per Class',
                     fontweight='bold')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

        fig.tight_layout()
        path = os.path.join(RESULTS_DIR, f'classification_report_{framework}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Saved → {path}")

    def _plot_per_class_accuracy(self, y_true, y_pred, framework):
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
        plt.show()
        print(f"Saved → {path}")

    def _save_metrics_txt(self, y_true, y_pred, accuracy, loss, framework):
        report = classification_report(y_true, y_pred, target_names=CLASSES)
        path   = os.path.join(RESULTS_DIR, f'metrics_{framework}.txt')
        with open(path, 'w') as f:
            f.write(f"Framework     : {framework}\n")
            f.write(f"Test Accuracy : {accuracy:.2f}%\n")
            f.write(f"Test Loss     : {loss:.4f}\n\n")
            f.write(report)
        print(f"Saved → {path}")


# ── TensorFlow Trainer ────────────────────────────────────────────────────────

class TFTrainer(Trainer):
    def __init__(self, model, train_gen, test_gen, lr, epochs, patience):
        self.model     = model
        self.train_gen = train_gen
        self.test_gen  = test_gen
        self.epochs    = epochs
        self.patience  = patience

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

    def train(self, save=False, plot=False):
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=4,
                min_lr=1e-6, verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=self.patience,
                restore_best_weights=True, verbose=1
            ),
        ]

        self.history = self.model.fit(
            self.train_gen,
            epochs=self.epochs,
            validation_data=self.test_gen,
            callbacks=callbacks,
            verbose=1,
        )

        if save:
            self.model.save('rosly_mamekem_model.keras')
            print("Model saved → rosly_mamekem_model.keras")
        if plot:
            self.plot_training_history()

    def evaluate(self):
        loss, accuracy = self.model.evaluate(self.test_gen, verbose=0)
        accuracy *= 100
        print(f"\nTest Accuracy: {accuracy:.2f}%  |  Test Loss: {loss:.4f}")

        self.test_gen.reset()
        y_pred_prob = self.model.predict(self.test_gen, verbose=0)
        y_pred      = np.argmax(y_pred_prob, axis=1)
        y_true      = self.test_gen.classes

        self.save_results(y_true, y_pred, accuracy, loss, 'tensorflow')
        return accuracy, loss

    def plot_training_history(self):
        _mkdir()
        hist   = self.history.history
        epochs = range(1, len(hist['loss']) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss — train (blue solid) vs val (orange dashed)
        ax1.plot(epochs, hist['loss'],     color='#2196F3',
                 marker='o', markersize=3, label='Train Loss')
        ax1.plot(epochs, hist['val_loss'], color='#FF9800',
                 marker='o', markersize=3, linestyle='--', label='Val Loss')
        ax1.set_title('Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy — train (green solid) vs val (red dashed)
        ax2.plot(epochs, hist['accuracy'],     color='#4CAF50',
                 marker='o', markersize=3, label='Train Accuracy')
        ax2.plot(epochs, hist['val_accuracy'], color='#F44336',
                 marker='o', markersize=3, linestyle='--', label='Val Accuracy')
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle('TensorFlow/Keras – Training vs Validation',
                     fontsize=14, fontweight='bold')
        fig.tight_layout()
        path = os.path.join(RESULTS_DIR, 'training_with_keras.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Saved → {path}")
