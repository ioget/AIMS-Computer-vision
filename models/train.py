import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf


# ── PyTorch Trainer (unchanged) ───────────────────────────────────────────────

class Trainer:
    def __init__(self, model, train_dataloader, test_dataloader, lr, wd, epochs, device):
        self.epochs = epochs
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, save=False, plot=False):
        self.model.train()
        self.train_acc = []
        self.train_loss = []
        for epoch in range(self.epochs):
            total_loss = 0
            total_correct = 0
            total_samples = 0

            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=False)

            for batch in progress_bar:
                input_datas, labels = batch
                input_datas, labels = input_datas.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(input_datas)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                _, preds = outputs.max(1)
                correct = (preds == labels).sum().item()
                total = labels.size(0)

                total_correct += correct
                total_samples += total
                total_loss += loss.item()

                batch_accuracy = 100.0 * correct / total
                average_accuracy = 100.0 * total_correct / total_samples
                average_loss = total_loss / total_samples

                progress_bar.set_postfix({
                    'Batch Acc': f'{batch_accuracy:.2f}%',
                    'Avg Acc': f'{average_accuracy:.2f}%',
                    'Loss': f'{average_loss:.4f}'
                })
            self.train_acc.append(average_accuracy)
            self.train_loss.append(average_loss)
        if save:
            torch.save(self.model.state_dict(), "rosly_model.pth")
            print("Model saved → rosly_model.pth")
        if plot:
            self.plot_training_history()

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for inputs, labels in tqdm(self.test_dataloader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            _, preds = outputs.max(1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)

        avg_loss = total_loss / total_samples
        accuracy = 100.0 * total_correct / total_samples

        print(f"\nTest Accuracy: {accuracy:.2f}%  |  Test Loss: {avg_loss:.4f}")
        return accuracy, avg_loss

    def plot_training_history(self):
        epochs = range(1, len(self.train_loss) + 1)

        fig, ax1 = plt.subplots(figsize=(8, 5))

        color_loss = 'tab:blue'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color_loss)
        ax1.plot(epochs, self.train_loss, color=color_loss, label='Loss')
        ax1.tick_params(axis='y', labelcolor=color_loss)

        ax2 = ax1.twinx()
        color_acc = 'tab:red'
        ax2.set_ylabel('Accuracy (%)', color=color_acc)
        ax2.plot(epochs, self.train_acc, color=color_acc, label='Accuracy')
        ax2.tick_params(axis='y', labelcolor=color_acc)

        plt.title('Training Loss and Accuracy')
        fig.tight_layout()
        plt.savefig('training_with_pytorch.png')
        plt.show()


# ── TensorFlow Trainer ────────────────────────────────────────────────────────

class TFTrainer:
    def __init__(self, model, train_gen, test_gen, lr, epochs):
        self.model = model
        self.train_gen = train_gen
        self.test_gen = test_gen
        self.epochs = epochs

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

    def train(self, save=False, plot=False):
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=3, verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=7, restore_best_weights=True, verbose=1
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
            self.model.save('rosly_model.keras')
            print("Model saved → rosly_model.keras")

        if plot:
            self.plot_training_history()

    def evaluate(self):
        loss, accuracy = self.model.evaluate(self.test_gen, verbose=0)
        print(f"\nTest Accuracy: {accuracy * 100:.2f}%  |  Test Loss: {loss:.4f}")
        return accuracy * 100, loss

    def plot_training_history(self):
        hist = self.history.history
        epochs = range(1, len(hist['loss']) + 1)

        fig, ax1 = plt.subplots(figsize=(8, 5))

        color_loss = 'tab:blue'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color_loss)
        ax1.plot(epochs, hist['loss'],     color=color_loss, label='Train Loss')
        ax1.plot(epochs, hist['val_loss'], color=color_loss, linestyle='--', label='Val Loss')
        ax1.tick_params(axis='y', labelcolor=color_loss)

        ax2 = ax1.twinx()
        color_acc = 'tab:red'
        ax2.set_ylabel('Accuracy', color=color_acc)
        ax2.plot(epochs, hist['accuracy'],     color=color_acc, label='Train Acc')
        ax2.plot(epochs, hist['val_accuracy'], color=color_acc, linestyle='--', label='Val Acc')
        ax2.tick_params(axis='y', labelcolor=color_acc)

        plt.title('Training Loss and Accuracy (TensorFlow)')
        fig.tight_layout()
        plt.savefig('training_with_keras.png')
        plt.show()
