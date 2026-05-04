import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# ---------------------- NEURAL NETWORK DEFINITION ----------------------
class NeuralNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        """
        Feedforward neural network for classification.
        """
        super(NeuralNetwork, self).__init__()

        layers = []
        prev_size = input_size

        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size

        # Output layer (raw logits — CrossEntropyLoss applies softmax internally)
        layers.append(nn.Linear(prev_size, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------- MODEL TRAINER ----------------------
class ModelTrainer:
    def __init__(self, model: nn.Module, learning_rate: float = 0.001,
                 max_lr: float = 0.01, epochs: int = 20, steps_per_epoch: int = None):
        """
        Trainer class for classification using CrossEntropyLoss and OneCycleLR.
        """
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Scheduler will be initialized after we know steps_per_epoch
        self.scheduler = None
        self.max_lr = max_lr
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

        self.train_losses, self.val_losses = [], []
        self.train_accuracies, self.val_accuracies = [], []

    def train_step(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(X)
        loss = self.criterion(outputs, y.squeeze())
        loss.backward()
        self.optimizer.step()

        # Accuracy
        _, preds = torch.max(outputs, 1)
        # Alternative fix for constant 1.0
        acc = (preds == y.squeeze()).float().mean().item()
        return loss.item(), acc

    def validate(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            loss = self.criterion(outputs, y.squeeze())
            _, preds = torch.max(outputs, 1)
            # Alternative fix for constant 1.0
            acc = (preds == y.squeeze()).float().mean().item()
        return loss.item(), acc

    def train(self, train_loader, val_loader, epochs: int) -> Tuple[list, list]:
        # Initialize OneCycleLR using the known number of batches per epoch
        if self.scheduler is None:
            self.steps_per_epoch = len(train_loader)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.max_lr,
                steps_per_epoch=self.steps_per_epoch,
                epochs=epochs
            )

        for epoch in range(epochs):
            train_losses, train_accs = [], []
            val_losses, val_accs = [], []

            for X_batch, y_batch in train_loader:
                loss, acc = self.train_step(X_batch, y_batch)
                train_losses.append(loss)
                train_accs.append(acc)

                # Step the scheduler after each batch
                self.scheduler.step()

            for X_batch, y_batch in val_loader:
                loss, acc = self.validate(X_batch, y_batch)
                val_losses.append(loss)
                val_accs.append(acc)

            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            avg_train_acc = np.mean(train_accs)
            avg_val_acc = np.mean(val_accs)

            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)
            self.train_accuracies.append(avg_train_acc)
            self.val_accuracies.append(avg_val_acc)

            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs}: "
                f"Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, "
                f"Train Acc={avg_train_acc:.3f}, Val Acc={avg_val_acc:.3f}, "
                f"LR={current_lr:.6f}")

        return self.train_losses, self.val_losses


    def plot_losses(self):
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(self.train_losses, label='Train Loss', color='tab:red')
        ax1.plot(self.val_losses, label='Val Loss', color='tab:orange')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='tab:red')
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.plot(self.train_accuracies, label='Train Acc', color='tab:blue')
        ax2.plot(self.val_accuracies, label='Val Acc', color='tab:cyan')
        ax2.set_ylabel('Accuracy', color='tab:blue')
        ax2.legend(loc='upper right')

        plt.title('Training & Validation Loss/Accuracy')
        plt.grid(True)
        plt.show()

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
