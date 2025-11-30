import torch
from tqdm import tqdm
import numpy as np

class Trainer:
    """
    A class to handle the training and validation of a model for a single run.
    """
    def __init__(self, model, optimizer, device, loss_calculator, metrics_calculator):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.loss_calculator = loss_calculator
        self.metrics_calculator = metrics_calculator

    def _train_epoch(self, data_loader):
        """Performs one epoch of training."""
        self.model.train()
        total_loss = 0.0
        all_metrics = {name: [] for name in self.metrics_calculator.metric_names}

        for eeg_features, labels in data_loader:
            eeg_features = eeg_features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(eeg_features)
            loss = self.loss_calculator.calculate(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            metrics = self.metrics_calculator.calculate(outputs, labels)
            for name, value in metrics.items():
                all_metrics[name].append(value)

        avg_loss = total_loss / len(data_loader)
        avg_metrics = {name: np.mean(values) for name, values in all_metrics.items()}
        return avg_loss, avg_metrics

    def _validate_epoch(self, data_loader):
        """Performs one epoch of validation."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_metrics = {name: [] for name in self.metrics_calculator.metric_names}

        with torch.no_grad():
            for eeg_features, labels in data_loader:
                eeg_features = eeg_features.to(self.device)
                labels = labels.to(self.device)

                # Handle models that may return attention weights during eval
                outputs = self.model(eeg_features)
                loss = self.loss_calculator.calculate(outputs, labels)

                total_loss += loss.item()
                metrics = self.metrics_calculator.calculate(outputs, labels)
                for name, value in metrics.items():
                    all_metrics[name].append(value)
                
                all_preds.append(torch.argmax(outputs, dim=1))
                all_labels.append(labels.squeeze())

        avg_loss = total_loss / len(data_loader)
        avg_metrics = {name: np.mean(values) for name, values in all_metrics.items()}
        
        # Concatenate all predictions and labels from the epoch
        return avg_loss, avg_metrics, torch.cat(all_preds), torch.cat(all_labels)

    def fit(self, train_loader, val_loader, num_epochs):
        """
        The main training loop.

        Args:
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation set.
            num_epochs (int): The number of epochs to train for.

        Returns:
            dict: A history dictionary containing epoch-wise loss and metrics.
            float: The best validation score achieved.
            np.array: The predictions from the best epoch.
            np.array: The true labels from the best epoch.
        """
        best_model_state = None
        best_val_accuracy = 0.0
        best_preds, best_labels = None, None
        self.loss_calculator.reset_history()
        self.metrics_calculator.reset_history()

        with tqdm(range(num_epochs), desc="Epochs") as pbar:
            for epoch in pbar:
                # --- Training ---
                train_loss, train_metrics = self._train_epoch(train_loader)
                self.loss_calculator.history['train'].append(train_loss)
                for name, value in train_metrics.items():
                    self.metrics_calculator.history[f'train_{name}'].append(value)
                
                # --- Validation ---                
                val_loss, val_metrics, epoch_preds, epoch_labels = self._validate_epoch(val_loader)
                self.loss_calculator.history['val'].append(val_loss)
                for name, value in val_metrics.items():
                    self.metrics_calculator.history[f'val_{name}'].append(value)

                # --- Update Progress Bar ---
                postfix_str = f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                for name in self.metrics_calculator.metric_names:
                    postfix_str += f", Val {name}: {val_metrics[name]:.4f}"
                pbar.set_postfix_str(postfix_str)
            
                # Simple checkpointing
                current_val_accuracy = val_metrics.get("Accuracy", 0)
                if current_val_accuracy > best_val_accuracy:
                    best_val_accuracy = current_val_accuracy
                    best_preds = epoch_preds.cpu().numpy()
                    best_labels = epoch_labels.cpu().numpy()
                    # Use deepcopy to save the state, as the model continues to train
                    best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        # Combine history from both calculators
        full_history = {**self.loss_calculator.history, **self.metrics_calculator.history}

        # Load the best model state before returning the model
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return full_history, best_val_accuracy, best_preds, best_labels, self.model
