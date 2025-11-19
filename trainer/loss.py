import torch
import torch.nn as nn


def _get_cross_entropy_loss(**kwargs):
    """Returns an instance of CrossEntropyLoss."""
    return nn.CrossEntropyLoss(**kwargs)

class LossCalculator:
    """
    A class to manage and calculate loss functions for the training process.
    It can handle a weighted sum of multiple loss functions.
    """
    def __init__(self, config: dict, device: str):
        """
        Initializes the LossCalculator based on a configuration dictionary.

        Args:
            config (dict): A dictionary from the config file, e.g., config['loss_params'].
            device (str): The device to move the loss functions to ('cuda' or 'cpu').
        """
        self.device = device
        self.loss_fns = []
        self.weights = []
        self.history = {'train': [], 'val': []}
        
        self._loss_mapping = {
            'CrossEntropyLoss': _get_cross_entropy_loss
        }

        for loss_config in config['losses']:
            name = loss_config['name']
            weight = loss_config['weight']
            
            if name in self._loss_mapping:
                self.loss_fns.append(self._loss_mapping[name]().to(self.device))
                self.weights.append(weight)
            else:
                raise ValueError(f"Loss function '{name}' not supported.")

    def calculate(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calculates the total weighted loss."""
        total_loss = 0.0
        for weight, loss_fn in zip(self.weights, self.loss_fns):
            total_loss += weight * loss_fn(y_pred, y_true.squeeze().long())
        return total_loss

    def reset_history(self):
        """Resets the history for a new training run (e.g., a new fold)."""
        self.history = {'train': [], 'val': []}
