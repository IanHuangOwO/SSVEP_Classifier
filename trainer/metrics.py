import torch
from sklearn.metrics import precision_recall_fscore_support

def _calculate_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Helper function to calculate classification accuracy."""
    preds = torch.argmax(y_pred, dim=1)
    correct = (preds == y_true.squeeze()).float()
    acc = correct.sum() / len(correct)
    return acc.item()

def _calculate_prf1(y_pred: torch.Tensor, y_true: torch.Tensor) -> dict:
    """Helper function to calculate precision, recall, and F1-score."""
    preds = torch.argmax(y_pred, dim=1).cpu().numpy()
    labels = y_true.squeeze().cpu().numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    return {"Precision": precision, "Recall": recall, "F1Score": f1}

class MetricsCalculator:
    """
    A class to manage and calculate performance metrics.
    """
    def __init__(self, config: dict):
        """
        Initializes the MetricsCalculator.

        Args:
            config (dict): A dictionary from the config file, e.g., config['metrics_params'].
        """
        self.metric_names = config.get('metrics', [])
        self.history = {}
        for name in self.metric_names:
            self.history[f'train_{name}'] = []
            self.history[f'val_{name}'] = []
        
        self._prf1_calculated = False
        self._prf1_results = {}

    def calculate(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> dict:
        """
        Calculates all configured metrics.

        Returns:
            dict: A dictionary with metric names as keys and their values.
        """
        results = {}
        self._prf1_calculated = False # Reset flag for each call

        if "Accuracy" in self.metric_names:
            results["Accuracy"] = _calculate_accuracy(y_pred, y_true)
        
        # Calculate P, R, F1 together for efficiency
        prf1_metrics = {"Precision", "Recall", "F1Score"}
        if any(metric in self.metric_names for metric in prf1_metrics):
            self._prf1_results = _calculate_prf1(y_pred, y_true)
            self._prf1_calculated = True

        for metric in prf1_metrics:
            if metric in self.metric_names:
                results[metric] = self._prf1_results[metric]
            
        return results

    def reset_history(self):
        """Resets the history for a new training run (e.g., a new fold)."""
        self.history = {key: [] for key in self.history}
