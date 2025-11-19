import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

def plot_kfold_results(histories: list, save_dir=None, subject_id=None):
    """
    Plots the mean and standard deviation of loss and accuracy curves over K folds.

    Args:
        histories (list): A list of history dictionaries, one from each fold.
        save_dir (str, optional): Directory to save the figure. If None, shows the plot. Defaults to None.
        subject_id (int, optional): The subject ID, used for creating a unique filename. Defaults to None.
    """
    if not histories:
        print("No history to plot.")
        return

    # Dynamically find all metrics that were tracked
    all_metric_keys = [key for key in histories[0].keys() if key not in ['train', 'val']]
    metric_names = sorted(list(set(key.split('_')[1] for key in all_metric_keys)))

    # --- Plotting ---
    num_metrics = len(metric_names)
    fig, axes = plt.subplots(1, num_metrics + 1, figsize=(6 * (num_metrics + 1), 5))
    fig.suptitle('K-Fold Cross-Validation Results', fontsize=16)

    # Extract data for all folds
    train_loss = np.array([h['train'] for h in histories])
    val_loss = np.array([h['val'] for h in histories])

    epochs = range(1, train_loss.shape[1] + 1)

    def plot_metric(ax, data, label, color):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        ax.plot(epochs, mean, color=color, label=f'Mean {label}')
        ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.2, label=f'Std Dev {label}')

    # Plot Loss
    plot_metric(axes[0], train_loss, 'Train Loss', 'blue')
    plot_metric(axes[0], val_loss, 'Val Loss', 'orange')
    axes[0].set_title('Loss over Epochs')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Plot all other metrics
    for i, metric_name in enumerate(metric_names):
        ax = axes[i + 1]
        train_data = np.array([h[f'train_{metric_name}'] for h in histories])
        val_data = np.array([h[f'val_{metric_name}'] for h in histories])
        
        plot_metric(ax, train_data, f'Train {metric_name}', 'blue')
        plot_metric(ax, val_data, f'Val {metric_name}', 'orange')
        ax.set_title(f'{metric_name} over Epochs')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric_name)
        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_dir:
        filename = f'kfold_results_S{subject_id}.png' if subject_id else 'kfold_results.png'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"K-Fold plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(y_true, y_pred, save_dir=None, subject_id=None, title_prefix=""):
    """
    Computes, plots, and saves a confusion matrix.

    Args:
        y_true (np.array): Array of true labels.
        y_pred (np.array): Array of predicted labels.
        save_dir (str, optional): Directory to save the figure. If None, does not save.
        subject_id (int, optional): The validation subject ID for the filename.
        train_subject_id (int, optional): The training subject ID for the filename (used in transfer mode).
        title_prefix (str, optional): A prefix for the plot title (e.g., "Inter-Subject").
    """
    if y_true is None or y_pred is None:
        print("Warning: Cannot generate confusion matrix because predictions or labels are missing.")
        return

    cm = confusion_matrix(y_true, y_pred)
    
    # Determine title and filename
    title = f"{title_prefix}: Validation on Subject {subject_id}"
    filename = f"conf_matrix_val_S{subject_id}.png"

    # Plotting
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap='Blues')
    ax.set_title(title)
    
    if save_dir:
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
        plt.close(fig)


def plot_subject_accuracies(accuracies, title='Per-Subject Accuracy', save_dir=None):
    """
    Plots a bar chart of accuracies for each subject.

    Args:
        accuracies (dict): A dictionary where keys are subject IDs and values are their accuracy scores.
        title (str): The title for the plot.
    """
    # ... (rest of the function is the same until the end)
    subjects = list(accuracies.keys())
    scores = list(accuracies.values())

    num_subjects = len(subjects)
    x_pos = np.arange(num_subjects)

    # Generate a list of distinct colors for each bar
    colors = plt.cm.viridis(np.linspace(0, 1, num_subjects))

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(x_pos, scores, align='center', alpha=0.8, capsize=10, color=colors)
    ax.set_ylabel('Accuracy')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'S{s}' for s in subjects])
    ax.set_title(title, y=1.05)
    ax.yaxis.grid(True)
    
    # Set Y-axis to percentage
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(0, 1) # Set ylim in terms of percentage

    # Add accuracy values on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    
    if save_dir:
        # Sanitize title for filename
        filename_title = title.replace(' ', '_').replace('(', '').replace(')', '')
        save_path = os.path.join(save_dir, f'{filename_title}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Subject accuracies plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_accuracy_matrix(accuracy_matrix, subject_list, title='Subject-to-Subject Transfer Accuracy', save_dir=None):
    """
    Plots a heatmap of the accuracy matrix.

    Args:
        accuracy_matrix (dict): Nested dictionary with accuracies.
        subject_list (list): List of subject IDs for ordering axes.
        title (str): The title for the plot.
        save_dir (str, optional): Directory to save the figure. If None, shows the plot. Defaults to None.
    """
    # Convert the dictionary to a 2D numpy array based on subject_list order
    matrix = np.array([[accuracy_matrix[train_id][val_id] for val_id in subject_list] for train_id in subject_list])

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap='hot', interpolation='nearest')

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Accuracy", rotation=-90, va="bottom")

    # Set ticks and labels
    tick_labels = [f'S{s}' for s in subject_list]
    ax.set_xticks(np.arange(len(subject_list)), labels=tick_labels)
    ax.set_yticks(np.arange(len(subject_list)), labels=tick_labels)
    ax.set_xlabel("Validation Subject")
    ax.set_ylabel("Training Subject")
    ax.set_title(title)

    # Loop over data dimensions and create text annotations.
    for i in range(len(subject_list)):
        for j in range(len(subject_list)):
            color = "w" if matrix[i, j] < 0.5 else "k" # Use white text for dark cells
            text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                           ha="center", va="center", color=color)

    fig.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, 'transfer_matrix_accuracy.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Accuracy matrix plot saved to {save_path}")
        plt.close()
    else:
        plt.show()
