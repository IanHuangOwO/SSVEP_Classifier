import numpy as np
import matplotlib.pyplot as plt
import os
import json
import torch

def _visualize_per_label(dataset, channel_names, stimulus_map, data_params):
    """Handles the logic for plotting one feature map per label."""
    sorted_stimuli = sorted(stimulus_map.items(), key=lambda item: float(item[1]))
    sorted_labels = [int(item[0]) for item in sorted_stimuli]

    # --- Get FFT parameters from data_params ---
    Fs = data_params['Sample_Frequency']
    Ws = data_params['Window_Size']
    Tp = int(Ws * Fs)

    resolution, start_freq, end_freq = 0.2, 8, 64
    nfft = round(Fs / resolution)
    fft_index_start = int(round(start_freq / resolution))
    fft_index_end = int(round(end_freq / resolution))

    # --- Pre-calculate all FFT data to find global color scale ---
    all_fft_data = []
    for label_to_find in sorted_labels:
        try:
            # Find the first trial with the desired label
            trial_index = (dataset.label_data == label_to_find).nonzero(as_tuple=True)[0][0].item()
            raw_eeg_tensor, _ = dataset[trial_index] # This now returns raw data
            
            # Perform the same FFT as the collate_fn
            fft_result = torch.fft.fft(raw_eeg_tensor, n=nfft, dim=-1) / (Tp / 2)
            magnitudes = torch.abs(fft_result[..., fft_index_start:fft_index_end]).squeeze(0).numpy()
            # Freq axis is constant, so we don't need to store it per sample
            all_fft_data.append({'magnitudes': magnitudes, 'label': label_to_find})
        except IndexError:
            continue # Skip if label not found

    if not all_fft_data:
        print("No trials found for the specified labels.")
        return

    # Calculate global min/max from the pre-calculated numerical data. No figures are created here.
    global_min = min(data['magnitudes'].min() for data in all_fft_data)
    global_max = max(data['magnitudes'].max() for data in all_fft_data)

    # --- Create the grid plot ---
    nrows, ncols = 4, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 15))
    fig.suptitle(f'Feature Magnitude Maps - Subject {dataset.subject_list} - One Trial Per Label', fontsize=16)
    axes = axes.flatten()

    for i, data in enumerate(all_fft_data):
        ax = axes[i]
        label = data['label']
        stimulus_hz = stimulus_map.get(str(label), "Unknown")
        
        num_freq_bins = data['magnitudes'].shape[-1]
        freq_axis = np.linspace(8, 64, num=num_freq_bins, endpoint=False)

        # Plot the pre-calculated FFT data
        im = ax.imshow(data['magnitudes'], aspect='auto', cmap='viridis',
                       extent=[freq_axis[0], freq_axis[-1], dataset.Nc - 0.5, -0.5],
                       vmin=global_min, vmax=global_max)
        
        ax.set_yticks(ticks=np.arange(dataset.Nc), labels=channel_names)
        if stimulus_hz != "Unknown":
            ax.axvline(x=float(stimulus_hz), color='r', linestyle='--')

        ax.set_title(f'Label: {label} ({stimulus_hz} Hz)')

    # --- Final figure adjustments ---
    fig.subplots_adjust(bottom=0.15, hspace=0.4, wspace=0.3)
    
    # Common labels
    fig.text(0.5, 0.04, 'Frequency (Hz)', ha='center', va='center', fontsize=14)
    fig.text(0.06, 0.5, 'Channel', ha='center', va='center', rotation='vertical', fontsize=14)

    # Shared colorbar at the bottom
    cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.02])
    norm = plt.Normalize(vmin=global_min, vmax=global_max)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')

def visualize_eeg_data(dataset, dataset_metadata_path, mode='per_label', save_dir=None):
    """
    Visualizes transformed EEG features from a dataset.

    Args:
        dataset (Dataset): The EEG dataset object containing RAW time-series data.
        dataset_metadata_path (str): Path to the metadata JSON file.
        mode (str): Currently only 'per_label' is supported for feature visualization.
        save_dir (str, optional): Directory to save the figure. If None, shows the plot. Defaults to None.
    """
    # --- Load metadata from the provided path ---
    with open(dataset_metadata_path, 'r') as f:
        metadata = json.load(f)
    
    data_params = metadata['data_param']    
    channel_names = dataset.channel_names # Use the selected channel names from the dataset object
    stimulus_map = data_params['stimulus_hz']

    if mode == 'per_label':
        _visualize_per_label(dataset, channel_names, stimulus_map, data_params)
    else:
        raise ValueError("Invalid mode. Only 'per_label' is currently supported for feature visualization.")

    if save_dir:
        subject_str = "_".join(map(str, dataset.subject_list))
        save_path = os.path.join(save_dir, f'feature_map_S{subject_str}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature map saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_attention_maps(model, dataset, dataset_metadata_path, preprocess_fn, save_dir=None, subject_id=None, train_subject_id=None):
    """
    Visualizes the model's attention weights on the input channels for one trial of each label.

    Args:
        model (torch.nn.Module): The trained model, which must return attention weights.
        dataset (Dataset): The dataset to visualize, with transforms applied.
        dataset_metadata_path (str): Path to the metadata JSON file.
        preprocess_fn (callable): The collate function used to preprocess raw data.
        save_dir (str, optional): Directory to save the figure. If None, shows the plot.
        subject_id (int, optional): The validation subject ID.
        train_subject_id (int, optional): The training subject ID (for inter-subject mode).
    """
    # --- Load metadata ---
    with open(dataset_metadata_path, 'r') as f:
        metadata = json.load(f)
    
    data_params = metadata['data_param']
    # channel_names are not needed for token-based attention plots
    stimulus_map = data_params['stimulus_hz']

    # --- Prepare for plotting ---
    sorted_stimuli = sorted(stimulus_map.items(), key=lambda item: float(item[1]))
    sorted_labels = [int(item[0]) for item in sorted_stimuli]

    # --- Set model to evaluation mode ---
    device = next(model.parameters()).device
    model.eval()

    # --- Collect data for each label ---
    all_attention_weights = []
    with torch.no_grad():
        for label_to_find in sorted_labels:
            try:
                trial_index = (dataset.label_data == label_to_find).nonzero(as_tuple=True)[0][0].item()
                raw_eeg_tensor, label_tensor = dataset[trial_index] # Shape: (1, Nc, Tp)

                # --- Use the provided preprocess_fn on a single-item batch ---
                # The collate_fn expects a list of samples (a batch)
                feature_tensor, _ = preprocess_fn([(raw_eeg_tensor, label_tensor)])

                # Assumes model returns (output, attention_weights)
                feature_tensor = feature_tensor.to(device) # The collate_fn already creates a batch
                _ = model(feature_tensor) # Run forward pass to populate attention weights

                # Directly access the stored weights from the model's internal layer
                attention_weights = model.encoder.layers[0].attn.attention_weights
                
                # The raw attention weights have shape (num_tokens, feature_dim).
                # We average across the feature dimension to get a single "importance score" per token
                # for high-level visualization.
                all_attention_weights.append(attention_weights.mean(dim=-1).squeeze().cpu().numpy())
            except (IndexError, TypeError): # TypeError if model doesn't return tuple
                print(f"Could not get attention for label {label_to_find}. Skipping.")
                continue

    if not all_attention_weights:
        print("Could not generate any attention maps. Ensure the model returns attention weights and labels are present.")
        return

    # --- Average the weights across all collected trials ---
    avg_attention_weights = np.mean(all_attention_weights, axis=0)
    num_tokens = len(avg_attention_weights)
    token_labels = [f'Token {j}' for j in range(num_tokens)]

    # --- Create a single plot for the average attention ---
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.bar(np.arange(num_tokens), avg_attention_weights, color='skyblue')
    ax.set_xticks(np.arange(num_tokens))
    ax.set_xticklabels(token_labels, rotation=45, ha="right")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylabel("Average Attention Weight")
    ax.set_xlabel("Token")

    # --- Final figure adjustments ---
    if train_subject_id:
        title = f"Average Attention Map: Inter-Subject on Val S{subject_id}"
        filename = f"attention_map_inter_subject_val_S{subject_id}.png"
    else:
        title = f"Average Attention Map: Intra-Subject on Val S{subject_id}"
        filename = f"attention_map_intra_subject_val_S{subject_id}.png"
    
    ax.set_title(title, fontsize=16)
    fig.tight_layout()

    if save_dir:
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention map plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_2d_attention_heatmap(model, dataset, dataset_metadata_path, preprocess_fn, save_dir=None, subject_id=None, train_subject_id=None):
    """
    Visualizes the full 2D attention map for trials 0 and 11 in a 2x2 subplot.
    The top row shows the full frequency range, the bottom row is zoomed to 8-20 Hz.

    Args:
        model (torch.nn.Module): The trained model.
        dataset (Dataset): The dataset containing raw data.
        dataset_metadata_path (str): Path to the metadata JSON file.
        preprocess_fn (callable): The collate function to preprocess the raw data.
        save_dir (str, optional): Directory to save the figure. Defaults to None.
        subject_id (int, optional): The validation subject ID.
        train_subject_id (int, optional): The training subject ID (for inter-subject mode).
    """
    with open(dataset_metadata_path, 'r') as f:
        metadata = json.load(f)
    stimulus_map = metadata['data_param']['stimulus_hz']

    device = next(model.parameters()).device
    model.eval()

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    labels_to_find = [0, 11]

    for i, label_to_find in enumerate(labels_to_find):
        try:
            # Find the index of the first trial with the desired label
            trial_index = (dataset.label_data == label_to_find).nonzero(as_tuple=True)[0][0].item()

            with torch.no_grad():
                raw_eeg_tensor, label_tensor = dataset[trial_index]
                actual_label = label_tensor.item()
                actual_hz = stimulus_map.get(str(actual_label), "Unknown")

                feature_tensor, _ = preprocess_fn([(raw_eeg_tensor, label_tensor)])
                feature_tensor = feature_tensor.to(device)

                output = model(feature_tensor) # Run forward pass
                predicted_label = torch.argmax(output, dim=1).item()
                predicted_hz = stimulus_map.get(str(predicted_label), "Unknown")

                # Directly access the stored weights
                raw_attention_weights = model.encoder.layers[0].attn.attention_weights.squeeze(0).cpu().numpy()

            num_tokens, num_freq_bins = raw_attention_weights.shape
            freq_axis = np.linspace(8, 64, num=num_freq_bins, endpoint=False)
            token_labels = [f'Token {j}' for j in range(num_tokens)]

            # --- Plot Full Heatmap (Top Row) ---
            ax_full = axes[0, i]
            im = ax_full.imshow(raw_attention_weights, aspect='auto', cmap='hot', extent=[freq_axis[0], freq_axis[-1], num_tokens - 0.5, -0.5])
            ax_full.set_yticks(ticks=np.arange(num_tokens), labels=token_labels)
            ax_full.set_title(f'Trial with Label {label_to_find} (Full)\nActual: {actual_hz} Hz | Predicted: {predicted_hz} Hz')
            ax_full.set_ylabel("Tokens")

            # --- Plot Zoomed Heatmap (Bottom Row) ---
            ax_zoom = axes[1, i]
            ax_zoom.imshow(raw_attention_weights, aspect='auto', cmap='hot', extent=[freq_axis[0], freq_axis[-1], num_tokens - 0.5, -0.5])
            ax_zoom.set_yticks(ticks=np.arange(num_tokens), labels=token_labels)
            ax_zoom.set_title(f'Trial with Label {label_to_find} (Zoomed 8-20 Hz)')
            ax_zoom.set_xlim(8, 20) # Zoom in on the specified frequency range
            ax_zoom.set_xlabel("Frequency Bins (Hz)")
            ax_zoom.set_ylabel("Tokens")

        except IndexError:
            print(f"Warning: Could not find a trial for label {label_to_find}. Skipping this subplot.")
            axes[0, i].set_title(f'Label {label_to_find} not found')
            axes[1, i].set_title(f'Label {label_to_find} not found')
            axes[0, i].axis('off')
            axes[1, i].axis('off')
            continue

    # --- Final Figure Adjustments ---
    if train_subject_id:
        mode_str = "Inter-Subject"
        filename = f"attention_heatmap_2d_inter_subject_val_S{subject_id}.png"
    else:
        mode_str = "Intra-Subject"
        filename = f"attention_heatmap_2d_intra_subject_val_S{subject_id}.png"

    fig.suptitle(f'2D Attention Heatmaps: {mode_str} on Val S{subject_id}', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_dir:
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"2D Attention heatmap saved to {save_path}")
        plt.close()
    else:
        plt.show()