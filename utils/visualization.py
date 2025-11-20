import numpy as np
import matplotlib.pyplot as plt
import os
import json
import torch

def _visualize_per_label(dataset, channel_names, stimulus_map, config):
    """Handles the logic for plotting one feature map per label."""
    sorted_stimuli = sorted(stimulus_map.items(), key=lambda item: float(item[1]))
    sorted_labels = [int(item[0]) for item in sorted_stimuli]

    # --- Get FFT parameters from data_params ---
    model_name = config['training_params']['model_name']
    model_params = config['model_params'][model_name]
    data_metadata = config['data_metadata']
    Fs = data_metadata['Sample_Frequency']
    Ws = data_metadata['Window_Size']
    Tp = int(Ws * Fs)
    
    preprocess_method = model_params.get('preprocess_method', 'fft')

    # --- Pre-calculate all feature data to find global color scale ---
    all_feature_data = []
    if preprocess_method == 'fft':
        resolution = model_params['resolution']
        start_freq, end_freq = model_params['start_freq'], model_params['end_freq']
        nfft = round(Fs / resolution)
        fft_index_start = int(round(start_freq / resolution))
        fft_index_end = int(round(end_freq / resolution))

        for label_to_find in sorted_labels:
            try:
                trial_index = (dataset.label_data == label_to_find).nonzero(as_tuple=True)[0][0].item()
                raw_eeg_tensor, _ = dataset[trial_index]
                
                fft_result = torch.fft.fft(raw_eeg_tensor, n=nfft, dim=-1) / (Tp / 2)
                magnitudes = torch.abs(fft_result[..., fft_index_start:fft_index_end]).squeeze(0).numpy()
                all_feature_data.append({'features': magnitudes, 'label': label_to_find})
            except IndexError:
                continue
    elif preprocess_method == 'raw':
        for label_to_find in sorted_labels:
            try:
                trial_index = (dataset.label_data == label_to_find).nonzero(as_tuple=True)[0][0].item()
                raw_eeg_tensor, _ = dataset[trial_index]
                all_feature_data.append({'features': raw_eeg_tensor.squeeze(0).numpy(), 'label': label_to_find})
            except IndexError:
                continue

    if not all_feature_data:
        print("No trials found for the specified labels.")
        return

    # Calculate global min/max from the pre-calculated numerical data.
    global_min = min(data['features'].min() for data in all_feature_data)
    global_max = max(data['features'].max() for data in all_feature_data)

    # --- Create the grid plot ---
    nrows, ncols = 4, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 15))
    fig.suptitle(f'Feature Maps (Method: {preprocess_method.upper()}) - Subject {dataset.subject_list} - One Trial Per Label', fontsize=16)
    axes = axes.flatten()

    for i, data in enumerate(all_feature_data):
        ax = axes[i]
        label = data['label']
        stimulus_hz = stimulus_map.get(str(label), "Unknown")
        
        if preprocess_method == 'fft':
            num_bins = data['features'].shape[-1]
            start_freq, end_freq = model_params['start_freq'], model_params['end_freq']
            axis = np.linspace(start_freq, end_freq, num=num_bins, endpoint=False)
            xlabel = 'Frequency (Hz)'
            if stimulus_hz != "Unknown":
                ax.axvline(x=float(stimulus_hz), color='r', linestyle='--')
        elif preprocess_method == 'raw':
            num_bins = data['features'].shape[-1]
            axis = np.linspace(0, Ws, num=num_bins, endpoint=False)
            xlabel = 'Time (s)'

        # Plot the pre-calculated feature data
        im = ax.imshow(data['features'], aspect='auto', cmap='viridis',
                       extent=[axis[0], axis[-1], dataset.Nc - 0.5, -0.5],
                       vmin=global_min, vmax=global_max)
        
        ax.set_yticks(ticks=np.arange(dataset.Nc), labels=channel_names)
        ax.set_title(f'Label: {label} ({stimulus_hz} Hz)')
        if i >= (nrows - 1) * ncols: # Only show x-label on bottom row
            ax.set_xlabel(xlabel)

    # --- Final figure adjustments ---
    fig.subplots_adjust(bottom=0.15, hspace=0.4, wspace=0.3)
    
    # Common labels
    fig.text(0.5, 0.04, xlabel, ha='center', va='center', fontsize=14)
    fig.text(0.06, 0.5, 'Channel', ha='center', va='center', rotation='vertical', fontsize=14)

    # Shared colorbar at the bottom
    cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.02])
    norm = plt.Normalize(vmin=global_min, vmax=global_max)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')

def visualize_eeg_data(dataset, config: dict, mode='per_label', save_dir=None):
    """
    Visualizes transformed EEG features from a dataset.

    Args:
        dataset (Dataset): The EEG dataset object containing RAW time-series data.
        config (dict): The unified configuration dictionary.
        mode (str): Currently only 'per_label' is supported for feature visualization.
        save_dir (str, optional): Directory to save the figure. If None, shows the plot. Defaults to None.
    """
    data_metadata = config['data_metadata']
    channel_names = dataset.channel_names # Use the selected channel names from the dataset object
    stimulus_map = data_metadata['stimulus_hz']

    if mode == 'per_label':
        _visualize_per_label(dataset, channel_names, stimulus_map, config)
    else:
        raise ValueError("Invalid mode. Only 'per_label' is currently supported for feature visualization.")

    if save_dir:
        subject_str = "_".join(map(str, dataset.subject_list))
        model_name = config['training_params']['model_name']
        preprocess_method = config['model_params'][model_name].get('preprocess_method', 'fft')
        save_path = os.path.join(save_dir, f'feature_map_{preprocess_method}_S{subject_str}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature map saved to {save_path}")
        plt.close()
    else:
        plt.show()
        
def plot_attention_visuals(model, dataset, config: dict, preprocess_fn, save_dir=None, subject_id=None, train_subject_id=None):
    """Generates comprehensive attention visualizations: a grid of individual 2D attention
    maps for each label, and a single plot of the average 'general strategy' attention."""
    stimulus_map = config['data_metadata']['stimulus_hz']
    model_name = config['training_params']['model_name']
    model_params = config['model_params'][model_name]
    preprocess_method = model_params.get('preprocess_method', 'fft')
    
    device = next(model.parameters()).device
    model.eval()

    # --- 1. Collect attention maps for all labels ---
    all_attention_maps = []
    sorted_stimuli = sorted(stimulus_map.items(), key=lambda item: float(item[1]))
    sorted_labels = [int(item[0]) for item in sorted_stimuli]

    with torch.no_grad():
        for label_to_find in sorted_labels:
            # Find ALL trial indices for the current label
            trial_indices = (dataset.label_data == label_to_find).nonzero(as_tuple=True)[0]
            
            if len(trial_indices) == 0:
                print(f"Warning: No trials found for label {label_to_find}. Skipping.")
                continue

            label_specific_maps = []
            for trial_index in trial_indices:
                try:
                    raw_eeg_tensor, label_tensor = dataset[trial_index.item()]
                    
                    feature_tensor, _ = preprocess_fn([(raw_eeg_tensor, label_tensor)])
                    feature_tensor = feature_tensor.to(device)
                    
                    _ = model(feature_tensor) # Forward pass to populate weights
                    
                    raw_attention_weights = model.encoder.layers[0].attn.attention_weights.squeeze(0).cpu().numpy()

                    if preprocess_method == 'fft':
                        # The attention is on the concatenated [real, imag] parts.
                        # We need to combine them to get attention per frequency bin.
                        num_freq_bins = raw_attention_weights.shape[1] // 2
                        attn_real = raw_attention_weights[:, :num_freq_bins]
                        attn_imag = raw_attention_weights[:, num_freq_bins:]
                        # Calculate the magnitude of attention for each frequency bin
                        attention_magnitude = np.sqrt(attn_real**2 + attn_imag**2)
                        label_specific_maps.append(attention_magnitude)
                    else: # 'raw' method
                        label_specific_maps.append(raw_attention_weights)
                except (IndexError, AttributeError):
                    # This handles cases where a specific trial fails or attention weights aren't available
                    continue
                
            if label_specific_maps:
                # Average the attention maps for this label and append
                avg_label_map = np.mean(label_specific_maps, axis=0)
                all_attention_maps.append({'label': label_to_find, 'weights': avg_label_map})
    
    if not all_attention_maps:
        print("Could not generate any 2D attention maps.")
        return

    # --- 2. Plot the grid of individual attention maps with marginals ---
    nrows, ncols = 4, 3
    nrows, ncols = 3, 4
    fig_grid = plt.figure(figsize=(30, 22), constrained_layout=True)
    fig_grid.suptitle(f'Individual 2D Attention Maps with Marginals ({preprocess_method.upper()}) - Val S{subject_id}', fontsize=20)
    main_gs = fig_grid.add_gridspec(nrows, ncols, wspace=0.5, hspace=0.6)

    for i, data in enumerate(all_attention_maps):
        weights = data['weights']
        label = data['label']
        actual_hz = stimulus_map.get(str(label), "Unknown")

        # Create a nested GridSpec for each subplot
        nested_gs = main_gs[i].subgridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.05, hspace=0.05)
        ax_heatmap = fig_grid.add_subplot(nested_gs[1, 0])
        ax_bar_x = fig_grid.add_subplot(nested_gs[0, 0], sharex=ax_heatmap)
        ax_bar_y = fig_grid.add_subplot(nested_gs[1, 1], sharey=ax_heatmap)

        # --- Calculate marginals for this specific map ---
        feature_attention = weights.mean(axis=0)
        token_attention = weights.mean(axis=1)

        num_tokens, num_bins = weights.shape
        token_labels = [f'T{j}' for j in range(num_tokens)]

        if preprocess_method == 'fft':
            start_freq, end_freq = model_params['start_freq'], model_params['end_freq']
            feature_axis = np.linspace(start_freq, end_freq, num=num_bins, endpoint=False)
            ax_heatmap.set_xlabel("Frequency (Hz)")
        else: # 'raw'
            Ws = config['data_metadata']['Window_Size']
            feature_axis = np.linspace(0, Ws, num=num_bins, endpoint=False)
            ax_heatmap.set_xlabel("Time (s)")

        # --- Plot the components ---
        # Heatmap
        ax_heatmap.imshow(weights, aspect='auto', cmap='coolwarm', extent=[feature_axis[0], feature_axis[-1], num_tokens - 0.5, -0.5])
        ax_heatmap.set_yticks(ticks=np.arange(num_tokens), labels=token_labels)
        ax_heatmap.set_ylabel("Tokens")

        # Top bar graph
        ax_bar_x.bar(feature_axis, feature_attention, width=(feature_axis[1]-feature_axis[0])*0.8, color='skyblue')
        ax_bar_x.tick_params(axis="x", labelbottom=False)
        ax_bar_x.set_ylabel("Avg Attention")
        ax_bar_x.set_title(f'Label: {label} ({actual_hz} Hz)')

        # Right bar graph
        ax_bar_y.barh(np.arange(num_tokens), token_attention, height=0.8, color='salmon')
        ax_bar_y.tick_params(axis="y", labelleft=False)
        ax_bar_y.set_xlabel("Avg Attention")

    if save_dir:
        mode_str = "inter" if train_subject_id else "intra"
        filename = f"attention_grid_2d_{preprocess_method}_{mode_str}_S{subject_id}.png"
        save_path = os.path.join(save_dir, filename)
        fig_grid.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"2D Attention grid saved to {save_path}")
        plt.close(fig_grid)
    else:
        plt.show()

    # --- 3. Calculate and plot the average "general strategy" attention map ---
    average_weights = np.mean([d['weights'] for d in all_attention_maps], axis=0)
    
    # Create the marginal plot for the average
    try:
        num_tokens, num_bins = average_weights.shape
        token_labels = [f'Token {j}' for j in range(num_tokens)]

        feature_attention = average_weights.mean(axis=0)
        token_attention = average_weights.mean(axis=1)

        if preprocess_method == 'fft':
            start_freq, end_freq = model_params['start_freq'], model_params['end_freq']
            feature_axis = np.linspace(start_freq, end_freq, num=num_bins, endpoint=False)
            xlabel = "Frequency Bins (Hz)"
        else: # 'raw'
            Ws = config['data_metadata']['Window_Size']
            feature_axis = np.linspace(0, Ws, num=num_bins, endpoint=False)
            xlabel = "Time (s)"

        fig_avg = plt.figure(figsize=(12, 12))
        gs = fig_avg.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                                left=0.1, right=0.9, bottom=0.1, top=0.9,
                                wspace=0.05, hspace=0.05)
        
        ax_heatmap = fig_avg.add_subplot(gs[1, 0])
        ax_bar_x = fig_avg.add_subplot(gs[0, 0], sharex=ax_heatmap)
        ax_bar_y = fig_avg.add_subplot(gs[1, 1], sharey=ax_heatmap)

        ax_heatmap.imshow(average_weights, aspect='auto', cmap='coolwarm', extent=[feature_axis[0], feature_axis[-1], num_tokens - 0.5, -0.5])
        ax_heatmap.set_yticks(ticks=np.arange(num_tokens), labels=token_labels)
        ax_heatmap.set_xlabel(xlabel)
        ax_heatmap.set_ylabel("Tokens")

        ax_bar_x.bar(feature_axis, feature_attention, width=(feature_axis[1]-feature_axis[0])*0.8, color='skyblue')
        ax_bar_x.tick_params(axis="x", labelbottom=False)
        ax_bar_x.set_ylabel("Avg Attention")

        ax_bar_y.barh(np.arange(num_tokens), token_attention, height=0.8, color='salmon')
        ax_bar_y.tick_params(axis="y", labelleft=False)
        ax_bar_y.set_xlabel("Avg Attention")

        mode_str = "Inter-Subject" if train_subject_id else "Intra-Subject"
        fig_avg.suptitle(f'Average 2D Attention (General Strategy, {preprocess_method.upper()})\n{mode_str} on Val S{subject_id}', fontsize=16)

        if save_dir:
            filename = f"attention_average_2d_{preprocess_method}_{mode_str}_S{subject_id}.png"
            save_path = os.path.join(save_dir, filename)
            fig_avg.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Average 2D Attention map saved to {save_path}")
            plt.close(fig_avg)
        else:
            plt.show()

    except Exception as e:
        print(f"Could not plot average attention map. Error: {e}")
        if 'fig_avg' in locals():
            plt.close(fig_avg)