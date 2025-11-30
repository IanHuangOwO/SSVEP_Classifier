import numpy as np
import matplotlib.pyplot as plt
import os
import torch

def _get_plot_params(config, num_bins):
    """
    Determines plot parameters based on the preprocessing method.
    This avoids code repetition in plotting functions.
    """
    model_name = config['training_params']['model_name']
    model_params = config['model_params'][model_name]
    preprocess_method = model_params.get('preprocess_method', 'frequency')

    if 'freq' in preprocess_method or 'fft' in preprocess_method:
        start_freq, end_freq = model_params['start_freq'], model_params['end_freq']
        feature_axis = np.linspace(start_freq, end_freq, num=num_bins, endpoint=False)
        xlabel = 'Frequency (Hz)'
    elif preprocess_method == 'raw':
        Ws = config['data_metadata']['Window_Size']
        feature_axis = np.linspace(0, Ws, num=num_bins, endpoint=False)
        xlabel = 'Time (s)'
    else:
        raise ValueError(f"Unsupported preprocess_method for plotting: {preprocess_method}")

    bin_width = feature_axis[1] - feature_axis[0] if len(feature_axis) > 1 else 0
    bar_width = bin_width * 0.8
    
    return feature_axis, xlabel, bin_width, bar_width, preprocess_method

def _get_feature_data(dataset, sorted_labels, config):
    """
    Extracts and preprocesses feature data for visualization.
    """
    all_feature_data = []
    model_name = config['training_params']['model_name']
    model_params = config['model_params'][model_name]
    preprocess_method = model_params.get('preprocess_method', 'frequency')

    Fs = config['data_metadata']['Sample_Frequency']
    Ws = config['data_metadata']['Window_Size']
    Tp = int(Ws * Fs)
    
    if 'freq' in preprocess_method or 'fft' in preprocess_method:
        resolution = model_params['resolution']
        start_freq, end_freq = model_params['start_freq'], model_params['end_freq']
        nfft = round(Fs / resolution)
        fft_index_start = int(round(start_freq / resolution))
        fft_index_end = int(round(end_freq / resolution))

        for label_to_find in sorted_labels:
            trial_indices = (dataset.label_data == label_to_find).nonzero(as_tuple=True)[0]
            if len(trial_indices) == 0:
                print(f"Visualization Warning: No trials found for label {label_to_find}. Skipping this plot.")
                continue

            all_magnitudes = []
            for trial_index in trial_indices:
                raw_eeg_tensor, _ = dataset[trial_index.item()]
                
                fft_result = torch.fft.fft(raw_eeg_tensor, n=nfft, dim=-1) / (Tp / 2)
                fft_slice = fft_result[..., fft_index_start:fft_index_end]
                magnitudes = torch.abs(fft_slice).squeeze(0).numpy()
                all_magnitudes.append(magnitudes)

            if all_magnitudes:
                # Average the magnitudes across all trials for this label
                avg_magnitudes = np.mean(all_magnitudes, axis=0)
                all_feature_data.append({'features': avg_magnitudes, 'label': label_to_find})

    elif preprocess_method == 'raw':
        for label_to_find in sorted_labels:
            trial_indices = (dataset.label_data == label_to_find).nonzero(as_tuple=True)[0]
            if len(trial_indices) == 0:
                print(f"Warning: No trial found for label {label_to_find} in raw feature extraction.")
                continue

            all_raw_tensors = []
            for trial_index in trial_indices:
                raw_eeg_tensor, _ = dataset[trial_index.item()]
                all_raw_tensors.append(raw_eeg_tensor.squeeze(0).numpy())
            
            try:
                # Average the raw signals across all trials
                avg_raw_signal = np.mean(all_raw_tensors, axis=0)
                all_feature_data.append({'features': avg_raw_signal, 'label': label_to_find})
            except IndexError:
                continue
    
    return all_feature_data, preprocess_method

def _visualize_per_label(dataset, channel_names, stimulus_map, config):
    """Handles the logic for plotting one feature map per label."""
    # The definitive list of labels to plot should come directly from the dataset itself,
    # not from the stimulus_map, which might be from the wrong metadata file.
    # We sort them to ensure a consistent plotting order.
    sorted_labels = sorted(torch.unique(dataset.label_data).tolist())

    all_feature_data, preprocess_method = _get_feature_data(dataset, sorted_labels, config)

    if not all_feature_data:
        print("No trials found for the specified labels.")
        return

    # --- Calculate global min/max for a consistent color scale ---
    global_min = min(data['features'].min() for data in all_feature_data)
    global_max = max(data['features'].max() for data in all_feature_data)

    # --- Create the grid plot ---
    # Dynamically determine grid size to fit all labels
    num_labels = len(all_feature_data)
    ncols = 4
    nrows = int(np.ceil(num_labels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(40, nrows * 5), sharex=True, sharey=True)
    fig.suptitle(f'Feature Maps (Method: {preprocess_method.upper()}) - Subject {dataset.subject_list}', fontsize=16)
    axes = axes.flatten()

    for i, data in enumerate(all_feature_data):
        ax = axes[i]
        label = data['label']
        stimulus_map_key = str(label)
        stimulus_hz = stimulus_map.get(stimulus_map_key, "Unknown")
        
        features = data['features']
        num_bins = features.shape[-1]
        feature_axis, xlabel, bin_width, _, _ = _get_plot_params(config, num_bins)

        # Plot the pre-calculated feature data
        # Adjust the extent to center the bins on the ticks.
        # The extent should start half a bin-width before the first tick and end half a bin-width after the last tick.
        x_start = feature_axis[0] - bin_width / 2
        x_end = feature_axis[-1] + bin_width / 2
        extent = [x_start, x_end, dataset.Nc - 0.5, -0.5]
        ax.imshow(features, aspect='auto', cmap='viridis', extent=extent, vmin=global_min, vmax=global_max)
        
        # Add vertical lines for fundamental and harmonics
        if ('freq' in preprocess_method or 'fft' in preprocess_method) and stimulus_hz != "Unknown":
            fundamental_hz = float(stimulus_hz)
            for h in range(1, 4): # Plot fundamental (h=1) and two harmonics
                harmonic_hz = fundamental_hz * h
                if len(feature_axis) > 0 and feature_axis[0] <= harmonic_hz <= feature_axis[-1]:
                    linestyle = '--' if h == 1 else ':'
                    ax.axvline(x=harmonic_hz, color='r', linestyle=linestyle, linewidth=1.2)
        
        ax.set_yticks(ticks=np.arange(dataset.Nc), labels=channel_names)
        ax.set_title(f'Label: {label} ({stimulus_hz} Hz)')

    # --- Hide unused subplots ---
    for j in range(num_labels, len(axes)):
        axes[j].set_visible(False)


    # --- Final figure adjustments ---
    fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.15, hspace=0.4, wspace=0.1)
    
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
        preprocess_method = config['model_params'][model_name].get('preprocess_method', 'frequency')
        save_path = os.path.join(save_dir, f'feature_map_{preprocess_method}_S{subject_str}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature map saved to {save_path}")
        plt.close()
    else:
        plt.show()
        
def _generate_grad_cam(model, input_tensor, target_class):
    """
    Generates a Grad-CAM heatmap for a given model and input.
    """
    model.eval()
    
    # 1. Hook the target layer (output of patch_embedding)
    target_layer = model.patch_embedding
    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    # 2. Forward pass
    output = model(input_tensor)
    
    # 3. Backward pass
    model.zero_grad()
    score = output[0, target_class]
    score.backward()

    # 4. Clean up hooks
    forward_handle.remove()
    backward_handle.remove()

    # 5. Compute Grad-CAM
    # Pool gradients across the feature dimension (width)
    pooled_gradients = torch.mean(gradients, dim=[2])
    
    # Weight the activations by the pooled gradients
    for i in range(activations.shape[1]): # Iterate over tokens/channels
        activations[:, i, :] *= pooled_gradients[:, i]
        
    # The weighted activations are the 2D heatmap. We just need to apply ReLU.
    heatmap = activations.squeeze(0).detach()
    heatmap = np.maximum(heatmap.cpu().numpy(), 0)
    
    # Normalize to [0, 1]
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
        
    return heatmap

def plot_grad_cam_visuals(model, dataset, config, preprocess_fn, save_dir, subject_id):
    """
    Generates and saves Grad-CAM visualizations for each target class.
    """
    device = config['training_params'].get('device', 'cpu')
    model.to(device)

    stimulus_map = config['data_metadata']['stimulus_hz']
    # The definitive list of labels to plot should come directly from the dataset itself,
    # not from the stimulus_map, which might be from the wrong metadata file.
    # We sort them to ensure a consistent plotting order.
    sorted_labels = sorted(torch.unique(dataset.label_data).tolist())
    # Dynamically determine grid size
    num_labels = len(sorted_labels)
    ncols = 4  # Keep a fixed number of columns for consistent layout
    nrows = int(np.ceil(num_labels / ncols))

    fig = plt.figure(figsize=(40, 30), constrained_layout=True)
    fig.suptitle(f'Grad-CAM Class Activation Maps - Subject {subject_id}', fontsize=20)
    main_gs = fig.add_gridspec(nrows, ncols, wspace=0.5, hspace=0.6)

    for i, label_to_find in enumerate(sorted_labels):
        try:
            if i >= nrows * ncols:
                print(f"Warning: Skipping label {label_to_find} as it exceeds the grid size.")
                continue
            # Find all trials for the current label
            trial_indices = (dataset.label_data == label_to_find).nonzero(as_tuple=True)[0]
            if len(trial_indices) == 0:
                print(f"Warning: No trials found for label {label_to_find}. Skipping.")
                continue
            
            # --- Generate Grad-CAM for all trials and average them ---
            all_grad_cams = []
            all_features = []
            for trial_index in trial_indices:
                raw_eeg_tensor, label_tensor = dataset[trial_index.item()]
                feature_tensor, _ = preprocess_fn([(raw_eeg_tensor, label_tensor)])
                feature_tensor = feature_tensor.to(device)
                feature_tensor.requires_grad_(True)

                # Generate Grad-CAM for this trial
                current_grad_cam = _generate_grad_cam(model, feature_tensor, target_class=label_to_find)
                all_grad_cams.append(current_grad_cam)
                
                # Store the features for averaging later
                input_features_np = feature_tensor.squeeze(0).squeeze(0).cpu().detach().numpy()
                all_features.append(input_features_np)

            if not all_grad_cams:
                print(f"Warning: Could not generate any Grad-CAM maps for label {label_to_find}. Skipping.")
                continue

            # Average the Grad-CAM maps and the input features
            grad_cam_map = np.mean(all_grad_cams, axis=0)
            input_features_np = np.mean(all_features, axis=0)

            # We still need one feature tensor for metadata, can use the last one
            feature_tensor, _ = preprocess_fn([(raw_eeg_tensor, label_tensor)])

            # --- Reverse map from tokens to channels if necessary ---
            num_tokens = grad_cam_map.shape[0]
            num_channels = len(dataset.channel_names)
            if num_tokens != num_channels:
                try:
                    # Get the weights from the embedding layer's convolution
                    conv_weights = model.patch_embedding.net[0].weight.detach().cpu()
                    # Average across the kernel dimension to get the mapping matrix
                    w_map = conv_weights.mean(dim=-1).numpy() # Shape: (token_num, chs_num)
                    # Use einsum to perform the weighted sum: (tokens, features) * (tokens, channels) -> (channels, features)
                    grad_cam_map = np.einsum('ij,ik->kj', grad_cam_map, w_map)
                except Exception as e:
                    print(f"Could not perform token-to-channel mapping: {e}. Visualization may be incorrect.")

            # --- Handle real+imag FFT data for visualization ---
            model_name = config['training_params']['model_name']
            preprocess_method = config['model_params'][model_name].get('preprocess_method', 'frequency')
            if preprocess_method == 'fft':
                # Convert features back to magnitude for plotting
                num_freq_bins = input_features_np.shape[1] // 2
                input_real, input_imag = input_features_np[:, :num_freq_bins], input_features_np[:, num_freq_bins:]
                input_features_np = np.sqrt(input_real**2 + input_imag**2)

                # Convert Grad-CAM map back to magnitude for plotting
                grad_cam_real, grad_cam_imag = grad_cam_map[:, :num_freq_bins], grad_cam_map[:, num_freq_bins:]
                grad_cam_map = np.sqrt(grad_cam_real**2 + grad_cam_imag**2)
            
            # --- Plotting Logic ---
            nested_gs = main_gs[i].subgridspec(2, 2, width_ratios=(8, 0.5), height_ratios=(1, 4), wspace=0.02, hspace=0.05)
            ax_heatmap = fig.add_subplot(nested_gs[1, 0])
            ax_bar_x = fig.add_subplot(nested_gs[0, 0], sharex=ax_heatmap)
            ax_bar_y = fig.add_subplot(nested_gs[1, 1], sharey=ax_heatmap)
            
            num_bins = input_features_np.shape[-1]
            feature_axis, xlabel, _, bar_width, preprocess_method = _get_plot_params(config, num_bins)
            heatmap_extent = [feature_axis[0], feature_axis[-1], dataset.Nc - 0.5, -0.5]

            ax_heatmap.imshow(input_features_np, aspect='auto', cmap='gray', extent=heatmap_extent)
            ax_heatmap.imshow(grad_cam_map, aspect='auto', cmap='coolwarm', extent=heatmap_extent, alpha=0.3)

            # --- Bar Plot Coloring ---
            actual_hz = stimulus_map.get(str(label_to_find), "Unknown")
            bar_colors = ['skyblue'] * len(feature_axis) # Default color
            if ('freq' in preprocess_method or 'fft' in preprocess_method) and actual_hz != "Unknown":
                fundamental_hz = float(actual_hz)
                # Loop for fundamental (h=1) and its harmonics
                for h in range(1, 4): # e.g., h=1, 2, 3 for fundamental and two harmonics
                    harmonic_hz = fundamental_hz * h
                    # Check if the harmonic is within the plot's frequency range
                    if feature_axis[0] <= harmonic_hz <= feature_axis[-1]:
                        # Find the index of the bar closest to the harmonic frequency
                        target_bin_index = np.argmin(np.abs(feature_axis - harmonic_hz))
                        bar_colors[target_bin_index] = 'red'

            ax_heatmap.set_yticks(ticks=np.arange(dataset.Nc), labels=dataset.channel_names)
            ax_heatmap.set_xlabel(xlabel)

            # Bar plots
            ax_bar_x.bar(feature_axis, grad_cam_map.mean(axis=0), color=bar_colors, width=bar_width)
            ax_bar_x.bar(feature_axis, grad_cam_map.mean(axis=0), color=bar_colors, width=bar_width, align='center')
            ax_bar_x.set_ylabel("Avg Importance")
            ax_bar_x.set_title(f'Target: {label_to_find} ({actual_hz} Hz)')
            ax_bar_x.tick_params(axis="x", labelbottom=False)

            ax_bar_y.barh(np.arange(dataset.Nc), grad_cam_map.mean(axis=1), color='salmon', height=0.8)
            ax_bar_y.barh(np.arange(dataset.Nc), grad_cam_map.mean(axis=1), color='salmon', height=0.8, align='center')
            ax_bar_y.tick_params(axis="y", labelleft=False)
            ax_bar_y.set_xlabel("Avg Importance")

        except Exception as e:
            print(f"Could not generate Grad-CAM for label {label_to_find}. Error: {e}")
            if i >= nrows * ncols:
                print(f"Error occurred for label {label_to_find}, which is outside the grid. Cannot plot error message.")
                continue
            ax = fig.add_subplot(main_gs[i])
            ax.text(0.5, 0.5, f'Error for Label {label_to_find}', ha='center', va='center')
            continue

    if save_dir:
        save_path = os.path.join(save_dir, f'grad_cam_grid_S{subject_id}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grad-CAM grid saved to {save_path}")
        plt.close(fig)