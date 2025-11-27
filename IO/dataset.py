import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io
import json
import os # Keep os for path joining in build_dataset_from_config


class SSVEPDataset(Dataset):
   """
   A PyTorch Dataset class for SSVEP data, configured via a metadata JSON file.
   """
   def __init__(self, data_root, data_structure, all_channel_names, desired_channels, Nf, Nt, Ws, Fs, trials_to_use, loading_style, subject_list=[1]):
       super(SSVEPDataset, self).__init__()
       
       # File paths and subject info
       self.data_root = data_root
       self.data_structure = data_structure
       self.loading_style = loading_style
       self.subject_list = subject_list
       self.channel_names = desired_channels

       # --- Dataset Parameters ---
       self.Nc = len(desired_channels) # number of channels being used
       self.Nf = Nf                 # number of target frequencies
       self.total_trials_per_freq = Nt # Total available trials per frequency in the raw file
       self.trials_to_use = trials_to_use # Number of trials to actually use
       self.Ws = Ws                 # Window Size in seconds
       self.Fs = Fs                 # Sample Frequency
       self.Nh = self.Nf * self.total_trials_per_freq  # Total trials in the raw file per subject
       self.Tp = int(Ws * Fs)       # number of time points

       # --- Map desired channel names to indices in the raw data ---
       # Create a reverse mapping from name to index for all available channels
       # The channel indices in metadata are 1-based, so we subtract 1 for 0-based Python indexing.
       all_channels_map = {name: int(i) - 1 for i, name in all_channel_names.items()}
       # Get the indices for the channels we want to use
       self.channel_indices = [all_channels_map[name] for name in desired_channels if name in all_channels_map]

       # Load all data for the subject
       self.eeg_data, self.label_data = self._load_subject_data()

   def __getitem__(self, index):
       eeg_sample = self.eeg_data[index]
       label_sample = self.label_data[index]

       return eeg_sample, label_sample

   def __len__(self):
       return len(self.label_data)

   def _load_from_separate_files(self, subject_id):
       """Loads data for a single subject from separate signal and label files (e.g., Dial dataset)."""
       subject_str = str(subject_id)
       # Construct full paths to data files for the current subject
       relative_signal_path = self.data_structure[subject_str]['signals'].lstrip('./')
       signal_path = os.path.join(self.data_root, relative_signal_path)
       relative_label_path = self.data_structure[subject_str]['labels'].lstrip('./')
       label_path = os.path.join(self.data_root, relative_label_path)

       # --- Load Signals ---
       signal_mat = scipy.io.loadmat(signal_path)
       samples = signal_mat['Data']

       # --- Load Labels ---
       label_mat = scipy.io.loadmat(label_path)
       # The raw labels from the .mat file are 1-indexed (1-12).
       raw_labels = torch.from_numpy(label_mat['Label']).flatten()

       # --- Select a subset of trials per frequency block ---
       # The labels in the file are 1-based, so we iterate from 1 to Nf.
       indices_to_keep = []
       for label_val in range(1, self.Nf + 1): # Iterate from 1 to 12
           # Find the indices of all trials that match the current 1-based label
           matching_indices = (raw_labels == label_val).nonzero(as_tuple=True)[0]
           # Keep only the number of trials specified by 'trials_to_use'
           block_start_index = matching_indices[0]
           block_end_index = block_start_index + self.trials_to_use
           indices_to_keep.extend(range(block_start_index, block_end_index))
       
       # Select only the desired channels and trials
       eeg_data = samples[self.channel_indices, :, :][:, :, indices_to_keep].swapaxes(1, 2)
       eeg_data = eeg_data.swapaxes(0, 1) # Shape: (trials, channels, time_points)

       # --- Pad or Crop time points to match self.Tp ---
       actual_time_points = eeg_data.shape[-1]
       if actual_time_points < self.Tp:
           pad_width = self.Tp - actual_time_points
           padding = ((0, 0), (0, 0), (0, pad_width))
           eeg_data = np.pad(eeg_data, padding, mode='constant', constant_values=0)
       elif actual_time_points > self.Tp:
           eeg_data = eeg_data[:, :, :self.Tp]
       
       # Select the corresponding labels and convert them to 0-indexed.
       final_labels = raw_labels[indices_to_keep] - 1
       
       return torch.from_numpy(eeg_data).float(), final_labels

   def _load_from_single_file(self, subject_id):
       """Loads data for a single subject from a single .mat file (e.g., BETA dataset)."""
       subject_str = str(subject_id)
       relative_path = self.data_structure[subject_str]['file'].lstrip('./')
       file_path = os.path.join(self.data_root, relative_path)

       # --- Check if file exists before trying to load ---
       if not os.path.exists(file_path):
           print(f"Warning: Data file not found for subject {subject_id} at {file_path}. Skipping.")
           return None, None

       # --- Load Data ---
       mat_data = scipy.io.loadmat(file_path)
       # The .mat file contains a struct. The actual data is in data['EEG'][0,0].
       # This is consistent with the MATLAB script using 'data.EEG'.
       if 'data' in mat_data and mat_data['data'].dtype.names and 'EEG' in mat_data['data'].dtype.names:
           raw_data = mat_data['data']['EEG'][0, 0]
       else:
           raise ValueError(f"Could not find the expected struct format data['EEG'] in file: {file_path}")

       # --- Validate data shape ---
       if raw_data.ndim != 4:
           print(f"Warning: Data for subject {subject_id} has incorrect dimensions (expected 4D, got {raw_data.ndim}D). Skipping.")
           return None, None

       # --- Dynamically get data parameters ---
       actual_time_points = raw_data.shape[1]
       actual_num_blocks = raw_data.shape[2]
       
       # 1. Select desired channels (indices are 0-based in numpy)
       eeg_data = raw_data[self.channel_indices, :, :, :]

       # 2. Select desired number of trials (blocks) from the 3rd dimension
       # Ensure we don't try to select more blocks than available
       if self.trials_to_use > actual_num_blocks:
           raise ValueError(f"Configuration error for subject {subject_id}: "
                            f"Requested to use {self.trials_to_use} trials, but the data file only contains {actual_num_blocks} trials (blocks). "
                            f"Please adjust 'trials_to_use' in your config file or check the data.")
       trials_to_load = self.trials_to_use
       eeg_data = eeg_data[:, :, :trials_to_load, :]

       # 3. Reshape the data.
       # The actual data shape is (channels, time_points, blocks, targets).
       # We permute it to (targets, blocks, channels, time_points).
       eeg_data = np.transpose(eeg_data, (3, 2, 0, 1))

       # Then, reshape to (total_trials, channels, time_points).
       eeg_data = eeg_data.reshape(-1, self.Nc, actual_time_points)

       # --- Pad or Crop time points to match self.Tp ---
       if actual_time_points < self.Tp:
           pad_width = self.Tp - actual_time_points
           padding = ((0, 0), (0, 0), (0, pad_width))
           eeg_data = np.pad(eeg_data, padding, mode='constant', constant_values=0)
        #    print(f"  [S{subject_id}] Padded trials from {actual_time_points} to {self.Tp} time points.")
       elif actual_time_points > self.Tp:
           eeg_data = eeg_data[:, :, :self.Tp]
        #    print(f"  [S{subject_id}] Cropped trials from {actual_time_points} to {self.Tp} time points.")

       # --- Create Labels ---
       # For each target (0 to Nf-1), create 'trials_to_load' number of labels
       labels = np.array([i for i in range(self.Nf) for _ in range(trials_to_load)])
       
       return torch.from_numpy(eeg_data).float(), torch.from_numpy(labels).long()

   def _load_subject_data(self):
      """Loads and concatenates signal and label data for all subjects in the subject_list."""
      all_eeg_data = []
      all_label_data = []

      # Determine which loading function to use
      if self.loading_style == "separate_files":
          load_func = self._load_from_separate_files
      elif self.loading_style == "single_file":
          load_func = self._load_from_single_file
      else:
          raise ValueError(f"Unknown loading_style: {self.loading_style}")

      for subject_id in self.subject_list:
           eeg_data, labels = load_func(subject_id)
           # Only append data if the loading was successful
           if eeg_data is not None and labels is not None:
               all_eeg_data.append(eeg_data)
               all_label_data.append(labels)
      
      if not all_eeg_data:
          # If no subjects were loaded, return empty tensors
          return torch.empty(0), torch.empty(0)

      # Concatenate data from all subjects
      final_eeg_data = torch.cat(all_eeg_data, dim=0).reshape(-1, 1, self.Nc, self.Tp)
      final_label_data = torch.cat(all_label_data, dim=0)

      # Both loaders now return 0-indexed labels, so no further adjustment is needed.
      return final_eeg_data, final_label_data


def build_dataset_from_config(config: dict, **kwargs):
    """
    Factory function to build the SSVEPDataset.
    It uses a unified config object to construct the dataset.
    """
    data_root = config['training_params']['dataset_path']
    
    # Get parameters from the appropriate config files
    dataset_params = config['dataset_params']
    data_metadata = config['data_metadata']
    training_params = config['training_params']

    # --- Determine Subject List (Priority: kwargs > config file) ---
    if 'subject_list' in kwargs:
        # If a specific list is passed to the function, use it.
        subject_list = kwargs['subject_list']
    else:
        # Otherwise, use the list from the config file, expanding "all" if present.
        if training_params.get('subjects') == ["all"]:
            num_subjects = data_metadata['Number_of_Subjects']
            subject_list = list(range(1, num_subjects + 1))
        else:
            subject_list = training_params.get('subjects', [1])

    # --- Determine Channel List ---
    # Check if "all" is specified for channels
    if dataset_params.get('channels') == ["all"]:
        desired_channels = list(data_metadata['channel_names'].values())
    else:
        desired_channels = dataset_params['channels']

    return SSVEPDataset(
        data_root=data_root,
        data_structure=config['data_structure'],
        all_channel_names=data_metadata['channel_names'],
        desired_channels=desired_channels,
        trials_to_use=dataset_params['trials_to_use'],
        Nf=data_metadata['Number_of_Targets'],
        Nt=data_metadata['Number_of_Trials'],
        Ws=data_metadata['Window_Size'],
        Fs=data_metadata['Sample_Frequency'],
        loading_style=data_metadata['loading_style'],
        subject_list=subject_list
    )