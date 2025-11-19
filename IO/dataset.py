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
   def __init__(self, data_root, data_structure, all_channel_names, desired_channels, Nf, Nt, Ws, Fs, trials_to_use, subject_list=[1]):
       super(SSVEPDataset, self).__init__()
       
       # File paths and subject info
       self.data_root = data_root
       self.data_structure = data_structure
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
       all_channels_map = {name: int(i) for i, name in all_channel_names.items()}
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

   def _load_subject_data(self):
      """Loads and concatenates signal and label data for all subjects in the subject_list."""
      all_eeg_data = []
      all_label_data = []

      for subject_id in self.subject_list:
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
          labels = torch.from_numpy(label_mat['Label'])

          # --- Select a subset of trials per frequency block ---
          indices_to_keep = []
          for i in range(self.Nf): # For each frequency
              block_start_index = i * self.total_trials_per_freq
              block_end_index = block_start_index + self.trials_to_use
              indices_to_keep.extend(range(block_start_index, block_end_index))
          
          # Select only the desired channels and trials
          eeg_data = samples[self.channel_indices, :, :][:, :, indices_to_keep].swapaxes(1, 2)
          eeg_data = torch.from_numpy(eeg_data.swapaxes(0, 1)).float()
          all_eeg_data.append(eeg_data)
          all_label_data.append(labels[indices_to_keep])

      # Concatenate data from all subjects
      final_eeg_data = torch.cat(all_eeg_data, dim=0).reshape(-1, 1, self.Nc, self.Tp)
      final_label_data = torch.cat(all_label_data, dim=0)

      return final_eeg_data, final_label_data - 1


def build_dataset_from_config(main_config_path: str, dataset_metadata_path: str, **kwargs):
    """
    Factory function to build the SSVEPDataset.
    It reads JSON config files, extracts dataset parameters, and constructs the dataset object.
    """
    with open(main_config_path, 'r') as f:
        main_config = json.load(f)
    with open(dataset_metadata_path, 'r') as f:
        metadata_config = json.load(f)
    
    # data_root is the parent directory of the directory containing the metadata.json file.
    data_root = os.path.dirname(os.path.dirname(dataset_metadata_path))
    
    # Get parameters from the appropriate config files
    config_params = main_config['data_params']
    data_params = metadata_config['data_param']
    subject_list = kwargs.get('subject_list', [1])

    return SSVEPDataset(
        data_root=data_root,
        data_structure=metadata_config['data_structure'],
        all_channel_names=data_params['channel_names'],
        desired_channels=config_params['channels'],
        trials_to_use=config_params['trials_to_use'],
        Nf=data_params['Number_of_Targets'],
        Nt=data_params['Number_of_Trials'],
        Ws=data_params['Window_Size'],
        Fs=data_params['Sample_Frequency'],
        subject_list=subject_list
    )