import json
import torch.nn as nn
from torch import Tensor
import torch

def preprocess_collate_fn(config: dict):
    """
    Creates a collate_fn for a DataLoader that preprocesses raw EEG data.
    This function uses a unified config object for FFT parameters.
    """
    
    model_name = config['training_params']['model_name']
    model_params = config['model_params'][model_name]
    data_metadata = config['data_metadata']

    Fs = data_metadata['Sample_Frequency']
    Ws = data_metadata['Window_Size']
    Tp = int(Ws * Fs)

    preprocess_method = model_params.get('preprocess_method', 'frequency') # Default to 'frequency'

    if 'freq' in preprocess_method or 'fft' in preprocess_method:
        # FFT parameters
        resolution = model_params['resolution']
        start_freq, end_freq = model_params['start_freq'], model_params['end_freq']
        nfft = round(Fs / resolution)
        fft_index_start = int(round(start_freq / resolution))
        fft_index_end = int(round(end_freq / resolution))

        def frequency_preprocess_collate_fn(batch):
            """
            Takes a batch of (raw_eeg, label), applies FFT to raw_eeg, and returns magnitude tensors.
            """
            eeg_data = torch.stack([item[0] for item in batch]) # Shape: (B, 1, Nc, Tp)
            labels = torch.stack([item[1] for item in batch])   # Shape: (B, 1)

            fft_result = torch.fft.fft(eeg_data, n=nfft, dim=-1) / (Tp / 2)
            fft_slice = fft_result[..., fft_index_start:fft_index_end]
            
            if preprocess_method == 'frequency': # Magnitude only
                features = torch.abs(fft_slice).float()
            elif preprocess_method == 'fft': # Real and Imaginary
                real_part = torch.real(fft_slice)
                imag_part = torch.imag(fft_slice)
                features = torch.cat([real_part, imag_part], dim=-1).float()
            else:
                raise ValueError(f"Unsupported FFT-based preprocess_method: '{preprocess_method}'")
            return features, labels
        return frequency_preprocess_collate_fn

    elif preprocess_method == 'raw':
        def raw_preprocess_collate_fn(batch):
            """
            Takes a batch of (raw_eeg, label) and returns them as stacked tensors without FFT.
            """
            eeg_data = torch.stack([item[0] for item in batch]).float()
            labels = torch.stack([item[1] for item in batch])
            return eeg_data, labels
        return raw_preprocess_collate_fn
    
    else:
        raise ValueError(f"Unsupported preprocess_method: '{preprocess_method}'")

class PatchEmbedding(nn.Module):
    """
    Expands EEG channels into a sequence of tokens using a 1x1 convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, feature_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

class FeedForward(nn.Module):
    """
    Standard Feed-Forward Network for a Transformer block.
    """
    def __init__(self, d_model: int, dropout: float, expansion_factor: int = 4):
        super().__init__()
        hidden_dim = int(d_model * expansion_factor)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SpatialConvAttention(nn.Module):
    """
    The original 'Attention' mechanism from SSVEPformer, which uses a 1D convolution
    to mix information across the token/channel dimension.
    """
    def __init__(self, num_tokens: int, d_model: int, kernel_size: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(num_tokens, num_tokens, kernel_size=kernel_size, padding=kernel_size // 2, groups=1)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class EncoderBlock(nn.Module):
    """
    A single encoder block combining SpatialConvAttention and a FeedForward network.
    """
    def __init__(self, d_model: int, num_tokens: int, kernel_size: int, dropout: float):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = SpatialConvAttention(num_tokens, d_model, kernel_size, dropout=dropout)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.attn(self.norm_attn(x)) + x
        x = self.ff(self.norm_ff(x)) + x
        return x


class Encoder(nn.Module):
    """A stack of N EncoderBlocks."""
    def __init__(self, depth: int, d_model: int, num_tokens: int, kernel_size: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(EncoderBlock(d_model, num_tokens, kernel_size, dropout=dropout))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class SSVEPformer(nn.Module):
    """The original SSVEPformer model, refactored for clarity."""
    def __init__(self, depth: int, attention_kernel_size: int, chs_num: int,
                 class_num: int, token_num: int, token_dim: int, dropout: float):
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            in_channels=chs_num, out_channels=token_num, feature_dim=token_dim, dropout=dropout
        )

        self.encoder = Encoder(
            depth=depth, d_model=token_dim, num_tokens=token_num, 
            kernel_size=attention_kernel_size, dropout=dropout
        )

        self.mlp_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * token_num, class_num * 6), # Note: Original had a typo here, should be token_dim
            nn.LayerNorm(class_num * 6),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(class_num * 6, class_num)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """Initializes weights of linear and convolutional layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x: Tensor) -> Tensor:
        # Squeeze singleton dimension if it exists
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)

        # Mix channels to create tokens
        x = self.patch_embedding(x)
        
        # Process sequence of tokens
        x = self.encoder(x)
        
        # Classify
        return self.mlp_head(x)


def build_model_from_config(config: dict) -> SSVEPformer:
    """
    Factory function to build the SSVEPformer model from a JSON config file.

    Args:
        config (dict): The unified configuration dictionary.

    Returns:
        SSVEPformer: An instance of the SSVEPformer model.
    """
    model_specific_params = config['model_params']['SSVEPformer']
    dataset_params = config['dataset_params']
    metadata_params = config['data_metadata']

    # --- Dynamically calculate token_dim based on preprocessing method ---
    preprocess_method = model_specific_params.get('preprocess_method', 'frequency')
    if preprocess_method == 'frequency':
        resolution = model_specific_params['resolution']
        start_freq = model_specific_params['start_freq']
        end_freq = model_specific_params['end_freq']
        token_dim = int(round(end_freq / resolution)) - int(round(start_freq / resolution))
    elif preprocess_method == 'fft':
        resolution = model_specific_params['resolution']
        start_freq = model_specific_params['start_freq']
        end_freq = model_specific_params['end_freq']
        token_dim = 2 * (int(round(end_freq / resolution)) - int(round(start_freq / resolution)))
    elif preprocess_method == 'raw':
        # For raw data, the feature dimension is the number of time points
        token_dim = int(metadata_params['Window_Size'] * metadata_params['Sample_Frequency'])
    else:
        raise ValueError(f"Unsupported preprocess_method: '{preprocess_method}'")

    return SSVEPformer(
        depth=model_specific_params['depth'],
        attention_kernel_size=model_specific_params['attention_kernel_size'],
        dropout=model_specific_params['dropout'],
        chs_num=len(dataset_params['channels']),
        class_num=metadata_params['Number_of_Targets'],
        token_num=model_specific_params.get('token_num', len(dataset_params['channels'])),
        token_dim=token_dim
    )

if __name__ == '__main__':
    print("Building model from config for visualization...")
    with open('./config.json', 'r') as f:
        config = json.load(f)

    model_specific_params = config['model_params']['SSVEPformer']
    data_dependent_params = config['data_params']
    
    all_params = {**model_specific_params, **data_dependent_params}
    model = SSVEPformer(**all_params)
    print(f"Model built successfully:\n{model}")

    def count_parameters(model: nn.Module):
        """Counts the number of trainable and non-trainable parameters in a model."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        return trainable_params, non_trainable_params
    
    trainable, non_trainable = count_parameters(model)
    print(f"\nTrainable parameters: {trainable:,}")
    print(f"Non-trainable parameters: {non_trainable:,}")