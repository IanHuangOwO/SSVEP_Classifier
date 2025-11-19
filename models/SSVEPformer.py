import json
import torch.nn as nn
from torch import Tensor
import torch

def preprocess_collate_fn(config_path: str):
    """
    Creates a collate_fn for a DataLoader that preprocesses raw EEG data.
    This function reads data parameters from the config to configure the FFT
    transformation required by the SSVEPformer model.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    data_params = config['data_param']
    Fs = data_params['Sample_Frequency']
    Ws = data_params['Window_Size']
    Tp = int(Ws * Fs)

    # FFT parameters
    resolution, start_freq, end_freq = 0.2, 8, 64
    nfft = round(Fs / resolution)
    fft_index_start = int(round(start_freq / resolution))
    fft_index_end = int(round(end_freq / resolution))

    def fft_preprocess_collate_fn(batch):
        """
        Takes a batch of (raw_eeg, label), applies FFT to raw_eeg, and returns tensors.
        """
        eeg_data = torch.stack([item[0] for item in batch]) # Shape: (B, 1, Nc, Tp)
        labels = torch.stack([item[1] for item in batch])   # Shape: (B, 1)

        fft_result = torch.fft.fft(eeg_data, n=nfft, dim=-1) / (Tp / 2)
        real_part = torch.real(fft_result[..., fft_index_start:fft_index_end])
        imag_part = torch.imag(fft_result[..., fft_index_start:fft_index_end])
        features = torch.cat([real_part, imag_part], dim=-1).float()
        return features, labels
    return fft_preprocess_collate_fn

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


def build_model_from_config(main_config_path: str) -> SSVEPformer:
    """
    Factory function to build the SSVEPformer model from a JSON config file.

    Args:
        main_config_path (str): Path to the main JSON configuration file (e.g., config.json).
        data_config_path (str): Path to the data-specific metadata file (e.g., metadata.json).

    Returns:
        SSVEPformer: An instance of the SSVEPformer model.
    """
    with open(main_config_path, 'r') as f:
        main_config = json.load(f)

    model_specific_params = main_config['model_params']['SSVEPformer']
    data_dependent_params = main_config['data_params']
    
    # Explicitly pass arguments for clarity and robustness
    return SSVEPformer(
        depth=model_specific_params['depth'],
        attention_kernel_size=model_specific_params['attention_kernel_size'],
        dropout=model_specific_params['dropout'],
        chs_num=len(data_dependent_params['channels']),
        class_num=data_dependent_params['class_num'],
        token_num=data_dependent_params['token_num'],
        token_dim=data_dependent_params['token_dim']
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