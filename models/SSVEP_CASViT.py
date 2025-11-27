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
        
        def frequency_collate_fn(batch):
            eeg_data = torch.stack([item[0] for item in batch])
            labels = torch.stack([item[1] for item in batch])

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
        
        return frequency_collate_fn

    elif preprocess_method == 'raw': # This is a non-FFT method
        def raw_preprocess_collate_fn(batch):
            eeg_data = torch.stack([item[0] for item in batch]).float()
            labels = torch.stack([item[1] for item in batch])
            return eeg_data, labels
        return raw_preprocess_collate_fn

    else:
        raise ValueError(f"Unsupported preprocess_method: '{preprocess_method}'")

class PatchEmbedding(nn.Module):
    """
    Expands EEG channels into a sequence of tokens using a convolution.
    This acts as the initial "patch embedding" layer for the Transformer.
    """
    def __init__(self, in_channels: int, out_channels: int, feature_dim: int, dropout: float, kernel_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        # Input x: (batch, in_channels, feature_dim)
        # The Conv1d mixes information ACROSS THE CHANNELS.
        # The LayerNorm normalizes ACROSS THE FEATURES for each new token.
        # Output: (batch, out_channels, feature_dim)
        return self.net(x)

class CASA(nn.Module):
    """
    Implements Convolutional Adaptive Separable Attention.
    This is an efficient attention mechanism that uses a dynamically generated
    depth-wise separable convolution to mix information between tokens (channels).
    The kernel is generated from the frequency content of each token.
    """
    def __init__(self, d_model: int, num_tokens: int, kernel_size: int, dropout: float):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_tokens = num_tokens

        # Kernel generator network: generates a kernel from each token's frequency content (d_model).
        self.kernel_generator = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            # nn.Softmax(dim=-1) # Force a probability distribution over the feature dimension
        )

        # Static depth-wise convolution to capture local patterns across tokens
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2, groups=d_model)

        # Point-wise convolution (Feed-Forward): mixes information across the feature dimension
        self.pointwise_conv = nn.Linear(d_model, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # 1. Generate adaptive kernel from frequency content.
        # The linear layers operate on the last dimension (D), which is the FEATURE dimension.
        # This creates a data-dependent scaling factor based on each token's frequency spectrum.
        adaptive_weights = self.kernel_generator(x) # -> (B, N, D)

        # 2. Apply static depth-wise convolution across the CHANNEL/TOKEN dimension
        # We permute x to (B, D, N) so convolution slides along the N tokens.
        x_permuted = x.permute(0, 2, 1) # -> (B, d_model, num_tokens)
        # The convolution now mixes information ACROSS THE TOKENS (CHANNELS) independently for each feature.
        x_conv = self.depthwise_conv(x_permuted)
        x_conv = x_conv.permute(0, 2, 1) # Permute back to (B, N, D)

        # 3. Combine adaptive weights with the convolution output
        # This multiplication makes the operation adaptive and is ONNX-friendly.
        x_adapted = adaptive_weights * x_conv
        
        # Store weights for visualization
        self.attention_weights = adaptive_weights

        # 4. Apply point-wise convolution (a linear layer) and activations.
        # This processes the FEATURES of each token independently.
        x = self.pointwise_conv(x_adapted)
        return self.drop(self.act(self.norm(x)))
    
    
class FeedForward(nn.Module):
    """
    Standard Feed-Forward Network for a Transformer block.
    It consists of two linear layers with a GELU activation in between.
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
        # The linear layers operate on the last dimension (d_model), processing the FEATURES of each token.
        return self.net(x)

class EncoderBlock(nn.Module):
    """
    A single block of the Transformer Encoder, combining CASA and a FeedForward network.
    Follows the Pre-LN (Layer Normalization) architecture.
    """
    def __init__(self, d_model: int, num_tokens: int, kernel_size: int, dropout: float):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = CASA(d_model, num_tokens, kernel_size, dropout=dropout)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        # The 'attn' block (CASA) mixes information ACROSS THE TOKENS/CHANNELS.
        x = self.attn(self.norm_attn(x)) + x
        
        # The 'ff' block (FeedForward) processes the FEATURES of each token independently.
        x = self.ff(self.norm_ff(x)) + x
        return x

class Encoder(nn.Module):
    """A stack of N EncoderBlocks to form the main body of the Transformer."""
    def __init__(self, depth: int, d_model: int, num_tokens: int, kernel_size: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(EncoderBlock(d_model, num_tokens, kernel_size, dropout=dropout))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class SSVEP_CASViT(nn.Module):
    """
    The main SSVEP_CASViT model, inspired by CAS-ViT and adapted for SSVEP classification.

    Args:
        depth (int): Number of EncoderBlocks to stack.
        attention_kernel_size (int): The size of the dynamically generated convolutional kernel in CASA.
        chs_num (int): Number of input EEG channels.
        class_num (int): Number of output classes (SSVEP targets).
        token_num (int): The number of tokens to create from the input channels.
        token_dim (int): The feature dimension of each token (should match FFT output).
        dropout (float): Dropout rate used throughout the model.
        embedding_kernel_size (int): The kernel size for the initial Conv1d embedding layer.
    """
    def __init__(self, depth: int, attention_kernel_size: int, chs_num: int,
                 class_num: int, token_num: int, token_dim: int, dropout: float, embedding_kernel_size: int):
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            in_channels=chs_num, out_channels=token_num, feature_dim=token_dim, 
            dropout=dropout, kernel_size=embedding_kernel_size
        )
        
        self.encoder = Encoder(
            depth=depth, d_model=token_dim, num_tokens=token_num, 
            kernel_size=attention_kernel_size, dropout=dropout
        )

        self.mlp_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * token_num, class_num * 6),
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
        # Input x shape: (batch, 1, num_channels, feature_dim)
        # Squeeze the singleton dimension if it exists.
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        # Now x shape: (batch, num_channels, feature_dim) -> e.g., (16, 8, 560)
        
        # --- 1. Channel Mixing ---
        # The PatchEmbedding mixes information ACROSS THE CHANNELS to create tokens.
        x = self.patch_embedding(x) # -> (batch, token_num, token_dim) -> (16, 16, 560)
        
        # --- 2. Feature Processing ---
        # The Encoder processes the sequence of tokens. Inside, CASA mixes information
        # ACROSS TOKENS/CHANNELS, while the FeedForward network processes the FEATURES of each token.
        x = self.encoder(x)

        # --- 3. Classification ---
        # The MLP head flattens all tokens and features together to make a final prediction.
        output = self.mlp_head(x)
        
        return output
    
def build_model_from_config(config: dict) -> SSVEP_CASViT:
    """
    Factory function to build the SSVEP_CASViT model from a JSON config file.

    Args:
        config (dict): The unified configuration dictionary.

    Returns:
        SSVEP_CASViT: An instance of the SSVEP_CASViT model.
    """
    model_specific_params = config['model_params']['SSVEP_CASViT']
    dataset_params = config['dataset_params']
    metadata_params = config['data_metadata']

    # --- Determine the actual number of channels ---
    # This logic must match the logic in IO/dataset.py
    if dataset_params.get('channels') == ["all"]:
        num_channels = len(metadata_params['channel_names'])
    else:
        num_channels = len(dataset_params['channels'])

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

    return SSVEP_CASViT(
        depth=model_specific_params['depth'],
        attention_kernel_size=model_specific_params['attention_kernel_size'],
        dropout=model_specific_params['dropout'],
        chs_num=num_channels,
        class_num=metadata_params['Number_of_Targets'],
        token_num=model_specific_params.get('token_num', len(dataset_params['channels'])),
        token_dim=token_dim,
        embedding_kernel_size=model_specific_params['embedding_kernel_size']
    )

if __name__ == '__main__':
    print("Building model from config for visualization...")
    with open('./config.json', 'r') as f:
        config = json.load(f)

    model_specific_params = config['model_params']['SSVEP_CASViT']
    data_dependent_params = config['data_params']
    
    all_params = {**model_specific_params, **data_dependent_params}
    model = SSVEP_CASViT(**all_params)
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