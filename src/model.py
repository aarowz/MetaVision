import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

"""
Vision Transformer Model for EM Field Prediction

Custom ViT encoder + CNN decoder architecture for predicting electromagnetic
field distributions from metasurface geometry images.

Architecture:
- Encoder: Custom Vision Transformer (4-channel input, no CLS token)
- Decoder: CNN with 3-stage upsampling (15×15 → 30×30 → 60×60 → 120×120)
- Output: 6-channel EM field prediction (Ex/Ey/Ez real+imaginary components)

Input: [B, 4, 120, 120] (R, H, D[0], D[1] geometry channels)
Output: [B, 6, 120, 120] (Ex_real, Ex_imag, Ey_real, Ey_imag, Ez_real, Ez_imag)
"""


class PatchEmbedding(nn.Module):
    """Convolutional patch embedding for 4-channel input."""
    
    def __init__(self, img_size: int = 120, patch_size: int = 8, 
                 in_chans: int = 4, embed_dim: int = 384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2  # 15×15 = 225
        
        # Convolutional patch embedding
        self.proj = nn.Conv2d(in_chans, embed_dim, 
                              kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert input image to patch tokens.
        
        Args:
            x: [B, 4, 120, 120]
        Returns:
            [B, 225, 384] (flattened patch tokens)
        """
        # [B, 4, 120, 120] -> [B, 384, 15, 15]
        x = self.proj(x)
        B, C, H, W = x.shape
        
        # Flatten spatial dimensions: [B, 384, 15, 15] -> [B, 225, 384]
        x = x.flatten(2).transpose(1, 2)  # [B, 225, 384]
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with pre-norm."""
    
    def __init__(self, embed_dim: int = 384, num_heads: int = 6, 
                 qkv_bias: bool = True, attn_drop: float = 0.1, 
                 proj_drop: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV projection
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head self-attention.
        
        Args:
            x: [B, N, embed_dim]
        Returns:
            [B, N, embed_dim]
        """
        B, N, C = x.shape
        
        # QKV projection: [B, N, embed_dim] -> [B, N, 3*embed_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, embed_dim]
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    """MLP with GELU activation."""
    
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, drop: float = 0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply MLP transformation.
        
        Args:
            x: [B, N, in_features]
        Returns:
            [B, N, out_features]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Stochastic depth (drop path) regularization."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply stochastic depth.
        
        Args:
            x: Input tensor
        Returns:
            Tensor with stochastic depth applied
        """
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        random_tensor = keep_prob + torch.rand(x.shape[0], 1, 1, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm, stochastic depth."""
    
    def __init__(self, embed_dim: int = 384, num_heads: int = 6,
                 mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 drop_rate: float = 0.1, attn_drop_rate: float = 0.1,
                 drop_path_rate: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop_rate, proj_drop=drop_rate
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(in_features=embed_dim, hidden_features=mlp_hidden_dim,
                       drop=drop_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply transformer block (pre-norm attention + MLP).
        
        Args:
            x: [B, N, embed_dim]
        Returns:
            [B, N, embed_dim]
        """
        # Pre-norm: norm -> attention -> residual
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # Pre-norm: norm -> mlp -> residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DecoderBlock(nn.Module):
    """Decoder block: upsample + two conv layers."""
    
    def __init__(self, in_channels: int, out_channels: int,
                 use_batch_norm: bool = True, activation: str = "relu"):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # First conv
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.act1 = self._get_activation(activation)
        
        # Second conv
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.act2 = self._get_activation(activation)
    
    def _get_activation(self, activation: str):
        """Get activation function from string."""
        if activation.lower() == "relu":
            return nn.ReLU(inplace=True)
        elif activation.lower() == "gelu":
            return nn.GELU()
        elif activation.lower() == "swish":
            return nn.SiLU()
        else:
            return nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply decoder block (upsample + conv layers).
        
        Args:
            x: Input feature map
        Returns:
            Upsampled and convolved feature map
        """
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x


class MetaVisionViT(nn.Module):
    """Vision Transformer for EM Field Prediction."""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Extract config
        encoder_cfg = config['model']['encoder']
        decoder_cfg = config['model']['decoder']
        output_cfg = config['model']['output_head']
        
        # ========== Encoder (ViT) ==========
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=encoder_cfg['img_size'],
            patch_size=encoder_cfg['patch_size'],
            in_chans=encoder_cfg['in_chans'],
            embed_dim=encoder_cfg['embed_dim']
        )
        
        # Positional encoding (learnable)
        num_patches = self.patch_embed.n_patches  # 225
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, encoder_cfg['embed_dim']))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=encoder_cfg['embed_dim'],
                num_heads=encoder_cfg['num_heads'],
                mlp_ratio=encoder_cfg['mlp_ratio'],
                qkv_bias=encoder_cfg['qkv_bias'],
                drop_rate=encoder_cfg['drop_rate'],
                attn_drop_rate=encoder_cfg['attn_drop_rate'],
                drop_path_rate=encoder_cfg['drop_path_rate']
            )
            for _ in range(encoder_cfg['depth'])
        ])
        
        self.encoder_norm = nn.LayerNorm(encoder_cfg['embed_dim'])
        
        # ========== Decoder (CNN) ==========
        # Initial projection: 384 -> 256
        embed_dim = encoder_cfg['embed_dim']
        decoder_start_channels = decoder_cfg['decoder_channels'][0]
        self.decoder_proj = nn.Conv2d(embed_dim, decoder_start_channels, kernel_size=1)
        
        # Decoder stages
        decoder_channels = decoder_cfg['decoder_channels']
        self.decoder_blocks = nn.ModuleList()
        
        for i in range(len(decoder_channels) - 1):
            in_ch = decoder_channels[i]
            out_ch = decoder_channels[i + 1]
            self.decoder_blocks.append(
                DecoderBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    use_batch_norm=decoder_cfg['use_batch_norm'],
                    activation=decoder_cfg['activation']
                )
            )
        
        # ========== Output Head ==========
        final_channels = output_cfg['final_channels']
        last_decoder_channels = decoder_channels[-1]  # 32
        self.output_head = nn.Conv2d(last_decoder_channels, final_channels, kernel_size=1)
        
        # Output activation (if specified)
        activation = output_cfg.get('activation', 'none').lower()
        if activation == 'tanh':
            self.output_act = nn.Tanh()
        elif activation == 'sigmoid':
            self.output_act = nn.Sigmoid()
        else:  # 'none' or default
            self.output_act = nn.Identity()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize positional embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Initialize patch embedding
        nn.init.kaiming_normal_(self.patch_embed.proj.weight, mode='fan_out', nonlinearity='relu')
        if self.patch_embed.proj.bias is not None:
            nn.init.constant_(self.patch_embed.proj.bias, 0)
        
        # Initialize transformer blocks
        for block in self.blocks:
            nn.init.normal_(block.norm1.weight, mean=1.0, std=0.02)
            nn.init.constant_(block.norm1.bias, 0)
            nn.init.normal_(block.norm2.weight, mean=1.0, std=0.02)
            nn.init.constant_(block.norm2.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [B, 4, 120, 120] (R, H, D[0], D[1])
        Returns:
            [B, 6, 120, 120] (Ex_real, Ex_imag, Ey_real, Ey_imag, Ez_real, Ez_imag)
        """
        B = x.shape[0]
        
        # ========== Encoder ==========
        # Patch embedding: [B, 4, 120, 120] -> [B, 225, 384]
        x = self.patch_embed(x)
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final norm
        x = self.encoder_norm(x)  # [B, 225, 384]
        
        # ========== Decoder ==========
        # Reshape to spatial grid: [B, 225, 384] -> [B, 384, 15, 15]
        H_patches = W_patches = int(self.patch_embed.n_patches ** 0.5)  # 15
        x = x.transpose(1, 2).reshape(B, -1, H_patches, W_patches)
        
        # Initial projection: [B, 384, 15, 15] -> [B, 256, 15, 15]
        x = self.decoder_proj(x)
        
        # Decoder stages: 15×15 -> 30×30 -> 60×60 -> 120×120
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x)
        
        # ========== Output Head ==========
        # [B, 32, 120, 120] -> [B, 6, 120, 120]
        x = self.output_head(x)
        x = self.output_act(x)
        
        return x


def create_model(config: Dict) -> MetaVisionViT:
    """
    Create model from config dictionary.
    
    Args:
        config: Configuration dictionary loaded from config.yaml
    Returns:
        MetaVisionViT model instance
    """
    return MetaVisionViT(config)

