import sys, os
path = '/home/kunanon.k/Sakon_works/faceDetections/RPNetPlus/src_original/lib/models'
sys.path.append(path)
from torch import nn
import torch
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import math
from Head.Head_test import Head
from Head.Head_test import Head_001
import numpy as np
from torchsummary import summary

# Import your MobileNetV4 implementation
# Adjust the import path as needed based on your project structure
from .centerface_mobilenet_v2_fpn import MBConvBlock, ConvBNSiLU, SELayer, fill_up_weights

__all__ = ['HybridMobileViT']

# ViT-related components
class PatchEmbed(nn.Module):
    """
    Split image into patches and embed them
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, embed_dim, H//patch_size, W//patch_size]
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

class Attention(nn.Module):
    """
    Multi-head Self-Attention module
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    """
    MLP block with GELU activation
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    """
    Transformer encoder block
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio * 0.75)  # 25% reduction in hidden dim
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# MobileNetV4 backbone (early layers)
class MobileNetV4_Early(nn.Module):
    def __init__(self, width_mult=1.0, round_nearest=8):
        super().__init__()
        input_channel = 32
        
        # Initial convolutional layer
        self.conv1 = ConvBNSiLU(3, input_channel, kernel_size=3, stride=2)
        
        # First stage MBConv blocks
        self.stage1 = nn.Sequential(
            MBConvBlock(input_channel, 16, stride=1, expand_ratio=1, use_se=True),
        )
        input_channel = 16
        
        # Second stage MBConv blocks
        self.stage2 = nn.Sequential(
            MBConvBlock(input_channel, 24, stride=2, expand_ratio=6, use_se=True),
            MBConvBlock(24, 24, stride=1, expand_ratio=6, use_se=True),
        )
        input_channel = 24
        
        # Third stage MBConv blocks
        self.stage3 = nn.Sequential(
            MBConvBlock(input_channel, 32, stride=2, expand_ratio=6, use_se=True),
            MBConvBlock(32, 32, stride=1, expand_ratio=6, use_se=True),
            MBConvBlock(32, 32, stride=1, expand_ratio=6, use_se=True),
        )
        
        # Store output channels for each stage
        self.out_channels = [16, 24, 32]
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        features = []
        
        # Extract features from each stage
        x = self.conv1(x)
        
        x = self.stage1(x)
        features.append(x)  # Stage 1 features
        
        x = self.stage2(x)
        features.append(x)  # Stage 2 features
        
        x = self.stage3(x)
        features.append(x)  # Stage 3 features
        
        return features

# Vision Transformer for later stages
class ViT_Later(nn.Module):
    def __init__(self, input_dim=32, embed_dim=192, depth=6, num_heads=3, 
                 input_size=52, patch_size=4, norm_layer=nn.LayerNorm):
        super().__init__()
        
        # Conv projection to change channels from input_dim to embed_dim
        self.conv_proj = nn.Sequential(
            nn.Conv2d(input_dim, embed_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True)
        )
        
        # Calculate new size after MobileNet stages
        self.input_size = input_size
        self.patch_size = patch_size
        num_patches = (input_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=input_size, patch_size=patch_size, 
            in_channels=embed_dim, embed_dim=embed_dim
        )
        
        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=4, qkv_bias=True,
                norm_layer=norm_layer)
            for i in range(depth)
        ])
        
        # Define feature extraction points
        self.feat_layers = [1, 3, 5]  # For a 6-layer transformer
        
        # Store feature channels for FPN
        self.feat_channel = [embed_dim] * len(self.feat_layers)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def interpolate_pos_encoding(self, x, pos_embed):
        npatch = x.shape[1] - 1  # exclude cls token
        N = pos_embed.shape[1] - 1  # exclude cls token
        
        if npatch == N:
            return pos_embed
        
        # Handle class token separately
        class_pos_embed = pos_embed[:, 0:1]
        patch_pos_embed = pos_embed[:, 1:]
        
        # Interpolate patch position embeddings
        dim = x.shape[-1]
        w = h = int(math.sqrt(npatch))
        w0 = h0 = int(math.sqrt(N))
        
        # Reshape for 2D interpolation
        patch_pos_embed = patch_pos_embed.reshape(1, h0, w0, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)  # [1, dim, h0, w0]
        
        # Interpolate using bicubic mode
        patch_pos_embed = torch.nn.functional.interpolate(
            patch_pos_embed, size=(h, w), mode='bicubic', align_corners=False)
        
        # Reshape back
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)  # [1, h, w, dim]
        patch_pos_embed = patch_pos_embed.flatten(1, 2)  # [1, h*w, dim]
        
        # Concat with class token position embedding
        pos_embed_new = torch.cat((class_pos_embed, patch_pos_embed), dim=1)
        
        return pos_embed_new
        
    def forward(self, x):
        # Apply initial convolution to change channels
        x = self.conv_proj(x)
        
        # Convert to patches
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        
        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add positional embedding
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        x = x + pos_embed
        x = self.pos_drop(x)
        
        # Collect features from transformer blocks
        features = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.feat_layers:
                # Remove class token and reshape to spatial feature map
                feature = x[:, 1:]  # Remove class token
                # Calculate height and width of feature map
                h = w = int(math.sqrt(feature.shape[1]))
                feature = feature.transpose(1, 2).reshape(B, -1, h, w)
                features.append(feature)
        
        return features

# IDAUp for feature fusion
class IDAUp(nn.Module):
    def __init__(self, out_dim, channel):
        super().__init__()
        self.out_dim = out_dim
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                out_dim, out_dim, kernel_size=2, stride=2, padding=0,
                output_padding=0, groups=out_dim, bias=False),
            nn.BatchNorm2d(out_dim, eps=0.001, momentum=0.1),
            nn.ReLU(inplace=True)
        )
        
        self.conv = nn.Sequential(
            nn.Conv2d(channel, out_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_dim, eps=0.001, momentum=0.1),
            nn.ReLU(inplace=True)
        )

    def forward(self, layers):
        layers = list(layers)
        x = self.up(layers[0])
        y = self.conv(layers[1])
        
        # Handle resolution mismatch with adaptive resizing
        if x.size(2) != y.size(2) or x.size(3) != y.size(3):
            # Resize to the larger resolution
            target_h = max(x.size(2), y.size(2))
            target_w = max(x.size(3), y.size(3))
            
            if x.size(2) != target_h or x.size(3) != target_w:
                x = torch.nn.functional.interpolate(
                    x, size=(target_h, target_w), mode='bilinear', align_corners=False)
            
            if y.size(2) != target_h or y.size(3) != target_w:
                y = torch.nn.functional.interpolate(
                    y, size=(target_h, target_w), mode='bilinear', align_corners=False)
                
        out = x + y
        return out

# Feature fusion module
class HybridUp(nn.Module):
    def __init__(self, channels_cnn, channels_vit, out_dim=24):
        super().__init__()
        # Reverse channels order for bottom-up processing
        channels_cnn = channels_cnn[::-1]
        channels_vit = channels_vit[::-1]
        
        # Calculate total feature maps
        self.num_cnn_features = len(channels_cnn)
        self.num_vit_features = len(channels_vit)
        self.total_features = self.num_cnn_features + self.num_vit_features
        
        # Conv for initial processing of the deepest CNN feature
        self.conv_cnn = nn.Sequential(
            nn.Conv2d(channels_cnn[0], out_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_dim, eps=0.001, momentum=0.1),
            nn.ReLU(inplace=True)
        )
        
        # Conv for initial processing of the deepest ViT feature
        self.conv_vit = nn.Sequential(
            nn.Conv2d(channels_vit[0], out_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_dim, eps=0.001, momentum=0.1),
            nn.ReLU(inplace=True)
        )
        
        # Additional conv to combine CNN and ViT features
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_dim, eps=0.001, momentum=0.1),
            nn.ReLU(inplace=True)
        )
        
        # Final processing
        self.conv_last = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_dim, eps=1e-5, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        
        # Feature fusion modules for CNN features
        for i, channel in enumerate(channels_cnn[1:]):
            setattr(self, f'up_cnn_{i}', IDAUp(out_dim, channel))
            
        # Feature fusion modules for ViT features
        for i, channel in enumerate(channels_vit[1:]):
            setattr(self, f'up_vit_{i}', IDAUp(out_dim, channel))

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                fill_up_weights(m)
                
    def fuse_features(self, feat1, feat2):
        """Fuse two feature maps with equal weighting"""
        # Resize features to match
        if feat1.size(2) != feat2.size(2) or feat1.size(3) != feat2.size(3):
            target_h = max(feat1.size(2), feat2.size(2))
            target_w = max(feat1.size(3), feat2.size(3))
            
            if feat1.size(2) != target_h or feat1.size(3) != target_w:
                feat1 = torch.nn.functional.interpolate(
                    feat1, size=(target_h, target_w), mode='bilinear', align_corners=False)
                
            if feat2.size(2) != target_h or feat2.size(3) != target_w:
                feat2 = torch.nn.functional.interpolate(
                    feat2, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        # Concatenate and apply fusion convolution
        fused = torch.cat([feat1, feat2], dim=1)
        fused = self.fusion_conv(fused)
        return fused

    def forward(self, cnn_features, vit_features):
        # Verify we have features
        assert len(cnn_features) >= 1, "Need at least one CNN feature"
        assert len(vit_features) >= 1, "Need at least one ViT feature"
        
        # Process CNN and ViT features separately first
        cnn_features = cnn_features[::-1]  # Reverse for bottom-up
        vit_features = vit_features[::-1]  # Reverse for bottom-up
        
        # Initial processing of deepest features
        x_cnn = self.conv_cnn(cnn_features[0])
        x_vit = self.conv_vit(vit_features[0])
        
        # Fuse the deepest features
        x = self.fuse_features(x_cnn, x_vit)
        
        # Bottom-up path for CNN features
        for i in range(len(cnn_features) - 1):
            up = getattr(self, f'up_cnn_{i}')
            x = up([x, cnn_features[i + 1]])
            
        # Bottom-up path for ViT features (if more than one)
        for i in range(len(vit_features) - 1):
            up = getattr(self, f'up_vit_{i}')
            x = up([x, vit_features[i + 1]])
            
        # Final processing
        x = self.conv_last(x)
        
        return x

# Main Hybrid Model
class HybridMobileViT(nn.Module):
    def __init__(self, heads, head_conv=24, input_size=416, output_size=128):
        super().__init__()
        self.heads = heads
        self.output_size = output_size
        
        # MobileNetV4 early layers
        self.mobilenet_early = MobileNetV4_Early(width_mult=1.0)
        cnn_channels = self.mobilenet_early.out_channels
        
        # Calculate input size for ViT
        # After 3 stages of MobileNet with strides 2, 2, 2
        vit_input_size = input_size // 8
        
        # Vision Transformer later layers
        self.vit_later = ViT_Later(
            input_dim=cnn_channels[-1],
            embed_dim=192,
            depth=6,
            num_heads=3,
            input_size=vit_input_size,
            patch_size=4
        )
        vit_channels = self.vit_later.feat_channel
        
        # Fusion module
        self.hybrid_up = HybridUp(cnn_channels, vit_channels, out_dim=head_conv)
        
        # Detection heads
        self.head = Head_001(in_channel=head_conv, heads=heads)
        
        # Feature normalization
        self.feature_norm = nn.BatchNorm2d(head_conv)
        
    def resize_to_output_size(self, x):
        """Ensure the feature map has the expected output size"""
        if x.shape[2] != self.output_size or x.shape[3] != self.output_size:
            x = torch.nn.functional.interpolate(
                x, size=(self.output_size, self.output_size), 
                mode='bilinear', align_corners=False)
        return x
    
    def forward(self, batch):
        x = batch['input']
        
        # Extract features from MobileNetV4 early layers
        cnn_features = self.mobilenet_early(x)
        
        # Extract features from Vision Transformer later layers
        # Pass the last CNN feature to ViT
        vit_features = self.vit_later(cnn_features[-1])
        
        # Fuse features from both branches
        x = self.hybrid_up(cnn_features, vit_features)
        
        # Normalize features
        x = self.feature_norm(x)
        
        # Ensure the feature map has the expected size
        x = self.resize_to_output_size(x)
        
        # Pass through detection heads
        ret = self.head(x, batch)
        
        return [ret]

# Function to create the hybrid model
def get_hybrid_net(heads=None, head_conv=24, output_size=128, **kwargs):
    """
    Get Hybrid MobileNetV4-ViT model with specified parameters
    
    Args:
        heads: Dictionary of detection heads and their output channels
        head_conv: Number of channels in the head convolution layers
        output_size: Size of the output feature map (default: 128)
        **kwargs: Extra parameters (for compatibility with other model factories)
    
    Returns:
        model: Hybrid model for detection
    """
    model = HybridMobileViT(heads, head_conv=head_conv, output_size=output_size)
    return model

if __name__ == '__main__':
    import torch
    # Create a sample input
    input_tensor = torch.zeros([1, 3, 416, 416])
    batch = {'input': input_tensor}
    
    # Create model
    model = get_hybrid_net({'hm': 1, 'hm_offset': 2, 'wh': 2, 'landmarks': 10}, head_conv=24)
    
    # Run inference
    res = model(batch)
    print(res[0].keys())  # Print the output keys