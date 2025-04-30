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

__all__ = ['VisionTransformer']

# ViT doesn't have a standard pretrained model URL yet for this specific task
model_urls = {
    'vit_base_patch16_224': 'https://download.pytorch.org/models/vit_b_16-c867db91.pth',
}

class PatchEmbed(nn.Module):
    """
    Split image into patches and embed them
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
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
        """
        Args:
            x: [B, C, H, W]
        Returns:
            patches: [B, num_patches, embed_dim]
        """
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
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, C//num_heads]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
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

# Optimized TransformerBlock with reduced computations
class TransformerBlock(nn.Module):
    """
    Transformer encoder block with optimizations for speed
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        # Reduced MLP ratio for faster computation
        mlp_hidden_dim = int(dim * mlp_ratio * 0.75)  # 25% reduction in hidden dim
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    """
    Vision Transformer model with multi-scale feature extraction for detection
    Optimized for faster training with ViT-Tiny
    """
    def __init__(self, img_size=416, patch_size=16, in_channels=3, embed_dim=192, 
             depth=8, num_heads=3, mlp_ratio=4., qkv_bias=True,
             norm_layer=nn.LayerNorm):
        super().__init__()
        
        # Calculate image size to be compatible with patch size
        self.img_size = img_size
        self.patch_size = patch_size
        
        # Calculate the actual number of patches
        num_patches = (img_size // patch_size) ** 2
        print(f"Creating ViT-Tiny with {num_patches} patches ({img_size}x{img_size} image, {patch_size}x{patch_size} patches)")
        
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
        
        # Ensure these match
        assert self.patch_embed.n_patches == num_patches, \
            f"Patch embed reports {self.patch_embed.n_patches} patches but calculation gives {num_patches}"
        
        # Add class token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)
    
        # Transformer blocks - REDUCED from 12 to 8 for speed
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                norm_layer=norm_layer)
            for i in range(depth)
        ])
        
        # Define intermediate feature extraction points
        # Adjusted for 8 layers instead of 12
        self.feat_layers = [1, 3, 5, 7]  # Reduced and evenly spaced
        
        # Store feature channels for the FPN
        self.feat_channel = [embed_dim] * len(self.feat_layers)
        
        # Normalization layer
        self.norm = norm_layer(embed_dim)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, pos_embed):
        """
        Interpolate position embeddings for different input sizes.
        
        Args:
            x: Input tensor [B, N, C]
            pos_embed: Position embedding tensor [1, N_orig, C]
        
        Returns:
            pos_embed_new: Interpolated position embedding [1, N_new, C]
        """
        npatch = x.shape[1] - 1  # exclude cls token
        N = pos_embed.shape[1] - 1  # exclude cls token
        
        if npatch == N:
            return pos_embed
        
        # Handle class token separately
        class_pos_embed = pos_embed[:, 0:1]
        patch_pos_embed = pos_embed[:, 1:]
        
        # Print debug info
        print(f"Interpolating pos embeddings: {N} patches to {npatch} patches")
        
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

    # Update the VisionTransformer's forward_features method to provide consistent feature map sizes
    def forward_features(self, x):
        """Extract features at different scales with consistent sizes"""
        # Get embeddings
        B, C, H, W = x.shape
        
        # Check if input size matches model's expected size
        input_resolution = H
        if input_resolution != self.img_size or W != self.img_size:
            print(f"Input resolution ({H}x{W}) doesn't match model's expected size ({self.img_size}x{self.img_size}).")
        
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        
        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Get position embeddings with proper size
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        
        # Add positional embedding
        x = x + pos_embed
        x = self.pos_drop(x)
        
        # Collect features from different transformer blocks
        features = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.feat_layers:
                # Remove class token and reshape to spatial feature map
                feature = x[:, 1:]  # Remove class token
                # Calculate height and width of feature map
                h = w = int(math.sqrt(feature.shape[1]))
                feature = feature.transpose(1, 2).reshape(B, -1, h, w)
                
                # Ensure the feature map has consistent dimensions by resizing if needed
                # This is to guarantee our feature maps have expected size for feature fusion
                # We resize to multiples of 16 for compatibility
                expected_size = (input_resolution // self.patch_size) // (2 ** (len(self.feat_layers) - self.feat_layers.index(i) - 1))
                expected_size = max(expected_size, 8)  # Ensure minimum size
                
                if h != expected_size or w != expected_size:
                    print(f"Resizing feature map at layer {i} from {h}x{w} to {expected_size}x{expected_size}")
                    feature = torch.nn.functional.interpolate(
                        feature, size=(expected_size, expected_size), mode='bilinear', align_corners=False)
                
                features.append(feature)
        
        print(f"Feature shapes: {[f.shape for f in features]}")
        return features

    def forward(self, x):
        return self.forward_features(x)

class TransformerFeatureProcessor(nn.Module):
    """
    Process features from ViT to be compatible with detection heads
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activate = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activate(x)
        return x

class ViTUp(nn.Module):
    """
    Process and upsample Vision Transformer features for detection
    """
    def __init__(self, channels, out_dim=24):
        super().__init__()
        channels = channels[::-1]  # Reverse channels to process from deep to shallow
        
        # Initial processing of the deepest feature
        self.conv = nn.Sequential(
            nn.Conv2d(channels[0], out_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_dim, eps=0.001, momentum=0.1),
            nn.ReLU(inplace=True)
        )
        
        # Final processing
        self.conv_last = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_dim, eps=1e-5, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        
        # Feature fusion modules
        for i, channel in enumerate(channels[1:]):
            setattr(self, f'up_{i}', IDAUp(out_dim, channel))

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

    def forward(self, layers):
        layers = list(layers)
        assert len(layers) > 1, "ViTUp requires at least 2 feature maps"
        
        print(f"Input feature shapes to ViTUp: {[layer.shape for layer in layers]}")
        
        # Process the deepest feature
        x = self.conv(layers[-1])
        print(f"After initial conv: {x.shape}")
        
        # Fuse features from deep to shallow
        for i in range(0, len(layers) - 1):
            up = getattr(self, f'up_{i}')
            x = up([x, layers[len(layers) - 2 - i]])
            print(f"After fusion step {i}: {x.shape}")
            
        # Final processing
        x = self.conv_last(x)
        print(f"Final output: {x.shape}")
        return x

# Update the IDAUp class in centerface_vit_fpn.py

# 5. Update the IDAUp class to handle mismatched feature sizes
class IDAUp(nn.Module):
    """
    Feature fusion module with adaptive resizing
    """
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
            print(f"Resizing feature maps - Up: {x.shape}, Lateral: {y.shape}")
            # Resize to the larger resolution
            target_h = max(x.size(2), y.size(2))
            target_w = max(x.size(3), y.size(3))
            
            if x.size(2) != target_h or x.size(3) != target_w:
                x = torch.nn.functional.interpolate(
                    x, size=(target_h, target_w), mode='bilinear', align_corners=False)
            
            if y.size(2) != target_h or y.size(3) != target_w:
                y = torch.nn.functional.interpolate(
                    y, size=(target_h, target_w), mode='bilinear', align_corners=False)
            
            print(f"After resize - Up: {x.shape}, Lateral: {y.shape}")
        
        out = x + y
        return out

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

# 1. Modify the ViTSeg.__init__ method to include output_size parameter
class ViTSeg(nn.Module):
    """
    Vision Transformer model for object detection
    """
    def __init__(self, base_name, heads, head_conv=24, pretrained=True, output_size=128):
        super().__init__()
        self.heads = heads
        self.output_size = output_size
        
        # Initialize Vision Transformer backbone
        if base_name == 'vit_base':
            self.base = VisionTransformer(
                img_size=416,
                patch_size=16,
                embed_dim=768,
                depth=12,
                num_heads=12
            )
        elif base_name == 'vit_small':
            self.base = VisionTransformer(
                img_size=416,
                patch_size=16,
                embed_dim=384,
                depth=10,  # Slightly reduced
                num_heads=6
            )
        elif base_name == 'vit_tiny':
            # Optimized tiny variant
            self.base = VisionTransformer(
                img_size=416,
                patch_size=16,
                embed_dim=192,
                depth=8,  # Reduced from 12 to 8
                num_heads=3
            )
        else:
            raise ValueError(f"Unknown model: {base_name}")
            
        # Rest of the initialization remains the same...
        channels = self.base.feat_channel
        self.vit_up = ViTUp(channels, out_dim=head_conv)
        self.head = Head_001(in_channel=head_conv, heads=heads)

    # 2. Add a new method to ensure output size matches expected size
    def resize_to_output_size(self, x):
        """Ensure that the feature map has the expected output size"""
        if x.shape[2] != self.output_size or x.shape[3] != self.output_size:
            print(f"Resizing final feature map from {x.shape[2]}x{x.shape[3]} to {self.output_size}x{self.output_size}")
            x = torch.nn.functional.interpolate(
                x, size=(self.output_size, self.output_size), 
                mode='bilinear', align_corners=False)
        return x

    # 3. Update the forward method to resize final output
    def forward(self, batch):
        x = batch['input']
        features = self.base(x)
        x = self.vit_up(features)
        
        # Ensure the feature map has the expected size before feeding to the head
        x = self.resize_to_output_size(x)
        
        ret = self.head(x, batch)
        return [ret]

def vit_base(pretrained=False, **kwargs):
    """
    ViT-Base model (ViT-B/16) with patch size 16x16
    """
    model = VisionTransformer(
        img_size=416,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        **kwargs
    )
    
    if pretrained:
        print("Pretrained weights need adaptation - loading skipped")
    
    return model

def vit_small(pretrained=False, **kwargs):
    """
    ViT-Small model with patch size 16x16
    """
    model = VisionTransformer(
        img_size=416,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        **kwargs
    )
    
    if pretrained:
        print("No pretrained weights available for ViT-Small")
    
    return model

def vit_tiny(pretrained=False, **kwargs):
    """
    ViT-Tiny model with patch size 16x16, optimized for speed
    """
    model = VisionTransformer(
        img_size=416,
        patch_size=16,
        embed_dim=192,
        depth=8,  # Reduced from 12 to 8
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        **kwargs
    )
    
    if pretrained:
        print("No pretrained weights available for ViT-Tiny")
    
    return model

# 6. Update the get_vit_net function to include output_size parameter
# Updated get_vit_net function
def get_vit_net(size='tiny', heads=None, head_conv=24, output_size=128, **kwargs):
    """
    Get Vision Transformer model with specified size and heads
    
    Args:
        size: Model size ('base', 'small', or 'tiny')
        heads: Dictionary of detection heads and their output channels
        head_conv: Number of channels in the head convolution layers
        output_size: Size of the output feature map (default: 128)
        **kwargs: Extra parameters (for compatibility with other model factories)
    
    Returns:
        model: ViT model for detection
    """
    # Handle any num_layers parameter passed for compatibility
    if 'num_layers' in kwargs:
        print(f"Note: num_layers parameter ({kwargs['num_layers']}) is ignored for ViT models")
    
    # Changed default to tiny
    model = ViTSeg(f'vit_{size}', heads, pretrained=False, head_conv=head_conv, output_size=output_size)
    return model
# Optional: Mixed Precision Training helper function
def enable_mixed_precision():
    """
    Enable mixed precision training if supported
    """
    if hasattr(torch.cuda, 'amp') and torch.cuda.is_available():
        print("Enabling mixed precision training")
        return torch.cuda.amp.GradScaler()
    else:
        print("Mixed precision training not available")
        return None

if __name__ == '__main__':
    import torch
    # Create a sample input
    input_tensor = torch.zeros([1, 3, 416, 416])
    batch = {'input': input_tensor}
    
    # Create model
    model = get_vit_net('base', {'hm': 1, 'hm_offset': 2, 'wh': 2, 'landmarks': 10}, head_conv=24)
    
    # Run inference
    res = model(batch)
    print(res[0].keys())  # Print the output keys