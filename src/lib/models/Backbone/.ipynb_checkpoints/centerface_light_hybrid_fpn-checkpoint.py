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
import timm # Import เพิ่มเติม


# Import MobileNetV4 components
from .centerface_mobilenet_v2_fpn import MBConvBlock, ConvBNSiLU, SELayer, fill_up_weights

__all__ = ['LightweightHybridModel']

# Efficient Attention Module (lighter than full self-attention)
class EfficientAttention(nn.Module):
    """
    Efficient attention mechanism that's faster than regular self-attention
    """
    def __init__(self, dim, key_dim=32, num_heads=4, attn_ratio=2, resolution=None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.attn_ratio = attn_ratio
        self.resolution = resolution
        
        # Input projection
        self.qkv = nn.Conv2d(dim, key_dim * (2 + attn_ratio) * num_heads, kernel_size=1)
        
        # Output projection
        self.proj = nn.Sequential(
            nn.Conv2d(key_dim * attn_ratio * num_heads, dim, kernel_size=1),
            nn.BatchNorm2d(dim)
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Generate queries, keys, and values
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, [
            self.key_dim * self.num_heads,
            self.key_dim * self.num_heads,
            self.key_dim * self.attn_ratio * self.num_heads
        ], dim=1)
        
        # Reshape for multi-head attention
        q = q.reshape(B, self.num_heads, self.key_dim, H*W)
        k = k.reshape(B, self.num_heads, self.key_dim, H*W)
        v = v.reshape(B, self.num_heads, self.key_dim * self.attn_ratio, H*W)
        
        # Calculate attention weights
        attn = torch.einsum('bhkn,bhkm->bhnm', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Apply attention weights to values
        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        out = out.reshape(B, self.key_dim * self.attn_ratio * self.num_heads, H, W)
        
        # Apply output projection
        out = self.proj(out)
        return out

# Lightweight Transformer Block
class LightTransformerBlock(nn.Module):
    """
    Lightweight transformer block using efficient attention
    """
    def __init__(self, dim, key_dim=32, num_heads=4, mlp_ratio=2., attn_ratio=2.,
                 drop=0., drop_path=0., act_layer=nn.SiLU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = EfficientAttention(
            dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio)
        self.norm2 = nn.BatchNorm2d(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, kernel_size=1),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# MobileNetV4 Backbone
class EfficientMobileNetV4(nn.Module):
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
            MBConvBlock(input_channel, 24, stride=2, expand_ratio=4, use_se=True),
            MBConvBlock(24, 24, stride=1, expand_ratio=4, use_se=True),
        )
        input_channel = 24
        
        # Third stage MBConv blocks (reduced depth)
        self.stage3 = nn.Sequential(
            MBConvBlock(input_channel, 40, stride=2, expand_ratio=4, use_se=True),
            MBConvBlock(40, 40, stride=1, expand_ratio=4, use_se=True),
        )
        input_channel = 40
        
        # Fourth stage MBConv blocks (reduced depth)
        self.stage4 = nn.Sequential(
            MBConvBlock(input_channel, 80, stride=2, expand_ratio=4, use_se=True),
            MBConvBlock(80, 80, stride=1, expand_ratio=4, use_se=True),
        )
        
        # Store output channels for each stage
        self.out_channels = [16, 24, 40, 80]
        
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
        
        x = self.stage4(x)
        features.append(x)  # Stage 4 features
        
        return features

# Lightweight Transformer Module
class LightTransformer(nn.Module):
    def __init__(self, input_dim=80, depth=3, num_heads=4, key_dim=32, attn_ratio=2):
        super().__init__()
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            LightTransformerBlock(
                dim=input_dim, key_dim=key_dim, num_heads=num_heads,
                mlp_ratio=2, attn_ratio=attn_ratio)
            for _ in range(depth)
        ])
    
    def forward(self, x):
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x)
        return x

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
            nn.SiLU(inplace=True)
        )
        
        self.conv = nn.Sequential(
            nn.Conv2d(channel, out_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_dim, eps=0.001, momentum=0.1),
            nn.SiLU(inplace=True)
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

# Feature Fusion Module
class LightHybridUp(nn.Module):
    def __init__(self, channels, out_dim=24):
        super().__init__()
        # Reverse channels order for bottom-up processing
        self.channels = channels[::-1]
        
        # Print channels for debugging
        print(f"Channel dimensions for FPN: {self.channels}")
        
        # Initial processing of the deepest feature
        self.conv = nn.Sequential(
            nn.Conv2d(self.channels[0], out_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_dim, eps=0.001, momentum=0.1),
            nn.SiLU(inplace=True)
        )
        
        # Final processing
        self.conv_last = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_dim, eps=1e-5, momentum=0.01),
            nn.SiLU(inplace=True)
        )
        
        # Feature fusion modules
        for i, channel in enumerate(self.channels[1:]):
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
        # Input: layers คือ list ของ features จาก backbone + transformer
        # เรียงจากตื้นไปลึก เช่น [s1(stride 2), s2(stride 4), s3(stride 8), s4_trans(stride 16)]
        # ตัวอย่าง shapes สำหรับ input 512x512:
        # [(16, 256, 256), (24, 128, 128), (40, 64, 64), (80, 32, 32)]
        layers = list(layers)
        if len(layers) != 4:
            raise ValueError("LightHybridUp expects exactly 4 input feature maps.")
    
        print(f"Input feature shapes to LightHybridUp (Original Order): {[layer.shape for layer in layers]}")
    
        # 1. ประมวลผล Feature ลึกสุด (s4_trans, index 3)
        # Output: (head_conv, H/16, W/16) เช่น (64, 32, 32) ถ้า head_conv=64
        x = self.conv(layers[3])
        print(f"After initial conv: {x.shape}")
    
        # 2. ทำ Feature Fusion จากลึกไปตื้น (Bottom-up)
        # เราต้องการ Output ที่ Stride 4 (เช่น 128x128)
        # i=0: ผสม x (s16) กับ layers[2] (s8) -> ผลลัพธ์ x เป็น s8 (64x64)
        # i=1: ผสม x (s8) กับ layers[1] (s4) -> ผลลัพธ์ x เป็น s4 (128x128)
        # เราจะหยุดแค่นี้ ไม่ทำ i=2 ซึ่งจะผสมกับ layers[0] (s2) และ Upsample ไป s2 (256x256)
    
        num_fusion_stages_to_run = 2 # ต้องการทำ 2 ขั้นตอน (i=0, i=1) เพื่อให้ได้ Stride 4
    
        for i in range(num_fusion_stages_to_run):
            up = getattr(self, f'up_{i}')
            # Index ของ layer ถัดไปที่จะนำมาผสม (จากตื้นลงไป)
            # i=0 -> ใช้ layers[2] (s3)
            # i=1 -> ใช้ layers[1] (s2)
            input_feature_index = len(layers) - 2 - i
            print(f"Fusing features: {x.shape} and {layers[input_feature_index].shape} (Stage index {input_feature_index})")
            x = up([x, layers[input_feature_index]])
            print(f"After fusion step {i}: {x.shape}")
    
        # ตอนนี้ x ควรจะมี Resolution ที่ต้องการแล้ว (เช่น 128x128, Stride 4)
    
        # 3. ผ่าน Conv สุดท้าย (ไม่เปลี่ยน Resolution)
        x = self.conv_last(x)
        print(f"Final output (before head): {x.shape}") # ควรเป็น (B, head_conv, H/4, W/4)
    
        return x

# Main Lightweight Hybrid Model
class LightweightHybridModel(nn.Module):
    def __init__(self, heads, head_conv=24, output_size=128):
        super().__init__()
        self.heads = heads
        self.output_size = output_size
        
        # MobileNetV4 backbone
        self.backbone = EfficientMobileNetV4(width_mult=1.0)
        backbone_channels = self.backbone.out_channels
        
        # Lightweight transformer module applied to the last feature map
        self.transformer = LightTransformer(
            input_dim=backbone_channels[-1],
            depth=3,
            num_heads=4,
            key_dim=32,
            attn_ratio=2
        )
        
        # Feature pyramid network
        self.fpn = LightHybridUp(backbone_channels, out_dim=head_conv)
        
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
        
        # Print input shape for debugging
        print(f"Input shape: {x.shape}")
        
        # Extract features from backbone
        features = self.backbone(x)
        
        # Print feature shapes for debugging
        print(f"Backbone feature shapes: {[f.shape for f in features]}")
        
        # Apply transformer to the last feature
        transformed_feature = self.transformer(features[-1])
        print(f"Transformed feature shape: {transformed_feature.shape}")
        
        # Replace the last feature with transformer-enhanced feature
        hybrid_features = features[:-1] + [transformed_feature]
        
        # Apply feature pyramid network
        x = self.fpn(hybrid_features)
        
        # Normalize features
        x = self.feature_norm(x)
        
        # Ensure the feature map has the expected size
        #x = self.resize_to_output_size(x)
        
        # Pass through detection heads
        ret = self.head(x, batch)
        
        return [ret]

# Function to create the lightweight hybrid model
def get_lightweight_hybrid_net(heads=None, head_conv=64, output_size=128, **kwargs):
    """
    Get Lightweight Hybrid MobileNetV4-ViT model with specified parameters
    
    Args:
        heads: Dictionary of detection heads and their output channels
        head_conv: Number of channels in the head convolution layers
        output_size: Size of the output feature map (default: 128)
        **kwargs: Extra parameters (for compatibility with other model factories)
    
    Returns:
        model: Lightweight hybrid model for detection
    """
    if 'num_layers' in kwargs:
        print(f"Note: num_layers parameter is ignored for hybrid model")
        
    model = LightweightHybridModel(heads, head_conv=head_conv, output_size=output_size)
    return model
