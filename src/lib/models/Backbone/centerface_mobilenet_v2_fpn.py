import sys, os
path = '/home/kunanon.k/Sakon_works/faceDetections/RPNetPlus/src_original/lib/models'
sys.path.append(path)
from torch import nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import math
from Head.Head_test import Head
from Head.Head_test import Head_001
import numpy as np
from torchsummary import summary

__all__ = ['MobileNetV4']

# MobileNetV4 doesn't have a standard pretrained model URL yet
# You'll need to update this when available
model_urls = {
    'mobilenet_v4': 'https://download.pytorch.org/models/mobilenet_v4-placeholder.pth',
}

def _make_divisible(v, divisor, min_value=None):
    """
    This function ensures that all layers have a channel number that is divisible by divisor
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNSiLU(nn.Sequential):
    """
    Modernized Conv-BN-SiLU block for MobileNetV4
    Uses SiLU (Swish) activation instead of ReLU6
    """
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNSiLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.SiLU(inplace=True)  # Using SiLU activation as in MobileNetV4
        )

class SELayer(nn.Module):
    """
    Squeeze-and-Excitation block for MobileNetV4
    """
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MBConvBlock(nn.Module):
    """
    MBConv (Mobile Block Convolutional) with SE (Squeeze-and-Excitation)
    This is the core building block of MobileNetV4
    """
    def __init__(self, inp, oup, stride, expand_ratio, use_se=True):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        # Expansion phase
        if expand_ratio != 1:
            layers.append(ConvBNSiLU(inp, hidden_dim, kernel_size=1))
            
        # Depthwise convolution
        layers.extend([
            # Depthwise conv
            ConvBNSiLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
        ])
        
        # SE layers
        if use_se:
            layers.append(SELayer(hidden_dim))
            
        # Projection phase
        layers.extend([
            # Pointwise linear projection
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV4(nn.Module):
    def __init__(self, width_mult=1.0, round_nearest=8):
        super(MobileNetV4, self).__init__()
        block = MBConvBlock
        input_channel = 32
        # MobileNetV4 has a different configuration with more blocks
        # t: expansion factor, c: output channels, n: number of blocks, s: stride
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],  # 0
            [6, 24, 2, 2],  # 1
            [6, 32, 3, 2],  # 2
            [6, 64, 4, 2],  # 3
            [6, 96, 3, 1],  # 4
            [6, 160, 3, 2], # 5
            [6, 320, 1, 1], # 6
        ]
        
        # Feature extraction points for FPN
        self.feat_id = [1, 2, 4, 6]
        self.feat_channel = []

        # Initial stem conv
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        features = [ConvBNSiLU(3, input_channel, stride=2)]

        # Building inverted residual blocks
        for id, (t, c, n, s) in enumerate(inverted_residual_setting):
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, use_se=True))
                input_channel = output_channel
                
            if id in self.feat_id:
                self.__setattr__("feature_%d" % id, nn.Sequential(*features))
                self.feat_channel.append(output_channel)
                features = []

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
        y = []
        for id in self.feat_id:
            x = self.__getattr__("feature_%d" % id)(x)
            y.append(x)
        return y

def load_model(model, state_dict):
    new_model = model.state_dict()
    new_keys = list(new_model.keys())
    old_keys = list(state_dict.keys())
    restore_dict = OrderedDict()
    for id in range(len(new_keys)):
        restore_dict[new_keys[id]] = state_dict[old_keys[id]]
    model.load_state_dict(restore_dict)

def dict2list(func):
    def wrap(*args, **kwargs):
        self = args[0]
        x = args[1]
        ret_list = []
        ret = func(self, x)
        for k, v in ret[0].items():
            ret_list.append(v)
        return ret_list
    return wrap

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

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class IDAUp(nn.Module):
    def __init__(self, out_dim, channel):
        super(IDAUp, self).__init__()
        self.out_dim = out_dim
        self.up = nn.Sequential(
                    nn.ConvTranspose2d(
                        out_dim, out_dim, kernel_size=2, stride=2, padding=0,
                        output_padding=0, groups=out_dim, bias=False),
                    nn.BatchNorm2d(out_dim, eps=0.001, momentum=0.1),
                    nn.SiLU(inplace=True))  # Changed to SiLU
        self.conv = nn.Sequential(
                    nn.Conv2d(channel, out_dim,
                              kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(out_dim, eps=0.001, momentum=0.1),
                    nn.SiLU(inplace=True))  # Changed to SiLU

    def forward(self, layers):
        layers = list(layers)
        x = self.up(layers[0])
        y = self.conv(layers[1])
        out = x + y
        return out

class MobileNetUp(nn.Module):
    def __init__(self, channels, out_dim=24):
        super(MobileNetUp, self).__init__()
        channels = channels[::-1]
        self.conv = nn.Sequential(
                    nn.Conv2d(channels[0], out_dim,
                              kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(out_dim, eps=0.001, momentum=0.1),
                    nn.SiLU(inplace=True))  # Changed to SiLU
        self.conv_last = nn.Sequential(
                    nn.Conv2d(out_dim, out_dim,
                              kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(out_dim, eps=1e-5, momentum=0.01),
                    nn.SiLU(inplace=True))  # Changed to SiLU

        for i, channel in enumerate(channels[1:]):
            setattr(self, 'up_%d' % (i), IDAUp(out_dim, channel))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                fill_up_weights(m)

    def forward(self, layers):
        layers = list(layers)
        assert len(layers) > 1
        x = self.conv(layers[-1])

        for i in range(0, len(layers)-1):
            up = getattr(self, 'up_{}'.format(i))
            x = up([x, layers[len(layers)-2-i]])
        x = self.conv_last(x)
        return x

class MobileNetSeg(nn.Module):
    def __init__(self, base_name, heads, head_conv=24, pretrained=True):
        super(MobileNetSeg, self).__init__()
        self.heads = heads
        self.base = globals()[base_name](pretrained=pretrained)
        channels = self.base.feat_channel
        self.dla_up = MobileNetUp(channels, out_dim=head_conv)
        self.head = Head_001(in_channel=head_conv, heads=heads)

    def forward(self, batch):
        batch = batch
        x = batch['input']
        x = self.base(x)
        x = self.dla_up(x)
        ret = self.head(x, batch)
        return [ret]

def mobilenetv4_10(pretrained=False, **kwargs):
    """
    Constructs a MobileNetV4 model with width_mult=1.0
    """
    model = MobileNetV4(width_mult=1.0)
    if pretrained:
        # Pretrained weights not available for MobileNetV4 yet
        print('Pretrained weights for MobileNetV4 are not available yet.')
    return model

def mobilenetv4_5(pretrained=False, **kwargs):
    """
    Constructs a MobileNetV4 model with width_mult=0.5
    """
    model = MobileNetV4(width_mult=0.5)
    if pretrained:
        print('Pretrained weights for MobileNetV4 are not available yet.')
    return model

def get_mobile_net(num_layers, heads, head_conv=24):
    """
    Get MobileNetV4 model with specified number of layers and heads
    """
    model = MobileNetSeg('mobilenetv4_{}'.format(num_layers), heads,
                  pretrained=False,  # Set to False since we don't have pretrained weights yet
                  head_conv=head_conv)
    return model

if __name__ == '__main__':
    import torch
    input = torch.zeros([1, 3, 416, 416])
    # Create model with MobileNetV4
    model = get_mobile_net(10, {'hm': 1, 'hm_offset': 2, 'wh': 2, 'landmarks': 10}, head_conv=24)
    
    # Create a dummy batch with 'input' key
    batch = {'input': input}
    res = model(batch)
    print(res[0].keys())  # Print keys of the output dictionary