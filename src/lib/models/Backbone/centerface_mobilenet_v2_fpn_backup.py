import sys,os
path = '/home/kunanon.k/Sakon_works/faceDetections/RPNetPlus/src_original/lib/models'
sys.path.append(path)
#sys.path.append(r'/home/LiLQ/glq/CenterFace.pytorch-master/src/lib/models')
#sys.path.append(r'/home/LiLQ/glq/CenterFace.pytorch-master/src/lib/models')
from torch import nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import math
#from Head.Head_test import Head
from Head.Head_test import Head
from Head.Head_test import Head_001          #修改查看引入的代码
import numpy as np
from torchsummary import summary
__all__ = ['MobileNetV2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):  #group=1是普通的卷积
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module): #倒残差结构
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # 3*3 dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # 1*1 pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,width_mult=1.0,round_nearest=8,):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1], # 0
            [6, 24, 2, 2], # 1
            [6, 32, 3, 2], # 2
            [6, 64, 4, 2], # 3
            [6, 96, 3, 1], # 4
            [6, 160, 3, 2],# 5
            [6, 320, 1, 1],# 6
        ]
        #self.feat_id = [,4,6]
        self.feat_id = [1,2,4,6]
        self.feat_channel = []

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]  #定义一个特征层，并且添加第一个特征层

        # building inverted residual blocks
        for id,(t, c, n, s) in enumerate(inverted_residual_setting):
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
                # if的这一部分是挑选使用的特征层进行fpn,这一部分可以进行仿写来选择浅层的特征，id就是使用的特定的层
            if id in self.feat_id:
                self.__setattr__("feature_%d"%id,nn.Sequential(*features))
                self.feat_channel.append(output_channel)
                features = []

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # 3、这个地方将fpn要使用的特征层存入y中
    def forward(self, x):
        y = []
        for id in self.feat_id:
            x = self.__getattr__("feature_%d"%id)(x)
            #print(x.shape)
            y.append(x)
        return y


'''这是自己定义的head class'''
'''
class Head(nn.Module):
    def __init__(self, in_channel, out_channel, heads):
        super(Head, self).__init__()
        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'hm':
                fc = nn.Sequential(
                    nn.Conv2d(in_channel, classes,
                              kernel_size=1, stride=1,
                              padding=0, bias=True),
                    nn.Sigmoid()
                )

            else:
                fc = nn.Conv2d(in_channel, classes,
                               kernel_size=1, stride=1,
                               padding=0, bias=True)
            self.__setattr__(head, fc)

    def forward(self, x):
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return ret
'''
def load_model(model,state_dict):
    new_model=model.state_dict()
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


class IDAUp(nn.Module):   #进行上采样
    def __init__(self, out_dim, channel):
        super(IDAUp, self).__init__()
        self.out_dim = out_dim
        self.up = nn.Sequential(
                    nn.ConvTranspose2d(  #进行特征层的上采样运算
                        out_dim, out_dim, kernel_size=2, stride=2, padding=0,
                        output_padding=0, groups=out_dim, bias=False),
                    nn.BatchNorm2d(out_dim,eps=0.001,momentum=0.1),
                    nn.ReLU())
        self.conv =  nn.Sequential(
                    nn.Conv2d(channel, out_dim,
                              kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(out_dim,eps=0.001,momentum=0.1),
                    nn.ReLU(inplace=True))
        # self.smooth = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, layers):
        layers = list(layers)
        x = self.up(layers[0])       #这一步就是进行上采样
        #print(x.shape)
        y = self.conv(layers[1])
        # out = self.smooth(x + y)
        out = x + y                 #这里进行的就是特征融合，实际上真的进行了四层的特征融合
        #print("out:",out.shape)
        return out

class MobileNetUp(nn.Module):   #实现上采样，获得最后的高精度的特征层
    def __init__(self, channels, out_dim = 24):
        super(MobileNetUp, self).__init__()
        channels =  channels[::-1]
        self.conv =  nn.Sequential(
                    nn.Conv2d(channels[0], out_dim,
                              kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(out_dim,eps=0.001,momentum=0.1),
                    nn.ReLU(inplace=True))
        self.conv_last =  nn.Sequential(
                    nn.Conv2d(out_dim,out_dim,
                              kernel_size=3, stride=1, padding=1 ,bias=False),
                    nn.BatchNorm2d(out_dim,eps=1e-5,momentum=0.01),
                    nn.ReLU(inplace=True))

        for i,channel in enumerate(channels[1:]):
            setattr(self,'up_%d'%(i),IDAUp(out_dim,channel)) #这里跳入IDAUp中去

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.ConvTranspose2d):
                fill_up_weights(m)   #这里跳入到

    def forward(self, layers):
        layers = list(layers) #layer传入的就是经过mobilenet提取到要进行fpn的特征层的内容
        assert len(layers) > 1
        x = self.conv(layers[-1])

        for i in range(0,len(layers)-1):
            up = getattr(self, 'up_{}'.format(i))
            x = up([x,layers[len(layers)-2-i]])    #进行特征融合所需要的上采样
        x = self.conv_last(x)
        return x               #此时输出的就不是一个数组，而是一个特定的特征层


class MobileNetSeg(nn.Module):
    def __init__(self, base_name,heads,head_conv=24, pretrained = True):
        super(MobileNetSeg, self).__init__()
        self.heads = heads
        self.base = globals()[base_name](
            pretrained=pretrained)   #1、这里跳入mobilenetv2_10里面选择使用的是mobilenetv2_10的网络结构 Here, we jump into mobilenetv2_10 and choose to use the mobilenetv2_10 network structure
        channels = self.base.feat_channel
        self.dla_up = MobileNetUp(channels, out_dim=head_conv)  #5、这里跳入网络的MobileNetUp,对特征层进行上采样 Here, we enter the MobileNetUp network to perform upsampling on the feature layers
        self.head = Head_001(in_channel=head_conv,heads = heads)  #这里就是修改调整的地方 - This is the place for modifications and adjustments.
        '''
        for head in self.heads:
            classes = self.heads[head]
            if head == 'hm':
                fc = nn.Sequential(
                    nn.Conv2d(head_conv, classes,
                              kernel_size=1, stride=1,
                              padding=0, bias=True),
                    nn.Sigmoid()
                )

            else:
                fc = nn.Conv2d(head_conv, classes,
                              kernel_size=1, stride=1,
                              padding=0, bias=True)
            # if 'hm' in head:
            #     fc.bias.data.fill_(-2.19)
            # else:
            #     nn.init.normal_(fc.weight, std=0.001)
            #     nn.init.constant_(fc.bias, 0)
            self.__setattr__(head, fc)
            '''
    # @dict2list         # 转onnx的时候需要将输出由dict转成list模式
    def forward(self, batch):
        batch = batch
        x = batch['input']  #这里只是使用了图片（input）-Here, only the image (input) is used.

        x = self.base(x)
        x = self.dla_up(x)    #此时输出的x就是一个特征层[8,64,200,200],这个特征层进行了特征的融合 -At this point, the output x is a feature layer [8,64,200,200], and this feature layer has undergone feature fusion.
        #print("x:",x.shape)

        ret = self.head(x,batch)
        #ret = self.head(x)
        return [ret]

        '''
        ret = {}              #建立一个新的空白的数组，这个数组的目的就是存储不同的head数组
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)   #这个x实际上就是要进行处理的最后的特征层，这个特征层的大小已经变成了原来的四分之一
        return [ret]
        '''

def mobilenetv2_10(pretrained=True, **kwargs):
    model = MobileNetV2(width_mult=1.0)   #2、跳入到真正的网络（MobileNetV2）中进行连接
    #summary(model,(3,800,800))            #这一行代码用来打印出网络的out_shape
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['mobilenet_v2'],
                                              progress=True)
        load_model(model,state_dict)     #4、这里要进入load_model里面进行与训练参数的加载
    return model

def mobilenetv2_5(pretrained=False, **kwargs):
    model = MobileNetV2(width_mult=0.5)
    if pretrained:
        print('This version does not have pretrain weights.')
    return model

# num_layers  : [10 , 5]
def get_mobile_net(num_layers, heads, head_conv=24):
  model = MobileNetSeg('mobilenetv2_{}'.format(num_layers), heads,
                 pretrained=True,
                 head_conv=head_conv)
  #summary(model, input_size=(3, 800, 800)) 这里也不行，也不输出outshape
  return model


if __name__ == '__main__':
    import torch
    input = torch.zeros([1,3,416,416])
    #这个地方就是设置出自己的模型
    model = get_mobile_net(10,{'hm':1, 'hm_offset':2, 'wh':2, 'landmarks':10},head_conv=24)           # hm reference for the classes of objects//这个头文件只能做矩形框检测
    res = model(input)
    print(res.shape)
