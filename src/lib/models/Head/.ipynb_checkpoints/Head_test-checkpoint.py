# import sys
import torch
from torch import nn
import torchvision
from opts_pose import opts
# sys.path.append(r'/home/LiLQ/glq/CenterFace.pytorch-master/src/lib/models')
from models.utils import _gather_feat, _tranpose_and_gather_feat
from models.decode import ctdet_decode
import torch.nn.functional as tf
import math
import torch.utils.checkpoint as cp






# --- CBAM Components ---

class ChannelAttention(nn.Module):
    """คำนวณ Channel Attention Map โดยใช้ Global Pooling และ Shared MLP"""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # Global Average Pooling และ Global Max Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP (ใช้ Conv2d ขนาด 1x1 แทน Fully Connected Layer)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # คำนวณผลลัพธ์จากทั้งสอง pooling ผ่าน MLP
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        # รวมผลลัพธ์เข้าด้วยกัน (บวกกันตามแบบ CBAM)
        out = avg_out + max_out
        # ผ่าน Sigmoid เพื่อให้ค่า attention อยู่ในช่วง [0, 1]
        return self.sigmoid(out) # Output shape: [B, C, 1, 1]

class SpatialAttention(nn.Module):
    """คำนวณ Spatial Attention Map โดยใช้ Pooling ข้าม Channel และ Convolution"""
    def __init__(self, kernel_size=7): # Kernel ขนาด 7x7 เป็นค่าที่นิยมใน CBAM
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        # Convolution layer รับ input ที่มี 2 channels (จาก avg และ max pooling ข้าม channel)
        # และให้ output เป็น 1 channel (spatial attention map)
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input x shape: [B, C, H, W]
        # Pooling ข้าม Channel dimension (dim=1)
        avg_out = torch.mean(x, dim=1, keepdim=True) # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True) # [B, 1, H, W]
        # Concatenate ผลลัพธ์ pooling ทั้งสองแบบตาม dimension ของ channel
        pooled = torch.cat([avg_out, max_out], dim=1) # [B, 2, H, W]
        # ผ่าน Convolution เพื่อรวมข้อมูลเชิงพื้นที่
        out = self.conv1(pooled) # [B, 1, H, W]
        # ผ่าน Sigmoid
        return self.sigmoid(out) # Output shape: [B, 1, H, W]

# --- ImprovedFeature โดยใช้ CBAM ---

class ImprovedFeature(nn.Module):
    """
    โมดูล Attention ที่รวม Channel Attention และ Spatial Attention (CBAM style)
    """
    def __init__(self, in_channels, ratio=16, spatial_kernel_size=7):
        super(ImprovedFeature, self).__init__()
        self.ca = ChannelAttention(in_channels, ratio=ratio)
        self.sa = SpatialAttention(kernel_size=spatial_kernel_size)

    def forward(self, x):
        # 1. ใช้ Channel Attention ก่อน
        ca_map = self.ca(x) # ได้ map ขนาด [B, C, 1, 1]
        # คูณ Channel Attention map เข้ากับ input feature map (Broadcasting)
        x_channel_refined = x * ca_map # หรือ x * ca_map.expand_as(x)

        # 2. ใช้ Spatial Attention กับ feature map ที่ผ่าน Channel Attention แล้ว
        sa_map = self.sa(x_channel_refined) # ได้ map ขนาด [B, 1, H, W]
        # คูณ Spatial Attention map เข้ากับ feature map ล่าสุด (Broadcasting)
        x_spatial_refined = x_channel_refined * sa_map # หรือ x_channel_refined * sa_map.expand_as(x_channel_refined)

        # ผลลัพธ์คือ feature map ที่ถูกปรับปรุงโดย attention ทั้งสองแบบ
        return x_spatial_refined




class ClassHead(nn.Module):
    def __init__(self, in_channel, classes=1):
        super(ClassHead, self).__init__()
        self.hm = nn.Sequential(
            nn.Conv2d(in_channel, classes,
                      kernel_size=1, stride=1,
                      padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.hm(x)
        return out


class BboxHead(nn.Module):
    def __init__(self, in_channel, classes=2):
        super(BboxHead, self).__init__()
        self.wh = nn.Conv2d(in_channel, classes,
                            kernel_size=1, stride=1,
                            padding=0, bias=True)

    def forward(self, x):
        out = self.wh(x)
        return out


class RegHead(nn.Module):
    def __init__(self, in_channel, classes=2):
        super(RegHead, self).__init__()
        self.reg = nn.Conv2d(in_channel, classes,
                             kernel_size=1, stride=1,
                             padding=0, bias=True)

    def forward(self, x):
        out = self.reg(x)
        return out


class LandmarkHead(nn.Module):
    def __init__(self, in_channel, classes=10):
        super(LandmarkHead, self).__init__()
        self.landmarks = nn.Conv2d(in_channel, classes,
                                   kernel_size=1, stride=1,
                                   padding=0, bias=True)

    def forward(self, x):
        out = self.landmarks(x)
        return out


class LandmarkHead_001(nn.Module):
    def __init__(self, in_channel=24, classes=5):  # 输入的in_channel是根据输入的feature的channel，class的数据是关键点的数据
        super(LandmarkHead_001, self).__init__()
        self.landmarks = nn.Sequential(
            nn.Conv2d(in_channel, classes,
                      kernel_size=1, stride=1,
                      padding=0, bias=True),
            nn.Sigmoid()  # 这一行至关重要没有的话，后面就出现nan,因为特征图出现了负数
        )

    def forward(self, x):  # 这个地方删除了一个batch
        # batch = batch
        out = self.landmarks(x)  # (512,5,64,64)
        return out


class Head(nn.Module):  # 原始版本000，这只是一个测试的Head
    def __init__(self, in_channel, heads):
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


class Head_000(nn.Module):  # Head分离版本001
    def __init__(self, in_channel, heads):
        super(Head_000, self).__init__()
        self.heads = heads
        # heads.pop('landmarks')      #这个地方就是删除landmarks的输入
        self.hw = ClassHead(in_channel=in_channel, classes=1)
        self.__setattr__('hm', self.hw)
        self.wh = BboxHead(in_channel=in_channel)
        self.__setattr__('wh', self.wh)
        self.reg = RegHead(in_channel=in_channel)
        self.__setattr__('hm_offset', self.reg)
        self.landmarks = LandmarkHead(in_channel=in_channel)
        self.__setattr__('landmarks', self.landmarks)

    def forward(self, x, feature, batch):
        batch = batch
        flag = len(batch)
        print(flag)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        '''测试代码部分'''
        '''
        if(flag>5):
            #部分就是训练的代码，使用的是batch的len来进行判断
            proposals_001 = batch['hp_one_bbox']
            landmarks_feature = real_feature(x, proposals_001)
        else:
            #这部分的代码使用的是模型输出的长宽进行判断的
            proposals_001 = proposals(self, ret, return_time=False)
            landmarks_feature =real_feature(x, proposals_001)
        ret['landmarks'] = self.__getattr__('landmarks')(landmarks_feature) #[512,5,64,64]  这个512 = 16*32   batch['hm_one_hp'] = [16,160,64,64]大小的
        '''
        return ret

'''
class Head_001(nn.Module):  # Head分离版本001
    def __init__(self, in_channel, heads):
        super(Head_001, self).__init__()
        self.heads = heads
        heads.pop('landmarks', None)  # 这个地方就是删除landmarks的输入
        self.hw = ClassHead(in_channel=in_channel, classes=1)
        self.__setattr__('hm', self.hw)
        self.wh = BboxHead(in_channel=in_channel)
        self.__setattr__('wh', self.wh)
        self.reg = RegHead(in_channel=in_channel)
        self.__setattr__('hm_offset', self.reg)
        self.landmarks = LandmarkHead_001(in_channel=in_channel)
        self.__setattr__('landmarks', self.landmarks)
      

    def forward(self, x, batch):
        batch = batch
        flag = len(batch)
        # print(flag)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        if (flag > 5):
            # 部分就是训练的代码，使用的是batch的len来进行判断
            proposals_001 = batch['hp_one_bbox'] # candidate faces
            landmarks_feature = real_feature(x, proposals_001) # RoIAlign
            landmarks_feature = polar(landmarks_feature, 64, 64)
           

        else:
            # 这部分的代码使用的是模型输出的长宽进行判断的
            proposals_001 = proposals(self, ret, return_time=False)
            landmarks_feature = real_feature(x, proposals_001)
            landmarks_feature = polar(landmarks_feature, 64, 64)  # 这个64是根据自己设置的大小设计的
            

        ret['landmarks'] = self.__getattr__('landmarks')(
            landmarks_feature)  # [512,5,64,64]  这个512 = 16*32   batch['hm_one_hp'] = [16,160,64,64]大小的
        return ret
'''

class Head_001(nn.Module):  # Head分离版本001
    def __init__(self, in_channel, heads):
        super(Head_001, self).__init__()
        self.heads = heads
        heads.pop('landmarks', None)  # 这个地方就是删除landmarks的输入
        self.hw = ClassHead(in_channel=in_channel, classes=1)
        self.__setattr__('hm', self.hw)
        self.wh = BboxHead(in_channel=in_channel)
        self.__setattr__('wh', self.wh)
        self.reg = RegHead(in_channel=in_channel)
        self.__setattr__('hm_offset', self.reg)
        self.landmarks = LandmarkHead_001(in_channel=in_channel)
        self.__setattr__('landmarks', self.landmarks)
        self.attention = ImprovedFeature(in_channel)  # Initialize Self-Attention module
        # Note: adjust the input size (in_dim) as needed.

    def forward(self, x, batch):
        batch = batch
        flag = len(batch)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        '''测试代码部分'''
        if (flag > 5):
            # 部分就是训练的代码，使用的是batch的len来进行判断 -This part is the training code, which uses the length of the batch for decision-making.
            proposals_001 = batch['hp_one_bbox'] # candidate faces
            landmarks_feature = real_feature(x, proposals_001) # RoIAlign
            landmarks_feature = polar(landmarks_feature, 64, 64)

        else:
            # 这部分的代码使用的是模型输出的长宽进行判断的
            proposals_001 = proposals(self, ret, return_time=False)
            landmarks_feature = real_feature(x, proposals_001)
            landmarks_feature = polar(landmarks_feature, 64, 64)  # 这个64是根据自己设置的大小设计的

        # Apply self-attention to landmarks_feature
        landmarks_feature = self.attention(landmarks_feature)

        ret['landmarks'] = self.__getattr__('landmarks')(
            landmarks_feature)  # [512,5,64,64]  这个512 = 16*32   batch['hm_one_hp'] = [16,160,64,64]大小的
        return ret

class Head_002(nn.Module):
    def __init__(self, in_channel, heads):
        super(Head_001, self).__init__()
        self.heads = heads
        heads.pop('landmarks')  # 这个地方就是删除landmarks 使heads={'hm','wh','hm_offset'}
        self.hw = ClassHead(in_channel=in_channel, classes=1)
        self.__setattr__('hm', self.hw)
        self.wh = BboxHead(in_channel=in_channel)
        self.__setattr__('wh', self.wh)
        self.reg = RegHead(in_channel=in_channel)
        self.__setattr__('hm_offset', self.reg)
        # self.landmarks = LandmarkHead_001(in_channel=in_channel)
        # self.__setattr__('landmarks',self.landmarks)

    def forward(self, x, batch):
        batch = batch
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        '''
        这个地方开始插入要处理的函数
        '''

        # proposals(self,ret,return_time=False)
        proposals_001 = proposals(self, ret, return_time=False)
        landmarks_feature = real_feature(x, proposals_001)

        return ret


'''
    以下部分的代码是用来模型输出代码的
'''


def polar(images, target_width: int, target_height: int) -> torch.Tensor:
    image_height, image_width = images.size()[-2:]
    roi_center = torch.tensor([image_height / 2.0, image_width / 2.0])
    # roi_center = images.size()[-2:] / 2.0
    grids = torch.zeros(images.size()[:1] + (target_height, target_width, 2), dtype=images.dtype,device=images.device)
    cx = image_height / 2
    cy = image_width / 2
    maxlen = math.sqrt(cx * cx + cy * cy)
    warped_radii = torch.arange(0, maxlen, maxlen / target_width, dtype=grids.dtype,device=grids.device).unsqueeze(-1).expand(
        (target_height, target_width))
    thetas = torch.arange(0.0, 2.0 * math.pi, 2.0 * math.pi / target_height, dtype=grids.dtype,device=grids.device).unsqueeze(0).expand(
        (target_height, target_width))

    orientation_x = torch.cos(thetas)
    orientation_y = torch.sin(thetas)

    warped_x_indices = warped_radii * orientation_x
    warped_y_indices = warped_radii * orientation_y
    grids[..., 0] = (roi_center[0] + warped_x_indices) / (image_width - 1.0) * 2.0 - 1.0
    grids[..., 1] = (roi_center[1] + warped_y_indices) / (image_height - 1.0) * 2.0 - 1.0

    return tf.grid_sample(images, grids, mode='bilinear', padding_mode='zeros', align_corners=True)


'''
def polar(images, target_width: int, target_height: int) -> torch.Tensor:
    image_height, image_width = images.size()[-2:]
    roi_center = torch.tensor([image_height / 2.0, image_width / 2.0])

    grids = torch.zeros(images.size()[:1] + (target_height, target_width, 2), dtype=images.dtype, device=images.device)
    cx = image_height / 2
    cy = image_width / 2
    maxlen = math.sqrt(cx * cx + cy * cy)
    # Change to log-polar
    warped_radii = torch.logspace(0, math.log(maxlen), target_width, base=math.exp(1), dtype=grids.dtype, device=grids.device).unsqueeze(-1).expand(
        (target_height, target_width))
    thetas = torch.linspace(0.0, 2.0 * math.pi, target_height, dtype=grids.dtype, device=grids.device).unsqueeze(0).expand(
        (target_height, target_width))

    orientation_x = torch.cos(thetas)
    orientation_y = torch.sin(thetas)

    warped_x_indices = warped_radii * orientation_x
    warped_y_indices = warped_radii * orientation_y
    grids[..., 0] = (roi_center[0] + warped_x_indices) / (image_width - 1.0) * 2.0 - 1.0
    grids[..., 1] = (roi_center[1] + warped_y_indices) / (image_height - 1.0) * 2.0 - 1.0

    return torch.nn.functional.grid_sample(images, grids, mode='bilinear', padding_mode='zeros', align_corners=True)
'''
def proposals(self, output, return_time=False):  # 输出head处理的代码，获取head输出的hm,wh和reg的预测数据，并且输入到ctdet_docode部分
    with torch.no_grad():
        hm = output['hm']
        wh = output['wh']

        reg = output['hm_offset'] if True else None  # self.opt.reg_offset opt里面的值就是True
        dets = ctdet_decode(hm, wh, reg=reg, K=600)  # 这个地方的k要自己进行设置后面看看可不可已自己引入整体函数
    return dets


def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)  # 3 * 3 区域的最大值滤波,得到中心点的数值

    scores, inds, clses, ys, xs = _topk(heat, K=K)  # 选择置信度最高的K个 （scores置信度，inds一维坐标，clses类别，ys,xs二维上的坐标）
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds)  # 人脸矩形框的宽高
    if cat_spec_wh:
        wh = wh.view(batch, K, cat, 2)
        clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
        wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
        wh = wh.view(batch, K, 2)  # wh的第一种方式
        wh = wh.exp() * 4.  # wh按照centerface论文的修改方式，还原预处理的数据到wh的格式
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    # 这里就是选择最大的wh
    pred_wh, idx_wh = wh.max(2)
    pred_wh = pred_wh.unsqueeze(-1)  # [1,200,1]
    # print(max(wh[..., 0:1],wh[..., 1:2]))
    bboxes = torch.cat([xs - pred_wh / 2,
                        ys - pred_wh / 2,
                        xs + pred_wh / 2,
                        ys + pred_wh / 2], dim=2)
    '''
    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)
    '''
    # detections = torch.cat([bboxes, scores, clses], dim=2)      # box:4+score:1+class:1=6 [1,200,6] [batch,top_k,数据信息]
    detections = torch.cat([bboxes], dim=2)  # [1.200.4]  这个地方就获得了proposals
    return detections


'''
    这里是数据处理函数
'''


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)  # 前100个点，变成1，80，128*128->1.80.k

    topk_inds = topk_inds % (height * width)
    # topk_ys   = (topk_inds / width).int().float()
    topk_ys = (torch.true_divide(topk_inds, width)).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    # topk_clses = (topk_ind / K).int()
    topk_clses = (torch.true_divide(topk_ind, K)).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _left_aggregate(heat):
    '''
        heat: batchsize x channels x h x w
    '''
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(1, heat.shape[0]):
        inds = (heat[i] >= heat[i - 1])
        ret[i] += ret[i - 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape)


def _right_aggregate(heat):
    '''
        heat: batchsize x channels x h x w
    '''
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(heat.shape[0] - 2, -1, -1):
        inds = (heat[i] >= heat[i + 1])
        ret[i] += ret[i + 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape)


def _top_aggregate(heat):
    '''
        heat: batchsize x channels x h x w
    '''
    heat = heat.transpose(3, 2)
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(1, heat.shape[0]):
        inds = (heat[i] >= heat[i - 1])
        ret[i] += ret[i - 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape).transpose(3, 2)


def _bottom_aggregate(heat):
    '''
        heat: batchsize x channels x h x w
    '''
    heat = heat.transpose(3, 2)
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(heat.shape[0] - 2, -1, -1):
        inds = (heat[i] >= heat[i + 1])
        ret[i] += ret[i + 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape).transpose(3, 2)


def _h_aggregate(heat, aggr_weight=0.1):
    return aggr_weight * _left_aggregate(heat) + \
           aggr_weight * _right_aggregate(heat) + heat


def _v_aggregate(heat, aggr_weight=0.1):
    return aggr_weight * _top_aggregate(heat) + \
           aggr_weight * _bottom_aggregate(heat) + heat


'''
    这部分是RPN的代码处理部分
'''


def real_feature(feature, detections):
    # batch, channels, height, width = feature.size()
    batch, top_k, box = detections.size()
    boxes = []
    for k in range(batch):  # 将获得的bbox信息转化为proposal的形式
        boxes.append(detections[k])
    pooler = torchvision.ops.RoIAlign(output_size=64, sampling_ratio=1, spatial_scale=1)
    output = pooler(feature, boxes)  # [200,1,64,64] batch
    return output
