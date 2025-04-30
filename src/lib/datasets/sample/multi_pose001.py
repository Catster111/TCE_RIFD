from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
from utils.utils import Data_anchor_sample
from utils.Randaugmentations import Randaugment
import math
from PIL import Image
import re
from torch._six import container_abcs, string_classes, int_classes

np_str_obj_array_pattern = re.compile(r'[SaUO]')

'''
预处理数据集，该类的主要功能是主要实现_getitem_方法，以便在train的时候被DataLoader加载
'''

class MultiPoseDataset(data.Dataset):   #将[x1,y1(左下角的坐标)，w,h]-->[x1,y1,x2,y2]
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

  def __getitem__(self, index):

    #读入图片和标注信息
    #img_id = self.images[index]
    img_id=45  #这个当时就是学习这个文件的时候固定了id为1
    file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
    img_path = os.path.join(self.img_dir, file_name)
    ann_ids = self.coco.getAnnIds(imgIds=[img_id]) #根据图片的id获取标注信息
    anns = self.coco.loadAnns(ids=ann_ids)        #这里获得的是所有的标注信息


    num_objs = len(anns)
    # num_objs = min(len(anns), self.max_objs)
    if num_objs > self.max_objs:    #这里的设置就是一些标注信息的比max_objs多的时候，只选择32，随机选择32个
        num_objs = self.max_objs
        anns = np.random.choice(anns, num_objs)

    img = cv2.imread(img_path) #读取图片

    img, anns = Data_anchor_sample(img, anns)   #处理一下图片的大小和标注信息的大小


    '''
        resize图片开始
    '''
    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)  #C就是图片的中心点
    s = max(img.shape[0], img.shape[1]) * 1.0   #找到最大的边
    rot = 0          #旋转角度

    flipped = False
    if self.split == 'train':
      if not self.opt.not_rand_crop:
        # s = s * np.random.choice(np.arange(0.8, 1.1, 0.1))
        s = s
        # _border = np.random.randint(128*0.4, 128*1.4)
        _border = s * np.random.choice([0.1, 0.2, 0.25])
        w_border = self._get_border(_border, img.shape[1])
        h_border = self._get_border(_border, img.shape[0])
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)    #修改中心的位置
      else:
        sf = self.opt.scale
        cf = self.opt.shift
        c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      if np.random.random() < self.opt.aug_rot:
        rf = self.opt.rotate
        rot = np.clip(np.random.randn()*rf, -rf*2, rf*2)

      if np.random.random() < self.opt.flip:
        flipped = True
        img = img[:, ::-1, :]
        c[0] =  width - c[0] - 1

    trans_input = get_affine_transform(
      c, s, rot, [self.opt.input_res, self.opt.input_res])
    inp = cv2.warpAffine(img, trans_input, 
                         (self.opt.input_res, self.opt.input_res),
                         flags=cv2.INTER_LINEAR)

    inp = (inp.astype(np.float32) / 255.)                    #归一化
    if self.split == 'train' and not self.opt.no_color_aug:                 # 随机进行图片增强
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
      # inp = Randaugment(self._data_rng, inp, self._eig_val, self._eig_vec)

    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)


    '''
    resize图片结束
    '''
    output_res = self.opt.output_res  #已经进行了下采样的缩放
    '''
    opt.output_h = opt.input_h // opt.down_ratio
    opt.output_w = opt.input_w // opt.down_ratio
    opt.input_res = max(opt.input_h, opt.input_w)
    opt.output_res = max(opt.output_h, opt.output_w)
    '''
    num_joints = self.num_joints
    trans_output_rot = get_affine_transform(c, s, rot, [output_res, output_res])  #获得变换矩阵（就是将原图缩放到128*128）

    trans_output = get_affine_transform(c, s, 0, [output_res, output_res])

    hm = np.zeros((self.num_classes, output_res, output_res), dtype=np.float32)   #heatmap的空白盒子[1,128,128]

    hm_hp = np.zeros((num_joints, output_res, output_res), dtype=np.float32)        #关键点的空白盒子[5,128,128]
    dense_kps = np.zeros((num_joints, 2, output_res, output_res),   #[5,2,128,128]  空白盒子感觉这个放的使关键点的坐标（猜测）
                          dtype=np.float32)
    dense_kps_mask = np.zeros((num_joints, output_res, output_res),  #[5,128,128]  空白盒子掩码不晓得是干啥的
                               dtype=np.float32)

    wh = np.zeros((self.max_objs, 2), dtype=np.float32)   #宽高的空白盒子  [32,2]
    kps = np.zeros((self.max_objs, num_joints * 2), dtype=np.float32)    #空白盒子[32,10]
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)  #中心点偏置的回归 空白盒子[32,2]
    ind = np.zeros((self.max_objs), dtype=np.int64)       #保存wh和reg 索引[32]
    #空白盒子和索引地址

    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)  #掩码 [32]
    wight_mask = np.ones((self.max_objs), dtype=np.float32)     #wight掩码，全是1[32]
    kps_mask = np.zeros((self.max_objs, self.num_joints * 2), dtype=np.uint8)  #[32,10]掩码
    hp_offset = np.zeros((self.max_objs * num_joints, 2), dtype=np.float32)  #[32*5,2]
    hp_ind = np.zeros((self.max_objs * num_joints), dtype=np.int64) #[32*5]
    hp_mask = np.zeros((self.max_objs * num_joints), dtype=np.int64) #[32*5] 掩码

    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian

    gt_det = []
    for k in range(num_objs):
      ann = anns[k]   #获得第K个的标注信息
      bbox = self._coco_box_to_bbox(ann['bbox'])  #转化为我们需要的bbox的坐标形式[x1,y1,x2,y2],这个地方就是存储一个bbox
      cls_id = int(ann['category_id']) - 1         #获取得到的类别信息
      pts = np.array(ann['keypoints'], np.float32).reshape(num_joints, 3) #单个ann的关键点的信息
      if flipped:                       #flipped翻转这个地方进行翻转的判断，flipped=false(没有启动)
        bbox[[0, 2]] = width - bbox[[2, 0]] - 1
        pts[:, 0] = width - pts[:, 0] - 1
        for e in self.flip_idx:
          pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()

        ##把对应的gt也进行相应的transform
      bbox[:2] = affine_transform(bbox[:2], trans_output)      #将原本的坐标缩放到128（原论文中使用的512的大小）的尺寸上，这时候标注信息存在问题进行修正
      bbox[2:] = affine_transform(bbox[2:], trans_output)
      bbox = np.clip(bbox, 0, output_res - 1)                  #避免坐标超出缩放后的大小
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]              #找出新的w,h

      if (h > 0 and w > 0) or (rot != 0):
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))   #
        radius = self.opt.hm_gauss if self.opt.mse_loss else max(0, int(radius)) 
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)       # 人脸的中心坐标
        ct_int = ct.astype(np.int32)                        # 整数化
        # wh[k] = 1. * w, 1. * h                                                    # 2. centernet的方式
        wh[k] = np.log(1. * w / 4), np.log(1. * h / 4)                              # 2. 人脸bbox的高度和宽度,centerface论文的方式
        ind[k] = ct_int[1] * output_res + ct_int[0]         # 人脸bbox在1/4特征图中的索引
        reg[k] = ct - ct_int                                # 3. 人脸bbox中心点整数化的偏差
        reg_mask[k] = 1                                     # 是否需要用于计算误差，掩码用于选择数据
        # if w*h <= 20:
        #     wight_mask[k] = 15

        num_kpts = pts[:, 2].sum()                           # 没有关键点标注的时哦
        if num_kpts == 0:                                    # 没有关键点标注的都是比较困难的样本
        #if num_kpts == 0 or w * h <= 8:
            # print('没有关键点标注')
          hm[cls_id, ct_int[1], ct_int[0]] = 0.9999   #如果没有关键点信息的话hm的对应值就设置为0.9999
          # reg_mask[k] = 0

        hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        hp_radius = self.opt.hm_gauss \
                    if self.opt.mse_loss else max(0, int(hp_radius))



        for j in range(num_joints):        #这个函数的作用就是得到整张图的关键点的坐标点的所有的信息
          if pts[j, 2] > 0:
            pts[j, :2] = affine_transform(pts[j, :2], trans_output_rot)    #关键点坐标缩放到128的尺寸上去
            if pts[j, 0] >= 0 and pts[j, 0] < output_res and \
               pts[j, 1] >= 0 and pts[j, 1] < output_res:
              kps[k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int                # 4. 关键点相对于人脸bbox的中心的偏差
              kps_mask[k, j * 2: j * 2 + 2] = 1
              pt_int = pts[j, :2].astype(np.int32)                          # 关键点整数化
              hp_offset[k * num_joints + j] = pts[j, :2] - pt_int           # 关键点整数化的偏差
              hp_ind[k * num_joints + j] = pt_int[1] * output_res + pt_int[0]   # 索引
              hp_mask[k * num_joints + j] = 1                                   # 计算损失的mask,这个值就是计算那里有值
              if self.opt.dense_hp:
                # must be before draw center hm gaussian
                draw_dense_reg(dense_kps[j], hm[cls_id], ct_int, 
                               pts[j, :2] - ct_int, radius, is_offset=True)
                draw_gaussian(dense_kps_mask[j], ct_int, radius)
              draw_gaussian(hm_hp[j], pt_int, hp_radius)                    # 1. 关键点高斯map eg:一张图上面有多少个人脸就有多少个鼻子的关键点（如果标注信息中有的话）
              if ann['bbox'][2]*ann['bbox'][3] <= 16.0:                   # 太小的人脸忽略
                kps_mask[k, j * 2: j * 2 + 2] = 0                                      ###
        draw_gaussian(hm[cls_id], ct_int, radius)                           #heatmap人脸框的heatmap
        gt_det.append([ct[0] - w / 2, ct[1] - h / 2, 
                       ct[0] + w / 2, ct[1] + h / 2, 1] + 
                       pts[:, :2].reshape(num_joints * 2).tolist() + [cls_id])
    if rot != 0:
      hm = hm * 0 + 0.9999
      reg_mask *= 0
      kps_mask *= 0
    ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh,
           'landmarks': kps, 'hps_mask': kps_mask, 'wight_mask': wight_mask}
    if self.opt.dense_hp:
      dense_kps = dense_kps.reshape(num_joints * 2, output_res, output_res)
      dense_kps_mask = dense_kps_mask.reshape(
        num_joints, 1, output_res, output_res)
      dense_kps_mask = np.concatenate([dense_kps_mask, dense_kps_mask], axis=1)
      dense_kps_mask = dense_kps_mask.reshape(
        num_joints * 2, output_res, output_res)
      ret.update({'dense_hps': dense_kps, 'dense_hps_mask': dense_kps_mask})
      del ret['hps'], ret['hps_mask']
    if self.opt.reg_offset:
      ret.update({'hm_offset': reg})                  # 人脸bbox中心点整数化的偏差
    if self.opt.hm_hp:
      ret.update({'hm_hp': hm_hp})
    if self.opt.reg_hp_offset:
      ret.update({'hp_offset': hp_offset, 'hp_ind': hp_ind, 'hp_mask': hp_mask})
    if self.opt.debug > 0 or not self.split == 'train':
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, 40), dtype=np.float32)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
      ret['meta'] = meta
    return ret


_use_shared_memory = False

error_msg_fmt = "batch must contain tensors, numbers, dicts or lists; found {}"

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(error_msg_fmt.format(elem.dtype))

            return default_collate([torch.from_numpy(b) for b in batch])
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(batch[0], int_classes):
        return torch.tensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], tuple) and hasattr(batch[0], '_fields'):  # namedtuple
        return type(batch[0])(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(batch[0], container_abcs.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError((error_msg_fmt.format(type(batch[0]))))


def multipose_collate(batch):
  objects_dims = [d.shape[0] for d in batch]
  index = objects_dims.index(max(objects_dims))

  # one_dim = True if len(batch[0].shape) == 1 else False
  res = []
  for i in range(len(batch)):
      tres = np.zeros_like(batch[index], dtype=batch[index].dtype)
      tres[:batch[i].shape[0]] = batch[i]
      res.append(tres)

  return res


def Multiposebatch(batch):
  sample_batch = {}
  for key in batch[0]:
    if key in ['hm', 'input']:
      sample_batch[key] = default_collate([d[key] for d in batch])
    else:
      align_batch = multipose_collate([d[key] for d in batch])
      sample_batch[key] = default_collate(align_batch)

  return sample_batch
