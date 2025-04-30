from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data

#这里的代码就是固定的格式，不用动

class FACEHP(data.Dataset):
  num_classes = 1   #数据集的种类数
  num_joints = 5
  #default_resolution = [800, 800]
  default_resolution = [640,640]          #这里设置的就是输入图片的大小
  #数据的均值和方差
  #'''
  #widerface
  mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
  #'''
  '''
  #imagenet
  mean = np.array([0.485, 0.456, 0.406],
                  dtype=np.float32).reshape(1, 1, 3)
  std = np.array([0.229, 0.224, 0.225],
                 dtype=np.float32).reshape(1, 1, 3)
                 '''
  flip_idx = [[0, 1], [3, 4]]             # 翻转的关键点在关键点矩阵中的索引


  def __init__(self, opt, split):
    super(FACEHP, self).__init__()
    self.edges = [[0, 1], [0, 2], [1, 3], [2, 4], 
                  [4, 6], [3, 5], [5, 6], 
                  [5, 7], [7, 9], [6, 8], [8, 10], 
                  [6, 12], [5, 11], [11, 12], 
                  [12, 14], [14, 16], [11, 13], [13, 15]]
    
    self.acc_idxs = [1, 2, 3, 4]    #这里对应的是关键点信息的索引
    #数据地址
    self.\
      data_dir = os.path.join(opt.data_dir, 'wider_face')
    #self.img_dir = os.path.join(self.data_dir, 'image')                 # 这个在载入图片的时候有用
    self.img_dir = os.path.join(self.data_dir,'WIDER_{}', 'images').format(split)
    _ann_name = {'train': 'train', 'val': 'val'}
    print(self.img_dir)

    #根据不同情况加载数据的json文件
    if split == 'test':       #选择是测试还是训练
      self.annot_path = os.path.join(
        self.data_dir, 'annotations', 
        '{}_wider_face.json').format(_ann_name[split])
    else:
      self.annot_path = os.path.join(
        self.data_dir, 'annotations', 
        '{}_wider_face.json').format(_ann_name[split])
    self.max_objs = 50  #最大的目标数
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)

    #特征变量 _eig_val
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    self.split = split  #训练还是测试的指示变量
    self.opt = opt      #配置文件

    print('==> initializing centerface key point {} data.'.format(split))
    self.coco = coco.COCO(self.annot_path)#加载数据集
    image_ids = self.coco.getImgIds()#获取图像的id

    if split == 'train':
      self.images = []
      for img_id in image_ids:
        idxs = self.coco.getAnnIds(imgIds=[img_id])   #根据image id获得所有的ann标注
        if len(idxs) > 0:
          self.images.append(img_id)
    else:
      self.images = image_ids
    self.num_samples = len(self.images) #获得图片的图像数目
    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):   #转化为浮点类型
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):    #转换为评价格式
    # import pdb; pdb.set_trace()
    detections = []
    for image_id in all_bboxes:         #遍历为所有的图像id
      for cls_ind in all_bboxes[image_id]:  #获取每个结果中的类别索引
        category_id = 1       #获得类别id

        #遍历每个图像的每个类别的结果
        for dets in all_bboxes[image_id][cls_ind]:
          bbox = dets[:4]
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = dets[4]
          bbox_out  = list(map(self._to_float, bbox))
          keypoints = np.concatenate([
            np.array(dets[5:39], dtype=np.float32).reshape(-1, 2), 
            np.ones((17, 1), dtype=np.float32)], axis=1).reshape(51).tolist()
          keypoints  = list(map(self._to_float, keypoints))

          #获得检测结果
          detection = {
              "image_id": int(image_id),
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.2f}".format(score)),
              "keypoints": keypoints
          }
          detections.append(detection)
    return detections

  def __len__(self):
    return self.num_samples


  def save_results(self, results, save_dir):    #保存结果
    json.dump(self.convert_eval_format(results), 
              open('{}/results.json'.format(save_dir), 'w'))


#进行评价结果
  def run_eval(self, results, save_dir):
    # result_json = os.path.join(opt.save_dir, "results.json")
    # detections  = convert_eval_format(all_boxes)
    # json.dump(detections, open(result_json, "w"))
    self.save_results(results, save_dir)
    coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
    coco_eval = COCOeval(self.coco, coco_dets, "keypoints")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()