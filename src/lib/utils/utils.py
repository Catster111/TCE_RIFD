from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import random
import cv2

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count


def Data_anchor_sample_backup(image, anns):
    maxSize = 12000
    infDistance = 9999999

    boxes = []
    for ann in anns:
        boxes.append([ann['bbox'][0], ann['bbox'][1], ann['bbox'][0]+ann['bbox'][2], ann['bbox'][1]+ann['bbox'][3]])
    boxes = np.asarray(boxes, dtype=np.float32)

    height, width, _ = image.shape

    random_counter = 0

    boxArea = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    rand_idx = random.randint(0, len(boxArea)-1)                #random.randint(start,shop)一个范围内的整数,就是随机选择一个area -random.randint(start, stop) selects an integer within a range, which means randomly choosing an area.
    rand_Side = boxArea[rand_idx] ** 0.5               #幂 - 返回boxArea[rand_idx]的0.5次幂 -power - returns the 0.5th power of boxArea[rand_idx].

    anchors = [16, 32, 48, 64, 96, 128, 256, 512]   #这里现实的anchor的边长，并不是面积 例如128实际上代表的是128*128所以上面的大小要求0.5次幂 -Here it shows the side length of the anchor, not the area. For example, 128 actually represents 128*128, so the size above requires the 0.5th power.
    distance = infDistance  #初时值设置一个很大的值 -Set a very large initial value.
    anchor_idx = 5   #我是这样想的：这里选择128的原因是因为使用的是512的输入后，缩放到四分之一后就成为了128，所以最大就是128 -I think this way: The reason for choosing 128 is because after using a 512 input, it is scaled down to a quarter, which becomes 128, so the maximum is 128.
    for i, anchor in enumerate(anchors):
        if abs(anchor - rand_Side) < distance:
            distance = abs(anchor - rand_Side)  # 选择最接近的anchors -Select the closest anchors
            anchor_idx = i                     #这个值就是最合适的一个大小的数值 -This value is the most suitable size number.

    target_anchor = random.choice(anchors[0:min(anchor_idx+1, 5) ])  # 随机选择一个相对较小的anchor，向下 -Randomly select a relatively smaller anchor, going downwards
    ratio = float(target_anchor) / rand_Side  # 缩放的尺度 scale of scaling
    ratio = ratio * (2 ** random.uniform(-1, 1))  # [ratio/2, 2ratio]  的均匀分布   #返回参数1和参数2之间的任意值 -[ratio/2, 2ratio] uniform distribution #returns any value between parameter 1 and parameter 2."

    if int(height * ratio * width * ratio) > maxSize * maxSize:
        ratio = (maxSize * maxSize / (height * width)) ** 0.5

    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = random.choice(interp_methods)
    image = cv2.resize(image, None, None, fx=ratio, fy=ratio, interpolation=interp_method)


    boxes[:, 0] *= ratio
    boxes[:, 1] *= ratio
    boxes[:, 2] *= ratio
    boxes[:, 3] *= ratio

    boxes = boxes.tolist()
    for i in range(len(anns)):
        anns[i]['bbox'] = [boxes[i][0], boxes[i][1], boxes[i][2]-boxes[i][0], boxes[i][3]-boxes[i][1]]      # 人脸bbox
        for j in range(5):
            anns[i]['keypoints'][j*3] *= ratio
            anns[i]['keypoints'][j*3+1] *= ratio

    return image, anns


def Data_anchor_sample(image, anns):
    maxSize = 12000
    infDistance = 9999999

    boxes = []
    for ann in anns:
        boxes.append([ann['bbox'][0], ann['bbox'][1], ann['bbox'][0]+ann['bbox'][2], ann['bbox'][1]+ann['bbox'][3]])
    boxes = np.asarray(boxes, dtype=np.float32)

    height, width, _ = image.shape
    random_counter = 0

    boxArea = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    rand_idx = random.randint(0, len(boxArea)-1)
    rand_Side = boxArea[rand_idx] ** 0.5

    anchors = [16, 32, 48, 64, 96, 128, 256, 512]
    distance = infDistance
    anchor_idx = 5

    for i, anchor in enumerate(anchors):
        if abs(anchor - rand_Side) < distance:
            distance = abs(anchor - rand_Side)
            anchor_idx = i

    target_anchor = random.choice(anchors[0:min(anchor_idx+1, 5)])
    ratio = float(target_anchor) / rand_Side
    ratio = ratio * (2 ** random.uniform(-1, 1))

    if int(height * ratio * width * ratio) > maxSize * maxSize:
        ratio = (maxSize * maxSize / (height * width)) ** 0.5

    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = random.choice(interp_methods)
    image = cv2.resize(image, None, None, fx=ratio, fy=ratio, interpolation=interp_method)

    boxes[:, 0] *= ratio
    boxes[:, 1] *= ratio
    boxes[:, 2] *= ratio
    boxes[:, 3] *= ratio

    boxes = boxes.tolist()
    for i in range(len(anns)):
        anns[i]['bbox'] = [boxes[i][0], boxes[i][1], boxes[i][2]-boxes[i][0], boxes[i][3]-boxes[i][1]]
        # Scale keypoints
        for j in range(5):
            anns[i]['keypoints'][j*3] *= ratio
            anns[i]['keypoints'][j*3+1] *= ratio
        # Scale face_center
        if 'face_center' in anns[i]:
            anns[i]['face_center'][0] *= ratio
            anns[i]['face_center'][1] *= ratio

    return image, anns