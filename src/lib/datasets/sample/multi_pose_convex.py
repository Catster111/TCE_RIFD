from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

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
from collections.abc import Container as container_abcs

np_str_obj_array_pattern = re.compile(r'[SaUO]')

'''
预处理数据集，该类的主要功能是主要实现_getitem_方法，以便在train的时候被DataLoader加载
Preprocess the dataset. 
The main function of this class is to implement the getitem method, 
so it can be loaded by DataLoader during training.
'''
'''
这一部分写自己的处理代码
Write your own processing code in this part
'''


class MultiPoseDataset(data.Dataset):  # 将[x1,y1(左下角的坐标)，w,h]-->[x1,y1,x2,y2]
    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    '''自己写的代码的模块
        The module of the code written by oneself.
    '''
    def _polar_label(self, a, cen):

            r = math.sqrt(math.pow(a[0] - cen, 2) + math.pow(a[1] - cen, 2))
            r = r / (math.sqrt(math.pow(cen, 2) + math.pow(cen, 2)) / (cen * 2))

            theta = math.atan2(a[1] - cen, a[0] - cen) / math.pi * 180
            if a[1] <= cen:
                theta = (theta + 360) / (360 / (cen * 2))
            else:
                theta = theta / (360 / (cen * 2))

            return np.array([theta, r])


    def new_polar_label(self, a, cen):
            """
            Converts Cartesian coordinates to normalized polar coordinates centered at (cen, cen).

            Parameters:
                a (np.ndarray): The (x, y) coordinates of the point.
                cen (float): The center coordinate value for both x and y axes.

            Returns:
                np.ndarray: An array containing the normalized angle (theta) and normalized radius (r).
            """
            x, y = a
            dx = x - cen
            dy = y - cen

            # Calculate Euclidean distance from center
            r = math.sqrt(dx**2 + dy**2)

            # Ensure the radius is non-negative
            if r == 0:
                r_normalized = 0.0
            else:
                max_radius = 32
                r_normalized = min(r / max_radius, 1.0)  # Clamp to [0, 1]

            # Calculate angle in degrees
            theta = math.atan2(dy, dx) * (180 / math.pi)  # Convert radians to degrees

            # Normalize angle to [0, 360)
            if theta < 0:
                theta += 360

            # Normalize angle to [0, 1)
            theta_normalized = theta / 360.0

            return np.array([theta_normalized, r_normalized], dtype=np.float32)

    def _rotate_img_bboxes(self, img, angle=5, scale=1.):
        # ---------------------- 旋转图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        # 角度变弧度
        rangle = np.deg2rad(angle)
        # 计算新图像的宽度和高度，分别为最高点和最低点的垂直距离
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        # 获取图像绕着某一点的旋转矩阵
        # getRotationMatrix2D(Point2f center, double angle, double scale)
        # Point2f center：表示旋转的中心点
        # double angle：表示旋转的角度
        # double scale：图像缩放因子
        # 参考：https://cloud.tencent.com/developer/article/1425373
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)  # 返回 2x3 矩阵
        # 新中心点与旧中心点之间的位置
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # 仿射变换
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))),
                                 flags=cv2.INTER_LANCZOS4)  # ceil向上取整

        # ---------------------- 矫正boundingbox ----------------------
        # rot_mat是最终的旋转矩阵
        # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下

        return rot_mat, rot_img

    def _rotate_bboxes(self, rot_mat, bbox):

        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]

        point1 = np.dot(rot_mat, np.array([(x_min + x_max) / 2, y_min, 1]))
        point2 = np.dot(rot_mat, np.array([x_max, (y_min + y_max) / 2, 1]))
        point3 = np.dot(rot_mat, np.array([(x_min + x_max) / 2, y_max, 1]))
        point4 = np.dot(rot_mat, np.array([x_min, (y_min + y_max) / 2, 1]))
        # 合并np.array
        concat = np.vstack((point1, point2, point3, point4))  # 在竖直方向上堆叠
        # 改变array类型
        concat = concat.astype(np.int32)
        # 得到旋转后的坐标
        rx, ry, rw, rh = cv2.boundingRect(concat)
        rx_min = rx
        ry_min = ry
        rx_max = rx + rw
        ry_max = ry + rh
        # 加入list中
        rot_bboxes = [rx_min, ry_min, rx_max, ry_max]
        rot_bboxes = np.array(rot_bboxes, dtype=np.float32)

        return rot_bboxes

    def _rotate_pts(self, rot_mat, pts):
        num_kpts = pts[:, 2].sum()
        rot_pts = list()
        if (num_kpts == 0):
            rot_pts = pts
        else:
            for rotpts in pts:
                rotpts[:2] = np.dot(rot_mat, np.array([rotpts[0], rotpts[1], 1]))
                rot_pts.append(rotpts)
        rot_pts = np.array(rot_pts, dtype=float)
        return rot_pts

    def save_result_image(ret, img_path, output_path):
                                                                                                    # Extract the original image
        inp = ret['input']
        inp = inp.transpose(1, 2, 0)  # Convert from CHW to HWC format
        inp = (inp * ret['std']) + ret['mean']  # Denormalize
        inp = (inp * 255).astype(np.uint8)

        # Draw bounding boxes and keypoints
        for k in range(len(ret['wh'])):
            if ret['reg_mask'][k] > 0:
                bbox = ret['hp_one_bbox'][k]
                x1, y1, x2, y2 = bbox
                cv2.rectangle(inp, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box

                # Draw keypoints
                for j in range(0, len(ret['landmarks'][k]), 2):
                    x, y = ret['landmarks'][k][j], ret['landmarks'][k][j + 1]
                    if x > 0 and y > 0:  # Check if the keypoint is valid
                        cv2.circle(inp, (x, y), 3, (0, 0, 255), -1)  # Draw keypoint

        # Save the result image
        cv2.imwrite(output_path, inp)       

    def __getitem__(self, index):
            #1. Loading Image and Annotations
            # Read in images and annotation information.
            img_id = self.images[index]  # Obtain the index of this point.
            # img_id=45  # At that time, when studying this file, the ID was fixed as 1.
            file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
            img_path = os.path.join(self.img_dir, file_name)
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])  # Retrieve annotation ID based on the image ID.
            anns = self.coco.loadAnns(ids=ann_ids)  # What we obtain here is all the annotation information.

            #2. Limiting the Number of Objects
            num_objs = len(anns)
            # num_objs = min(len(anns), self.max_objs)
            if num_objs > self.max_objs:  # 这里的设置就是一些标注信息的比max_objs多的时候，只选择32，随机选择32个 -The setting here is that when there are more annotation information than max_objs, only 32 are selected, randomly choosing 32.
                num_objs = self.max_objs
                anns = np.random.choice(anns, num_objs)
            
            #3. Reading and Preprocessing the Image
            img = cv2.imread(img_path)  # 读取图片 -read the image
            temp_path = '/workspace/RPNet/temp/'
            #cv2.imwrite(temp_path + '1.png', img)
            # print(img.shape)
            # cv2.imwrite(temp_path+'1.png',img)
            img, anns = Data_anchor_sample(img, anns)  # Resize the images and annotation information to the desired sizes.
            # Code for rotating images and obtaining rotation matrices.
            #rand_num = random.randint(0, 3)
            ''' Original code, cover only 180
            rand_num = random.uniform(-4, 4)
            if rand_num <= -3:
                rand_num = 0
            if rand_num >= 2:
                rand_num =2
            # print(rand_num)
            rot_mat, img = self._rotate_img_bboxes(img, 90 * rand_num)
            '''

            #4. Data Augmentation
            #a. Random Rotation
            # Generate a random integer between 0 and 35
            rand_step = random.randint(0, 35)

            # Multiply by 10 to get the rotation angle in steps of 10 degrees
            rotation_angle = rand_step * 10

            # Now use this rotation_angle for your rotation
            rot_mat, img = self._rotate_img_bboxes(img, rotation_angle)


            #cv2.imwrite(temp_path + '2.png', img)
            #print(img.shape)

            '''
            resize图片开始
            start resizing the image.
        '''
            #b. Random Scaling and Shifting
            height, width = img.shape[0], img.shape[1]
            c = np.array([img.shape[1] / 2., img.shape[0] / 2.],
                        dtype=np.float32)  # C is the center point of the original image
            s = max(img.shape[0], img.shape[1]) * 1.0  # Find the maximum edge of the original image.
            rot = 0  # 旋转角度

            flipped = False

            #c. Additional Augmentations (Training Mode Only)
            if self.split == 'train':
                if not self.opt.not_rand_crop:
                    s = s * np.random.choice(np.arange(0.3, 1.2, 0.1))
                    # s = s
                    # _border = np.random.randint(128*0.4, 128*1.4)
                    _border = s * np.random.choice([0.1, 0.2, 0.25])
                    w_border = self._get_border(_border, img.shape[1])
                    h_border = self._get_border(_border, img.shape[0])
                    c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                    c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)  # 修改中心的位置
                else:
                    sf = self.opt.scale
                    cf = self.opt.shift
                    c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                    c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                    s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
                if np.random.random() < self.opt.aug_rot:
                    rf = self.opt.rotate
                    rot = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)

                # 进行
                if np.random.random() < self.opt.flip:
                    flipped = True
                    img = img[:, ::-1, :]
                    c[0] = width - c[0] - 1

            #5. Applying Affine Transformation
            trans_input = get_affine_transform(
                c, s, rot, [self.opt.input_res, self.opt.input_res])

            inp = cv2.warpAffine(img, trans_input,
                                (self.opt.input_res, self.opt.input_res),
                                flags=cv2.INTER_LINEAR)
            #cv2.imwrite(temp_path + '3.png', inp)
            # cv2.imwrite('/home/newz/CenterFace_new_mod/output/temp/'+ '3.png', inp)

            #6. Normalizing the Image
            inp = (inp.astype(np.float32) / 255.)  # 归一化 -normalization
            if self.split == 'train' and not self.opt.no_color_aug:  # 随机进行图片增强 - randomly perform image enhancement
                color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
                # inp = Randaugment(self._data_rng, inp, self._eig_val, self._eig_vec)

            inp = (inp - self.mean) / self.std
            inp = inp.transpose(2, 0, 1)

            '''
        resize图片结束
        '''
            #7. Preparing Target Tensors
            #a. Initializing Tensors
            output_res = self.opt.output_res  # 已经进行了下采样的缩放 -scaling has already been performed with downsampling.
            '''
        opt.output_h = opt.input_h // opt.down_ratio
        opt.output_w = opt.input_w // opt.down_ratio
        opt.input_res = max(opt.input_h, opt.input_w)
        opt.output_res = max(opt.output_h, opt.output_w)
        '''
            num_joints = self.num_joints
            trans_output_rot = get_affine_transform(c, s, rot, [output_res, output_res])  # 获得变换矩阵（就是将原图缩放到128*128）- Obtain the transformation matrix (which scales the original image to 128*128).

            trans_output = get_affine_transform(c, s, 0, [output_res, output_res])

            '''Heatmaps (hm and hm_hp):'''

            hm = np.zeros((self.num_classes, output_res, output_res), dtype=np.float32)  # heatmap的空白盒子[1,128,128] # Creating heatmap
            #This is used to create a heatmap for each class, indicating the presence of objects in different regions of the image.

            hm_hp = np.zeros((num_joints, output_res, output_res), dtype=np.float32)  # 关键点的空白盒子[5,128,128] -blank box for keypoints [5,128,128].
            #This heatmap is specific to keypoints, with one layer per joint, representing the likelihood of each joint's presence at each pixel.

            '''Dense Keypoint Representations (dense_kps and dense_kps_mask):'''
            dense_kps = np.zeros((num_joints, 2, output_res, output_res),  # [5,2,128,128]  空白盒子感觉这个放的使关键点的坐标（猜测）-The blank box seems to contain the coordinates of the keypoints (guess)
                                dtype=np.float32)
            #used to store the coordinates of keypoints in a dense format, where each keypoint has its spatial map.

            dense_kps_mask = np.zeros((num_joints, output_res, output_res),  # [5,128,128]  空白盒子掩码不晓得是干啥的 -Not sure what the blank box mask is for.
                                    dtype=np.float32)
            #used to mask or indicate the validity of the keypoints in dense_kps.

            '''Bounding Box Width and Height (wh):
            used to store the width and height of the bounding boxes for each detected object.
            '''
            wh = np.zeros((self.max_objs, 2), dtype=np.float32)  # 宽高的空白盒子  [32,2] -blank box for width and height [32,2].

            '''Keypoint Coordinates (kps): storing the coordinates of keypoints for each object.'''
            kps = np.zeros((self.max_objs, num_joints * 2), dtype=np.float32)  # 空白盒子[32,10] -blank box [32,10].

            '''Center Point Offset Regression (reg):  
            used for regressing the offset of the center points of the bounding boxes to refine their positions.'''
            reg = np.zeros((self.max_objs, 2), dtype=np.float32)  # 中心点偏置的回归 空白盒子[32,2] - regression of center point offset, blank box [32,2].

            '''Indices (ind): 
            storing indices for locating the center points of objects in the heatmap.'''
            ind = np.zeros((self.max_objs), dtype=np.int64)  # 保存wh和reg 索引[32] save wh and reg indices [32].
            # 空白盒子和索引地址 - blank box and index address

            '''Masks (reg_mask, wight_mask, kps_mask, hp_mask):'''
            reg_mask = np.zeros((self.max_objs), dtype=np.uint8)  # 掩码 [32]
            #indicating which objects have valid regression offsets.

            wight_mask = np.ones((self.max_objs), dtype=np.float32)  # wight掩码，全是1[32]
            #, initialized to ones, possibly used for weighting objects during loss calculation.

            kps_mask = np.zeros((self.max_objs, self.num_joints * 2), dtype=np.uint8)  # [32,10]掩码
            #indicating the validity of each keypoint.
            hp_mask = np.zeros((self.max_objs * num_joints), dtype=np.int64)  # [32*5] 掩码
            #used to mask the keypoints in heatmap predictions.

            '''Keypoint Offset (hp_offset) and Indices (hp_ind):'''
            hp_offset = np.zeros((self.max_objs * num_joints, 2), dtype=np.float32)  # [32*5,2]
            # storing the offset for each keypoint to refine its position.


            hp_ind = np.zeros((self.max_objs * num_joints), dtype=np.int64)  # [32*5]
            #storing indices for locating keypoints in the heatmap.
            

            '''这一部分是我新加的单张人脸的关键点估计的target
            一个用来存储单张人脸的关键点的heatmap，用于训练landmarks head的targe t
            实际上就是截取出人脸大小区域的图片进行特征变换
            输入的大小为人脸wh的最大值，然后放缩到64*64
            开始部分
        '''
            # hp_one_information = np.zeros((2,2), dtype=np.float32)
            hp_one_bbox = np.zeros((self.max_objs, 4), dtype=np.float32)
            #This array is used to store the bounding box coordinates for each detected object.
            #Each bounding box is represented by four values (typically x1, y1, x2, y2 - the coordinates of the top-left and bottom-right corners).

            hp_one_center = np.zeros((self.max_objs, 2), dtype=np.float32)  # 获取人脸在128*128对应的中心点
            # It stores the center points of the bounding boxes in the scaled-down image space (128x128). 
            #Each center point is represented by two values (x, y coordinates).

            hp_one_wh = np.zeros((self.max_objs, 2), dtype=np.float32)  # 获取人脸实际上在128*128的对应的长宽距离
            # to store the width and height of each bounding box in the scaled-down image space (128x128).

            hm_hp_one = np.zeros((self.max_objs * num_joints, 64, 64), dtype=np.float32)  # heatmap
            # This array is a set of heatmaps for each keypoint of each object, with each heatmap having a resolution of 64x64. 
            #It's used to represent the likelihood of each keypoint's position within the bounding box.

            hp_one_ind = np.zeros((self.max_objs * num_joints), dtype=np.int64)  # 索引（这个索引实际上就是在当前heatmap的索引值）
            #This array stores the index values for keypoints in the heatmap, 
            #helping to locate the specific position of each keypoint within the heatmap grid.

            hp_one_offset = np.zeros((self.max_objs * num_joints, 2), dtype=np.float32)
            #It's used to store the offset values for each keypoint, which are the small adjustments 
            #needed to pinpoint the exact location of keypoints within the bounding box.

            hp_one_mask = np.zeros((self.max_objs * num_joints), dtype=np.int64)  # [32*5] 掩码
            #This array acts as a mask to indicate which keypoints are valid and should be considered in further processing or loss calculation.
            '''
            结束部分
        '''

            draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                draw_umich_gaussian

            #8. Processing Each Annotation
            gt_det = [] # groundtruth details
            for k in range(num_objs): # each object
                ann = anns[k]  # 获得第K个的标注信息 Obtain the annotation information of the Kth
                bbox = self._coco_box_to_bbox(ann['bbox'])  # 转化为我们需要的bbox的坐标形式[x1,y1,x2,y2],这个地方就是存储一个bboxl -  Convert to the bbox coordinate form we need [x1, y1, x2, y2], this place is where a bbox is stored
                # 这个就是用来旋转数据集bbox标注信息的 - This is used to rotate the bbox annotation information of the dataset.
                bbox = self._rotate_bboxes(rot_mat, bbox)

                cls_id = int(ann['category_id']) - 1  # 获取得到的类别信息 - Obtain the obtained category information.
                pts = np.array(ann['keypoints'], np.float32).reshape(num_joints, 3)  # 单个ann的关键点的信息 Information on key points of a single ann
                # print(pts[0][:2])
                # 用来旋转数据集的关键点信息的
                pts = self._rotate_pts(rot_mat, pts) # Rotate points according to rotation matrix

                #a. Flipping Adjustments (if Applied)
                if flipped:  # flipped翻转这个地方进行翻转的判断，flipped=false(没有启动)
                    bbox[[0, 2]] = width - bbox[[2, 0]] - 1
                    pts[:, 0] = width - pts[:, 0] - 1
                    for e in self.flip_idx:
                        pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()

                '''test'''
                bbox_one = np.copy(bbox) # Checking which domain of bbox_one
                pts_one = np.copy(pts) # Which domain of pts?
                '''
        x1 = pts[1][0] - pts[0][0]
        y1 = pts[][0] - pts[0][0]
        '''
                ##把对应的gt也进行相应的transform - Perform the corresponding transform on the ground truth (gt) as well
                
                #9. Applying Affine Transformation to Bounding Box and Keypoints
                bbox[:2] = affine_transform(bbox[:2], trans_output)  # 将原本的坐标缩放到128（原论文中使用的512的大小）的尺寸上，这时候标注信息存在问题进行修正 -Scale the original coordinates to a size of 128 (the original paper used a size of 512), and at this time, correct any issues with the annotation information.
                bbox[2:] = affine_transform(bbox[2:], trans_output)
                bbox = np.clip(bbox, 0, output_res - 1)  # 避免坐标超出缩放后的大小- Avoid coordinates exceeding the size after scaling

                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]  # 找出新的w,h
                
                #10. Generating Targets for Valid Objects
                if (h > 0 and w > 0) or (rot != 0):
                    radius = gaussian_radius((math.ceil(h), math.ceil(w)))  #
                    radius = self.opt.hm_gauss if self.opt.mse_loss else max(0, int(radius))
                    ct = np.array(
                        [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)  # 人脸的中心坐标 the center coordinates of the face
                    # Is it the center of boundding box and face ? 
                    ct_int = ct.astype(np.int32)  # 整数化  to make integer
                    # wh[k] = 1. * w, 1. * h                                                    # 2. centernet的方式
                    wh[k] = np.log(1. * w / 4), np.log(1. * h / 4)  # 2. 人脸bbox的高度和宽度,centerface论文的方式  -The height and width of the face bounding box, the method of the CenterFace pape
                    ind[k] = ct_int[1] * output_res + ct_int[0]  # 人脸bbox在1/4特征图中的索引 - Index of the face bounding box in the 1/4 feature map
                    reg[k] = ct - ct_int  # 3. 人脸bbox中心点整数化的偏差 - The deviation of the face bounding box center point after integerization
                    reg_mask[k] = 1  # 是否需要用于计算误差，掩码用于选择数据 - Is it necessary to calculate the error, and are masks used to select data?
                    # if w*h <= 20:
                    #     wight_mask[k] = 15

                    #11. Handling Objects Without Keypoints
                    num_kpts = pts[:, 2].sum()  # 没有关键点标注的时哦 - When there are no keypoint annotations
                    if num_kpts == 0:  # 没有关键点标注的都是比较困难的样本- Samples without keypoint annotations are generally more difficult.
                        # if num_kpts == 0 or w * h <= 8:
                        # print('没有关键点标注')
                        hm[cls_id, ct_int[1], ct_int[0]] = 0.9999  # 如果没有关键点信息的话hm的对应值就设置为0.9999 If there is no keypoint information, then the corresponding value in hm is set to 0.9999.
                        # reg_mask[k] = 0

                    #12. Generating Keypoint Targets
                    hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                    hp_radius = self.opt.hm_gauss \
                        if self.opt.mse_loss else max(0, int(hp_radius))

                    '''这里实现的是单张的人脸的关键点的检测target,start''' # Here is implemented the detection of keypoints for a single face, target, start
                    if num_kpts != 0:
                        '''这部分有问题，貌似没用''' # There seems to be a problem with this part, it appears to be unused
                        hp_one_bbox[k] = bbox  # 这个部分的代码可以直接得到框的数据信息，初始的框的数据，调整一下输入框的大小 - The code in this part can directly obtain the data information of the box, the initial data of the box, adjust the size of the input box
                        hp_one_center[k] = ct  # 这个地方可以得到数据中心点 - This place can obtain the data center point
                        hp_one_wh[k] = 1. * w, 1. * h  # 这个地方记录一下图片的 wh,(记录的是预处理之后的值，这个地方没有错误) - Record the width and height (wh) of the image here (the recorded values are after preprocessing, there is no error in this part

                        # 这部分的代码是进行调整，可以把矩形的人脸图片变换为正方形的人脸区域，这个部分有问题 - This part of the code is for adjustment, it can transform the rectangular face image into a square face area, there is a problem with this part.
                        whmax = max(hp_one_wh[k][0], hp_one_wh[k][1])
                        # 这个地方得到的是修正成为正方形的bbox - This place obtains the bbox corrected to a square shape.
                        x1 = hp_one_center[k][0] - whmax / 2
                        y1 = hp_one_center[k][1] - whmax / 2
                        x2 = hp_one_center[k][0] + whmax / 2
                        y2 = hp_one_center[k][1] + whmax / 2
                        '''
                #这个地方得到的实际上是已知的人脸框的bbox
                x1 =  hp_one_center[k][0] - hp_one_wh[k][0]/2
                y1 =  hp_one_center[k][1] - hp_one_wh[k][1]/2
                x2 =  hp_one_center[k][0] + hp_one_wh[k][0]/2
                y2 =  hp_one_center[k][1] + hp_one_wh[k][1]/2
                '''
                        # 调整一下输入框的大 - Adjust the size of the input box.
                        hp_one_bbox[k][0] = x1
                        hp_one_bbox[k][1] = y1
                        hp_one_bbox[k][2] = x2
                        hp_one_bbox[k][3] = y2
                        '''end'''
                        # 这个地方有问题，当时设计的时候是存在问题的，貌似没啥大问题暂时不管2022.8.14 - There is a problem here; there was an issue with the design at the time, but it seems not to be a big problem, so it will be ignored for now, 2022.8.14.
                        '''这个地方的设置是吧关键点转化为''' # The setting here is to convert the keypoints into
                        ct_one = np.array(
                            [(bbox_one[0] + bbox_one[2]) / 2, (bbox_one[1] + bbox_one[3]) / 2], dtype=np.float32)
                        h_one, w_one = bbox_one[3] - bbox_one[1], bbox_one[2] - bbox_one[0]
                        temp_max = max(h_one, w_one)
                        scale_one = 64 / max(w_one, h_one)  # 这里的64就是设置输出的标签的大 - Here, 64 is the size of the output label set
                        center_one = [64 / 2, 64 / 2]

                        # Import necessary libraries
                        from scipy.spatial import ConvexHull
                        from shapely.geometry import Polygon

                        # Compute ct_one based on the convex hull of facial keypoints
                        visible_pts = pts_one[pts_one[:, 2] > 0, :2]  # Get visible keypoints

                        if visible_pts.shape[0] >= 3:
                            try:
                                # Compute the convex hull
                                hull = ConvexHull(visible_pts)
                                hull_points = visible_pts[hull.vertices]

                                # Create a polygon and compute centroid
                                polygon = Polygon(hull_points)

                                if not polygon.is_valid or polygon.area == 0:
                                    # Polygon is invalid or area is zero, fallback to mean
                                    ct_one = np.mean(visible_pts, axis=0)
                                else:
                                    # Compute the centroid
                                    centroid = polygon.centroid
                                    ct_one = np.array(centroid.coords[0], dtype=np.float32)
                                    
                                    # Check if centroid coordinates are finite numbers
                                    if not np.isfinite(ct_one).all():
                                        # Centroid has NaN or Inf values, fallback to mean
                                        ct_one = np.mean(visible_pts, axis=0)
                            except Exception as e:
                                print(f"Error computing convex hull or centroid: {e}")
                                # Fallback to mean of visible keypoints
                                ct_one = np.mean(visible_pts, axis=0)
                        elif visible_pts.shape[0] > 0:
                            # Compute mean of visible points if not enough for convex hull
                            ct_one = np.mean(visible_pts, axis=0)
                        else:
                            # Fallback to bbox center if no keypoints are visible
                            ct_one = np.array(
                                [(bbox_one[0] + bbox_one[2]) / 2, (bbox_one[1] + bbox_one[3]) / 2], dtype=np.float32)

                        # Ensure ct_one is finite
                        if not np.isfinite(ct_one).all():
                            print("ct_one contains NaN or Inf values, using bbox center as fallback.")
                            ct_one = np.array(
                                [(bbox_one[0] + bbox_one[2]) / 2, (bbox_one[1] + bbox_one[3]) / 2], dtype=np.float32)


                        hp_one_radius = math.ceil(hp_radius / 2)
                        for j in range(num_joints):
                            if (pts[j, 2] > 0):
                                pts_one[j, :2] = center_one + (pts_one[j, :2] - ct_one) * scale_one
                                pts_one[j, :2] = self._polar_label(pts_one[j, :2], 32) # is 32 is center point
                                # pts_one_int = pts_one[j, :2].astype(np.int32)  # 关键点整数化 - Keypoint integerization
                                pts_one_int = pts_one[j, :2]
                                     # Ensure pts_one_int contains valid indices
                                if np.isfinite(pts_one_int).all():
                                    hp_one_offset[k * num_joints + j] = pts_one[j, :2] - pts_one_int  # 关键点整数化的偏差 - The deviation of keypoint integerization
                                    hp_one_ind[k * num_joints + j] = pts_one_int[1] * 64 + pts_one_int[0]  # 索引 - index
                                    hp_one_mask[k * num_joints + j] = 1
                                    draw_gaussian(hm_hp_one[k * num_joints + j], pts_one_int, hp_one_radius)  # 画高斯点 - draw Gaussian points
                                else:
                                    print(f"Invalid keypoint indices at joint {j}, skipping this keypoint.")
                                
                                # Assuming hm_hp_one is defined and contains your heatmaps
                                #index = k * num_joints + j
                                #filename = 'heatmap%d.png' % index
                                #save_heatmap_image(hm_hp_one, filename, index)



                        # pts_one = np.array(ann['keypoints '], np.float32).reshape(num_joints, 3)
                        '''这个循环有问题，貌似不应该这样写'''
                        '''
                for j in range(num_joints):
                    if((pts)[j,2]>0):
                        pts_one[j][0] -=bbox_one[0]
                        pts_one[j][1] -=bbox_one[1]
                        pts_one[j][0] *=scale_one
                        pts_one[j][1] *=scale_one
                        pts_one_int = pts_one[j, :2].astype(np.int32)       #关键点整数化
                        hp_one_offset[k * num_joints + j] = pts_one[j, :2] - pts_one_int           # 关键点整数化的偏差
                        hp_one_ind[k * num_joints + j] = pts_one_int[1] * 64 + pts_one_int[0]   # 索引
                        hp_one_mask[k * num_joints + j] = 1
                        draw_gaussian(hm_hp_one[k*num_joints+j], pts_one_int, hp_one_rad ius)          #画高斯点
                        '''
                    '''end'''

                    for j in range(num_joints):  # 这个函数的作用就是得到整张图的关键点的坐标点的所有的信息 -The function's purpose is to obtain all the information of the keypoint coordinates for the entire image.
                        if pts[j, 2] > 0:
                            pts[j, :2] = affine_transform(pts[j, :2], trans_output_rot)  # 关键点坐标缩放到128的尺寸上去 - Scale the keypoint coordinates to a size of 128.
                            # print(pts[j][0])
                            # print(pts[j][1])
                            if 0 <= pts[j, 0] < output_res and \
                                    0 <= pts[j, 1] < output_res:
                                kps[k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int  # 4. 关键点相对于人脸bbox的中心的偏差
                                kps_mask[k, j * 2: j * 2 + 2] = 1
                                pt_int = pts[j, :2].astype(np.int32)  # 关键点整数化
                                hp_offset[k * num_joints + j] = pts[j, :2] - pt_int  # 关键点整数化的偏差
                                hp_ind[k * num_joints + j] = pt_int[1] * output_res + pt_int[0]  # 索引
                                hp_mask[k * num_joints + j] = 1  # 计算损失的mask,这个值就是计算那里有值
                                if self.opt.dense_hp:
                                    # must be before draw center hm gaussian
                                    draw_dense_reg(dense_kps[j], hm[cls_id], ct_int,
                                                pts[j, :2] - ct_int, radius, is_offset=True)
                                    draw_gaussian(dense_kps_mask[j], ct_int, radius)
                                draw_gaussian(hm_hp[j], pt_int,
                                            hp_radius)  # 1. 关键点高斯map eg:一张图上面有多少个人脸就有多少个鼻子的关键点（如果标注信息中有的话）
                                if ann['bbox'][2] * ann['bbox'][3] <= 16.0:  # 太小的人脸忽略
                                    kps_mask[k, j * 2: j * 2 + 2] = 0  ###

                    #13. Drawing Object Center Heatmap
                    draw_gaussian(hm[cls_id], ct_int, radius)  # heatmap人脸框的heatmap
                    gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                                ct[0] + w / 2, ct[1] + h / 2, 1] +
                                pts[:, :2].reshape(num_joints * 2).tolist() + [cls_id])
            #14. Handling Rotations
            if rot != 0:
                hm = hm * 0 + 0.9999
                reg_mask *= 0
                kps_mask *= 0


            #15. Preparing the Output Dictionary

            ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh,
                'landmarks': kps, 'hps_mask': kps_mask, 'wight_mask': wight_mask,
                'hp_one_center': hp_one_center, 'hp_one_wh': hp_one_wh, 'hm_hp_one': hm_hp_one,
                'hp_one_ind': hp_one_ind, 'hp_one_bbox': hp_one_bbox}
            
            '''
            # Assuming ret is your result dictionary
            inp = ret['input']  # The preprocessed image tensor
            landmarks = ret['landmarks']  # The keypoints
            hp_one_bbox = ret['hp_one_bbox']  # The bounding boxes

            # Convert inp from CHW to HWC format and then to RGB
            inp = inp.transpose(1, 2, 0)  # CHW to HWC
            inp = (inp * self.std + self.mean) * 255.0  # Unnormalize
            inp = inp.astype(np.uint8)
            inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

            # Draw bounding boxes and keypoints
            for i in range(hp_one_bbox.shape[0]):
                bbox = hp_one_bbox[i]
                # Draw bounding box
                cv2.rectangle(inp, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

                # Draw keypoints
                for j in range(0, landmarks.shape[1], 2):
                    x = int(landmarks[i, j])
                    y = int(landmarks[i, j + 1])
                    if x > 0 and y > 0:  # Check if the keypoint is valid
                        cv2.circle(inp, (x, y), 5, (255, 0, 0), -1)  # Draw keypoint

            # Save the final image
            cv2.imwrite('result_image.png', cv2.cvtColor(inp, cv2.COLOR_RGB2BGR))  # Convert back to BGR for saving

            '''

            #16. Adding Optional Targets
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
                ret.update({'hm_offset': reg})  # 人脸bbox中心点整数化的偏差
            if self.opt.hm_hp:
                ret.update({'hm_hp': hm_hp})
            if self.opt.reg_hp_offset:
                ret.update({'hp_offset': hp_offset, 'hp_ind': hp_ind, 'hp_mask': hp_mask})
            if self.opt.debug > 0 or not self.split == 'train':
                gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                    np.zeros((1, 40), dtype=np.float32)
                meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
                ret['meta'] = meta

            #17. Returning the Result
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
