import random
import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
import math
import re
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
from utils.utils import Data_anchor_sample
from utils.Randaugmentations import Randaugment
from collections.abc import Container as container_abcs

np_str_obj_array_pattern = re.compile(r'[SaUO]')

_use_shared_memory = False

class MultiPoseDataset(data.Dataset):
    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def _polar_label(self, a, cen):
        r = math.sqrt(math.pow(a[0] - cen, 2) + math.pow(a[1] - cen, 2))
        r = r / (math.sqrt(math.pow(cen, 2) + math.pow(cen, 2)) / (cen * 2))
        theta = math.atan2(a[1] - cen, a[0] - cen) / math.pi * 180
        if a[1] <= cen:
            theta = (theta + 360) / (360 / (cen * 2))
        else:
            theta = theta / (360 / (cen * 2))
        return np.array([theta, r])

    def _rotate_img_bboxes(self, img, angle=5, scale=1.):
        w = img.shape[1]
        h = img.shape[0]
        rangle = np.deg2rad(angle)
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
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
        concat = np.vstack((point1, point2, point3, point4))
        concat = concat.astype(np.int32)
        rx, ry, rw, rh = cv2.boundingRect(concat)
        rx_min = rx
        ry_min = ry
        rx_max = rx + rw
        ry_max = ry + rh
        rot_bboxes = [rx_min, ry_min, rx_max, ry_max]
        rot_bboxes = np.array(rot_bboxes, dtype=np.float32)
        return rot_bboxes

    def _rotate_pts(self, rot_mat, pts):
        num_kpts = pts[:, 2].sum()
        rot_pts = list()
        if num_kpts == 0:
            rot_pts = pts
        else:
            for rotpts in pts:
                rotpts[:2] = np.dot(rot_mat, np.array([rotpts[0], rotpts[1], 1]))
                rot_pts.append(rotpts)
        rot_pts = np.array(rot_pts, dtype=float)
        return rot_pts

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = len(anns)
        if num_objs > self.max_objs:
            num_objs = self.max_objs
            anns = np.random.choice(anns, num_objs)

        img = cv2.imread(img_path)
        img, anns = Data_anchor_sample(img, anns)
        rand_step = random.randint(0, 35)
        rotation_angle = rand_step * 10
        rot_mat, img = self._rotate_img_bboxes(img, rotation_angle)

        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.0
        rot = 0

        flipped = False
        if self.split == 'train':
            if not self.opt.not_rand_crop:
                s = s * np.random.choice(np.arange(0.3, 1.2, 0.1))
                _border = s * np.random.choice([0.1, 0.2, 0.25])
                w_border = self._get_border(_border, img.shape[1])
                h_border = self._get_border(_border, img.shape[0])
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            else:
                sf = self.opt.scale
                cf = self.opt.shift
                c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            if np.random.random() < self.opt.aug_rot:
                rf = self.opt.rotate
                rot = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)

            if np.random.random() < self.opt.flip:
                flipped = True
                img = img[:, ::-1, :]
                c[0] = width - c[0] - 1

        trans_input = get_affine_transform(c, s, rot, [self.opt.input_res, self.opt.input_res])
        inp = cv2.warpAffine(img, trans_input, (self.opt.input_res, self.opt.input_res), flags=cv2.INTER_LINEAR)

        inp = (inp.astype(np.float32) / 255.)
        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)

        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        output_res = self.opt.output_res
        num_joints = self.num_joints
        trans_output_rot = get_affine_transform(c, s, rot, [output_res, output_res])
        trans_output = get_affine_transform(c, s, 0, [output_res, output_res])

        hm = np.zeros((self.num_classes, output_res, output_res), dtype=np.float32)
        hm_hp = np.zeros((num_joints, output_res, output_res), dtype=np.float32)
        dense_kps = np.zeros((num_joints, 2, output_res, output_res), dtype=np.float32)
        dense_kps_mask = np.zeros((num_joints, output_res, output_res), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        kps = np.zeros((self.max_objs, num_joints * 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        wight_mask = np.ones((self.max_objs), dtype=np.float32)
        kps_mask = np.zeros((self.max_objs, self.num_joints * 2), dtype=np.uint8)
        hp_mask = np.zeros((self.max_objs * num_joints), dtype=np.int64)
        hp_offset = np.zeros((self.max_objs * num_joints, 2), dtype=np.float32)
        hp_ind = np.zeros((self.max_objs * num_joints), dtype=np.int64)

        hp_one_bbox = np.zeros((self.max_objs, 4), dtype=np.float32)
        hp_one_center = np.zeros((self.max_objs, 2), dtype=np.float32)
        hp_one_wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        hm_hp_one = np.zeros((self.max_objs * num_joints, 64, 64), dtype=np.float32)
        hp_one_ind = np.zeros((self.max_objs * num_joints), dtype=np.int64)
        hp_one_offset = np.zeros((self.max_objs * num_joints, 2), dtype=np.float32)
        hp_one_mask = np.zeros((self.max_objs * num_joints), dtype=np.int64)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian

        gt_det = []
        for k in range(num_objs):
            ann = anns[k]
            bbox = self._coco_box_to_bbox(ann['bbox'])
            bbox = self._rotate_bboxes(rot_mat, bbox)
            cls_id = int(ann['category_id']) - 1
            pts = np.array(ann['keypoints'], np.float32).reshape(num_joints, 3)
            pts = self._rotate_pts(rot_mat, pts)
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
                pts[:, 0] = width - pts[:, 0] - 1
                for e in self.flip_idx:
                    pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()

            bbox_one = np.copy(bbox)
            pts_one = np.copy(pts)

            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox = np.clip(bbox, 0, output_res - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]

            if (h > 0 and w > 0) or (rot != 0):
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = self.opt.hm_gauss if self.opt.mse_loss else max(0, int(radius))
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                wh[k] = np.log(1. * w / 4), np.log(1. * h / 4)
                ind[k] = ct_int[1] * output_res + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1

                num_kpts = pts[:, 2].sum()
                if num_kpts == 0:
                    hm[cls_id, ct_int[1], ct_int[0]] = 0.9999

                hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                hp_radius = self.opt.hm_gauss if self.opt.mse_loss else max(0, int(hp_radius))

                if num_kpts != 0:
                    hp_one_bbox[k] = bbox
                    hp_one_center[k] = ct
                    hp_one_wh[k] = 1. * w, 1. * h

                    whmax = max(hp_one_wh[k][0], hp_one_wh[k][1])
                    x1 = hp_one_center[k][0] - whmax / 2
                    y1 = hp_one_center[k][1] - whmax / 2
                    x2 = hp_one_center[k][0] + whmax / 2
                    y2 = hp_one_center[k][1] + whmax / 2

                    hp_one_bbox[k][0] = x1
                    hp_one_bbox[k][1] = y1
                    hp_one_bbox[k][2] = x2
                    hp_one_bbox[k][3] = y2

                    ct_one = np.array([(bbox_one[0] + bbox_one[2]) / 2, (bbox_one[1] + bbox_one[3]) / 2], dtype=np.float32)
                    h_one, w_one = bbox_one[3] - bbox_one[1], bbox_one[2] - bbox_one[0]
                    scale_one = 64 / max(w_one, h_one)
                    center_one = [64 / 2, 64 / 2]
                    hp_one_radius = math.ceil(hp_radius / 2)
                    for j in range(num_joints):
                        if (pts[j, 2] > 0):
                            pts_one[j, :2] = center_one + (pts_one[j, :2] - ct_one) * scale_one
                            pts_one[j, :2] = self._polar_label(pts_one[j, :2], 32)
                            pts_one_int = pts_one[j, :2].astype(np.int32)
                            hp_one_offset[k * num_joints + j] = pts_one[j, :2] - pts_one_int
                            hp_one_ind[k * num_joints + j] = pts_one_int[1] * 64 + pts_one_int[0]
                            hp_one_mask[k * num_joints + j] = 1
                            draw_gaussian(hm_hp_one[k * num_joints + j], pts_one_int, hp_one_radius)

                for j in range(num_joints):
                    if pts[j, 2] > 0:
                        pts[j, :2] = affine_transform(pts[j, :2], trans_output_rot)
                        if 0 <= pts[j, 0] < output_res and 0 <= pts[j, 1] < output_res:
                            kps[k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int
                            kps_mask[k, j * 2: j * 2 + 2] = 1
                            pt_int = pts[j, :2].astype(np.int32)
                            hp_offset[k * num_joints + j] = pts[j, :2] - pt_int
                            hp_ind[k * num_joints + j] = pt_int[1] * output_res + pt_int[0]
                            hp_mask[k * num_joints + j] = 1
                            if self.opt.dense_hp:
                                draw_dense_reg(dense_kps[j], hm[cls_id], ct_int, pts[j, :2] - ct_int, radius, is_offset=True)
                                draw_gaussian(dense_kps_mask[j], ct_int, radius)
                            draw_gaussian(hm_hp[j], pt_int, hp_radius)
                            if ann['bbox'][2] * ann['bbox'][3] <= 16.0:
                                kps_mask[k, j * 2: j * 2 + 2] = 0

                draw_gaussian(hm[cls_id], ct_int, radius)
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2, ct[0] + w / 2, ct[1] + h / 2, 1] + pts[:, :2].reshape(num_joints * 2).tolist() + [cls_id])

        if rot != 0:
            hm = hm * 0 + 0.9999
            reg_mask *= 0
            kps_mask *= 0

        ret = {
            'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh,
            'landmarks': kps, 'hps_mask': kps_mask, 'wight_mask': wight_mask,
            'hp_one_center': hp_one_center, 'hp_one_wh': hp_one_wh, 'hm_hp_one': hm_hp_one,
            'hp_one_ind': hp_one_ind, 'hp_one_bbox': hp_one_bbox
        }

        if self.opt.dense_hp:
            dense_kps = dense_kps.reshape(num_joints * 2, output_res, output_res)
            dense_kps_mask = dense_kps_mask.reshape(num_joints, 1, output_res, output_res)
            dense_kps_mask = np.concatenate([dense_kps_mask, dense_kps_mask], axis=1)
            dense_kps_mask = dense_kps_mask.reshape(num_joints * 2, output_res, output_res)
            ret.update({'dense_hps': dense_kps, 'dense_hps_mask': dense_kps_mask})
            del ret['hps'], ret['hps_mask']

        if self.opt.reg_offset:
            ret.update({'hm_offset': reg})
        if self.opt.hm_hp:
            ret.update({'hm_hp': hm_hp})
        if self.opt.reg_hp_offset:
            ret.update({'hp_offset': hp_offset, 'hp_ind': hp_ind, 'hp_mask': hp_mask})

        if self.opt.debug > 0 or not self.split == 'train':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else np.zeros((1, 40), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
            ret['meta'] = meta

        return ret

def default_collate(batch):
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(error_msg_fmt.format(elem.dtype))
            return default_collate([torch.from_numpy(b) for b in batch])
        if elem.shape == ():
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
    elif isinstance(batch[0], tuple) and hasattr(batch[0], '_fields'):
        return type(batch[0])(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(batch[0], container_abcs.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError((error_msg_fmt.format(type(batch[0]))))

def multipose_collate(batch):
    objects_dims = [d.shape[0] for d in batch]
    index = objects_dims.index(max(objects_dims))

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
