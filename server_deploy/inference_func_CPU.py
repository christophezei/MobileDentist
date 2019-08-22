'''
@Description: In User Settings Edit
@Author: Yuan Liang 
@Date: 2019-08-08 16:16:25
@LastEditTime: 2019-08-08 17:30:08
@LastEditors: Fan, Hsuan-Wei 
'''
import os
from server_deploy.SSD_VGG16_Detector_server import SSDDetector
from utils.checkpoint import load_checkpoint
from cores.pre_processing.image_transform import ImageTransform
from utils.image_utils import to_tensor
import torch
import gc


class Detector(object):
    def __init__(self):
        self.device = 'cpu'
        dir_path = os.path.dirname(os.getcwd())
        checkpoint_file = dir_path + '/flask-app/MobileDentist/checkpoints/dental_711_w_pretrained_wt_fix/epoch_300.pth'
        self.img_transform_cfg = \
            dict(
                mean=[123.675, 116.28, 103.53],
                std=[1, 1, 1],
                to_rgb=True,
                size_divisor=None,
                scale=(480, 320),
                flip=False,
                resize_keep_ratio=False
            )
        # init model
        self.model = SSDDetector(
            pretrained=None,
            # basic
            input_size=(480, 320),
            num_classes=4,
            in_channels=(512, 1024, 512, 256, 256, 256),
            # anchor generate
            anchor_ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
            anchor_strides=((8, 8), (18, 19), (34, 36), (69, 64), (96, 107), (160, 320)),
            basesize_ratios=(0.02, 0.05, 0.08, 0.12, 0.15, 0.18),
            allowed_border=-1,
            # regression
            target_means=(.0, .0, .0, .0),
            target_stds=(0.1, 0.1, 0.2, 0.2),
            # box assign
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.,
            gt_max_assign_all=False,
            # sampling
            sampling=False,
            # balancing the loss
            neg_pos_ratio=3,
            # loss
            smoothl1_beta=1.,
            # inference nms
            nms_pre=-1,
            score_thr=0.02,
            nms_cfg=['nms', 0.45, None],
            max_per_img=200,
            device='cpu'
        )
        # load checkpoint
        self.checkpoint = load_checkpoint(
            model=self.model,
            filename=checkpoint_file,
            map_location='cpu',
            strict=False,
            logger=None
        )

    def inference_func_CPU(self, img):
        # load classes
        if 'CLASSES' in self.checkpoint['meta']:
            self.model.CLASSES = self.checkpoint['meta']['CLASSES']
        else:
            raise RuntimeError('classes not found in self.checkpoint file meta. ')
        self.model.to(self.device)
        self.model.eval()
        # preprocess img
        img_transform = ImageTransform(
            mean=self.img_transform_cfg['mean'],
            std=self.img_transform_cfg['std'],
            to_rgb=self.img_transform_cfg['to_rgb'],
            size_divisor=self.img_transform_cfg['size_divisor'],
        )
        # inference
        ori_shape = img.shape
        img, img_shape, pad_shape, scale_factor = img_transform(
            img=img,
            scale=self.img_transform_cfg['scale'],
            flip=self.img_transform_cfg['flip'],
            keep_ratio=self.img_transform_cfg['resize_keep_ratio'],
        )
        img = to_tensor(img).to(self.device).unsqueeze(0)
        img_meta = [
            dict(
                ori_shape=ori_shape,
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=self.img_transform_cfg['flip']
            )
        ]
        with torch.no_grad():
            result = self.model.forward_test(img=img, img_meta=img_meta, rescale=True)

        for x in locals().keys():
            del locals()[x]
        gc.collect()
        return result


detector = Detector()
