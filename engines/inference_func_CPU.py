import os

from detectors.SSD_VGG16_Detector import SSDDetector
from utils.checkpoint import load_checkpoint
from cores.pre_processing.image_transform import ImageTransform
from utils.image_utils import to_tensor
import torch

dir_path = os.path.dirname(os.getcwd())
checkpoint_file = dir_path+'/work_dirs/dental_711_w_pretrained_wt_fix/epoch_100.pth'

device = 'cpu'

img_transform_cfg = \
    dict(
        mean=[123.675, 116.28, 103.53],
        std=[1, 1, 1],
        to_rgb=True,
        size_divisor=None,
        scale=(480, 320),
        flip=False,
        resize_keep_ratio=False
    )


def inference_func_CPU(img):

    # init model
    model = SSDDetector(
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
    checkpoint = load_checkpoint(
        model=model,
        filename=checkpoint_file,
        map_location=None,
        strict=False,
        logger=None
    )

    # load classes
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        raise RuntimeError('classes not found in checkpoint file meta. ')

    model.to(device)
    model.eval()

    # preprocess img
    img_transform = ImageTransform(
        mean=img_transform_cfg['mean'],
        std=img_transform_cfg['std'],
        to_rgb=img_transform_cfg['to_rgb'],
        size_divisor=img_transform_cfg['size_divisor'],
    )

    # inference
    ori_shape = img.shape
    img, img_shape, pad_shape, scale_factor = img_transform(
        img=img,
        scale=img_transform_cfg['scale'],
        flip=img_transform_cfg['flip'],
        keep_ratio=img_transform_cfg['resize_keep_ratio'],
    )

    img = to_tensor(img).to(device).unsqueeze(0)
    img_meta = [
        dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=img_transform_cfg['flip']
        )
    ]

    with torch.no_grad():
        result = model.forward_test(img=img, img_meta=img_meta, rescale=True)

    return result