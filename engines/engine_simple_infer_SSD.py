import warnings

import os

# from detectors.SSD_VGG16_Detector import SSDDetector
from detectors.SSD_UNET_Detector import SSDDetector
from utils.checkpoint import load_checkpoint
from cores.misc import get_classes
from cores.pre_processing.image_transform import ImageTransform
from utils.inference import inference_single, show_result

dir_path = os.path.dirname(os.getcwd())
checkpoint_file = dir_path+'/work_dirs/dental_711_w_pretrained_wt_fix/epoch_100.pth'
# checkpoint_file = dir_path+'/work_dirs/dental_711_unet/epoch_300.pth'
device = 'cuda:0'
img = dir_path+'/demo/example.jpg'
out_file = dir_path+'/demo/example_result.jpg'
# out_file = dir_path+'/demo/example_result_2.jpg'
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


def main():
    # build the model from a config file and a checkpoint file
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
    )

    # model = SSDDetector(
    #     pretrained=None,
    #     # basic
    #     input_size=(480, 320),
    #     num_classes=4,
    #     in_channels=(512, 256, 128, 64, 32),
    #     # anchor generate
    #     anchor_ratios=([2, 3], [2, 3], [2, 3], [2, 3], [2, 3]),
    #     anchor_strides=((16, 16), (8, 8), (4, 4), (2, 2), (1, 1)),
    #     basesize_ratios=(0.16, 0.13, 0.10, 0.06, 0.02),
    #     allowed_border=-1,
    #     # regression
    #     target_means=(.0, .0, .0, .0),
    #     target_stds=(0.1, 0.1, 0.2, 0.2),
    #     # box assign
    #     pos_iou_thr=0.5,
    #     neg_iou_thr=0.5,
    #     min_pos_iou=0.,
    #     gt_max_assign_all=False,
    #     # sampling
    #     sampling=False,
    #     # balancing the loss
    #     neg_pos_ratio=3,
    #     # loss
    #     smoothl1_beta=1.,
    #     # inference nms
    #     nms_pre=3000,
    #     score_thr=0.02,
    #     nms_cfg=['nms', 0.2, None],
    #     max_per_img=200,
    # )

    checkpoint = load_checkpoint(
        model=model,
        filename=checkpoint_file,
        map_location=None,
        strict=False,
        logger=None
    )

    if 'CLASSES' in checkpoint['meta']:
        warnings.warn('Class names are saved in the checkpoint.')
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        warnings.warn('Class names are called from dataset.')
        model.CLASSES = get_classes('coco')

    model.to(device)
    model.eval()

    img_transform = ImageTransform(
        mean=img_transform_cfg['mean'],
        std=img_transform_cfg['std'],
        to_rgb=img_transform_cfg['to_rgb'],
        size_divisor=img_transform_cfg['size_divisor'],
    )

    result = inference_single(
        model=model,
        img=img,
        img_transform=img_transform,
        scale=img_transform_cfg['scale'],
        flip=img_transform_cfg['flip'],
        resize_keep_ratio=img_transform_cfg['resize_keep_ratio'],
        rescale=True,
        device=next(model.parameters()).device,
    )
    print(result)

    show_result(img=img, result=result[0], class_names=model.CLASSES, score_thr=0.3, out_file=out_file)
    # show_result(img=img, result=result[0], class_names=model.CLASSES, score_thr=0.23, out_file=out_file)


if __name__ == '__main__':
    main()

