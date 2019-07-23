import warnings

import os

from detectors.SSD_VGG16_300_Detector import SSDDetector
from utils.checkpoint import load_checkpoint
from cores.misc import get_classes
from cores.pre_processing.image_transform import ImageTransform
from utils.inference import inference_single, show_result

dir_path = os.path.dirname(os.getcwd())
checkpoint_file = dir_path+'/checkpoints/ssd300_coco_vgg16_caffe_120e_20181221-84d7110b.pth'
device = 'cuda:0'
img = dir_path+'/demo/csdc.jpg'
out_file = dir_path+'/demo/result.jpg'
img_transform_cfg = \
    dict(
        mean=[123.675, 116.28, 103.53],
        std=[1, 1, 1],
        to_rgb=True,
        size_divisor=None,
        scale=(300, 300),
        flip=False,
        resize_keep_ratio=False
    )


def main():
    # build the model from a config file and a checkpoint file
    model = SSDDetector(
        pretrained=None
    )
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
        warnings.warn('Class names are not saved in the checkpoint\'s '
                      'meta data, use COCO classes by default.')
        model.CLASSES = get_classes('coco')

    model.to(device)
    model.eval()

    print(model)

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

    show_result(img=img, result=result[0], class_names=model.CLASSES, score_thr=0.3, out_file=out_file)


if __name__ == '__main__':
    main()

